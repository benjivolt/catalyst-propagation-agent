from __future__ import annotations

import json
import logging
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.config import settings
from app.db import execute, rows
from app.services.biomed_context import apply_biomedical_context_guardrails, extract_biomedical_context
from app.services.evidence import apply_evidence_guardrails, is_verified_source, source_tier
from app.services.openai_utils import openai_model_candidates
from app.services.openai_utils import output_text as openai_output_text
from app.services.openai_utils import response_usage as openai_response_usage
from app.services.utils import redact_secrets

logger = logging.getLogger(__name__)

ALLOWED_DIRECTIONS = {"positive", "negative", "neutral"}
ALLOWED_LAYERS = {"target", "indication", "peer"}
ALLOWED_READTHROUGH = {"validation", "competitive", "partner", "neutral", "none"}

SYSTEM_PROMPT = """You are a biotech catalyst propagation analyst.
Analyze one catalyst using ONLY the provided source snippets and watchlist/peer context.

Return JSON only.

Rules:
- Every material factual claim must cite a source_index from the provided sources.
- If the sources do not prove a factual claim, mark that part uncertain/neutral and say what evidence is missing.
- Confidence must reflect source quality, directness of evidence, target/indication overlap, and whether the
  peer relationship is explicit or inferred.
- Direction must be positive, negative, or neutral.
- Never assume blanket contagion across biotech. Peer read-through requires a stated or inferred relationship.
- Separate validation read-through from competitive read-through:
  * validation: same target/mechanism success or failure changes confidence in biology.
  * competitive: same indication event changes standard of care, product-profile bar, regulatory bar,
    or market positioning.
  * partner: contractual/economic relationship makes the event relevant.
  * neutral/none: overlap is weak, offsetting, or unsupported.
- Analyze the target, indication, and peer layers when supported.
- For same-target + same-indication peers, explicitly discuss offsetting validation vs competition and choose
  neutral when the net effect is unclear.
"""

USER_PROMPT_INSTRUCTIONS = (
    "Analyze the propagation effects now. Cite factual claims using source_index values above. "
    "If a peer relationship is inferred rather than explicit, keep confidence lower. "
    "Do not make high-confidence target or indication claims unless the checklist shows endpoints, safety, "
    "trial design, statistics, regulatory details, or deal terms that actually support them."
)

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["trial_readout", "fda_decision", "licensing_deal", "acquisition", "other_catalyst"],
        },
        "plain_language_summary": {"type": "string"},
        "overall_direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "effects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "layer": {"type": "string", "enum": ["target", "indication", "peer"]},
                    "entity": {"type": "string"},
                    "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "readthrough_type": {
                        "type": "string",
                        "enum": ["validation", "competitive", "partner", "neutral", "none"],
                    },
                    "rationale": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "claim": {"type": "string"},
                                "source_index": {"type": "integer", "minimum": 1},
                            },
                            "required": ["claim", "source_index"],
                        },
                    },
                },
                "required": ["layer", "entity", "direction", "confidence", "readthrough_type", "rationale", "claims"],
            },
        },
    },
    "required": ["event_type", "plain_language_summary", "overall_direction", "overall_confidence", "effects"],
}

LLM_CALL_FLOOR_USD = 0.012


def _company_context(company: dict[str, Any]) -> str:
    keys = ("name", "ticker", "aliases", "targets", "indications", "peers", "inferred_peer_count", "notes")
    return json.dumps({k: company.get(k) for k in keys}, ensure_ascii=False, indent=2)


def _sources_context(source_docs: list[dict[str, Any]]) -> str:
    keys = ("title", "url", "publisher", "source_type", "published_at", "excerpt")
    compact = [{"source_index": idx, **{k: doc.get(k) for k in keys}} for idx, doc in enumerate(source_docs, start=1)]
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _fallback_analysis(
    company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]
) -> dict[str, Any]:
    from app.services.expert_fallback import expert_fallback_analysis

    return expert_fallback_analysis(company, catalyst, source_docs)


def _finalize(
    data: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]], fallback_claim: str
) -> dict[str, Any]:
    metadata = data.get("analysis_metadata", {})
    repaired = _repair_and_validate(
        data, catalyst.get("event_type", "other_catalyst"), len(source_docs), fallback_claim
    )
    repaired["analysis_metadata"] = metadata
    contextualized = apply_biomedical_context_guardrails(repaired, catalyst, source_docs)
    return apply_evidence_guardrails(contextualized, source_docs)


def _repair_and_validate(
    data: dict[str, Any],
    fallback_event_type: str,
    source_count: int = 0,
    fallback_claim: str = "Source-backed catalyst detected.",
) -> dict[str, Any]:
    data.setdefault("event_type", fallback_event_type)
    data.setdefault("plain_language_summary", "No summary returned.")
    data.setdefault("overall_direction", "neutral")
    data.setdefault("overall_confidence", 0.4)
    data.setdefault("effects", [])
    if data["event_type"] not in ANALYSIS_SCHEMA["properties"]["event_type"]["enum"]:
        data["event_type"] = fallback_event_type
    if data["overall_direction"] not in ALLOWED_DIRECTIONS:
        data["overall_direction"] = "neutral"
    try:
        data["overall_confidence"] = float(data["overall_confidence"])
    except (TypeError, ValueError):
        data["overall_confidence"] = 0.4
    data["overall_confidence"] = min(max(data["overall_confidence"], 0.0), 1.0)

    cleaned_effects = []
    for effect in data.get("effects", []):
        if not isinstance(effect, dict):
            continue
        layer = effect.get("layer", "peer")
        direction = effect.get("direction", "neutral")
        readthrough = effect.get("readthrough_type", "none")
        if layer not in ALLOWED_LAYERS:
            layer = "peer"
        if direction not in ALLOWED_DIRECTIONS:
            direction = "neutral"
        if readthrough not in ALLOWED_READTHROUGH:
            readthrough = "none"
        try:
            conf = float(effect.get("confidence", 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        claims = effect.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        clean_claims: list[dict[str, Any]] = []
        for c in claims:
            if not isinstance(c, dict) or not c.get("claim"):
                continue
            try:
                idx = int(c.get("source_index"))
            except (TypeError, ValueError):
                idx = 1
            if source_count:
                idx = min(max(idx, 1), source_count)
            clean_claims.append({"claim": str(c.get("claim"))[:500], "source_index": idx})
        if not clean_claims and source_count:
            clean_claims.append({"claim": fallback_claim[:500], "source_index": 1})
        cleaned_effects.append(
            {
                "layer": layer,
                "entity": str(effect.get("entity", "Unknown"))[:240],
                "direction": direction,
                "confidence": min(max(conf, 0.0), 1.0),
                "readthrough_type": readthrough,
                "rationale": str(effect.get("rationale", ""))[:2200],
                "claims": clean_claims,
            }
        )
    data["effects"] = cleaned_effects
    return data


def _record_openai_usage(
    catalyst_id: int,
    response: Any | None,
    status: str = "success",
    error: str = "",
    model: str | None = None,
    attempts: list[dict[str, str]] | None = None,
) -> None:
    usage = (
        openai_response_usage(response)
        if response is not None
        else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    used_model = model or settings.openai_model
    raw: dict[str, Any] = {"status": status, "model": used_model}
    if attempts:
        raw["model_attempts"] = attempts[-8:]
    if error:
        raw["error"] = redact_secrets(error)[:1000]
    cost = (usage["input_tokens"] / 1_000_000 * float(settings.openai_input_cost_per_1m)) + (
        usage["output_tokens"] / 1_000_000 * float(settings.openai_output_cost_per_1m)
    )
    if status == "success" and cost <= 0:
        cost = LLM_CALL_FLOOR_USD
    execute(
        """
        INSERT INTO llm_usage (
            catalyst_id, model, provider, input_tokens, output_tokens, total_tokens, estimated_cost, status, raw_json
        )
        VALUES (?, ?, 'openai', ?, ?, ?, ?, ?, ?)
        """,
        (
            catalyst_id,
            used_model,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
            float(cost),
            status,
            json.dumps(raw, ensure_ascii=False),
        ),
    )
    execute(
        """
        INSERT INTO budget_ledger (
            run_id, job_id, company_id, source_name, model, input_tokens, output_tokens, estimated_cost, reason
        )
        VALUES (NULL, NULL, NULL, 'openai_analysis', ?, ?, ?, ?, ?)
        """,
        (used_model, usage["input_tokens"], usage["output_tokens"], float(cost), f"actual_token_usage:{status}"),
    )


def _call_openai(user_prompt: str, catalyst_id: int, attempts: list[dict[str, str]]) -> tuple[str, str, bool]:
    client = OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=1)
    responses_api_used = hasattr(client, "responses")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    last_exc: Exception | None = None
    for model_name in openai_model_candidates(settings.openai_model):
        try:
            if responses_api_used:
                response = client.responses.create(
                    model=model_name,
                    input=messages,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "catalyst_propagation_report",
                            "schema": ANALYSIS_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                content = openai_output_text(response) or "{}"
            else:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content or "{}"
        except Exception as exc:
            last_exc = exc
            attempts.append({"model": model_name, "status": "failed", "error": redact_secrets(str(exc))[:400]})
            continue
        attempts.append({"model": model_name, "status": "ok"})
        _record_openai_usage(catalyst_id, response, status="success", model=model_name, attempts=attempts)
        return content, model_name, responses_api_used
    raise last_exc or RuntimeError("OpenAI analysis failed for all configured model candidates")


def analyze_catalyst(
    company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]
) -> dict[str, Any]:
    from app.services.scheduler import _budget_allows

    fallback_claim = catalyst.get("title") or (
        source_docs[0].get("title") if source_docs else "Source-backed catalyst detected."
    )
    base_metadata = {
        "openai_configured": bool(settings.openai_api_key),
        "openai_sdk_installed": OpenAI is not None,
        "openai_model": settings.openai_model,
        "strict_analysis": bool(settings.openai_strict_analysis),
    }

    def fallback(metadata: dict[str, Any], note: str = "") -> dict[str, Any]:
        data = _fallback_analysis(company, catalyst, source_docs)
        data["analysis_metadata"] = {**base_metadata, **metadata}
        if note:
            data["plain_language_summary"] += f" {note}"
        return _finalize(data, catalyst, source_docs, fallback_claim)

    if not settings.openai_api_key or OpenAI is None:
        missing = "OPENAI_API_KEY is not set" if not settings.openai_api_key else "the openai package is not installed"
        if settings.openai_strict_analysis:
            raise RuntimeError(f"OpenAI strict analysis is enabled, but {missing}.")
        note = (
            ""
            if not settings.openai_api_key
            else "The openai package is not installed, so this is a rule-based report."
        )
        return fallback({"provider_used": "fallback", "reason": missing}, note)

    if not _budget_allows(LLM_CALL_FLOOR_USD):
        return fallback(
            {"provider_used": "fallback_budget_cap", "reason": "daily LLM budget reached"},
            "Today's LLM budget is used up, so this is a rule-based report.",
        )

    clinical_context = extract_biomedical_context(catalyst, source_docs)
    user_prompt = "\n\n".join(
        [
            f"COMPANY WATCHLIST AND PEER-GRAPH CONTEXT:\n{_company_context(company)}",
            f"CATALYST:\n{json.dumps(catalyst, ensure_ascii=False, indent=2)}",
            "EXTRACTED CLINICAL / REGULATORY / DEAL FACT CHECKLIST:\n"
            + json.dumps(clinical_context, ensure_ascii=False, indent=2),
            f"SOURCES:\n{_sources_context(source_docs)}",
            USER_PROMPT_INSTRUCTIONS,
        ]
    )
    attempts: list[dict[str, str]] = []
    model_candidates = openai_model_candidates(settings.openai_model)
    catalyst_id = int(catalyst.get("id") or 0)
    try:
        content, used_model, responses_api_used = _call_openai(user_prompt, catalyst_id, attempts)
        data = json.loads(content)
    except Exception as exc:
        error = redact_secrets(str(exc))
        _record_openai_usage(catalyst_id, None, status="failed", error=error, attempts=attempts)
        if settings.openai_strict_analysis:
            raise RuntimeError(f"OpenAI analysis failed in strict mode: {error}") from exc
        return fallback(
            {
                "provider_used": "fallback_after_openai_error",
                "model_candidates": model_candidates,
                "model_attempts": attempts,
                "openai_error": error[:1000],
            },
            "The OpenAI call failed, so this is a rule-based report.",
        )
    data["analysis_metadata"] = {
        **base_metadata,
        "provider_used": "openai",
        "openai_model": used_model,
        "configured_primary_model": settings.openai_model,
        "model_candidates": model_candidates,
        "model_attempts": attempts,
        "responses_api_used": responses_api_used,
    }
    return _finalize(data, catalyst, source_docs, fallback_claim)


def save_report(catalyst_id: int, analysis: dict[str, Any]) -> int:
    source_docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    source_by_index = {idx: doc for idx, doc in enumerate(source_docs, start=1)}

    existing = rows("SELECT id FROM reports WHERE catalyst_id = ?", (catalyst_id,))
    for r in existing:
        execute("DELETE FROM reports WHERE id = ?", (r["id"],))
    execute("DELETE FROM source_correlations WHERE catalyst_id = ?", (catalyst_id,))

    report_id = execute(
        """
        INSERT INTO reports (catalyst_id, overall_direction, overall_confidence, plain_language_summary, raw_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            catalyst_id,
            analysis.get("overall_direction", "neutral"),
            float(analysis.get("overall_confidence", 0.4)),
            analysis.get("plain_language_summary", ""),
            json.dumps(analysis, ensure_ascii=False),
        ),
    )
    execute("UPDATE llm_usage SET report_id = ? WHERE catalyst_id = ? AND report_id IS NULL", (report_id, catalyst_id))
    try:
        from app.services.outcome_learning import attach_prediction_to_report

        attach_prediction_to_report(catalyst_id, report_id)
    except Exception:
        logger.exception("Could not link prediction to report %s", report_id)
    audit = analysis.get("evidence_audit") or {}
    if audit:
        execute(
            """
            INSERT INTO analysis_audits (report_id, evidence_grade, citation_coverage, unsupported_effect_count,
                weak_claim_count, audit_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                str(audit.get("evidence_grade", "D")),
                float(audit.get("citation_coverage", 0.0) or 0.0),
                int(audit.get("unsupported_effect_count", 0) or 0),
                int(audit.get("weak_claim_count", 0) or 0),
                json.dumps(audit, ensure_ascii=False),
            ),
        )
    review_status = (
        "approved" if audit.get("evidence_grade") == "A" and not audit.get("requires_human_review") else "needs_review"
    )
    execute(
        """
        INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
        VALUES (?, ?, '', ?, CURRENT_TIMESTAMP)
        """,
        (
            report_id,
            review_status,
            "Waiting for review. Approve only after checking that the sources support each claim."
            if review_status == "needs_review"
            else "Approved automatically: every effect passed the evidence checks.",
        ),
    )

    for effect in analysis.get("effects", []):
        cited_docs_for_effect = []
        for claim_for_audit in effect.get("claims", []) or []:
            try:
                idx_for_audit = int(claim_for_audit.get("source_index"))
            except (TypeError, ValueError):
                idx_for_audit = 0
            doc_for_audit = source_by_index.get(idx_for_audit)
            if doc_for_audit:
                cited_docs_for_effect.append(doc_for_audit)
        distinct_keys = {str(d.get("canonical_url") or d.get("url") or d.get("id")) for d in cited_docs_for_effect}
        official_count = sum(1 for d in cited_docs_for_effect if source_tier(d) == "official")
        verified_count = sum(1 for d in cited_docs_for_effect if is_verified_source(d))
        execute(
            """
            INSERT INTO source_correlations (catalyst_id, entity, layer, distinct_source_count, official_source_count,
                verified_source_count, corroboration_status, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalyst_id,
                str(effect.get("entity", "Unknown"))[:240],
                str(effect.get("layer", "peer")),
                len([k for k in distinct_keys if k]),
                official_count,
                verified_count,
                "strong"
                if len([k for k in distinct_keys if k]) >= 2 and (official_count or verified_count)
                else "weak",
                json.dumps(
                    {"readthrough_type": effect.get("readthrough_type"), "direction": effect.get("direction")},
                    ensure_ascii=False,
                ),
            ),
        )

        effect_id = execute(
            """
            INSERT INTO effects (report_id, layer, entity, direction, confidence, readthrough_type, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                effect.get("layer", "peer"),
                effect.get("entity", "Unknown"),
                effect.get("direction", "neutral"),
                float(effect.get("confidence", 0.4)),
                effect.get("readthrough_type", "none"),
                effect.get("rationale", ""),
            ),
        )
        for claim in effect.get("claims", []):
            source_index = claim.get("source_index")
            try:
                source_index = int(source_index) if source_index is not None else None
            except (TypeError, ValueError):
                source_index = None
            doc = source_by_index.get(source_index) if source_index else None
            execute(
                """
                INSERT INTO claims (effect_id, claim, source_doc_id, citation_url)
                VALUES (?, ?, ?, ?)
                """,
                (
                    effect_id,
                    claim.get("claim", ""),
                    doc.get("id") if doc else None,
                    doc.get("url") if doc else None,
                ),
            )
    try:
        from app.services.verified_evidence_graph import build_verified_evidence_graph

        build_verified_evidence_graph(report_id)
    except Exception as exc:
        try:
            execute(
                """
                INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
                VALUES (?, 'needs_review', '', ?, CURRENT_TIMESTAMP)
                """,
                (report_id, f"Evidence verification failed, so this report needs a manual check: {str(exc)[:600]}"),
            )
        except Exception:
            logger.exception("Could not mark report %s for review", report_id)
    try:
        from app.services.outcome_learning import record_peer_effect_outcomes_for_report

        record_peer_effect_outcomes_for_report(report_id)
    except Exception:
        logger.exception("Could not record peer outcomes for report %s", report_id)
    execute("UPDATE catalysts SET status = 'analyzed' WHERE id = ?", (catalyst_id,))
    if settings.alerts_enabled:
        try:
            from app.services.alerting import send_report_alert

            send_report_alert(report_id)
        except Exception:
            logger.exception("Alert delivery failed for report %s", report_id)
    return report_id
