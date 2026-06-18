from __future__ import annotations

import json
import re
from typing import Any

try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  

from app.config import settings
from app.db import execute, rows
from app.services.evidence import apply_evidence_guardrails, is_verified_source, source_tier
from app.services.biomed_context import apply_biomedical_context_guardrails, extract_biomedical_context
from app.services.peer_graph import indication_overlap_terms, target_overlap_terms
from app.services.utils import infer_direction
from app.services.openai_utils import openai_model_candidates, output_text as openai_output_text, response_usage as openai_response_usage

ALLOWED_DIRECTIONS = {"positive", "negative", "neutral"}
ALLOWED_LAYERS = {"target", "indication", "peer"}
ALLOWED_READTHROUGH = {"validation", "competitive", "partner", "neutral", "none"}

SYSTEM_PROMPT = """You are a biotech catalyst propagation analyst.
Analyze one catalyst using ONLY the provided source snippets and watchlist/peer context.

Return JSON only.

Rules:
- Every material factual claim must cite a source_index from the provided sources.
- If the sources do not prove a factual claim, mark that part uncertain/neutral and say what evidence is missing.
- Confidence must reflect source quality, directness of evidence, target/indication overlap, and whether the peer relationship is explicit or inferred.
- Direction must be positive, negative, or neutral.
- Never assume blanket contagion across biotech. Peer read-through requires a stated or inferred relationship.
- Separate validation read-through from competitive read-through:
  * validation: same target/mechanism success or failure changes confidence in biology.
  * competitive: same indication event changes standard of care, product-profile bar, regulatory bar, or market positioning.
  * partner: contractual/economic relationship makes the event relevant.
  * neutral/none: overlap is weak, offsetting, or unsupported.
- Analyze the target, indication, and peer layers when supported.
- For same-target + same-indication peers, explicitly discuss offsetting validation vs competition and choose neutral when the net effect is unclear.
"""

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

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_GENERIC = {
    "same", "target", "mechanism", "indication", "competitor", "partner", "therapy", "therapies", "disease",
    "trial", "study", "phase", "program", "data", "results", "company", "expected", "readthrough",
    "receptor", "receptors", "protein", "proteins", "hormone", "agonist", "agonists", "biology", "pathway",
}


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        text = " ".join(str(x) for x in value)
    else:
        text = str(value)
    return {t.strip("-/").lower() for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t.lower() not in _GENERIC}


def _overlap(left: Any, right: Any) -> set[str]:
    return _tokens(left) & _tokens(right)


def _company_context(company: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": company.get("name"),
            "ticker": company.get("ticker"),
            "aliases": company.get("aliases", []),
            "targets": company.get("targets", []),
            "indications": company.get("indications", []),
            "peers": company.get("peers", []),
            "inferred_peer_count": company.get("inferred_peer_count", 0),
            "notes": company.get("notes", ""),
        },
        ensure_ascii=False,
        indent=2,
    )


def _sources_context(source_docs: list[dict[str, Any]]) -> str:
    compact = []
    for idx, doc in enumerate(source_docs, start=1):
        compact.append(
            {
                "source_index": idx,
                "title": doc.get("title"),
                "url": doc.get("url"),
                "publisher": doc.get("publisher"),
                "source_type": doc.get("source_type"),
                "published_at": doc.get("published_at"),
                "excerpt": doc.get("excerpt"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _source_claim(catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if source_docs:
        title = catalyst.get("title") or source_docs[0].get("title") or "Catalyst source was added."
        return [{"claim": title, "source_index": 1}]
    return []


def _evidence_claim_from_source(source_docs: list[dict[str, Any]], fallback: str) -> list[dict[str, Any]]:
    if not source_docs:
        return []
    doc = source_docs[0]
    title = str(doc.get("title") or fallback or "Source-backed catalyst detected.")
    excerpt = str(doc.get("excerpt") or "")
    claim = title
    if excerpt and title.lower() not in excerpt.lower():
        claim = f"{title}: {excerpt[:220]}"
    return [{"claim": claim[:500], "source_index": 1}]


def _peer_relationship_flags(peer: dict[str, Any], company: dict[str, Any], text: str = "") -> tuple[bool, bool, bool, str]:
    relationship_type = str(peer.get("relationship_type") or peer.get("relationship") or "").lower()
    relation_blob = " ".join(
        str(peer.get(k, ""))
        for k in [
            "relationship", "relationship_type", "target", "indication", "expected_readthrough",
            "target_overlap", "indication_overlap", "source",
        ]
    ).lower()

    peer_targets = [peer.get("target", ""), peer.get("target_overlap", "")]
    peer_indications = [peer.get("indication", ""), peer.get("indication_overlap", "")]
    computed_target_overlap = target_overlap_terms(company.get("targets", []) + company.get("aliases", []), peer_targets)
    computed_indication_overlap = indication_overlap_terms(company.get("indications", []), peer_indications)

    declared_different_target = any(x in relation_blob for x in ["different target", "not same-target", "not same target", "unrelated target"])
    same_target = (
        ("same_target" in relationship_type or "same target" in relationship_type or "same-target" in relationship_type)
        or bool(peer.get("target_overlap"))
        or bool(computed_target_overlap)
    ) and not declared_different_target
    same_indication = (
        "same_indication" in relationship_type
        or "same indication" in relationship_type
        or "same-indication" in relationship_type
        or "competitor" in relationship_type
        or "competition" in relationship_type
        or bool(peer.get("indication_overlap"))
        or bool(computed_indication_overlap)
    )
    partner = any(x in relationship_type or x in relation_blob for x in ["partner", "collaboration", "royalt", "milestone", "licensee", "licensor", "commercial partner"])

    evidence = "source-cited" if peer.get("_relationship_source_index") else "relationship-source-missing"
    if same_target and same_indication:
        strength = f"{evidence}, mixed same-target/same-indication overlap"
    elif same_target:
        strength = f"{evidence}, same-target overlap"
    elif same_indication:
        strength = f"{evidence}, same-indication overlap"
    elif partner:
        strength = f"{evidence}, partner relationship"
    else:
        strength = f"{evidence}, weak/unsupported overlap"
    return same_target, same_indication, partner, strength


def _peer_claims(peer: dict[str, Any], catalyst_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    try:
        source_index = int(peer.get("_relationship_source_index") or 0)
    except Exception:
        source_index = 0
    if source_index > 0:
        peer_name = peer.get("company") or peer.get("name") or "Peer company"
        relationship = peer.get("relationship") or peer.get("relationship_type") or "peer relationship"
        target = peer.get("target") or peer.get("target_overlap") or "target not specified"
        indication = peer.get("indication") or peer.get("indication_overlap") or "indication not specified"
        claims.append({
            "claim": f"{peer_name} peer relationship evidence: {relationship}; target/program overlap: {target}; indication overlap: {indication}.",
            "source_index": source_index,
        })
    
    
    for claim in catalyst_claims[:1]:
        if isinstance(claim, dict) and claim.get("claim"):
            claims.append(claim)
    return claims or catalyst_claims


def _peer_effect(peer: dict[str, Any], event_direction: str, base_confidence: float, claim: list[dict[str, Any]], company: dict[str, Any] | None = None, catalyst_text: str = "") -> dict[str, Any]:
    company = company or {}
    peer_name = peer.get("company") or peer.get("name") or str(peer)
    same_target, same_indication, partner, strength = _peer_relationship_flags(peer, company, catalyst_text)
    peer_claims = _peer_claims(peer, claim)

    
    evidence_penalty = 0.0 if peer.get("_relationship_source_index") else 0.10

    if partner:
        return {
            "layer": "peer",
            "entity": peer_name,
            "direction": event_direction,
            "confidence": round(max(0.30, base_confidence - 0.12 - evidence_penalty), 2),
            "readthrough_type": "partner",
            "rationale": f"Partner/economic relationship detected ({strength}). The catalyst may propagate through shared program economics; exact magnitude needs contract terms.",
            "claims": peer_claims,
        }
    if same_target and same_indication:
        return {
            "layer": "peer",
            "entity": peer_name,
            "direction": "neutral",
            "confidence": round(0.46 - evidence_penalty, 2),
            "readthrough_type": "neutral",
            "rationale": f"Mixed read-through ({strength}): target biology may be validated/weakened, while same-indication competition can move the product bar in the opposite direction. Net direction is held neutral without stronger evidence.",
            "claims": peer_claims,
        }
    if same_target:
        return {
            "layer": "peer",
            "entity": peer_name,
            "direction": event_direction,
            "confidence": round(max(0.30, base_confidence - 0.18 - evidence_penalty), 2),
            "readthrough_type": "validation",
            "rationale": f"Validation read-through ({strength}): overlapping target/mechanism means the catalyst can support or weaken confidence in the underlying biology.",
            "claims": peer_claims,
        }
    if same_indication:
        if event_direction == "positive":
            peer_direction = "negative"
            rationale = f"Competitive read-through ({strength}): a positive catalyst in the same indication can raise the efficacy, safety, convenience, or regulatory bar for rivals."
        elif event_direction == "negative":
            peer_direction = "positive"
            rationale = f"Competitive read-through ({strength}): a negative catalyst for an indication competitor can reduce the immediate competitive threat, though it can also signal shared development risk."
        else:
            peer_direction = "neutral"
            rationale = f"Same-indication relationship detected ({strength}), but the catalyst direction is too uncertain to infer a strong competitive effect."
        return {
            "layer": "peer",
            "entity": peer_name,
            "direction": peer_direction,
            "confidence": round(max(0.25, base_confidence - 0.22 - evidence_penalty), 2),
            "readthrough_type": "competitive",
            "rationale": rationale,
            "claims": peer_claims,
        }
    return {
        "layer": "peer",
        "entity": peer_name,
        "direction": "neutral",
        "confidence": 0.25,
        "readthrough_type": "none",
        "rationale": "The watchlist/peer graph does not provide enough same-target, same-indication, or partner overlap to infer a reliable peer read-through.",
        "claims": claim,
    }

def _fallback_analysis(company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    
    
    from app.services.expert_fallback import expert_fallback_analysis

    return expert_fallback_analysis(company, catalyst, source_docs)


def _repair_and_validate(data: dict[str, Any], fallback_event_type: str, source_count: int = 0, fallback_claim: str = "Source-backed catalyst detected.") -> dict[str, Any]:
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
    except Exception:
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
        except Exception:
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
            except Exception:
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




def _response_usage(response: Any) -> dict[str, int]:
    return openai_response_usage(response)


def _record_openai_usage(catalyst_id: int, response: Any | None, status: str = "success", error: str = "", model: str | None = None, attempts: list[dict[str, str]] | None = None) -> None:
    usage = openai_response_usage(response) if response is not None else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    used_model = model or settings.openai_model
    raw = {"status": status, "model": used_model}
    if attempts:
        raw["model_attempts"] = attempts[-8:]
    if error:
        raw["error"] = error[:1000]
    cost = (usage["input_tokens"] / 1_000_000 * float(settings.openai_input_cost_per_1m)) + (
        usage["output_tokens"] / 1_000_000 * float(settings.openai_output_cost_per_1m)
    )
    execute(
        """
        INSERT INTO llm_usage (catalyst_id, model, provider, input_tokens, output_tokens, total_tokens, estimated_cost, status, raw_json)
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
        INSERT INTO budget_ledger (run_id, job_id, company_id, source_name, model, input_tokens, output_tokens, estimated_cost, reason)
        VALUES (NULL, NULL, NULL, 'openai_analysis', ?, ?, ?, ?, ?)
        """,
        (used_model, usage["input_tokens"], usage["output_tokens"], float(cost), f"actual_token_usage:{status}"),
    )


def _parse_response_text(response: Any) -> str:
    return openai_output_text(response)


def analyze_catalyst(company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_claim = catalyst.get("title") or (source_docs[0].get("title") if source_docs else "Source-backed catalyst detected.")
    if not settings.openai_api_key or OpenAI is None:
        if settings.openai_strict_analysis:
            missing = "OPENAI_API_KEY is not set" if not settings.openai_api_key else "openai package is not installed"
            raise RuntimeError(f"OpenAI strict analysis is enabled, but {missing}.")
        data = _fallback_analysis(company, catalyst, source_docs)
        data["analysis_metadata"] = {
            "provider_used": "fallback",
            "openai_configured": bool(settings.openai_api_key),
            "openai_sdk_installed": OpenAI is not None,
            "openai_model": settings.openai_model,
            "reason": "missing OPENAI_API_KEY" if not settings.openai_api_key else "openai package not installed",
            "strict_analysis": bool(settings.openai_strict_analysis),
        }
        if settings.openai_api_key and OpenAI is None:
            data["plain_language_summary"] += " The openai package is not installed, so fallback analysis was used."
        repaired = _repair_and_validate(data, catalyst.get("event_type", "other_catalyst"), len(source_docs), fallback_claim)
        repaired["analysis_metadata"] = data.get("analysis_metadata", {})
        contextualized = apply_biomedical_context_guardrails(repaired, catalyst, source_docs)
        return apply_evidence_guardrails(contextualized, source_docs)

    clinical_context = extract_biomedical_context(catalyst, source_docs)
    user_prompt = f"""
COMPANY WATCHLIST AND PEER-GRAPH CONTEXT:
{_company_context(company)}

CATALYST:
{json.dumps(catalyst, ensure_ascii=False, indent=2)}

EXTRACTED CLINICAL / REGULATORY / DEAL FACT CHECKLIST:
{json.dumps(clinical_context, ensure_ascii=False, indent=2)}

SOURCES:
{_sources_context(source_docs)}

Analyze the propagation effects now. Cite factual claims using source_index values above. If a peer relationship is inferred rather than explicit, keep confidence lower. Do not make high-confidence target or indication claims unless the checklist shows endpoints, safety, trial design, statistics, regulatory details, or deal terms that actually support them.
"""
    attempts: list[dict[str, str]] = []
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        responses_api_used = hasattr(client, "responses")
        last_exc: Exception | None = None
        response = None
        content = "{}"
        used_model = settings.openai_model
        for model_name in openai_model_candidates(settings.openai_model):
            try:
                if responses_api_used:
                    response = client.responses.create(
                        model=model_name,
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "catalyst_propagation_report",
                                "schema": ANALYSIS_SCHEMA,
                                "strict": True,
                            }
                        },
                    )
                    content = _parse_response_text(response) or "{}"
                else:  
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or "{}"
                used_model = model_name
                attempts.append({"model": model_name, "status": "ok"})
                _record_openai_usage(int(catalyst.get("id") or 0), response, status="success", model=used_model, attempts=attempts)
                break
            except Exception as model_exc:
                last_exc = model_exc
                attempts.append({"model": model_name, "status": "failed", "error": str(model_exc)[:400]})
        else:
            raise last_exc or RuntimeError("OpenAI analysis failed for all configured model candidates")
        data = json.loads(content)
        data["analysis_metadata"] = {
            "provider_used": "openai",
            "openai_configured": True,
            "openai_sdk_installed": True,
            "openai_model": used_model,
            "configured_primary_model": settings.openai_model,
            "model_candidates": openai_model_candidates(settings.openai_model),
            "model_attempts": attempts,
            "responses_api_used": responses_api_used,
            "strict_analysis": bool(settings.openai_strict_analysis),
        }
        repaired = _repair_and_validate(data, catalyst.get("event_type", "other_catalyst"), len(source_docs), fallback_claim)
        repaired["analysis_metadata"] = data.get("analysis_metadata", {})
        contextualized = apply_biomedical_context_guardrails(repaired, catalyst, source_docs)
        return apply_evidence_guardrails(contextualized, source_docs)
    except Exception as exc:
        _record_openai_usage(int(catalyst.get("id") or 0), None, status="failed", error=str(exc), attempts=attempts)
        if settings.openai_strict_analysis:
            raise RuntimeError(f"OpenAI analysis failed in strict mode: {exc}") from exc
        data = _fallback_analysis(company, catalyst, source_docs)
        data["analysis_metadata"] = {
            "provider_used": "fallback_after_openai_error",
            "openai_configured": True,
            "openai_sdk_installed": OpenAI is not None,
            "openai_model": settings.openai_model,
            "model_candidates": openai_model_candidates(settings.openai_model),
            "model_attempts": attempts,
            "openai_error": str(exc)[:1000],
            "strict_analysis": bool(settings.openai_strict_analysis),
        }
        data["plain_language_summary"] += f" OpenAI analysis failed and fallback was used: {exc}"
        repaired = _repair_and_validate(data, catalyst.get("event_type", "other_catalyst"), len(source_docs), fallback_claim)
        repaired["analysis_metadata"] = data.get("analysis_metadata", {})
        contextualized = apply_biomedical_context_guardrails(repaired, catalyst, source_docs)
        return apply_evidence_guardrails(contextualized, source_docs)


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
        pass
    audit = analysis.get("evidence_audit") or {}
    if audit:
        execute(
            """
            INSERT INTO analysis_audits (report_id, evidence_grade, citation_coverage, unsupported_effect_count, weak_claim_count, audit_json)
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
    review_status = "approved" if audit.get("evidence_grade") == "A" and not audit.get("requires_human_review") else "needs_review"
    execute(
        """
        INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
        VALUES (?, ?, '', ?, CURRENT_TIMESTAMP)
        """,
        (
            report_id,
            review_status,
            "Auto-created review gate. Approve only after checking source support." if review_status == "needs_review" else "Auto-approved by evidence gate.",
        ),
    )

    for effect in analysis.get("effects", []):
        
        
        
        cited_docs_for_effect = []
        for claim_for_audit in effect.get("claims", []) or []:
            try:
                idx_for_audit = int(claim_for_audit.get("source_index"))
            except Exception:
                idx_for_audit = 0
            doc_for_audit = source_by_index.get(idx_for_audit)
            if doc_for_audit:
                cited_docs_for_effect.append(doc_for_audit)
        distinct_keys = {str(d.get("canonical_url") or d.get("url") or d.get("id")) for d in cited_docs_for_effect}
        official_count = sum(1 for d in cited_docs_for_effect if source_tier(d) == "official")
        verified_count = sum(1 for d in cited_docs_for_effect if is_verified_source(d))
        execute(
            """
            INSERT INTO source_correlations (catalyst_id, entity, layer, distinct_source_count, official_source_count, verified_source_count, corroboration_status, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalyst_id,
                str(effect.get("entity", "Unknown"))[:240],
                str(effect.get("layer", "peer")),
                len([k for k in distinct_keys if k]),
                official_count,
                verified_count,
                "strong" if len([k for k in distinct_keys if k]) >= 2 and (official_count or verified_count) else "weak",
                json.dumps({"readthrough_type": effect.get("readthrough_type"), "direction": effect.get("direction")}, ensure_ascii=False),
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
            except Exception:
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
                (report_id, f"Verified Evidence Graph failed and needs review: {str(exc)[:700]}"),
            )
        except Exception:
            pass
    try:
        from app.services.outcome_learning import record_peer_effect_outcomes_for_report
        record_peer_effect_outcomes_for_report(report_id)
    except Exception:
        pass
    execute("UPDATE catalysts SET status = 'analyzed' WHERE id = ?", (catalyst_id,))
    if settings.alerts_enabled:
        try:
            from app.services.alerting import send_report_alert
            send_report_alert(report_id)
        except Exception:
            pass
    return report_id
