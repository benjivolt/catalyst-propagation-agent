from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.db import execute, row, rows
from app.services.utils import classify_catalyst, clean_text, infer_direction, normalize_url, source_key

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


def domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host or "unknown"
    except (AttributeError, TypeError, ValueError):
        return "unknown"


def confidence_bucket(confidence: float) -> str:
    value = clamp(confidence)
    lower = math.floor(value * 10) / 10
    upper = min(lower + 0.1, 1.0)
    return f"{lower:.1f}-{upper:.1f}"


def source_reputation_key(source_type: str, publisher: str = "", url: str = "") -> str:
    domain = domain_from_url(url)
    source_type = (source_type or "unknown").lower()
    publisher = (publisher or domain or "unknown").strip().lower()
    if domain != "unknown":
        return f"{source_type}:{domain}"
    return f"{source_type}:{publisher}"


@dataclass
class CandidateReview:
    decision: str
    event_type: str = "none"
    confidence: float = 0.0
    direction: str = "unknown"
    company_detected: bool = False
    asset_detected: bool = False
    required_fields_present: bool = False
    missing_fields: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    evidence_needed_later: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    model_name: str = "deterministic_policy"

    def to_prediction_fields(self) -> dict[str, Any]:
        return {
            "predicted_decision": self.decision,
            "predicted_event_type": self.event_type,
            "predicted_direction": self.direction,
            "predicted_confidence": clamp(self.confidence),
            "model_name": self.model_name,
            "ai_reasoning_summary": clean_text(self.reasoning_summary, 1200),
            "risk_flags_json": safe_json_dumps(self.risk_flags),
            "evidence_needed_later_json": safe_json_dumps(self.evidence_needed_later),
        }


def learned_threshold(source_type: str, event_type: str, company_id: int | None = None) -> dict[str, float | str]:
    source_type = source_type or "unknown"
    event_type = event_type or "none"
    candidates = []
    if company_id:
        candidates.append((source_type, event_type, company_id))
    candidates.append((source_type, event_type, None))
    candidates.append((source_type, "all", None))
    for source, event, company in candidates:
        if company is None:
            item = row(
                "SELECT * FROM gate_policy_thresholds WHERE source_type = ? AND event_type = ? AND company_id IS NULL",
                (source, event),
            )
        else:
            item = row(
                "SELECT * FROM gate_policy_thresholds WHERE source_type = ? AND event_type = ? AND company_id = ?",
                (source, event, company),
            )
        if item:
            return {
                "promote_threshold": float(item.get("promote_threshold") or 0.80),
                "review_threshold": float(item.get("review_threshold") or 0.60),
                "context_threshold": float(item.get("context_threshold") or 0.40),
                "reason": str(item.get("policy_reason") or "Learned policy."),
            }
    return {
        "promote_threshold": 0.80,
        "review_threshold": 0.60,
        "context_threshold": 0.40,
        "reason": "Default learned threshold.",
    }


def reputation_score(
    source_type: str, event_type: str = "all", company_id: int | None = None, publisher: str = "", url: str = ""
) -> float:
    key = source_reputation_key(source_type, publisher, url)
    candidates = []
    if company_id:
        candidates.append((key, source_type, event_type, company_id))
    candidates.append((key, source_type, event_type, None))
    candidates.append((key, source_type, "all", None))
    candidates.append((f"{source_type}:global", source_type, event_type, None))
    for source_key_value, source, event, company in candidates:
        if company is None:
            item = row(
                "SELECT reliability_score FROM source_reputation WHERE source_key = ? AND source_type = ? "
                "AND event_type = ? AND company_id IS NULL",
                (source_key_value, source, event),
            )
        else:
            item = row(
                "SELECT reliability_score FROM source_reputation WHERE source_key = ? AND source_type = ? "
                "AND event_type = ? AND company_id = ?",
                (source_key_value, source, event, company),
            )
        if item:
            return clamp(float(item.get("reliability_score") or 0.5))
    defaults = {
        "sec_edgar": 0.88,
        "company_news": 0.82,
        "fda_press": 0.88,
        "openfda": 0.80,
        "clinicaltrials_gov": 0.62,
        "pubmed": 0.42,
        "gdelt_news": 0.38,
        "manual": 0.72,
    }
    return defaults.get((source_type or "").lower(), 0.50)


def calibration_multiplier(event_type: str, source_type: str, confidence: float) -> float:
    bucket = confidence_bucket(confidence)
    item = (
        row(
            "SELECT recommended_multiplier FROM confidence_calibration WHERE event_type = ? AND "
            "source_type = ? AND confidence_bucket = ?",
            (event_type or "all", source_type or "all", bucket),
        )
        or row(
            "SELECT recommended_multiplier FROM confidence_calibration WHERE event_type = 'all' AND "
            "source_type = ? AND confidence_bucket = ?",
            (source_type or "all", bucket),
        )
        or row(
            "SELECT recommended_multiplier FROM confidence_calibration WHERE event_type = 'all' AND "
            "source_type = 'all' AND confidence_bucket = ?",
            (bucket,),
        )
    )
    if not item:
        return 1.0
    return max(0.35, min(1.15, float(item.get("recommended_multiplier") or 1.0)))


def deterministic_candidate_review(company: dict[str, Any], candidate: Any, gate_score: float = 0.0) -> CandidateReview:
    text = (
        f"{getattr(candidate, 'title', '')} {getattr(candidate, 'publisher', '')} "
        f"{getattr(candidate, 'excerpt', '')} {getattr(candidate, 'url', '')}"
    ).lower()
    event_type = classify_catalyst(getattr(candidate, "title", ""), getattr(candidate, "excerpt", ""))
    source_type = getattr(candidate, "source_type", "unknown") or "unknown"
    company_terms = [str(company.get("name") or ""), str(company.get("ticker") or "")]
    company_terms += [str(x) for x in company.get("aliases") or []]
    company_terms = [x.lower() for x in company_terms if x and len(x.strip()) >= 2]
    company_detected = any(term in text for term in company_terms)
    asset_terms = [
        str(x).lower() for x in (company.get("aliases") or []) + (company.get("targets") or []) if str(x).strip()
    ]
    asset_detected = any(term in text for term in asset_terms if len(term) >= 3)
    direction, direction_conf = infer_direction(getattr(candidate, "title", ""), getattr(candidate, "excerpt", ""))

    missing: list[str] = []
    risk_flags: list[str] = []
    evidence_needed: list[str] = []
    strong_event_terms = {
        "trial_readout": [
            "met primary",
            "missed primary",
            "topline",
            "readout",
            "endpoint",
            "results",
            "statistically significant",
        ],
        "fda_decision": [
            "approved",
            "complete response",
            "crl",
            "pdufa",
            "advisory committee",
            "clinical hold",
            "label expansion",
        ],
        "licensing_deal": [
            "license",
            "licensing",
            "collaboration",
            "upfront",
            "milestone",
            "royalties",
            "exclusive rights",
        ],
        "acquisition": ["acquire", "acquisition", "merger", "definitive agreement", "tender offer", "all-cash"],
    }
    if event_type == "other_catalyst":
        event_type = "none"
        missing.append("recognized catalyst type")
    else:
        terms = strong_event_terms.get(event_type, [])
        if not any(term in text for term in terms):
            missing.append("explicit event action/result language")
    if not company_detected:
        missing.append("named watched company or ticker")
    if event_type == "trial_readout" and not any(x in text for x in ["phase", "nct", "trial", "study"]):
        missing.append("trial phase, NCT ID, or study identifier")
    if source_type in {"pubmed", "gdelt_news"}:
        risk_flags.append(f"{source_type}_is_noisy_for_direct_catalyst_detection")
        evidence_needed.append("official company, SEC, FDA, or trial-registry confirmation")
    if source_type == "company_news":
        risk_flags.append("company_source_can_overstate_interpretation")
        evidence_needed.append("independent or structured-source confirmation for strong propagation claims")

    rep = reputation_score(
        source_type,
        event_type,
        int(company.get("id") or 0) or None,
        getattr(candidate, "publisher", ""),
        getattr(candidate, "url", ""),
    )
    base_score = max(gate_score, 0.0)
    if company_detected:
        base_score += 0.16
    if asset_detected:
        base_score += 0.08
    if event_type != "none":
        base_score += 0.16
    if not missing:
        base_score += 0.10
    confidence = clamp((base_score * 0.72) + (rep * 0.28))
    policy = learned_threshold(source_type, event_type, int(company.get("id") or 0) or None)
    if not company_detected and source_type not in {"sec_edgar", "fda_press", "openfda"}:
        decision = "store_as_context" if event_type != "none" else "reject_irrelevant"
    elif missing and confidence >= float(policy["review_threshold"]):
        decision = "needs_ai_review"
    elif confidence >= float(policy["promote_threshold"]) and not missing:
        decision = "promote_to_catalyst"
    elif confidence >= float(policy["context_threshold"]):
        decision = "store_as_context"
    else:
        decision = "reject_irrelevant"

    return CandidateReview(
        decision=decision,
        event_type=event_type,
        confidence=confidence * calibration_multiplier(event_type, source_type, confidence),
        direction=direction if direction_conf >= 0.45 else "unknown",
        company_detected=company_detected,
        asset_detected=asset_detected,
        required_fields_present=not missing,
        missing_fields=missing,
        risk_flags=risk_flags,
        evidence_needed_later=evidence_needed,
        reasoning_summary=(
            f"Deterministic learning review: decision={decision}; event_type={event_type}; "
            f"company_detected={company_detected}; asset_detected={asset_detected}; "
            f"missing={missing or 'none'}; source_reputation={rep:.2f}; policy={policy.get('reason')}"
        ),
        model_name="deterministic_learning_policy",
    )


AI_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["promote_to_catalyst", "store_as_context", "reject_irrelevant", "needs_ai_review"],
        },
        "event_type": {
            "type": "string",
            "enum": ["trial_readout", "fda_decision", "licensing_deal", "acquisition", "none"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "direction": {"type": "string", "enum": ["positive", "negative", "neutral", "unknown"]},
        "company_detected": {"type": "boolean"},
        "asset_detected": {"type": "boolean"},
        "required_fields_present": {"type": "boolean"},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "evidence_needed_later": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "decision",
        "event_type",
        "confidence",
        "direction",
        "company_detected",
        "asset_detected",
        "required_fields_present",
        "missing_fields",
        "risk_flags",
        "evidence_needed_later",
        "reasoning_summary",
    ],
}


def ai_candidate_review(company: dict[str, Any], candidate: Any, gate_score: float = 0.0) -> CandidateReview:
    fallback = deterministic_candidate_review(company, candidate, gate_score)
    if not settings.openai_api_key or OpenAI is None:
        return fallback
    payload = {
        "company": {
            "id": company.get("id"),
            "name": company.get("name"),
            "ticker": company.get("ticker"),
            "aliases": company.get("aliases") or [],
            "targets": company.get("targets") or [],
            "indications": company.get("indications") or [],
        },
        "candidate": candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate),
        "deterministic_review": asdict(fallback),
        "learned_policy": learned_threshold(
            getattr(candidate, "source_type", "unknown"), fallback.event_type, int(company.get("id") or 0) or None
        ),
        "source_reputation": reputation_score(
            getattr(candidate, "source_type", "unknown"),
            fallback.event_type,
            int(company.get("id") or 0) or None,
            getattr(candidate, "publisher", ""),
            getattr(candidate, "url", ""),
        ),
    }
    system = (
        "You are a conservative biotech catalyst reviewer. Classify whether a source item is a real catalyst "
        "or only context. Only promote trial readouts, FDA decisions, licensing deals, and acquisitions. "
        "Do not promote broad PubMed papers, vague trial status pages, rumors, or target/indication-only matches. "
        "Return strict JSON only."
    )
    try:
        client = OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=1)
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "candidate_review",
                        "schema": AI_REVIEW_SCHEMA,
                        "strict": True,
                    }
                },
            )
            from app.services.openai_utils import output_text

            content = output_text(response) or "{}"
        else:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
        data = safe_json_loads(content, {})
        review = CandidateReview(
            decision=str(data.get("decision") or fallback.decision),
            event_type=str(data.get("event_type") or fallback.event_type),
            confidence=clamp(float(data.get("confidence", fallback.confidence))),
            direction=str(data.get("direction") or fallback.direction),
            company_detected=bool(data.get("company_detected", fallback.company_detected)),
            asset_detected=bool(data.get("asset_detected", fallback.asset_detected)),
            required_fields_present=bool(data.get("required_fields_present", fallback.required_fields_present)),
            missing_fields=[str(x) for x in data.get("missing_fields") or []],
            risk_flags=[str(x) for x in data.get("risk_flags") or []],
            evidence_needed_later=[str(x) for x in data.get("evidence_needed_later") or []],
            reasoning_summary=clean_text(str(data.get("reasoning_summary") or fallback.reasoning_summary), 1200),
            model_name=settings.openai_model,
        )
        if review.decision == "promote_to_catalyst" and (
            not review.company_detected or not review.required_fields_present
        ):
            review.risk_flags.append("ai_promotion_blocked_by_deterministic_validator")
            review.decision = "needs_ai_review" if review.company_detected else "store_as_context"
        return review
    except Exception as exc:
        fallback.risk_flags.append("openai_reviewer_failed_fallback_used")
        fallback.reasoning_summary = clean_text(f"{fallback.reasoning_summary} OpenAI reviewer failed: {exc}", 1200)
        return fallback


def schedule_outcome_checks(prediction_id: int) -> None:
    due_offsets = [24, 7 * 24, 30 * 24, 90 * 24]
    for hours in due_offsets:
        due = iso(utc_now() + timedelta(hours=hours))
        execute(
            """
            INSERT INTO ai_prediction_outcome_checks (prediction_id, due_at, status, check_type)
            VALUES (?, ?, 'pending', ?)
            """,
            (prediction_id, due, f"delayed_{hours}h"),
        )


def record_prediction(
    company_id: int,
    source_type: str,
    source_key_value: str,
    source_url: str,
    source_title: str,
    review: CandidateReview,
    gate_score: float = 0.0,
    candidate_id: int | None = None,
    catalyst_id: int | None = None,
    report_id: int | None = None,
    peer_effects: dict[str, Any] | None = None,
) -> int:
    fields = review.to_prediction_fields()
    existing = row(
        """
        SELECT id FROM ai_predictions
        WHERE company_id = ? AND source_key = ? AND predicted_decision = ? AND predicted_event_type = ?
        ORDER BY id DESC LIMIT 1
        """,
        (company_id, source_key_value or "", fields["predicted_decision"], fields["predicted_event_type"]),
    )
    if existing:
        prediction_id = int(existing["id"])
        execute(
            """
            UPDATE ai_predictions
            SET catalyst_id = COALESCE(?, catalyst_id), report_id = COALESCE(?, report_id), predicted_confidence = ?,
                gate_score = ?, ai_reasoning_summary = ?, risk_flags_json = ?, evidence_needed_later_json = ?,
                    updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                catalyst_id,
                report_id,
                float(fields["predicted_confidence"]),
                float(gate_score),
                fields["ai_reasoning_summary"],
                fields["risk_flags_json"],
                fields["evidence_needed_later_json"],
                prediction_id,
            ),
        )
        return prediction_id
    prediction_id = execute(
        """
        INSERT INTO ai_predictions (
            candidate_id, catalyst_id, report_id, company_id, source_type, source_key, source_url, source_title,
            predicted_decision, predicted_event_type, predicted_direction, predicted_confidence,
                predicted_peer_effects_json,
            model_name, gate_score, ai_reasoning_summary, risk_flags_json, evidence_needed_later_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            catalyst_id,
            report_id,
            company_id,
            source_type or "unknown",
            source_key_value or "",
            normalize_url(source_url or "") or source_url or "",
            clean_text(source_title or "", 500),
            fields["predicted_decision"],
            fields["predicted_event_type"],
            fields["predicted_direction"],
            float(fields["predicted_confidence"]),
            safe_json_dumps(peer_effects or {}),
            fields["model_name"],
            float(gate_score),
            fields["ai_reasoning_summary"],
            fields["risk_flags_json"],
            fields["evidence_needed_later_json"],
        ),
    )
    schedule_outcome_checks(prediction_id)
    return prediction_id


def record_scheduler_candidate_prediction(
    company: dict[str, Any],
    candidate: Any,
    gate_score: float,
    review: CandidateReview | None = None,
    catalyst_id: int | None = None,
) -> int:
    review = review or deterministic_candidate_review(company, candidate, gate_score)
    return record_prediction(
        company_id=int(company.get("id") or 0),
        source_type=getattr(candidate, "source_type", "unknown"),
        source_key_value=source_key(str(company.get("id") or ""), getattr(candidate, "key", "")),
        source_url=getattr(candidate, "url", ""),
        source_title=getattr(candidate, "title", ""),
        review=review,
        gate_score=gate_score,
        catalyst_id=catalyst_id,
    )


def attach_prediction_to_report(catalyst_id: int, report_id: int) -> None:
    execute(
        """
        UPDATE ai_predictions
        SET report_id = ?, updated_at = CURRENT_TIMESTAMP
        WHERE catalyst_id = ? AND (report_id IS NULL OR report_id = 0)
        """,
        (report_id, catalyst_id),
    )


def prediction_outcome_from_local_evidence(prediction: dict[str, Any]) -> dict[str, Any]:
    decision = str(prediction.get("predicted_decision") or "")
    catalyst_id = prediction.get("catalyst_id")
    report_id = prediction.get("report_id")
    evidence: list[dict[str, Any]] = []
    official_count = 0
    contradiction_count = 0
    label = "pending"
    confidence = 0.35
    status = "unknown"
    reason = "No delayed evidence checked yet."
    if catalyst_id:
        docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ?", (int(catalyst_id),))
        audit = row("SELECT * FROM analysis_audits WHERE report_id = ?", (int(report_id),)) if report_id else None
        for doc in docs:
            stype = str(doc.get("source_type") or "").lower()
            url = str(doc.get("url") or "")
            official = any(x in stype for x in ["sec", "fda", "company", "clinicaltrials", "openfda"]) or any(
                host in url.lower() for host in ["sec.gov", "fda.gov", "clinicaltrials.gov"]
            )
            if official:
                official_count += 1
            evidence.append(
                {
                    "source_doc_id": doc.get("id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "source_type": doc.get("source_type"),
                    "verification_status": doc.get("verification_status"),
                    "verification_score": doc.get("verification_score"),
                    "official_like": official,
                }
            )
        grade = str(audit.get("evidence_grade") or "") if audit else ""
        coverage = float(audit.get("citation_coverage") or 0.0) if audit else 0.0
        if decision == "promote_to_catalyst":
            if official_count and grade in {"A", "B"} and coverage >= 0.65:
                label = "true_positive"
                confidence = 0.88
                status = "confirmed"
                reason = "Promoted catalyst has official/structured source support and a strong evidence audit."
            elif docs and report_id:
                label = "partial_positive"
                confidence = 0.62
                status = "partially_confirmed"
                reason = "Promoted catalyst produced a report but evidence is not strong enough for full confirmation."
            else:
                label = "unsupported_prediction"
                confidence = 0.55
                status = "unsupported"
                reason = "Promoted candidate has not produced enough source/report evidence."
        else:
            if docs or report_id:
                label = (
                    "false_negative" if decision in {"reject_irrelevant", "store_as_context"} else "partial_positive"
                )
                confidence = 0.65
                status = "later_confirmed"
                reason = "A non-promoted or borderline prediction later became attached to catalyst/report evidence."
            else:
                label = "true_context" if decision == "store_as_context" else "true_negative"
                confidence = 0.58
                status = "not_confirmed_as_catalyst"
                reason = "No later local official/report evidence showed this item became a catalyst."
    else:
        if decision == "promote_to_catalyst":
            label = "unsupported_prediction"
            confidence = 0.50
            status = "no_catalyst_created"
            reason = "Promotion prediction never became a saved catalyst."
        elif decision == "store_as_context":
            label = "true_context"
            confidence = 0.55
            status = "not_confirmed_as_catalyst"
            reason = "Context item remained outside the catalyst table."
        else:
            label = "true_negative"
            confidence = 0.55
            status = "not_confirmed_as_catalyst"
            reason = "Rejected item remained outside the catalyst table."
    risk_flags = safe_json_loads(prediction.get("risk_flags_json"), [])
    if any("rumor" in str(flag).lower() for flag in risk_flags) and label in {
        "unsupported_prediction",
        "false_positive",
    }:
        contradiction_count += 1
    return {
        "outcome_label": label,
        "outcome_confidence": confidence,
        "confirmation_status": status,
        "official_confirmation_count": official_count,
        "contradiction_count": contradiction_count,
        "evidence_json": evidence,
        "verifier_model": "local_outcome_verifier_v1",
        "verifier_reason": reason,
    }


def save_prediction_outcome(prediction_id: int, outcome: dict[str, Any]) -> int:
    outcome_id = execute(
        """
        INSERT INTO prediction_outcomes (
            prediction_id, outcome_label, outcome_confidence, confirmation_status, official_confirmation_count,
            contradiction_count, evidence_json, verifier_model, verifier_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            outcome["outcome_label"],
            float(outcome.get("outcome_confidence") or 0.0),
            outcome.get("confirmation_status") or "unknown",
            int(outcome.get("official_confirmation_count") or 0),
            int(outcome.get("contradiction_count") or 0),
            safe_json_dumps(outcome.get("evidence_json") or []),
            outcome.get("verifier_model") or "local_outcome_verifier_v1",
            clean_text(outcome.get("verifier_reason") or "", 1500),
        ),
    )
    execute(
        """
        UPDATE ai_predictions
        SET outcome_status = 'checked', current_outcome_label = ?, last_outcome_checked_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (outcome["outcome_label"], prediction_id),
    )
    return outcome_id


def upsert_source_reputation_from_prediction(prediction: dict[str, Any], outcome: dict[str, Any]) -> None:
    source_type = prediction.get("source_type") or "unknown"
    event_type = prediction.get("predicted_event_type") or "all"
    source_key_value = source_reputation_key(source_type, "", prediction.get("source_url") or "")
    company_id = prediction.get("company_id")
    existing = row(
        """
        SELECT * FROM source_reputation
        WHERE source_key = ? AND source_type = ? AND event_type = ? AND company_id = ?
        """,
        (source_key_value, source_type, event_type, company_id),
    )
    if not existing:
        execute(
            """
            INSERT OR IGNORE INTO source_reputation (source_key, source_type, publisher, event_type, company_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_key_value,
                source_type,
                domain_from_url(prediction.get("source_url") or ""),
                event_type,
                company_id,
            ),
        )
        existing = row(
            """
            SELECT * FROM source_reputation
            WHERE source_key = ? AND source_type = ? AND event_type = ? AND company_id = ?
            """,
            (source_key_value, source_type, event_type, company_id),
        )
    label = outcome.get("outcome_label") or "pending"
    true_add = 1 if label == "true_positive" else 0
    false_add = 1 if label in {"false_positive", "unsupported_prediction"} else 0
    partial_add = 1 if label in {"partial_positive", "premature_prediction"} else 0
    contra_add = int(outcome.get("contradiction_count") or 0)
    unsupported_add = 1 if label == "unsupported_prediction" else 0
    false_negative_add = 1 if label == "false_negative" else 0
    totals = {
        "total": int(existing.get("total_candidates") or 0) + 1,
        "tp": int(existing.get("true_positives") or 0) + true_add,
        "fp": int(existing.get("false_positives") or 0) + false_add,
        "partial": int(existing.get("partial_positives") or 0) + partial_add,
        "contra": int(existing.get("contradicted_count") or 0) + contra_add,
        "unsupported": int(existing.get("unsupported_count") or 0) + unsupported_add,
        "fn": int(existing.get("false_negative_count") or 0) + false_negative_add,
    }
    useful = totals["tp"] + 0.5 * totals["partial"]
    noise = totals["fp"] + totals["unsupported"] + 0.75 * totals["contra"]
    conversion = useful / max(totals["total"], 1)
    reliability = clamp(
        0.35
        + 0.55 * conversion
        - 0.08 * (noise / max(totals["total"], 1))
        - 0.06 * (totals["fn"] / max(totals["total"], 1))
    )
    execute(
        """
        UPDATE source_reputation
        SET total_candidates = ?, true_positives = ?, false_positives = ?, partial_positives = ?,
            contradicted_count = ?,
            unsupported_count = ?, false_negative_count = ?, reliability_score = ?, catalyst_conversion_rate = ?,
                noise_score = ?,
            last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            totals["total"],
            totals["tp"],
            totals["fp"],
            totals["partial"],
            totals["contra"],
            totals["unsupported"],
            totals["fn"],
            reliability,
            conversion,
            clamp(noise / max(totals["total"], 1)),
            existing["id"],
        ),
    )


def update_confidence_calibration(prediction: dict[str, Any], outcome: dict[str, Any]) -> None:
    event_type = prediction.get("predicted_event_type") or "all"
    source_type = prediction.get("source_type") or "all"
    bucket = confidence_bucket(float(prediction.get("predicted_confidence") or 0.0))
    existing = row(
        "SELECT * FROM confidence_calibration WHERE event_type = ? AND source_type = ? AND confidence_bucket = ?",
        (event_type, source_type, bucket),
    )
    if not existing:
        execute(
            """
            INSERT OR IGNORE INTO confidence_calibration (event_type, source_type, confidence_bucket)
            VALUES (?, ?, ?)
            """,
            (event_type, source_type, bucket),
        )
        existing = row(
            "SELECT * FROM confidence_calibration WHERE event_type = ? AND source_type = ? AND confidence_bucket = ?",
            (event_type, source_type, bucket),
        )
    label = outcome.get("outcome_label") or "pending"
    correct_add = 1 if label in {"true_positive", "true_negative", "true_context"} else 0
    partial_add = 1 if label in {"partial_positive", "premature_prediction"} else 0
    count = int(existing.get("prediction_count") or 0) + 1
    correct = int(existing.get("correct_count") or 0) + correct_add
    partial = int(existing.get("partial_count") or 0) + partial_add
    observed = (correct + 0.5 * partial) / max(count, 1)
    midpoint = float(bucket.split("-")[0]) + 0.05
    error = observed - midpoint
    multiplier = clamp(1.0 + error, 0.45, 1.10)
    execute(
        """
        UPDATE confidence_calibration
        SET prediction_count = ?, correct_count = ?, partial_count = ?, observed_accuracy = ?, calibration_error = ?,
            recommended_multiplier = ?, last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (count, correct, partial, observed, error, multiplier, existing["id"]),
    )


def record_error_memory(prediction: dict[str, Any], outcome: dict[str, Any]) -> int | None:
    label = outcome.get("outcome_label") or ""
    source_type = prediction.get("source_type") or "unknown"
    event_type = prediction.get("predicted_event_type") or "unknown"
    error_type = ""
    pattern = ""
    rule = ""
    severity = 0.5
    if label == "unsupported_prediction":
        error_type = "unsupported_promotion"
        pattern = f"{source_type} candidate predicted as {prediction.get('predicted_decision')} without later support."
        rule = "Require stronger official evidence or lower confidence before promotion for this source/event pair."
        severity = 0.78
    elif label == "false_negative":
        error_type = "missed_catalyst"
        pattern = f"{source_type} candidate was not promoted but later gained catalyst evidence."
        rule = "Lower review threshold and route similar candidates to AI review instead of rejection."
        severity = 0.72
    elif label == "partial_positive":
        error_type = "partial_evidence"
        pattern = f"{source_type} candidate had some support but insufficient confirmation strength."
        rule = "Keep as review/context until official confirmation or exact citation evidence is available."
        severity = 0.45
    if not error_type:
        return None
    existing = row(
        """
        SELECT * FROM ai_error_memory
        WHERE error_type = ? AND event_type = ? AND source_type = ? AND pattern_description = ? AND active = 1
        """,
        (error_type, event_type, source_type, pattern),
    )
    if existing:
        execute(
            """
            UPDATE ai_error_memory
            SET seen_count = seen_count + 1, severity = ?, corrective_rule = ?, example_prediction_id = ?,
                last_seen_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (severity, rule, prediction.get("id"), existing["id"]),
        )
        return int(existing["id"])
    return execute(
        """
        INSERT INTO ai_error_memory (error_type, event_type, source_type, pattern_description, example_prediction_id,
            corrective_rule, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (error_type, event_type, source_type, pattern, prediction.get("id"), rule, severity),
    )


def optimize_threshold_for_source_event(
    source_type: str, event_type: str, company_id: int | None, run_id: int | None = None
) -> bool:
    stats = row(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN current_outcome_label = 'true_positive' THEN 1 ELSE 0 END) AS tp,
               SUM(CASE WHEN current_outcome_label IN ('unsupported_prediction','false_positive') THEN 1 ELSE 0 END)
                   AS fp,
               SUM(CASE WHEN current_outcome_label = 'false_negative' THEN 1 ELSE 0 END) AS fn,
               SUM(CASE WHEN current_outcome_label IN ('partial_positive','premature_prediction') THEN 1 ELSE 0 END)
                   AS partial
        FROM ai_predictions
        WHERE source_type = ? AND predicted_event_type = ? AND company_id = ? AND current_outcome_label != 'pending'
        """,
        (source_type, event_type, company_id),
    )
    if not stats or int(stats.get("n") or 0) < 3:
        return False
    n = int(stats.get("n") or 0)
    tp = int(stats.get("tp") or 0)
    fp = int(stats.get("fp") or 0)
    fn = int(stats.get("fn") or 0)
    partial = int(stats.get("partial") or 0)
    false_positive_rate = fp / max(tp + fp + partial, 1)
    false_negative_rate = fn / max(n, 1)
    precision = (tp + 0.5 * partial) / max(tp + fp + partial, 1)
    recall = tp / max(tp + fn, 1)
    existing = row(
        "SELECT * FROM gate_policy_thresholds WHERE source_type = ? AND event_type = ? AND company_id = ?",
        (source_type, event_type, company_id),
    )
    if not existing:
        execute(
            """
            INSERT OR IGNORE INTO gate_policy_thresholds (source_type, event_type, company_id)
            VALUES (?, ?, ?)
            """,
            (source_type, event_type, company_id),
        )
        existing = row(
            "SELECT * FROM gate_policy_thresholds WHERE source_type = ? AND event_type = ? AND company_id = ?",
            (source_type, event_type, company_id),
        )
    promote = float(existing.get("promote_threshold") or 0.80)
    review = float(existing.get("review_threshold") or 0.60)
    old_promote = promote
    if false_positive_rate > 0.25:
        promote = min(0.96, promote + 0.04)
        review = min(0.86, review + 0.02)
        reason = f"Raised thresholds because false-positive/unsupported rate is {false_positive_rate:.2f}."
    elif false_negative_rate > 0.18 and precision >= 0.50:
        promote = max(0.64, promote - 0.03)
        review = max(0.42, review - 0.04)
        reason = (
            f"Lowered review threshold because false-negative rate is {false_negative_rate:.2f} with "
            "acceptable precision."
        )
    elif precision > 0.80 and n >= 6:
        promote = max(0.66, promote - 0.02)
        reason = f"Slightly lowered promotion threshold because historical precision is {precision:.2f}."
    else:
        reason = (
            f"Policy refreshed without major change; precision={precision:.2f}, "
            f"fp_rate={false_positive_rate:.2f}, fn_rate={false_negative_rate:.2f}."
        )
    execute(
        """
        UPDATE gate_policy_thresholds
        SET promote_threshold = ?, review_threshold = ?, false_positive_rate = ?, false_negative_rate = ?,
            precision_score = ?, recall_score = ?,
            policy_reason = ?, last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (promote, review, false_positive_rate, false_negative_rate, precision, recall, reason, existing["id"]),
    )
    if abs(promote - old_promote) >= 0.015:
        execute(
            """
            INSERT INTO policy_updates (run_id, source_type, event_type, company_id, update_type, old_value,
                new_value, reason)
            VALUES (?, ?, ?, ?, 'promote_threshold', ?, ?, ?)
            """,
            (run_id, source_type, event_type, company_id, old_promote, promote, reason),
        )
    return True


def run_due_outcome_checks(limit: int = 50, include_not_due: bool = False) -> dict[str, Any]:
    now_iso = iso(utc_now())
    if include_not_due:
        checks = rows(
            """
            SELECT ch.id AS check_id, ch.prediction_id, ch.due_at,
                ch.status AS check_status FROM ai_prediction_outcome_checks ch
            JOIN ai_predictions p ON p.id = ch.prediction_id
            WHERE ch.status = 'pending'
            ORDER BY ch.due_at ASC LIMIT ?
            """,
            (limit,),
        )
    else:
        checks = rows(
            """
            SELECT ch.id AS check_id, ch.prediction_id, ch.due_at,
                ch.status AS check_status FROM ai_prediction_outcome_checks ch
            JOIN ai_predictions p ON p.id = ch.prediction_id
            WHERE ch.status = 'pending' AND ch.due_at <= ?
            ORDER BY ch.due_at ASC LIMIT ?
            """,
            (now_iso, limit),
        )
    processed = 0
    labels: dict[str, int] = {}
    for item in checks:
        prediction = row("SELECT * FROM ai_predictions WHERE id = ?", (item["prediction_id"],))
        if not prediction:
            execute(
                "UPDATE ai_prediction_outcome_checks SET status = 'failed', checked_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (item["check_id"],),
            )
            continue
        outcome = prediction_outcome_from_local_evidence(prediction)
        save_prediction_outcome(int(prediction["id"]), outcome)
        upsert_source_reputation_from_prediction(prediction, outcome)
        update_confidence_calibration(prediction, outcome)
        record_error_memory(prediction, outcome)
        execute(
            "UPDATE ai_prediction_outcome_checks SET status = 'checked', checked_at = "
            "CURRENT_TIMESTAMP, result_json = ? WHERE id = ?",
            (safe_json_dumps(outcome), item["check_id"]),
        )
        labels[outcome["outcome_label"]] = labels.get(outcome["outcome_label"], 0) + 1
        processed += 1
    return {"status": "ok", "processed": processed, "labels": labels, "due_remaining": len(checks) - processed}


def run_policy_update() -> dict[str, Any]:
    run_id = execute("INSERT INTO policy_update_runs (status) VALUES ('running')")
    checked = run_due_outcome_checks(limit=100, include_not_due=False)
    combos = rows(
        """
        SELECT DISTINCT source_type, predicted_event_type AS event_type, company_id
        FROM ai_predictions
        WHERE current_outcome_label != 'pending'
        """
    )
    threshold_updates = 0
    for combo in combos:
        if optimize_threshold_for_source_event(
            combo["source_type"], combo["event_type"], combo.get("company_id"), run_id=run_id
        ):
            threshold_updates += 1
    reputations = rows("SELECT COUNT(*) AS n FROM source_reputation")
    calibration = rows("SELECT COUNT(*) AS n FROM confidence_calibration")
    errors = rows("SELECT COUNT(*) AS n FROM ai_error_memory WHERE active = 1")
    summary = {
        "outcome_checks": checked,
        "source_event_company_combinations": len(combos),
        "threshold_updates": threshold_updates,
        "active_error_memories": int(errors[0]["n"] if errors else 0),
    }
    execute(
        """
        UPDATE policy_update_runs
        SET status = 'finished', finished_at = CURRENT_TIMESTAMP, predictions_reviewed = ?, thresholds_updated = ?,
            reputations_updated = ?, calibration_rows_updated = ?, error_memories_updated = ?, summary_json = ?
        WHERE id = ?
        """,
        (
            int(checked.get("processed") or 0),
            threshold_updates,
            int(reputations[0]["n"] if reputations else 0),
            int(calibration[0]["n"] if calibration else 0),
            int(errors[0]["n"] if errors else 0),
            safe_json_dumps(summary),
            run_id,
        ),
    )
    return {"status": "ok", "run_id": run_id, **summary}


def record_peer_effect_outcomes_for_report(report_id: int) -> int:
    effects = rows("SELECT * FROM effects WHERE report_id = ? AND layer = 'peer'", (report_id,))
    count = 0
    for effect in effects:
        existing = row(
            "SELECT id FROM peer_effect_outcomes WHERE report_id = ? AND effect_id = ?", (report_id, effect["id"])
        )
        if existing:
            continue
        confidence = float(effect.get("confidence") or 0.0)
        readthrough = str(effect.get("readthrough_type") or "none")
        evidence = rows(
            """
            SELECT c.*, sd.source_type, sd.url, sd.verification_status, sd.verification_score
            FROM claims c LEFT JOIN source_docs sd ON sd.id = c.source_doc_id
            WHERE c.effect_id = ?
            """,
            (effect["id"],),
        )
        official_like = sum(
            1
            for e in evidence
            if str(e.get("source_type") or "").lower()
            in {"sec_edgar", "company_news", "fda_press", "openfda", "clinicaltrials_gov", "peer_relationship"}
        )
        if readthrough == "none" or confidence < 0.35:
            label = "low_confidence_no_strong_readthrough"
            reason = "Peer effect was intentionally low-confidence or no-read-through."
            outcome_conf = 0.50
        elif official_like >= 2:
            label = "supported_by_relationship_and_catalyst_evidence"
            reason = "Peer effect cites multiple relationship/catalyst-supporting sources."
            outcome_conf = 0.72
        elif official_like == 1:
            label = "partially_supported"
            reason = "Peer effect has at least one structured or official source but no second source to back it up."
            outcome_conf = 0.58
        else:
            label = "needs_more_evidence"
            reason = "Peer effect lacks sufficient official or relationship-specific support."
            outcome_conf = 0.42
        execute(
            """
            INSERT INTO peer_effect_outcomes (
                report_id, effect_id, peer_name, predicted_direction, predicted_readthrough_type, predicted_confidence,
                outcome_label, outcome_confidence, outcome_evidence_json, observed_news_confirmation, verifier_reason,
                    checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                report_id,
                effect["id"],
                clean_text(effect.get("entity") or "", 240),
                effect.get("direction") or "neutral",
                readthrough,
                confidence,
                label,
                outcome_conf,
                safe_json_dumps(evidence[:8]),
                1 if official_like else 0,
                reason,
            ),
        )
        count += 1
    return count


def learning_dashboard() -> dict[str, Any]:
    prediction_counts = (
        row(
            """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN current_outcome_label = 'pending' THEN 1 ELSE 0 END) AS pending,
               SUM(CASE WHEN current_outcome_label = 'true_positive' THEN 1 ELSE 0 END) AS true_positive,
               SUM(CASE WHEN current_outcome_label IN ('false_positive','unsupported_prediction') THEN 1 ELSE 0 END)
                   AS false_positive,
               SUM(CASE WHEN current_outcome_label = 'false_negative' THEN 1 ELSE 0 END) AS false_negative
        FROM ai_predictions
        """
        )
        or {}
    )
    return {
        "prediction_counts": prediction_counts,
        "recent_predictions": rows("SELECT * FROM ai_predictions ORDER BY id DESC LIMIT 25"),
        "reputations": rows(
            "SELECT * FROM source_reputation ORDER BY reliability_score DESC, total_candidates DESC LIMIT 30"
        ),
        "thresholds": rows("SELECT * FROM gate_policy_thresholds ORDER BY last_updated DESC LIMIT 30"),
        "calibration": rows("SELECT * FROM confidence_calibration ORDER BY last_updated DESC LIMIT 30"),
        "errors": rows(
            "SELECT * FROM ai_error_memory WHERE active = 1 ORDER BY severity DESC, seen_count DESC LIMIT 30"
        ),
        "policy_runs": rows("SELECT * FROM policy_update_runs ORDER BY id DESC LIMIT 10"),
        "policy_updates": rows("SELECT * FROM policy_updates ORDER BY id DESC LIMIT 30"),
        "peer_outcomes": rows("SELECT * FROM peer_effect_outcomes ORDER BY id DESC LIMIT 30"),
        "due_checks": rows(
            "SELECT * FROM ai_prediction_outcome_checks WHERE status = 'pending' ORDER BY due_at ASC LIMIT 20"
        ),
    }
