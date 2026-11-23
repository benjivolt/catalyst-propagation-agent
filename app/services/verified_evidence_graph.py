from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from app.db import execute, row, rows
from app.services.evidence import is_verified_source, source_tier
from app.services.utils import clean_text

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "than",
    "then",
    "there",
    "their",
    "will",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "because",
    "source",
    "company",
    "companies",
    "trial",
    "study",
    "results",
    "data",
    "drug",
    "therapy",
    "therapies",
    "program",
    "programs",
    "target",
    "targets",
    "indication",
    "indications",
    "effect",
    "effects",
    "readthrough",
    "read",
    "through",
    "positive",
    "negative",
    "neutral",
    "supports",
    "support",
    "may",
    "can",
    "likely",
    "potentially",
    "suggests",
    "evidence",
    "peer",
    "peers",
    "catalyst",
    "event",
    "announced",
    "announces",
    "reported",
    "reports",
}
_NEGATIVE_TERMS = {
    "failed",
    "failure",
    "missed",
    "terminated",
    "discontinued",
    "halted",
    "hold",
    "crl",
    "complete response",
    "not statistically significant",
    "did not meet",
    "safety concern",
    "adverse",
    "withdrawn",
    "rejected",
}
_POSITIVE_TERMS = {
    "approved",
    "approval",
    "met",
    "statistically significant",
    "positive",
    "successful",
    "granted",
    "accepted",
    "priority review",
    "breakthrough",
    "clearance",
    "authorization",
    "acquire",
    "acquisition",
    "licensed",
}
_OFFICIAL_TYPES = {
    "sec_edgar",
    "fda_press_announcements",
    "openfda_drugs",
    "clinicaltrials_gov",
    "pubmed",
    "company_news",
}

SOURCE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "regulatory_fact": {
        "preferred": {"fda_press_announcements", "openfda_drugs", "company_news", "sec_edgar"},
        "max_without_preferred": 0.60,
    },
    "trial_result_fact": {
        "preferred": {"company_news", "clinicaltrials_gov", "pubmed", "sec_edgar"},
        "max_without_preferred": 0.65,
    },
    "deal_fact": {
        "preferred": {"sec_edgar", "company_news"},
        "max_without_preferred": 0.60,
    },
    "drug_target_fact": {
        "preferred": {"company_news", "clinicaltrials_gov", "pubmed", "peer_relationship", "openfda_drugs"},
        "max_without_preferred": 0.70,
    },
    "peer_relationship_fact": {
        "preferred": {
            "peer_relationship",
            "watchlist_peer_evidence",
            "company_news",
            "clinicaltrials_gov",
            "sec_edgar",
        },
        "max_without_preferred": 0.55,
    },
    "propagation_inference": {
        "preferred": {
            "company_news",
            "sec_edgar",
            "fda_press_announcements",
            "openfda_drugs",
            "peer_relationship",
            "clinicaltrials_gov",
            "pubmed",
        },
        "max_without_preferred": 0.65,
    },
    "background_fact": {
        "preferred": {
            "company_news",
            "clinicaltrials_gov",
            "pubmed",
            "peer_relationship",
            "sec_edgar",
            "openfda_drugs",
        },
        "max_without_preferred": 0.72,
    },
}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _tokens(text: Any) -> list[str]:
    raw = " ".join(str(x) for x in text) if isinstance(text, list) else str(text or "")
    out: list[str] = []
    for token in _TOKEN_RE.findall(raw.lower()):
        token = token.strip("-/")
        if len(token) > 1 and token not in _STOPWORDS:
            out.append(token)
    return out


def _token_set(text: Any) -> set[str]:
    return set(_tokens(text))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _raw_json(doc: dict[str, Any]) -> dict[str, Any]:
    raw = doc.get("raw_json") or {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


def _snapshot_text(doc: dict[str, Any]) -> str:
    raw = _raw_json(doc)
    raw_text = ""
    for key in ["text", "page_text", "content", "body", "abstract", "summary", "description"]:
        value = raw.get(key)
        if isinstance(value, str) and len(value.strip()) > len(raw_text):
            raw_text = value
    pieces = [str(doc.get("title") or ""), str(doc.get("publisher") or ""), str(doc.get("excerpt") or ""), raw_text]
    text = clean_text("\n\n".join(p for p in pieces if p), 50000)
    return text or clean_text(str(doc.get("title") or "Stored source document"), 2000)


def _paragraph_candidates(text: str) -> list[tuple[int, int, str]]:
    text = text or ""
    pieces: list[tuple[int, int, str]] = []
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^\n]+(?:\n+|$)", text):
        chunk = match.group(0).strip()
        if chunk:
            spans.append((match.start(), match.start() + len(match.group(0)), chunk))
    if len(spans) <= 2:
        offset = 0
        for sentence in _SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            start = text.find(sentence, offset)
            if start < 0:
                start = offset
            end = start + len(sentence)
            offset = end
            pieces.append((start, end, sentence))
    else:
        pieces = spans
    merged: list[tuple[int, int, str]] = []
    buffer_text = ""
    buffer_start = 0
    buffer_end = 0
    for start, end, piece in pieces:
        if not buffer_text:
            buffer_text, buffer_start, buffer_end = piece, start, end
            continue
        if len(buffer_text) < 220:
            buffer_text = f"{buffer_text} {piece}".strip()
            buffer_end = end
        else:
            merged.append((buffer_start, buffer_end, buffer_text[:2400]))
            buffer_text, buffer_start, buffer_end = piece, start, end
    if buffer_text:
        merged.append((buffer_start, buffer_end, buffer_text[:2400]))
    return merged or [(0, len(text), text[:2400])]


def _claim_type(claim: str, effect: dict[str, Any] | None = None) -> str:
    low = claim.lower()
    layer = str((effect or {}).get("layer") or "").lower()
    readthrough = str((effect or {}).get("readthrough_type") or "").lower()
    if any(x in low for x in ["fda", "approval", "approved", "crl", "complete response", "pdufa", "adcom", "label"]):
        return "regulatory_fact"
    if any(
        x in low
        for x in [
            "endpoint",
            "phase",
            "trial",
            "readout",
            "statistically",
            "p-value",
            "p value",
            "met",
            "failed",
            "safety",
        ]
    ):
        return "trial_result_fact"
    if any(
        x in low
        for x in ["acquisition", "acquire", "merger", "license", "licensing", "royalty", "milestone", "collaboration"]
    ):
        return "deal_fact"
    if layer == "peer" or readthrough in {"validation", "competitive", "partner"}:
        if any(
            x in low
            for x in ["peer", "competitor", "partner", "same-target", "same target", "same indication", "overlap"]
        ):
            return "peer_relationship_fact"
        return "propagation_inference"
    if layer in {"target", "indication"} or any(
        x in low for x in ["validates", "undermines", "biology", "standard of care", "competitive bar"]
    ):
        return "propagation_inference"
    if any(x in low for x in ["agonist", "inhibitor", "receptor", "target", "mechanism", "modality"]):
        return "drug_target_fact"
    return "background_fact"


def _required_tokens_for_claim(claim: str) -> set[str]:
    tokens = _token_set(claim)
    short_ok = {t for t in tokens if any(c.isdigit() for c in t) or t.isupper()}
    return {t for t in tokens if len(t) >= 3} | short_ok


def _source_quality(doc: dict[str, Any] | None) -> float:
    if not doc:
        return 0.05
    tier = source_tier(doc)
    verification_score = _safe_float(doc.get("verification_score"), 0.0)
    if tier == "official":
        return max(0.84, verification_score)
    if is_verified_source(doc):
        return max(0.78, verification_score)
    if tier in {"relationship_verified", "trusted_verified", "manual_verified"}:
        return max(0.72, verification_score)
    if str(doc.get("source_type") or "") in _OFFICIAL_TYPES:
        return max(0.66, verification_score)
    if tier.endswith("unverified"):
        return max(0.35, min(0.62, verification_score or 0.45))
    return max(0.25, min(0.58, verification_score or 0.38))


def _source_type_ok(claim_type: str, doc: dict[str, Any] | None) -> bool:
    if not doc:
        return False
    source_type = str(doc.get("source_type") or "").lower()
    preferred = SOURCE_REQUIREMENTS.get(claim_type, SOURCE_REQUIREMENTS["background_fact"])["preferred"]
    return source_type in preferred or source_tier(doc) == "official"


def _has_contradiction(claim: str, span: str) -> float:
    claim_low = claim.lower()
    span_low = span.lower()
    claim_positive = any(t in claim_low for t in _POSITIVE_TERMS) or any(
        t in claim_low for t in ["validates", "supports", "helps", "positive"]
    )
    claim_negative = any(t in claim_low for t in _NEGATIVE_TERMS) or any(
        t in claim_low for t in ["undermines", "hurts", "negative"]
    )
    span_negative = any(t in span_low for t in _NEGATIVE_TERMS)
    span_positive = any(t in span_low for t in _POSITIVE_TERMS)
    if claim_positive and span_negative and not span_positive:
        return 0.82
    if claim_negative and span_positive and not span_negative:
        return 0.76
    if "not " in span_low and any(t in claim_low for t in ["met", "approved", "validated", "validates"]):
        return 0.55
    return 0.0


def _coverage(claim: str, span: str) -> tuple[float, list[str]]:
    required = _required_tokens_for_claim(claim)
    if not required:
        return 0.0, []
    span_tokens = _token_set(span)
    covered = required & span_tokens
    coverage = len(covered) / max(len(required), 1)
    missing = sorted(required - covered)
    return min(1.0, coverage), missing[:12]


def _entity_match_score(claim: str, span: str) -> float:
    claim_tokens = _required_tokens_for_claim(claim)
    if not claim_tokens:
        return 0.0
    span_tokens = _token_set(span)
    important = {t for t in claim_tokens if any(ch.isdigit() for ch in t) or len(t) >= 5}
    if not important:
        important = claim_tokens
    return len(important & span_tokens) / max(len(important), 1)


def _retrieval_score(
    claim: str, paragraph: dict[str, Any], doc: dict[str, Any] | None, cited_source_doc_id: int | None
) -> float:
    coverage, _ = _coverage(claim, paragraph.get("text", ""))
    entity = _entity_match_score(claim, paragraph.get("text", ""))
    source_bonus = 0.12 if cited_source_doc_id and doc and int(doc.get("id") or 0) == cited_source_doc_id else 0.0
    quality = _source_quality(doc) * 0.18
    exact = 0.12 if clean_text(claim, 180).lower() in str(paragraph.get("text") or "").lower() else 0.0
    return min(1.0, coverage * 0.45 + entity * 0.33 + quality + source_bonus + exact)


def _verification_label(claim: str, span: str, claim_type: str, doc: dict[str, Any] | None) -> dict[str, Any]:
    coverage, missing = _coverage(claim, span)
    entity = _entity_match_score(claim, span)
    contradiction = _has_contradiction(claim, span)
    preferred_ok = _source_type_ok(claim_type, doc)
    entailment = min(1.0, coverage * 0.55 + entity * 0.30 + _source_quality(doc) * 0.15)
    if contradiction >= 0.55:
        label = "contradicted"
    elif coverage >= 0.62 and entity >= 0.55 and preferred_ok:
        label = "supported"
    elif coverage >= 0.48 and entity >= 0.40:
        label = "partially_supported"
    elif coverage >= 0.25 and entity >= 0.25:
        label = "not_enough_information"
    else:
        label = "unsupported"
    if label == "supported" and claim_type == "propagation_inference" and coverage < 0.80:
        label = "partially_supported"
    return {
        "label": label,
        "entailment_score": round(entailment, 3),
        "coverage_score": round(coverage, 3),
        "entity_match_score": round(entity, 3),
        "contradiction_score": round(contradiction, 3),
        "missing_elements": missing,
        "source_type_ok": preferred_ok,
    }


def _confidence_cap(
    label: str, claim_type: str, doc: dict[str, Any] | None, contradiction: float, has_quote: bool
) -> float:
    if not doc:
        return 0.20
    if not has_quote:
        return 0.45
    if contradiction >= 0.55 or label == "contradicted":
        return 0.35
    if label == "unsupported":
        return 0.35
    if label == "not_enough_information":
        return 0.50
    cap = 0.60 if label == "partially_supported" else 0.88
    req = SOURCE_REQUIREMENTS.get(claim_type, SOURCE_REQUIREMENTS["background_fact"])
    if not _source_type_ok(claim_type, doc):
        cap = min(cap, float(req.get("max_without_preferred", 0.65)))
    if not is_verified_source(doc) and source_tier(doc) != "official":
        cap = min(cap, 0.72)
    if claim_type == "peer_relationship_fact" and str(doc.get("source_type") or "") not in {
        "peer_relationship",
        "watchlist_peer_evidence",
    }:
        cap = min(cap, 0.55)
    if claim_type == "propagation_inference" and label != "supported":
        cap = min(cap, 0.65)
    return round(max(0.20, min(0.95, cap)), 3)


def create_source_snapshots_for_catalyst(catalyst_id: int) -> list[dict[str, Any]]:
    docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    snapshots: list[dict[str, Any]] = []
    for doc in docs:
        text = _snapshot_text(doc)
        content_hash = _hash(text)
        existing = row(
            "SELECT * FROM source_snapshots WHERE source_doc_id = ? AND content_hash = ? ORDER BY id DESC LIMIT 1",
            (doc["id"], content_hash),
        )
        if existing:
            snapshot = existing
        else:
            snapshot_id = execute(
                """
                INSERT INTO source_snapshots (
                    source_doc_id, url, source_type, title, publisher, fetched_at,
                    raw_html, raw_text, content_hash, archive_path, http_status, final_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["id"],
                    doc.get("url") or "",
                    doc.get("source_type") or "unknown",
                    doc.get("title") or "",
                    doc.get("publisher") or "",
                    _now(),
                    "",
                    text,
                    content_hash,
                    f"sqlite://source_snapshots/{content_hash}",
                    200 if doc.get("url") else None,
                    doc.get("canonical_url") or doc.get("url") or "",
                ),
            )
            snapshot = row("SELECT * FROM source_snapshots WHERE id = ?", (snapshot_id,)) or {}
        execute("DELETE FROM source_paragraphs WHERE snapshot_id = ?", (snapshot["id"],))
        for idx, (start, end, paragraph_text) in enumerate(_paragraph_candidates(text), start=1):
            execute(
                """
                INSERT INTO source_paragraphs (snapshot_id, paragraph_index, text, start_char, end_char, text_hash,
                    embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["id"],
                    idx,
                    paragraph_text,
                    start,
                    end,
                    _hash(paragraph_text),
                    f"local-token:{_hash(paragraph_text)[:20]}",
                ),
            )
        snapshots.append(snapshot)
    return snapshots


def _candidate_paragraphs(
    catalyst_id: int, claim: str, cited_source_doc_id: int | None, claim_type: str
) -> list[dict[str, Any]]:
    paragraphs = rows(
        """
        SELECT sp.*, ss.source_doc_id, ss.source_type, ss.title AS snapshot_title, ss.url AS snapshot_url,
               sd.title AS source_title, sd.publisher, sd.verification_status, sd.verification_score,
               sd.source_type AS doc_source_type, sd.url, sd.id AS doc_id
        FROM source_paragraphs sp
        JOIN source_snapshots ss ON ss.id = sp.snapshot_id
        JOIN source_docs sd ON sd.id = ss.source_doc_id
        WHERE sd.catalyst_id = ?
        """,
        (catalyst_id,),
    )
    scored: list[dict[str, Any]] = []
    doc_cache = {d["id"]: d for d in rows("SELECT * FROM source_docs WHERE catalyst_id = ?", (catalyst_id,))}
    for paragraph in paragraphs:
        doc_id = int(paragraph.get("source_doc_id") or 0)
        doc = doc_cache.get(doc_id)
        score = _retrieval_score(claim, paragraph, doc, cited_source_doc_id)
        if (
            claim_type in {"regulatory_fact", "deal_fact", "trial_result_fact"}
            and doc
            and _source_type_ok(claim_type, doc)
        ):
            score += 0.05
        item = dict(paragraph)
        item["retrieval_score"] = round(min(1.0, score), 3)
        item["doc"] = doc
        scored.append(item)
    scored.sort(key=lambda p: p["retrieval_score"], reverse=True)
    return scored[:8]


def _insert_atomic_claim(report_id: int, effect: dict[str, Any], claim: dict[str, Any], claim_type: str) -> int:
    text = clean_text(str(claim.get("claim") or ""), 1200)
    source_claim_id = int(claim.get("id") or 0)
    existing = row(
        "SELECT id FROM atomic_claims WHERE source_claim_id = ? AND atomic_claim = ?", (source_claim_id, text)
    )
    if existing:
        return int(existing["id"])
    return execute(
        """
        INSERT INTO atomic_claims (
            report_id, effect_id, source_claim_id, original_sentence, atomic_claim, claim_type,
            entity_company, entity_drug, entity_target, entity_indication, event_type, direction, importance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_id,
            effect.get("id"),
            source_claim_id,
            text,
            text,
            claim_type,
            "",
            "",
            str(effect.get("entity") or "") if str(effect.get("layer") or "") == "target" else "",
            str(effect.get("entity") or "") if str(effect.get("layer") or "") == "indication" else "",
            "",
            str(effect.get("direction") or "neutral"),
            "high"
            if claim_type
            in {"regulatory_fact", "trial_result_fact", "deal_fact", "peer_relationship_fact", "propagation_inference"}
            else "medium",
        ),
    )


def _final_status(best: dict[str, Any] | None, contradiction_score: float) -> str:
    if not best:
        return "unsupported"
    label = best.get("entailment_label")
    if contradiction_score >= 0.55 or label == "contradicted":
        return "contradicted"
    if label == "supported":
        return "verified"
    if label == "partially_supported":
        return "weakly_verified"
    if label == "not_enough_information":
        return "needs_human_review"
    return "unsupported"


def _summary_reason(status: str, best: dict[str, Any] | None, missing: list[str], source_type_ok: bool) -> str:
    if not best:
        return "No source paragraph was close enough to this claim to check it."
    parts = [
        f"Closest source passage: {str(best.get('entailment_label') or 'unknown').replace('_', ' ')} "
        f"(coverage {best.get('coverage_score'):.2f}, entailment {best.get('entailment_score'):.2f})."
    ]
    if missing:
        parts.append("Not found in the source: " + ", ".join(missing[:8]) + ".")
    if not source_type_ok:
        parts.append("This kind of claim needs a different type of source, so confidence is capped.")
    if status in {"unsupported", "contradicted", "needs_human_review"}:
        parts.append("Needs an analyst to check it before anyone relies on it.")
    return " ".join(parts)


def build_verified_evidence_graph(report_id: int) -> dict[str, Any]:
    report = row("SELECT * FROM reports WHERE id = ?", (report_id,))
    if not report:
        raise ValueError(f"Unknown report_id {report_id}")
    catalyst_id = int(report["catalyst_id"])
    create_source_snapshots_for_catalyst(catalyst_id)
    effects = rows("SELECT * FROM effects WHERE report_id = ? ORDER BY id", (report_id,))
    source_docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    source_by_id = {int(d["id"]): d for d in source_docs}
    execute(
        "DELETE FROM claim_evidence_spans WHERE claim_id IN (SELECT id FROM atomic_claims WHERE report_id = ?)",
        (report_id,),
    )
    execute(
        "DELETE FROM claim_verification_summary WHERE claim_id IN (SELECT id FROM atomic_claims WHERE report_id = ?)",
        (report_id,),
    )
    execute("DELETE FROM atomic_claims WHERE report_id = ?", (report_id,))

    total = 0
    verified = 0
    weak = 0
    unsupported = 0
    contradicted = 0
    needs_review = 0
    caps: list[float] = []
    per_effect_caps: dict[int, list[float]] = {}

    for effect in effects:
        claims = rows("SELECT * FROM claims WHERE effect_id = ? ORDER BY id", (effect["id"],))
        for claim in claims:
            total += 1
            text = str(claim.get("claim") or "")
            cited_doc_id = int(claim.get("source_doc_id") or 0) or None
            cited_doc = source_by_id.get(cited_doc_id or -1)
            claim_type = _claim_type(text, effect)
            atomic_id = _insert_atomic_claim(report_id, effect, claim, claim_type)
            candidates = _candidate_paragraphs(catalyst_id, text, cited_doc_id, claim_type)
            best_span: dict[str, Any] | None = None
            best_span_id: int | None = None
            best_missing: list[str] = []
            best_source_type_ok = False
            best_contradiction = 0.0
            for candidate in candidates[:5]:
                doc = candidate.get("doc") or source_by_id.get(int(candidate.get("source_doc_id") or 0))
                span_text = str(candidate.get("text") or "")
                verdict = _verification_label(text, span_text, claim_type, doc)
                best_contradiction = max(best_contradiction, float(verdict.get("contradiction_score") or 0.0))
                span_id = execute(
                    """
                    INSERT INTO claim_evidence_spans (
                        claim_id, source_doc_id, snapshot_id, paragraph_id, quote_text, start_char, end_char,
                        retrieval_score, entity_match_score, entailment_label, entailment_score,
                        contradiction_score, coverage_score, verifier_model, verification_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        atomic_id,
                        int(candidate.get("source_doc_id") or 0),
                        int(candidate.get("snapshot_id") or 0),
                        int(candidate.get("id") or 0),
                        span_text[:3000],
                        int(candidate.get("start_char") or 0),
                        int(candidate.get("end_char") or 0),
                        float(candidate.get("retrieval_score") or 0.0),
                        float(verdict.get("entity_match_score") or 0.0),
                        verdict.get("label"),
                        float(verdict.get("entailment_score") or 0.0),
                        float(verdict.get("contradiction_score") or 0.0),
                        float(verdict.get("coverage_score") or 0.0),
                        "local_verified_evidence_graph_v1",
                        json.dumps(
                            {
                                "missing_elements": verdict.get("missing_elements") or [],
                                "source_type_ok": verdict.get("source_type_ok"),
                                "claim_type": claim_type,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                score = (
                    float(verdict.get("entailment_score") or 0.0)
                    - float(verdict.get("contradiction_score") or 0.0)
                    + float(candidate.get("retrieval_score") or 0.0) * 0.15
                )
                current_best_score = -1.0
                if best_span:
                    current_best_score = (
                        float(best_span.get("entailment_score") or 0.0)
                        - float(best_span.get("contradiction_score") or 0.0)
                        + float(best_span.get("retrieval_score") or 0.0) * 0.15
                    )
                if score > current_best_score:
                    best_span = {
                        "id": span_id,
                        "entailment_label": verdict.get("label"),
                        "entailment_score": float(verdict.get("entailment_score") or 0.0),
                        "contradiction_score": float(verdict.get("contradiction_score") or 0.0),
                        "coverage_score": float(verdict.get("coverage_score") or 0.0),
                        "retrieval_score": float(candidate.get("retrieval_score") or 0.0),
                        "quote_text": span_text[:3000],
                        "doc": doc,
                    }
                    best_span_id = span_id
                    best_missing = verdict.get("missing_elements") or []
                    best_source_type_ok = bool(verdict.get("source_type_ok"))
            status = _final_status(best_span, best_contradiction)
            if status == "verified":
                verified += 1
            elif status == "weakly_verified":
                weak += 1
            elif status == "contradicted":
                contradicted += 1
            elif status == "unsupported":
                unsupported += 1
            else:
                needs_review += 1
            best_doc = best_span.get("doc") if best_span else cited_doc
            cap = _confidence_cap(
                str(best_span.get("entailment_label") if best_span else "unsupported"),
                claim_type,
                best_doc,
                best_contradiction,
                bool(best_span and best_span.get("quote_text")),
            )
            caps.append(cap)
            per_effect_caps.setdefault(int(effect["id"]), []).append(cap)
            publishable = 1 if status in {"verified", "weakly_verified"} else 0
            reason = _summary_reason(status, best_span, best_missing, best_source_type_ok)
            execute(
                """
                INSERT INTO claim_verification_summary (
                    claim_id, final_status, best_span_id, best_support_score, best_contradiction_score,
                    required_source_type, has_required_source_type, confidence_cap, publishable, failure_reason,
                        verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    atomic_id,
                    status,
                    best_span_id,
                    float(best_span.get("entailment_score") if best_span else 0.0),
                    best_contradiction,
                    ",".join(
                        sorted(SOURCE_REQUIREMENTS.get(claim_type, SOURCE_REQUIREMENTS["background_fact"])["preferred"])
                    ),
                    1 if best_source_type_ok else 0,
                    cap,
                    publishable,
                    reason,
                    _now(),
                ),
            )
            execute(
                """
                UPDATE claims
                SET verification_status = ?, verification_score = ?, confidence_cap = ?, exact_quote = ?,
                    source_paragraph_id = ?, verifier_reason = ?, verified_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    float(best_span.get("entailment_score") if best_span else 0.0),
                    cap,
                    str(best_span.get("quote_text") if best_span else "")[:2400],
                    int(best_span_id or 0) if best_span_id else None,
                    reason[:1200],
                    int(claim["id"]),
                ),
            )

    for effect_id, effect_caps in per_effect_caps.items():
        if not effect_caps:
            continue
        current = row("SELECT confidence, rationale FROM effects WHERE id = ?", (effect_id,))
        if not current:
            continue
        cap = max(0.25, min(effect_caps))
        new_conf = min(_safe_float(current.get("confidence"), 0.4), cap)
        rationale = str(current.get("rationale") or "")
        if "Claim check:" not in rationale and new_conf < _safe_float(current.get("confidence"), 0.4):
            rationale = (
                rationale
                + f" Claim check: capped at {cap:.2f} by how well the exact source passages support the claims."
            ).strip()
        execute("UPDATE effects SET confidence = ?, rationale = ? WHERE id = ?", (new_conf, rationale, effect_id))

    coverage = (verified + weak) / total if total else 0.0
    unsupported_rate = (unsupported + contradicted + needs_review) / total if total else 1.0
    avg_cap = sum(caps) / len(caps) if caps else 0.35
    grade = (
        "A"
        if coverage >= 0.95 and unsupported_rate == 0 and avg_cap >= 0.78
        else "B"
        if coverage >= 0.85 and unsupported_rate <= 0.10
        else "C"
        if coverage >= 0.65
        else "D"
    )
    report_cap = {"A": 0.92, "B": 0.80, "C": 0.65, "D": 0.48}[grade]
    execute("UPDATE reports SET overall_confidence = MIN(overall_confidence, ?) WHERE id = ?", (report_cap, report_id))
    audit = row("SELECT * FROM analysis_audits WHERE report_id = ?", (report_id,))
    veg_audit = {
        "verification_layer": "Verified Evidence Graph",
        "grade": grade,
        "atomic_claim_count": total,
        "verified_claim_count": verified,
        "weakly_verified_claim_count": weak,
        "unsupported_claim_count": unsupported,
        "contradicted_claim_count": contradicted,
        "needs_human_review_claim_count": needs_review,
        "claim_publishable_coverage": round(coverage, 3),
        "average_confidence_cap": round(avg_cap, 3),
        "report_confidence_cap": report_cap,
        "guardrail": (
            "claims require exact stored source spans, paragraph IDs, character offsets, entailment labels, "
            "contradiction risk checks, source-type requirements, and confidence caps"
        ),
    }
    if audit:
        try:
            payload = json.loads(audit.get("audit_json") or "{}")
        except (TypeError, ValueError):
            payload = {}
        payload["verified_evidence_graph"] = veg_audit
        combined_grade = min(str(audit.get("evidence_grade") or "D"), grade)
        execute(
            """
            UPDATE analysis_audits
            SET evidence_grade = ?, citation_coverage = ?, unsupported_effect_count = ?, weak_claim_count = ?,
                audit_json = ?
            WHERE report_id = ?
            """,
            (
                combined_grade,
                min(_safe_float(audit.get("citation_coverage"), 0.0), coverage)
                if _safe_float(audit.get("citation_coverage"), 0.0)
                else coverage,
                int(audit.get("unsupported_effect_count") or 0) + unsupported + contradicted,
                int(audit.get("weak_claim_count") or 0) + weak + needs_review,
                json.dumps(payload, ensure_ascii=False),
                report_id,
            ),
        )
    else:
        execute(
            """
            INSERT INTO analysis_audits (report_id, evidence_grade, citation_coverage, unsupported_effect_count,
                weak_claim_count, audit_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                grade,
                coverage,
                unsupported + contradicted,
                weak + needs_review,
                json.dumps({"verified_evidence_graph": veg_audit}, ensure_ascii=False),
            ),
        )
    if grade != "A":
        execute(
            """
            INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
            VALUES (?, 'needs_review', '', ?, CURRENT_TIMESTAMP)
            """,
            (
                report_id,
                "Some claims are weak, unsupported or contradicted by their sources. Check them before approving.",
            ),
        )
    return veg_audit


def evidence_graph_for_report(report_id: int) -> dict[str, Any]:
    summary = (
        row(
            """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN cvs.final_status = 'verified' THEN 1 ELSE 0 END) AS verified,
               SUM(CASE WHEN cvs.final_status = 'weakly_verified' THEN 1 ELSE 0 END) AS weak,
               SUM(CASE WHEN cvs.final_status = 'unsupported' THEN 1 ELSE 0 END) AS unsupported,
               SUM(CASE WHEN cvs.final_status = 'contradicted' THEN 1 ELSE 0 END) AS contradicted,
               SUM(CASE WHEN cvs.final_status = 'needs_human_review' THEN 1 ELSE 0 END) AS needs_review,
               AVG(cvs.confidence_cap) AS avg_cap
        FROM atomic_claims ac
        LEFT JOIN claim_verification_summary cvs ON cvs.claim_id = ac.id
        WHERE ac.report_id = ?
        """,
            (report_id,),
        )
        or {}
    )
    claims = rows(
        """
        SELECT ac.*, cvs.final_status, cvs.best_support_score, cvs.best_contradiction_score,
               cvs.confidence_cap, cvs.publishable, cvs.failure_reason,
               ces.quote_text, ces.start_char, ces.end_char, ces.paragraph_id,
               sd.title AS source_title, sd.url AS source_url, sd.publisher, sd.source_type
        FROM atomic_claims ac
        LEFT JOIN claim_verification_summary cvs ON cvs.claim_id = ac.id
        LEFT JOIN claim_evidence_spans ces ON ces.id = cvs.best_span_id
        LEFT JOIN source_docs sd ON sd.id = ces.source_doc_id
        WHERE ac.report_id = ?
        ORDER BY ac.id
        """,
        (report_id,),
    )
    return {"summary": summary, "claims": claims}
