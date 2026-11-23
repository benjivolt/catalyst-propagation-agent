from __future__ import annotations

import re
from typing import Any

OFFICIAL_SOURCE_TYPES = {
    "sec_edgar",
    "fda_press_announcements",
    "openfda_drugs",
    "clinicaltrials_gov",
    "pubmed",
}
RELATIONSHIP_SOURCE_TYPES = {"peer_relationship", "watchlist_peer_evidence"}
TRUSTED_SOURCE_TYPES = OFFICIAL_SOURCE_TYPES | RELATIONSHIP_SOURCE_TYPES | {"company_news", "manual", "gdelt_news"}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "was",
    "were",
    "are",
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
    "trial",
    "study",
    "results",
    "peer",
    "relationship",
    "evidence",
    "target",
    "program",
    "overlap",
    "indication",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS}


def claim_support_score(claim: str, doc: dict[str, Any] | None) -> float:
    if not claim or not doc:
        return 0.0
    claim_terms = _tokens(claim)
    if not claim_terms:
        return 0.0
    doc_text = " ".join(str(doc.get(k) or "") for k in ["title", "excerpt", "publisher", "source_type"])
    doc_terms = _tokens(doc_text)
    if not doc_terms:
        return 0.0
    overlap = claim_terms & doc_terms

    return min(1.0, len(overlap) / max(len(claim_terms), 1))


def _verification_status(source_doc: dict[str, Any] | None) -> str:
    if not source_doc:
        return "missing"
    return str(source_doc.get("verification_status") or "not_checked").lower()


def _verification_score(source_doc: dict[str, Any] | None) -> float:
    if not source_doc:
        return 0.0
    try:
        return float(source_doc.get("verification_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_verified_source(source_doc: dict[str, Any] | None) -> bool:
    status = _verification_status(source_doc)
    score = _verification_score(source_doc)
    return status in {"official_metadata_trusted", "url_verified"} or score >= 0.72


def source_tier(source_doc: dict[str, Any] | None) -> str:
    if not source_doc:
        return "missing"
    source_type = str(source_doc.get("source_type") or "").lower()
    publisher = str(source_doc.get("publisher") or "").lower()
    status = _verification_status(source_doc)
    verified = is_verified_source(source_doc)
    official_like = source_type in OFFICIAL_SOURCE_TYPES or any(
        x in publisher for x in ["fda", "sec", "clinicaltrials", "pubmed", "ncbi"]
    )
    if official_like and status != "fetch_failed":
        return (
            "official" if verified or status in {"not_checked", "official_metadata_trusted"} else "official_unverified"
        )
    if source_type in RELATIONSHIP_SOURCE_TYPES:
        return "relationship_verified" if verified else "relationship_unverified"
    if source_type == "manual":
        return "manual_verified" if verified else "manual_unverified"
    if source_type in TRUSTED_SOURCE_TYPES and status != "fetch_failed":
        return "trusted_verified" if verified else "trusted_unverified"
    return "discovery_verified" if verified else "discovery_unverified"


def _effect_claim_docs(effect: dict[str, Any], source_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for claim in effect.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        try:
            idx = int(claim.get("source_index") or 0)
        except (AttributeError, TypeError, ValueError):
            idx = 0
        doc = source_by_index.get(idx)
        if doc:
            docs.append(doc)
    return docs


def _distinct_source_count(docs: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    for doc in docs:
        key = str(doc.get("canonical_url") or doc.get("url") or doc.get("id") or "").strip().lower()
        if key:
            seen.add(key)
    return len(seen)


def _verified_url_count(docs: list[dict[str, Any]]) -> int:
    return sum(
        1
        for doc in docs
        if _verification_status(doc) == "url_verified" or source_tier(doc) == "official" or is_verified_source(doc)
    )


def apply_evidence_guardrails(analysis: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    source_by_index = {idx: doc for idx, doc in enumerate(source_docs, start=1)}
    official_count = sum(1 for doc in source_docs if source_tier(doc) == "official")
    verified_source_count = sum(1 for doc in source_docs if is_verified_source(doc))
    relationship_count = sum(1 for doc in source_docs if source_tier(doc).startswith("relationship"))
    verified_relationship_count = sum(1 for doc in source_docs if source_tier(doc) == "relationship_verified")
    trusted_count = sum(
        1
        for doc in source_docs
        if source_tier(doc)
        in {"official", "trusted_verified", "relationship_verified", "manual_verified", "discovery_verified"}
    )
    total_claims = 0
    cited_claims = 0
    weak_claims = 0
    unsupported_effects = 0
    peer_relationship_missing = 0
    peer_relationship_unverified = 0
    source_verification_gaps = 0
    peer_corroboration_gaps = 0
    high_confidence_without_verified_url = 0
    support_scores: list[float] = []

    for effect in analysis.get("effects", []) or []:
        claims = effect.get("claims", []) if isinstance(effect, dict) else []
        if not claims:
            unsupported_effects += 1
            effect["confidence"] = min(float(effect.get("confidence", 0.35) or 0.35), 0.35)
            effect["rationale"] = (
                str(effect.get("rationale", "")) + " No claim cites a source, so confidence is capped."
            ).strip()
            continue
        effect_scores: list[float] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            total_claims += 1
            try:
                source_index = int(claim.get("source_index") or 0)
            except (AttributeError, TypeError, ValueError):
                source_index = 0
            doc = source_by_index.get(source_index)
            if doc:
                cited_claims += 1
            lexical_score = claim_support_score(str(claim.get("claim") or ""), doc)
            verification_score = _verification_score(doc)
            tier = source_tier(doc)

            verification_floor = (
                0.80 if is_verified_source(doc) or tier == "official" else max(0.35, verification_score)
            )
            score = min(lexical_score, verification_floor)
            claim["support_score"] = round(score, 3)
            claim["lexical_support_score"] = round(lexical_score, 3)
            claim["source_tier"] = tier
            claim["source_verification_status"] = _verification_status(doc)
            claim["source_verification_score"] = round(verification_score, 3)
            support_scores.append(score)
            effect_scores.append(score)
            threshold = 0.08 if tier.startswith("relationship") else 0.12
            if score < threshold:
                weak_claims += 1
            if tier.endswith("unverified") or _verification_status(doc) in {
                "fetch_failed",
                "manual_unverified",
                "manual_url_unchecked",
                "relationship_url_unchecked",
                "discovery_url_unchecked",
            }:
                source_verification_gaps += 1
        docs = _effect_claim_docs(effect, source_by_index)
        distinct_sources = _distinct_source_count(docs)
        verified_or_official_sources = _verified_url_count(docs)

        try:
            current_conf = float(effect.get("confidence", 0.4) or 0.4)
        except (AttributeError, TypeError, ValueError):
            current_conf = 0.4
        if current_conf > 0.74 and verified_or_official_sources == 0:
            high_confidence_without_verified_url += 1
            effect["confidence"] = min(current_conf, 0.74)
            effect["rationale"] = (
                str(effect.get("rationale", "")) + " Capped: none of the cited sources is verified or official."
            ).strip()

        if str(effect.get("layer")) == "peer" and str(effect.get("readthrough_type")) in {
            "validation",
            "competitive",
            "partner",
        }:
            has_relationship_source = any(source_tier(doc).startswith("relationship") for doc in docs)
            has_verified_relationship_source = any(source_tier(doc) == "relationship_verified" for doc in docs)
            if not has_relationship_source:
                peer_relationship_missing += 1
                effect["confidence"] = min(float(effect.get("confidence", 0.45) or 0.45), 0.55)
                effect["rationale"] = (
                    str(effect.get("rationale", "")) + " Capped: no source documents the peer relationship."
                ).strip()
            elif not has_verified_relationship_source:
                peer_relationship_unverified += 1
                effect["confidence"] = min(float(effect.get("confidence", 0.45) or 0.45), 0.50)
                effect["rationale"] = (
                    str(effect.get("rationale", ""))
                    + " Capped until reviewed: the peer relationship source has not been verified."
                ).strip()

            if distinct_sources < 2 or verified_or_official_sources < 1:
                peer_corroboration_gaps += 1
                effect["confidence"] = min(float(effect.get("confidence", 0.45) or 0.45), 0.60)
                effect["rationale"] = (
                    str(effect.get("rationale", ""))
                    + " Capped: a peer read-through needs two distinct sources, one of them verified or official."
                ).strip()
        if not effect_scores or max(effect_scores) < 0.10:
            unsupported_effects += 1
            effect["confidence"] = min(float(effect.get("confidence", 0.4) or 0.4), 0.45)
            effect["rationale"] = (
                str(effect.get("rationale", ""))
                + " Capped until reviewed: the cited text only weakly supports the claims."
            ).strip()
        elif sum(effect_scores) / len(effect_scores) < 0.18:
            effect["confidence"] = min(float(effect.get("confidence", 0.5) or 0.5), 0.65)

    citation_coverage = cited_claims / total_claims if total_claims else 0.0
    avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.0
    if (
        unsupported_effects == 0
        and peer_relationship_missing == 0
        and peer_relationship_unverified == 0
        and citation_coverage >= 0.98
        and (official_count or verified_source_count)
        and avg_support >= 0.22
        and source_verification_gaps == 0
        and peer_corroboration_gaps == 0
        and high_confidence_without_verified_url == 0
    ):
        grade = "A"
    elif (
        unsupported_effects <= 1
        and peer_relationship_missing <= 1
        and citation_coverage >= 0.90
        and trusted_count
        and avg_support >= 0.16
        and source_verification_gaps <= max(2, total_claims // 2)
        and peer_corroboration_gaps <= 1
    ):
        grade = "B"
    elif citation_coverage >= 0.75 and total_claims:
        grade = "C"
    else:
        grade = "D"

    grade_caps = {"A": 0.95, "B": 0.82, "C": 0.68, "D": 0.50}
    try:
        analysis["overall_confidence"] = min(float(analysis.get("overall_confidence", 0.4) or 0.4), grade_caps[grade])
    except (AttributeError, TypeError, ValueError):
        analysis["overall_confidence"] = min(0.4, grade_caps[grade])
    requires_human_review = (
        grade != "A"
        or unsupported_effects > 0
        or weak_claims > 0
        or peer_relationship_missing > 0
        or peer_relationship_unverified > 0
        or source_verification_gaps > 0
        or peer_corroboration_gaps > 0
        or high_confidence_without_verified_url > 0
    )

    audit = {
        "evidence_grade": grade,
        "source_count": len(source_docs),
        "official_source_count": official_count,
        "relationship_source_count": relationship_count,
        "verified_relationship_source_count": verified_relationship_count,
        "trusted_source_count": trusted_count,
        "verified_source_count": verified_source_count,
        "claim_count": total_claims,
        "cited_claim_count": cited_claims,
        "citation_coverage": round(citation_coverage, 3),
        "average_claim_support": round(avg_support, 3),
        "weak_claim_count": weak_claims,
        "unsupported_effect_count": unsupported_effects,
        "peer_relationship_missing_count": peer_relationship_missing,
        "peer_relationship_unverified_count": peer_relationship_unverified,
        "source_verification_gap_count": source_verification_gaps,
        "peer_corroboration_gap_count": peer_corroboration_gaps,
        "high_confidence_without_verified_url_count": high_confidence_without_verified_url,
        "requires_human_review": requires_human_review,
        "guardrail": (
            "confidence capped when effects lacked citations, peer relationship evidence, "
            "multi-source corroboration, URL/source verification, or source support"
        ),
    }
    analysis["evidence_audit"] = audit
    return analysis
