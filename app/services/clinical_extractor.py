from __future__ import annotations

import re
from typing import Any

PHASE_RE = re.compile(r"\bphase\s*(1/2|2/3|1|2|3|4|i/ii|ii/iii|i|ii|iii|iv)\b", re.I)
N_RE = re.compile(r"\b(?:n\s*=|enrolled|randomi[sz]ed|treated|patients?\s*\(?n\s*=?)\s*([0-9]{2,6})", re.I)
P_VALUE_RE = re.compile(r"\bp\s*(?:=|<|>|≤|>=|<=)\s*0?\.\d+", re.I)
HR_RE = re.compile(r"\b(?:hazard ratio|HR)\s*(?:=|of|:)?\s*(0?\.\d+|1\.\d+|\d+\.\d+)", re.I)
ORR_RE = re.compile(r"\b(?:ORR|overall response rate|response rate)\s*(?:=|of|:)?\s*([0-9]{1,3}(?:\.[0-9]+)?%)", re.I)
PFS_OS_RE = re.compile(r"\b(?:PFS|progression-free survival|OS|overall survival)\b[^.;]{0,80}", re.I)
EFFECT_SIZE_RE = re.compile(r"\b(?:improvement|reduction|increase|decrease|decline|lowered|raised)\b[^.;]{0,80}(?:[0-9]{1,3}(?:\.[0-9]+)?%|points?|months?)", re.I)
SAFETY_RE = re.compile(r"\b(?:adverse events?|AEs?|serious adverse events?|SAEs?|grade\s*[3-5]|death|deaths|discontinuations?|tolerability|safety profile|clinical hold)\b[^.;]{0,120}", re.I)
ENDPOINT_RE = re.compile(r"\b(?:primary endpoint|secondary endpoint|met endpoint|failed to meet|statistically significant|noninferiority|superiority|complete response|objective response|overall survival|progression-free survival|MACE|ACR20|ACR50|HbA1c|LDL|ALT|AST)\b[^.;]{0,120}", re.I)
DESIGN_RE = re.compile(r"\b(?:randomi[sz]ed|double-blind|placebo-controlled|controlled|open-label|single-arm|dose-ranging|multicenter|noninferiority|superiority)\b", re.I)
REGULATORY_RE = re.compile(r"\b(?:approved|approval|PDUFA|complete response letter|CRL|priority review|accelerated approval|breakthrough therapy|label|black box|boxed warning|advisory committee|FDA|EMA)\b[^.;]{0,120}", re.I)
DEAL_RE = re.compile(r"\b(?:upfront|milestone|royalt(?:y|ies)|collaboration|license|licensing|acquisition|merger|tender offer|cash|equity)\b[^.;]{0,120}", re.I)


def _blob(catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> str:
    parts = [str(catalyst.get("title") or ""), str(catalyst.get("summary") or ""), str(catalyst.get("event_type") or "")]
    for doc in source_docs:
        parts.extend([str(doc.get("title") or ""), str(doc.get("excerpt") or "")])
    return " ".join(parts)


def _find_all(regex: re.Pattern[str], text: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    for match in regex.finditer(text):
        value = match.group(0).strip()
        if value and value not in out:
            out.append(value[:240])
        if len(out) >= limit:
            break
    return out


def extract_clinical_facts(catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract concrete clinical/regulatory/deal facts from cited snippets.

    This is deliberately transparent and regex-based. It does not invent expert
    conclusions; it gives the analyzer/audit a checklist of endpoints, effect
    sizes, safety signals, trial design, regulatory actions, and deal terms that
    must be present before high-confidence propagation is allowed.
    """
    text = _blob(catalyst, source_docs)
    low = text.lower()
    phase = PHASE_RE.search(text)
    sample = N_RE.search(text)
    facts = {
        "phase": phase.group(0) if phase else "",
        "sample_size": sample.group(1) if sample else "",
        "p_values": _find_all(P_VALUE_RE, text),
        "hazard_ratios": _find_all(HR_RE, text),
        "response_rates": _find_all(ORR_RE, text),
        "survival_mentions": _find_all(PFS_OS_RE, text),
        "effect_size_mentions": _find_all(EFFECT_SIZE_RE, text),
        "endpoint_mentions": _find_all(ENDPOINT_RE, text),
        "safety_mentions": _find_all(SAFETY_RE, text),
        "design_mentions": _find_all(DESIGN_RE, text),
        "regulatory_mentions": _find_all(REGULATORY_RE, text),
        "deal_terms": _find_all(DEAL_RE, text),
    }
    facts["has_quantified_effect"] = bool(facts["p_values"] or facts["hazard_ratios"] or facts["response_rates"] or facts["effect_size_mentions"] or facts["sample_size"])
    facts["has_endpoint_detail"] = bool(facts["endpoint_mentions"])
    facts["has_safety_detail"] = bool(facts["safety_mentions"])
    facts["has_design_detail"] = bool(facts["design_mentions"])
    facts["has_regulatory_detail"] = bool(facts["regulatory_mentions"])
    facts["has_deal_terms"] = bool(facts["deal_terms"])
    
    facts["mentions_positive_stat_sig"] = "statistically significant" in low or bool(facts["p_values"])
    facts["mentions_failed_endpoint"] = "failed to meet" in low or "did not meet" in low or "missed" in low
    facts["mentions_clinical_hold_or_crl"] = "clinical hold" in low or "complete response letter" in low or " crl" in low
    return facts


def clinical_summary_sentence(facts: dict[str, Any]) -> str:
    pieces: list[str] = []
    if facts.get("phase"):
        pieces.append(f"phase={facts['phase']}")
    if facts.get("sample_size"):
        pieces.append(f"n={facts['sample_size']}")
    if facts.get("endpoint_mentions"):
        pieces.append("endpoint detail present")
    if facts.get("has_quantified_effect"):
        pieces.append("quantified effect/statistic present")
    if facts.get("safety_mentions"):
        pieces.append("safety detail present")
    if facts.get("design_mentions"):
        pieces.append("trial design detail present")
    if facts.get("regulatory_mentions"):
        pieces.append("regulatory detail present")
    if facts.get("deal_terms"):
        pieces.append("deal terms present")
    return "; ".join(pieces) if pieces else "no concrete clinical/regulatory/deal facts were extracted from the snippets"
