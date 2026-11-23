from __future__ import annotations

from typing import Any

from app.services.clinical_extractor import clinical_summary_sentence, extract_clinical_facts


def extract_biomedical_context(catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    facts = extract_clinical_facts(catalyst, source_docs)
    event_type = str(catalyst.get("event_type") or "other_catalyst")
    missing: list[str] = []

    if event_type == "trial_readout":
        if not facts.get("phase"):
            missing.append("trial phase")
        if not facts.get("has_endpoint_detail"):
            missing.append("endpoint detail")
        if not facts.get("has_safety_detail"):
            missing.append("safety/tolerability detail")
        if not facts.get("has_design_detail"):
            missing.append("trial design/control detail")
        if not facts.get("has_quantified_effect"):
            missing.append("sample size, p-value, effect size, HR, response rate, or other quantified statistic")
    elif event_type == "fda_decision":
        if not facts.get("has_regulatory_detail"):
            missing.append("specific FDA/regulatory action detail")
        if not facts.get("has_safety_detail") and any(
            "approved" in str(x).lower() for x in facts.get("regulatory_mentions", [])
        ):
            missing.append("label/safety detail")
    elif event_type in {"licensing_deal", "acquisition"} and not facts.get("has_deal_terms"):
        missing.append("deal terms such as upfront, milestones, royalty, cash/equity, or tender value")

    if not missing:
        level = "high"
    elif len(missing) <= 2:
        level = "moderate"
    else:
        level = "low"

    return {
        "event_type": event_type,
        "phase_detected": facts.get("phase", ""),
        "sample_size_detected": facts.get("sample_size", ""),
        "has_endpoint_detail": bool(facts.get("has_endpoint_detail")),
        "has_safety_detail": bool(facts.get("has_safety_detail")),
        "has_trial_design_detail": bool(facts.get("has_design_detail")),
        "has_p_value_or_statistic": bool(facts.get("has_quantified_effect")),
        "has_regulatory_detail": bool(facts.get("has_regulatory_detail")),
        "has_standard_of_care_detail": any(
            "standard of care" in str(x).lower() or "first-line" in str(x).lower()
            for x in facts.get("endpoint_mentions", [])
        ),
        "has_deal_terms": bool(facts.get("has_deal_terms")),
        "clinical_facts": facts,
        "clinical_summary": clinical_summary_sentence(facts),
        "missing_critical_fields": missing,
        "biomedical_evidence_level": level,
    }


def apply_biomedical_context_guardrails(
    analysis: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]
) -> dict[str, Any]:
    context = extract_biomedical_context(catalyst, source_docs)
    level = context["biomedical_evidence_level"]
    event_type = context["event_type"]
    if event_type not in {"trial_readout", "fda_decision", "licensing_deal", "acquisition"}:
        analysis["biomedical_context_audit"] = context
        return analysis

    caps = {"high": 0.93, "moderate": 0.76, "low": 0.56}
    effect_caps = {"high": 0.93, "moderate": 0.74, "low": 0.54}
    cap = caps[level]
    ecap = effect_caps[level]
    missing = ", ".join(context.get("missing_critical_fields") or [])
    clinical_summary = context.get("clinical_summary") or "no concrete clinical detail extracted"

    if level != "high":
        for effect in analysis.get("effects", []) or []:
            try:
                effect["confidence"] = min(float(effect.get("confidence", 0.4) or 0.4), ecap)
            except (AttributeError, TypeError, ValueError):
                effect["confidence"] = min(0.4, ecap)
            layer = str(effect.get("layer") or "")
            if layer in {"target", "indication", "peer"}:
                effect["rationale"] = (
                    str(effect.get("rationale", ""))
                    + f" Capped because the source detail is {level}. Found: {clinical_summary}."
                    + f" Missing: {missing or 'clinical or deal specifics'}."
                ).strip()
        try:
            analysis["overall_confidence"] = min(float(analysis.get("overall_confidence", 0.4) or 0.4), cap)
        except (AttributeError, TypeError, ValueError):
            analysis["overall_confidence"] = min(0.4, cap)
    else:
        for effect in analysis.get("effects", []) or []:
            effect["rationale"] = (
                str(effect.get("rationale", "")) + f" Source detail found: {clinical_summary}."
            ).strip()

    analysis["biomedical_context_audit"] = context
    return analysis
