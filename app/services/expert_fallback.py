from __future__ import annotations

"""Expert-grade deterministic fallback analysis for Catalyst Propagation.

The OpenAI path should produce the richest narrative when an API key is present.
This module exists for production resilience: when the model is unavailable, the
app still produces transparent, source-grounded propagation analysis using an
explicit clinical/regulatory/deal checklist rather than generic sentiment rules.
"""

import re
from typing import Any

from app.services.biomed_context import extract_biomedical_context
from app.services.clinical_extractor import extract_clinical_facts
from app.services.evidence import is_verified_source, source_tier
from app.services.peer_graph import indication_overlap_terms, target_overlap_terms
from app.services.utils import clean_text, infer_direction

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_GENERIC = {
    "same", "target", "mechanism", "indication", "competitor", "partner", "therapy", "therapies",
    "disease", "trial", "study", "phase", "program", "data", "results", "company", "expected",
    "readthrough", "receptor", "receptors", "protein", "proteins", "hormone", "agonist", "agonists",
    "biology", "pathway", "drug", "clinical", "patient", "patients", "treatment", "candidate",
}


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        text = " ".join(str(x) for x in value)
    else:
        text = str(value)
    return {t.strip("-/").lower() for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t.strip("-/").lower() not in _GENERIC}


def _blob(catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> str:
    parts = [str(catalyst.get("title") or ""), str(catalyst.get("summary") or ""), str(catalyst.get("event_type") or "")]
    for doc in source_docs:
        parts.extend([str(doc.get("title") or ""), str(doc.get("excerpt") or ""), str(doc.get("publisher") or ""), str(doc.get("source_type") or "")])
    return " ".join(parts)


def _source_claim(source_docs: list[dict[str, Any]], fallback: str, source_index: int = 1) -> list[dict[str, Any]]:
    if not source_docs:
        return []
    idx = min(max(int(source_index or 1), 1), len(source_docs))
    doc = source_docs[idx - 1]
    title = clean_text(doc.get("title") or fallback or "Source-backed catalyst detected", 260)
    excerpt = clean_text(doc.get("excerpt") or "", 220)
    claim = title if not excerpt or title.lower() in excerpt.lower() else f"{title}: {excerpt}"
    return [{"claim": claim[:500], "source_index": idx}]


def _specific_claim(source_index: int, claim: str, source_count: int) -> list[dict[str, Any]]:
    if source_count <= 0:
        return []
    return [{"claim": clean_text(claim, 500), "source_index": min(max(int(source_index or 1), 1), source_count)}]


def _best_catalyst_source_index(source_docs: list[dict[str, Any]]) -> int:
    """Prefer fetched/official catalyst sources over relationship-only docs."""
    if not source_docs:
        return 1
    for idx, doc in enumerate(source_docs, start=1):
        if str(doc.get("source_type") or "").lower() not in {"peer_relationship", "watchlist_peer_evidence"} and is_verified_source(doc):
            return idx
    for idx, doc in enumerate(source_docs, start=1):
        if str(doc.get("source_type") or "").lower() not in {"peer_relationship", "watchlist_peer_evidence"}:
            return idx
    return 1


def _evidence_quality(source_docs: list[dict[str, Any]], context: dict[str, Any]) -> float:
    if not source_docs:
        return 0.30
    verified = sum(1 for d in source_docs if is_verified_source(d))
    official = sum(1 for d in source_docs if source_tier(d) == "official")
    relationship = sum(1 for d in source_docs if source_tier(d).startswith("relationship"))
    clinical_bonus = {"high": 0.16, "moderate": 0.08, "low": -0.04}.get(str(context.get("biomedical_evidence_level")), 0.0)
    score = 0.46 + min(verified * 0.08, 0.18) + min(official * 0.07, 0.16) + min(max(len(source_docs) - relationship, 0) * 0.025, 0.08) + clinical_bonus
    return max(0.25, min(score, 0.92))


def _event_strength(event_type: str, facts: dict[str, Any], direction: str, context: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    strength = 0.40
    level = str(context.get("biomedical_evidence_level") or "low")
    if event_type == "trial_readout":
        strength += {"high": 0.22, "moderate": 0.10, "low": -0.06}.get(level, 0.0)
        if facts.get("phase"):
            phase = str(facts.get("phase"))
            reasons.append(phase)
            if any(x in phase.lower() for x in ["3", "iii", "2/3", "ii/iii"]):
                strength += 0.12
            elif any(x in phase.lower() for x in ["2", "ii"]):
                strength += 0.06
        if facts.get("has_endpoint_detail"):
            strength += 0.08
            reasons.append("endpoint detail")
        if facts.get("has_quantified_effect"):
            strength += 0.08
            reasons.append("quantified effect/statistic")
        if facts.get("has_design_detail"):
            strength += 0.05
            reasons.append("trial design/control detail")
        if facts.get("has_safety_detail"):
            strength += 0.04
            reasons.append("safety detail")
        if direction == "negative" and facts.get("mentions_failed_endpoint"):
            strength += 0.07
            reasons.append("failed endpoint language")
    elif event_type == "fda_decision":
        strength += 0.22 if facts.get("has_regulatory_detail") else 0.08
        reasons.append("regulatory action detail" if facts.get("has_regulatory_detail") else "regulatory catalyst with limited label detail")
        if facts.get("has_safety_detail"):
            strength += 0.04
            reasons.append("label/safety detail")
    elif event_type in {"licensing_deal", "acquisition"}:
        strength += 0.16 if facts.get("has_deal_terms") else 0.04
        reasons.append("deal economics/terms" if facts.get("has_deal_terms") else "strategic transaction detail")
    else:
        strength += 0.02
    return max(0.20, min(strength, 0.95)), reasons


def _fact_phrase(facts: dict[str, Any], context: dict[str, Any]) -> str:
    pieces: list[str] = []
    if facts.get("phase"):
        pieces.append(str(facts["phase"]))
    if facts.get("sample_size"):
        pieces.append(f"n={facts['sample_size']}")
    if facts.get("design_mentions"):
        pieces.append("design: " + clean_text(facts["design_mentions"][0], 120))
    if facts.get("endpoint_mentions"):
        pieces.append("endpoint: " + clean_text(facts["endpoint_mentions"][0], 140))
    if facts.get("effect_size_mentions"):
        pieces.append("effect: " + clean_text(facts["effect_size_mentions"][0], 140))
    elif facts.get("p_values"):
        pieces.append("stat: " + clean_text(facts["p_values"][0], 80))
    elif facts.get("hazard_ratios"):
        pieces.append("HR: " + clean_text(facts["hazard_ratios"][0], 80))
    elif facts.get("response_rates"):
        pieces.append("response: " + clean_text(facts["response_rates"][0], 80))
    if facts.get("safety_mentions"):
        pieces.append("safety: " + clean_text(facts["safety_mentions"][0], 140))
    if facts.get("regulatory_mentions"):
        pieces.append("regulatory: " + clean_text(facts["regulatory_mentions"][0], 150))
    if facts.get("deal_terms"):
        pieces.append("deal: " + clean_text(facts["deal_terms"][0], 150))
    if not pieces:
        pieces.append(context.get("clinical_summary") or "limited concrete details extracted")
    return "; ".join(pieces[:7])


def _entity_directness(entity: str, company: dict[str, Any], text: str) -> float:
    entity_terms = _tokens(entity)
    if not entity_terms:
        return 0.42
    text_terms = _tokens(text)
    company_terms = _tokens(company.get("aliases", [])) | _tokens(company.get("targets", [])) | _tokens(company.get("indications", []))
    overlap = len(entity_terms & (text_terms | company_terms)) / max(len(entity_terms), 1)
    if overlap >= 0.50:
        return 1.0
    if overlap > 0:
        return 0.72
    return 0.48


def _confidence(base: float, event_strength: float, directness: float, adjustment: float = 0.0, cap: float = 0.88) -> float:
    value = (base * 0.36) + (event_strength * 0.42) + (directness * 0.22) + adjustment
    return round(max(0.22, min(value, cap)), 2)


def _target_effects(company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]], direction: str, facts: dict[str, Any], context: dict[str, Any], event_strength: float, base_quality: float, source_index: int) -> list[dict[str, Any]]:
    targets = company.get("targets") or ["underlying drug target"]
    text = _blob(catalyst, source_docs)
    event_type = str(catalyst.get("event_type") or "other_catalyst")
    fact_phrase = _fact_phrase(facts, context)
    effects: list[dict[str, Any]] = []
    for target in targets[:3]:
        directness = _entity_directness(str(target), company, text)
        readthrough = "validation" if direction != "neutral" and event_type in {"trial_readout", "fda_decision"} else "neutral"
        target_direction = direction
        if event_type in {"licensing_deal", "acquisition"}:
            
            target_direction = "positive" if direction == "positive" and context.get("biomedical_evidence_level") != "low" else "neutral"
            readthrough = "validation" if target_direction == "positive" else "neutral"
        cap = 0.90 if context.get("biomedical_evidence_level") == "high" else 0.74 if context.get("biomedical_evidence_level") == "moderate" else 0.55
        conf = _confidence(base_quality, event_strength, directness, cap=cap)
        if target_direction == "neutral":
            conf = min(conf, 0.55)
        if event_type == "trial_readout":
            rationale = (
                f"Fallback expert checklist ties the target call to trial quality instead of simple sentiment. "
                f"Extracted facts: {fact_phrase}. "
                f"A {direction} trial readout {'supports' if direction == 'positive' else 'weakens' if direction == 'negative' else 'does not clearly change'} confidence in {target} only to the extent the source directly links the program/target; directness={directness:.2f}."
            )
        elif event_type == "fda_decision":
            rationale = (
                f"Regulatory decision read-through for {target}: extracted facts: {fact_phrase}. "
                f"Approval/CRL language can validate or undermine the development path, but target-biology confidence is capped unless the label/source explains mechanism, safety, and patient population."
            )
        elif event_type in {"licensing_deal", "acquisition"}:
            rationale = (
                f"Strategic transaction read-through for {target}: extracted facts: {fact_phrase}. "
                f"Deal interest can be external validation of an asset/platform, but it is not treated as clinical proof unless the cited source gives clinical endpoint, safety, or regulatory details."
            )
        else:
            rationale = f"Other catalyst mapped conservatively to {target}; extracted facts: {fact_phrase}. Direction is capped because catalyst specificity is limited."
        effects.append({
            "layer": "target",
            "entity": str(target),
            "direction": target_direction,
            "confidence": conf,
            "readthrough_type": readthrough,
            "rationale": rationale,
            "claims": _source_claim(source_docs, catalyst.get("title", ""), source_index),
        })
    return effects


def _indication_effects(company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]], direction: str, facts: dict[str, Any], context: dict[str, Any], event_strength: float, base_quality: float, source_index: int) -> list[dict[str, Any]]:
    indications = company.get("indications") or ["relevant indication"]
    text = _blob(catalyst, source_docs)
    event_type = str(catalyst.get("event_type") or "other_catalyst")
    fact_phrase = _fact_phrase(facts, context)
    effects: list[dict[str, Any]] = []
    for indication in indications[:3]:
        directness = _entity_directness(str(indication), company, text)
        cap = 0.88 if context.get("biomedical_evidence_level") == "high" else 0.73 if context.get("biomedical_evidence_level") == "moderate" else 0.54
        conf = _confidence(base_quality, event_strength, directness, adjustment=-0.03, cap=cap)
        readthrough = "competitive" if direction == "positive" and event_type in {"trial_readout", "fda_decision"} else "neutral" if direction == "neutral" else "none"
        if event_type == "fda_decision" and direction == "positive":
            rationale = (
                f"Indication impact for {indication}: {fact_phrase}. "
                f"A positive FDA decision can change the treatment landscape and raise the clinical/regulatory bar, but standard-of-care impact is capped unless label population, line of therapy, safety, and comparators are explicit."
            )
        elif event_type == "trial_readout":
            bar = "raises" if direction == "positive" else "lowers immediate competitive pressure but may signal shared disease/trial risk" if direction == "negative" else "does not clearly move"
            rationale = (
                f"Indication impact for {indication}: {fact_phrase}. "
                f"The fallback analyzes endpoint strength, effect size, safety, sample/design, and phase before saying the event {bar} the competitive bar."
            )
        elif event_type in {"licensing_deal", "acquisition"}:
            rationale = (
                f"Indication impact for {indication}: {fact_phrase}. "
                f"The transaction may increase resources or strategic focus in this disease area, but it is not assumed to alter standard of care without clinical/regulatory evidence."
            )
            readthrough = "neutral" if direction != "negative" else "none"
            if context.get("biomedical_evidence_level") == "low":
                conf = min(conf, 0.50)
        else:
            rationale = f"Indication impact for {indication} is conservative because the event type/source detail is limited; extracted facts: {fact_phrase}."
        effects.append({
            "layer": "indication",
            "entity": str(indication),
            "direction": direction,
            "confidence": conf,
            "readthrough_type": readthrough,
            "rationale": rationale,
            "claims": _source_claim(source_docs, catalyst.get("title", ""), source_index),
        })
    return effects


def _relationship_source_index(peer: dict[str, Any]) -> int:
    try:
        return int(peer.get("_relationship_source_index") or 0)
    except Exception:
        return 0


def _peer_flags(peer: dict[str, Any], company: dict[str, Any]) -> tuple[bool, bool, bool, str]:
    relationship_type = str(peer.get("relationship_type") or peer.get("relationship") or "").lower()
    relation_blob = " ".join(str(peer.get(k, "")) for k in ["relationship", "relationship_type", "target", "indication", "expected_readthrough", "target_overlap", "indication_overlap", "source"]).lower()
    declared_different_target = any(x in relation_blob for x in ["different target", "not same-target", "not same target", "unrelated target"])
    peer_targets = [peer.get("target", ""), peer.get("target_overlap", "")]
    peer_indications = [peer.get("indication", ""), peer.get("indication_overlap", "")]
    computed_target_overlap = target_overlap_terms((company.get("targets") or []) + (company.get("aliases") or []), peer_targets)
    computed_indication_overlap = indication_overlap_terms(company.get("indications", []), peer_indications)
    same_target = (
        "same_target" in relationship_type or "same target" in relationship_type or "same-target" in relationship_type
        or "target_validation" in relationship_type or "mixed_target" in relationship_type
        or bool(peer.get("target_overlap")) or bool(computed_target_overlap)
    ) and not declared_different_target
    same_indication = (
        "same_indication" in relationship_type or "same indication" in relationship_type or "same-indication" in relationship_type
        or "competitor" in relationship_type or "competition" in relationship_type or "mixed_target_and_indication" in relationship_type
        or bool(peer.get("indication_overlap")) or bool(computed_indication_overlap)
    )
    partner = any(x in relationship_type or x in relation_blob for x in ["partner", "collaboration", "royalt", "milestone", "licensee", "licensor", "commercial partner"])
    evidence = "source-cited" if _relationship_source_index(peer) > 0 else "relationship-source-missing"
    if same_target and same_indication:
        strength = f"{evidence}; same target/mechanism plus same indication overlap"
    elif same_target:
        strength = f"{evidence}; same target/mechanism overlap"
    elif same_indication:
        strength = f"{evidence}; same indication/market overlap"
    elif partner:
        strength = f"{evidence}; partner/economic relationship"
    else:
        strength = f"{evidence}; weak or unsupported overlap"
    return same_target, same_indication, partner, strength


def _peer_claims(peer: dict[str, Any], event_claim: list[dict[str, Any]], source_count: int) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    rel_idx = _relationship_source_index(peer)
    if rel_idx > 0 and source_count:
        peer_name = peer.get("company") or peer.get("name") or "Peer company"
        relationship = peer.get("relationship") or peer.get("relationship_type") or "peer relationship"
        target = peer.get("target") or peer.get("target_overlap") or "target not specified"
        indication = peer.get("indication") or peer.get("indication_overlap") or "indication not specified"
        claims.extend(_specific_claim(rel_idx, f"{peer_name} relationship evidence: {relationship}; target/program overlap: {target}; indication overlap: {indication}.", source_count))
    claims.extend(event_claim[:1])
    return claims or event_claim


def _peer_effect(peer: dict[str, Any], company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]], direction: str, event_strength: float, base_quality: float, event_claim: list[dict[str, Any]]) -> dict[str, Any]:
    peer_name = peer.get("company") or peer.get("name") or "Peer company"
    same_target, same_indication, partner, strength = _peer_flags(peer, company)
    source_count = len(source_docs)
    rel_idx = _relationship_source_index(peer)
    has_rel = rel_idx > 0
    evidence_penalty = 0.0 if has_rel else 0.12
    event_type = str(catalyst.get("event_type") or "other_catalyst")
    expected = clean_text(peer.get("expected_readthrough") or "", 360)
    claims = _peer_claims(peer, event_claim, source_count)

    if partner:
        conf = _confidence(base_quality, event_strength, 0.78, adjustment=-0.10 - evidence_penalty, cap=0.78)
        return {
            "layer": "peer", "entity": str(peer_name), "direction": direction, "confidence": conf,
            "readthrough_type": "partner",
            "rationale": f"Partner/economic read-through ({strength}). Direction follows the catalyst, but magnitude depends on contract economics, territory, royalties, opt-ins, and commercialization responsibility. {expected}",
            "claims": claims,
        }
    if same_target and same_indication:
        
        if direction == "positive" and event_type in {"fda_decision", "trial_readout"}:
            peer_direction = "neutral"
            rationale = (
                f"Mixed peer read-through ({strength}). Same target creates validation for the class, while same indication creates competitive pressure and a higher efficacy/safety/convenience bar. "
                f"Net is neutral unless cited evidence proves product differentiation or commercial displacement. {expected}"
            )
        elif direction == "negative":
            peer_direction = "neutral"
            rationale = (
                f"Mixed negative read-through ({strength}). A failure may weaken class biology but also reduce competitive threat; net is neutral without evidence separating target-specific risk from asset-specific execution. {expected}"
            )
        else:
            peer_direction = "neutral"
            rationale = f"Mixed same-target/same-indication relationship ({strength}); catalyst direction is too uncertain for a directional peer call. {expected}"
        return {"layer": "peer", "entity": str(peer_name), "direction": peer_direction, "confidence": round(max(0.24, min(0.50, 0.48 - evidence_penalty)), 2), "readthrough_type": "neutral", "rationale": rationale, "claims": claims}
    if same_target:
        conf = _confidence(base_quality, event_strength, 0.72, adjustment=-0.16 - evidence_penalty, cap=0.72)
        return {
            "layer": "peer", "entity": str(peer_name), "direction": direction, "confidence": conf,
            "readthrough_type": "validation",
            "rationale": f"Validation read-through ({strength}). The event can strengthen or weaken confidence in the shared biology/mechanism, but confidence is below the direct company effect because peer asset design, stage, dose, and population may differ. {expected}",
            "claims": claims,
        }
    if same_indication:
        if direction == "positive":
            peer_direction = "negative"
            rationale = f"Competitive read-through ({strength}). A positive event in the same indication can raise endpoint, safety, convenience, label, or commercial expectations for rivals. {expected}"
        elif direction == "negative":
            peer_direction = "positive"
            rationale = f"Competitive read-through ({strength}). A negative event can reduce immediate competitive pressure, but confidence is capped because it may also reveal shared disease/trial risk. {expected}"
        else:
            peer_direction = "neutral"
            rationale = f"Same-indication relationship exists ({strength}), but the event is not directional enough to infer competitive displacement. {expected}"
        conf = _confidence(base_quality, event_strength, 0.66, adjustment=-0.20 - evidence_penalty, cap=0.68)
        return {"layer": "peer", "entity": str(peer_name), "direction": peer_direction, "confidence": conf, "readthrough_type": "competitive", "rationale": rationale, "claims": claims}
    return {
        "layer": "peer", "entity": str(peer_name), "direction": "neutral", "confidence": 0.24,
        "readthrough_type": "none",
        "rationale": "No reliable target, indication, or partner relationship was found. The fallback does not assume blanket biotech contagion.",
        "claims": event_claim,
    }


def expert_fallback_analysis(company: dict[str, Any], catalyst: dict[str, Any], source_docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a deterministic propagation report with clinical-quality reasoning."""
    text = _blob(catalyst, source_docs)
    direction, sentiment_conf = infer_direction(catalyst.get("title", ""), text)
    context = extract_biomedical_context(catalyst, source_docs)
    facts = extract_clinical_facts(catalyst, source_docs)
    event_type = str(catalyst.get("event_type") or "other_catalyst")
    event_strength, strength_reasons = _event_strength(event_type, facts, direction, context)
    base_quality = _evidence_quality(source_docs, context)
    source_index = _best_catalyst_source_index(source_docs)
    event_claim = _source_claim(source_docs, catalyst.get("title", ""), source_index)

    
    overall_conf = round(max(0.24, min((sentiment_conf * 0.22) + (base_quality * 0.38) + (event_strength * 0.40), 0.90)), 2)
    if context.get("biomedical_evidence_level") == "low" and event_type in {"trial_readout", "fda_decision", "licensing_deal", "acquisition"}:
        overall_conf = min(overall_conf, 0.56)

    effects: list[dict[str, Any]] = []
    effects.extend(_target_effects(company, catalyst, source_docs, direction, facts, context, event_strength, base_quality, source_index))
    effects.extend(_indication_effects(company, catalyst, source_docs, direction, facts, context, event_strength, base_quality, source_index))
    for peer in (company.get("peers") or [])[:18]:
        if isinstance(peer, dict):
            effects.append(_peer_effect(peer, company, catalyst, source_docs, direction, event_strength, base_quality, event_claim))

    missing = context.get("missing_critical_fields") or []
    strength_text = ", ".join(strength_reasons) if strength_reasons else "limited concrete catalyst detail"
    summary = (
        f"Expert fallback report: the system detected a {event_type} with {direction} direction. "
        f"It reviewed trial/regulatory/deal specifics before propagating effects: {context.get('clinical_summary')}. "
        f"Evidence level is {context.get('biomedical_evidence_level')} based on {strength_text}."
    )
    if missing:
        summary += " Missing fields that limit confidence: " + ", ".join(missing) + "."

    return {
        "event_type": event_type,
        "plain_language_summary": summary,
        "overall_direction": direction,
        "overall_confidence": overall_conf,
        "effects": effects,
        "expert_fallback_checklist": {
            "direction_signal_confidence": sentiment_conf,
            "evidence_quality": round(base_quality, 3),
            "event_strength": round(event_strength, 3),
            "strength_reasons": strength_reasons,
            "clinical_context": context,
        },
    }
