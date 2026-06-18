from __future__ import annotations

import json
import re
from typing import Any

from app.db import execute, rows

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_GENERIC = {
    
    "inc", "corp", "corporation", "therapeutics", "pharmaceuticals", "pharma", "biotech", "company",
    "therapy", "therapies", "disease", "diseases", "study", "trial", "phase", "program", "drug", "drugs",
    "same", "target", "indication", "competitor", "partner", "collaboration", "expected", "readthrough",
    
    "receptor", "receptors", "protein", "proteins", "hormone", "agonist", "agonists", "antibody", "antibodies",
    "modality", "mechanism", "pathway", "biology", "agent", "asset", "candidate", "investigational",
    "oral", "subcutaneous", "intravenous", "selective", "novel", "approved", "commercial", "metabolic",
}




_TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "thr-beta": (
        "thr beta", "thr-beta", "thrβ", "tr beta", "tr-beta", "trβ", "thrb", "thrbeta",
        "thyroid hormone receptor beta", "thyroid hormone receptor-beta", "thyroid receptor beta",
    ),
    "fcrn": (
        "fcrn", "fc-rn", "neonatal fc receptor", "neonatal-fc-receptor", "fc receptor neonatal",
        "anti fcrn", "anti-fcrn", "fc neonatal receptor",
    ),
    "fgf21": ("fgf21", "fgf-21", "fibroblast growth factor 21", "fgf21 analog", "fgf21 analogue"),
    "bcl11a": ("bcl11a", "bcl11a enhancer", "bcl11a erythroid enhancer"),
    "crispr-cas9": ("crispr/cas9", "crispr cas9", "crispr-cas9", "cas9", "exa-cel", "exagamglogene"),
    "base-editing": ("base editing", "base editor", "adenine base editor", "beam-101", "beam 101"),
    "lentiviral-gene-therapy": ("lentiviral", "lovo-cel", "lovotibeglogene", "lyfgenia"),
    "dgat2": ("dgat2", "dgat-2", "diacylglycerol o-acyltransferase 2"),
    "siRNA": ("sirna", "small interfering rna", "rna interference"),
    "tyk2": ("tyk2", "tyk-2", "tyrosine kinase 2"),
    "pcsk9": ("pcsk9", "pcsk-9", "proprotein convertase subtilisin kexin type 9"),
    "glp-1": ("glp 1", "glp-1", "glp1", "glucagon like peptide 1"),
    "bcma": ("bcma", "b cell maturation antigen", "b-cell maturation antigen"),
    "her2": ("her2", "her-2", "erbb2"),
}


_INDICATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mash": (
        "mash", "nash", "metabolic dysfunction associated steatohepatitis",
        "metabolic dysfunction-associated steatohepatitis", "nonalcoholic steatohepatitis",
        "non-alcoholic steatohepatitis",
    ),
    "sickle-cell-disease": ("sickle cell disease", "scd", "sickle-cell disease"),
    "tdt": ("transfusion dependent beta thalassemia", "transfusion-dependent beta thalassemia", "tdt", "beta thalassemia"),
    "myasthenia-gravis": ("myasthenia gravis", "mg", "gmg", "generalized myasthenia gravis"),
    "cidp": ("cidp", "chronic inflammatory demyelinating polyneuropathy"),
    "thyroid-eye-disease": ("thyroid eye disease", "ted", "graves ophthalmopathy"),
    "autoimmune": ("autoimmune", "autoimmune diseases", "igg mediated autoimmune"),
    "psoriasis": ("psoriasis", "plaque psoriasis"),
    "obesity": ("obesity", "weight management", "overweight"),
    "multiple-myeloma": ("multiple myeloma", "rrmm", "relapsed refractory multiple myeloma"),
    "breast-cancer": ("breast cancer", "her2 positive breast cancer", "her2-positive breast cancer"),
}



def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def _normalize(value: Any) -> str:
    text = " ".join(_listify(value)).lower()
    text = text.replace("β", " beta ").replace("α", " alpha ").replace("γ", " gamma ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    text = _normalize(value)
    out: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        token = token.strip("-/").lower()
        if len(token) >= 2 and token not in _GENERIC:
            out.add(token)
    return out


def _canonical_matches(terms: Any, synonym_map: dict[str, tuple[str, ...]]) -> set[str]:
    normalized_terms = [_normalize(t) for t in _listify(terms)]
    matches: set[str] = set()
    for canonical, synonyms in synonym_map.items():
        for term in normalized_terms:
            for synonym in synonyms:
                syn = _normalize(synonym)
                if not syn:
                    continue
                if term == syn or re.search(rf"(^|\s){re.escape(syn)}($|\s)", term):
                    matches.add(canonical)
    return matches


def _exact_phrase_matches(left_terms: list[str], right_terms: list[str]) -> set[str]:
    """Return conservative exact phrase matches after removing generic words.

    We require either an exact normalized phrase match or at least two distinctive
    shared tokens. A one-token match such as "receptor" or "beta" is not enough.
    """
    matches: set[str] = set()
    for left in left_terms:
        left_norm = _normalize(left)
        left_tokens = _tokens(left)
        if not left_norm or not left_tokens:
            continue
        for right in right_terms:
            right_norm = _normalize(right)
            right_tokens = _tokens(right)
            if not right_norm or not right_tokens:
                continue
            if left_norm == right_norm and len(left_norm) >= 3:
                matches.add(str(left))
                continue
            shared = left_tokens & right_tokens
            if len(shared) >= 2:
                matches.add(str(left))
                matches.add(str(right))
    return matches


def target_overlap_terms(left_terms: Any, right_terms: Any) -> list[str]:
    left_list = _listify(left_terms)
    right_list = _listify(right_terms)
    left_canon = _canonical_matches(left_list, _TARGET_SYNONYMS)
    right_canon = _canonical_matches(right_list, _TARGET_SYNONYMS)
    canonical_overlap = left_canon & right_canon
    matches: list[str] = []
    if canonical_overlap:
        for term in [*left_list, *right_list]:
            term_canon = _canonical_matches([term], _TARGET_SYNONYMS)
            if term_canon & canonical_overlap and term not in matches:
                matches.append(str(term))
    else:
        matches = list(_exact_phrase_matches(left_list, right_list))
    return matches[:6]


def indication_overlap_terms(left_terms: Any, right_terms: Any) -> list[str]:
    left_list = _listify(left_terms)
    right_list = _listify(right_terms)
    left_canon = _canonical_matches(left_list, _INDICATION_SYNONYMS)
    right_canon = _canonical_matches(right_list, _INDICATION_SYNONYMS)
    canonical_overlap = left_canon & right_canon
    matches: list[str] = []
    if canonical_overlap:
        for term in [*left_list, *right_list]:
            term_canon = _canonical_matches([term], _INDICATION_SYNONYMS)
            if term_canon & canonical_overlap and term not in matches:
                matches.append(str(term))
    else:
        
        exact = {_normalize(x) for x in left_list} & {_normalize(x) for x in right_list}
        if exact:
            for term in [*left_list, *right_list]:
                if _normalize(term) in exact and term not in matches:
                    matches.append(str(term))
        else:
            matches = list(_exact_phrase_matches(left_list, right_list))
    return matches[:6]


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _company_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "name": row.get("name") or "",
        "ticker": row.get("ticker") or "",
        "aliases": _json_loads(row.get("aliases_json"), []),
        "targets": _json_loads(row.get("targets_json"), []),
        "indications": _json_loads(row.get("indications_json"), []),
        "peers": _json_loads(row.get("peers_json"), []),
        "source_urls": _json_loads(row.get("source_urls_json"), []),
        "notes": row.get("notes") or "",
    }



def _peer_asset_rows_for_company(company: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert watchlist/manual peer context into asset-index rows.

    The project cannot depend on a hand-curated three-company graph only. This
    index turns every watched company and every manual peer entry into a dynamic
    target/indication/asset record that can be re-synced after watchlist edits.
    """
    rows_out: list[dict[str, Any]] = []
    company_name = company.get("name") or company.get("company") or "Unknown"
    ticker = company.get("ticker") or ""
    source_urls = _listify(company.get("source_urls"))
    base_url = source_urls[0] if source_urls else ""
    targets = _listify(company.get("targets")) or [""]
    indications = _listify(company.get("indications")) or [""]
    aliases = _listify(company.get("aliases"))
    asset_name = aliases[0] if aliases else company_name
    for target in targets[:8]:
        for indication in indications[:8]:
            rows_out.append({
                "company_id": int(company.get("id") or 0),
                "company_name": company_name,
                "ticker": ticker,
                "asset_name": asset_name,
                "target": target,
                "indication": indication,
                "modality": "watchlist profile",
                "development_stage": "watchlist",
                "source_url": base_url,
                "source_title": f"{company_name} watchlist profile",
                "evidence_excerpt": f"Watchlist profile lists targets {targets} and indications {indications}.",
                "confidence": 0.62,
            })
    
    for peer in company.get("peers") or []:
        if not isinstance(peer, dict):
            continue
        peer_name = peer.get("company") or peer.get("name") or "Unknown peer"
        rows_out.append({
            "company_id": int(company.get("id") or 0),
            "company_name": peer_name,
            "ticker": peer.get("ticker") or "",
            "asset_name": peer.get("asset") or peer.get("program") or peer.get("target") or peer_name,
            "target": peer.get("target") or peer.get("target_overlap") or "",
            "indication": peer.get("indication") or peer.get("indication_overlap") or "",
            "modality": peer.get("modality") or peer.get("relationship_type") or peer.get("relationship") or "manual peer",
            "development_stage": peer.get("stage") or peer.get("development_stage") or "manual peer evidence",
            "source_url": peer.get("evidence_url") or peer.get("source_url") or peer.get("relationship_url") or "",
            "source_title": peer.get("evidence_title") or f"{peer_name} peer evidence",
            "evidence_excerpt": peer.get("evidence_excerpt") or peer.get("expected_readthrough") or f"Manual peer graph entry for {peer_name}.",
            "confidence": float(peer.get("confidence") or 0.66),
        })
    return rows_out


def refresh_peer_asset_index() -> int:
    """Rebuild the dynamic asset graph from current watchlist and peer entries."""
    all_rows = rows("SELECT * FROM companies WHERE active = 1")
    execute("DELETE FROM peer_asset_index")
    count = 0
    for r in all_rows:
        company = _company_from_row(r)
        for asset in _peer_asset_rows_for_company(company):
            try:
                execute(
                    """
                    INSERT OR REPLACE INTO peer_asset_index (
                        company_id, company_name, ticker, asset_name, target, indication, modality,
                        development_stage, source_url, source_title, evidence_excerpt, confidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        asset["company_id"], asset["company_name"], asset["ticker"], asset["asset_name"],
                        asset["target"], asset["indication"], asset["modality"], asset["development_stage"],
                        asset["source_url"], asset["source_title"], asset["evidence_excerpt"], asset["confidence"],
                    ),
                )
                count += 1
            except Exception:
                pass
    return count


def _asset_index_relationship(company: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any] | None:
    peer_name = asset.get("company_name") or "Unknown peer"
    if peer_name.lower() == str(company.get("name") or "").lower():
        return None
    target_overlap = target_overlap_terms(_listify(company.get("targets")) + _listify(company.get("aliases")), [asset.get("target"), asset.get("asset_name"), asset.get("modality")])
    indication_overlap = indication_overlap_terms(company.get("indications"), [asset.get("indication")])
    if not target_overlap and not indication_overlap:
        return None
    if target_overlap and indication_overlap:
        relationship_type = "mixed_target_and_indication"
        confidence = min(float(asset.get("confidence") or 0.6) + 0.08, 0.82)
        expected = "Mixed: target validation can help the peer class, while same-indication competition can raise the bar."
    elif target_overlap:
        relationship_type = "same_target_validation"
        confidence = min(float(asset.get("confidence") or 0.6) + 0.04, 0.76)
        expected = "Validation read-through from same/adjacent target or mechanism."
    else:
        relationship_type = "same_indication_competition"
        confidence = min(float(asset.get("confidence") or 0.55), 0.68)
        expected = "Competitive read-through from disease/market overlap."
    return {
        "company": peer_name,
        "ticker": asset.get("ticker") or "",
        "relationship": relationship_type,
        "relationship_type": relationship_type,
        "target": asset.get("target") or ", ".join(target_overlap) or "not overlapping",
        "indication": asset.get("indication") or ", ".join(indication_overlap) or "not overlapping",
        "target_overlap": target_overlap,
        "indication_overlap": indication_overlap,
        "asset": asset.get("asset_name") or "",
        "modality": asset.get("modality") or "",
        "stage": asset.get("development_stage") or "",
        "confidence": confidence,
        "source": "dynamic_peer_asset_index",
        "evidence_url": asset.get("source_url") or "",
        "evidence_title": asset.get("source_title") or f"{peer_name} asset-index evidence",
        "evidence_excerpt": asset.get("evidence_excerpt") or f"Dynamic peer asset index links {peer_name} through target/indication overlap.",
        "expected_readthrough": expected,
    }

def infer_watchlist_relationship(company: dict[str, Any], peer: dict[str, Any]) -> dict[str, Any] | None:
    """Infer a read-through edge between two watched companies.

    The inference is deliberately conservative. It requires canonical target
    matches, exact/controlled indication matches, or explicit manual peer data.
    Generic words such as "receptor" do not create same-target relationships.
    """
    if company.get("id") == peer.get("id"):
        return None
    company_targets = _listify(company.get("targets")) + _listify(company.get("aliases"))
    peer_targets = _listify(peer.get("targets")) + _listify(peer.get("aliases"))
    company_indications = _listify(company.get("indications"))
    peer_indications = _listify(peer.get("indications"))
    target_overlap = target_overlap_terms(company_targets, peer_targets)
    indication_overlap = indication_overlap_terms(company_indications, peer_indications)
    if not target_overlap and not indication_overlap:
        return None
    if target_overlap and indication_overlap:
        relationship_type = "mixed_target_and_indication"
        confidence = 0.74
        expected = "Mixed: same-target validation can help, while same-indication competition can raise the product bar."
    elif target_overlap:
        relationship_type = "same_target_validation"
        confidence = 0.68
        expected = "Validation read-through: same/overlapping target or product biology may be strengthened or weakened."
    else:
        relationship_type = "same_indication_competition"
        confidence = 0.58
        expected = "Competitive read-through: same/overlapping indication may change the standard-of-care or competitive bar."
    peer_source_urls = _listify(peer.get("source_urls"))
    evidence_url = peer_source_urls[0] if peer_source_urls else ""
    return {
        "company": peer.get("name") or peer.get("company") or "Unknown peer",
        "ticker": peer.get("ticker") or "",
        "relationship": relationship_type,
        "relationship_type": relationship_type,
        "target": ", ".join(target_overlap) if target_overlap else "not overlapping",
        "indication": ", ".join(indication_overlap) if indication_overlap else "not overlapping",
        "target_overlap": target_overlap,
        "indication_overlap": indication_overlap,
        "confidence": confidence,
        "source": "conservative_watchlist_overlap",
        "evidence_url": evidence_url,
        "evidence_title": f"{peer.get('name') or peer.get('company') or 'Peer'} watchlist/source evidence",
        "evidence_excerpt": (
            f"Watched peer profile lists targets {peer.get('targets', [])} and indications {peer.get('indications', [])}. "
            f"Inferred relationship: {relationship_type}."
        ),
        "expected_readthrough": expected,
    }


def sync_peer_relationships_for_company(company: dict[str, Any]) -> list[dict[str, Any]]:
    
    
    refresh_peer_asset_index()
    all_rows = rows("SELECT * FROM companies WHERE active = 1")
    companies = [_company_from_row(r) for r in all_rows]
    asset_rows = rows("SELECT * FROM peer_asset_index")
    relationships: list[dict[str, Any]] = []

    for peer in companies:
        rel = infer_watchlist_relationship(company, peer)
        if rel:
            relationships.append(rel)
    for asset in asset_rows:
        rel = _asset_index_relationship(company, asset)
        if rel:
            relationships.append(rel)

    
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel in relationships:
        key = f"{str(rel.get('company')).lower()}::{str(rel.get('relationship_type')).lower()}::{_normalize(rel.get('target'))}::{_normalize(rel.get('indication'))}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(rel)
        try:
            execute(
                """
                INSERT OR REPLACE INTO peer_relationships (
                    company_id, peer_name, peer_ticker, relationship_type,
                    target_overlap_json, indication_overlap_json, confidence, source, rationale, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    company["id"],
                    rel["company"],
                    rel.get("ticker", ""),
                    rel["relationship_type"],
                    json.dumps(rel.get("target_overlap", []), ensure_ascii=False),
                    json.dumps(rel.get("indication_overlap", []), ensure_ascii=False),
                    float(rel.get("confidence", 0.0)),
                    rel.get("source", "dynamic_peer_asset_index"),
                    rel.get("expected_readthrough", ""),
                ),
            )
        except Exception:
            pass
    return unique

def enrich_company_peers(company: dict[str, Any]) -> dict[str, Any]:
    manual = company.get("peers") or []
    inferred = sync_peer_relationships_for_company(company) if company.get("id") else []
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for peer in [*manual, *inferred]:
        if not isinstance(peer, dict):
            continue
        key = f"{str(peer.get('company') or peer.get('name') or '').lower()}::{str(peer.get('relationship_type') or peer.get('relationship') or '').lower()}"
        if key.strip(":") in seen:
            continue
        seen.add(key.strip(":"))
        merged.append(peer)
    out = dict(company)
    out["peers"] = merged
    out["inferred_peer_count"] = len(inferred)
    return out


def sync_all_peer_relationships() -> int:
    refresh_peer_asset_index()
    count = 0
    for r in rows("SELECT * FROM companies WHERE active = 1"):
        count += len(sync_peer_relationships_for_company(_company_from_row(r)))
    return count
