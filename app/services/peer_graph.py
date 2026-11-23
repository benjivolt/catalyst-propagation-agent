from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from app.db import company_from_row, execute, rows, transaction
from app.services.utils import json_dumps, tokens

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_GENERIC = {
    "inc",
    "corp",
    "corporation",
    "therapeutics",
    "pharmaceuticals",
    "pharma",
    "biotech",
    "company",
    "therapy",
    "therapies",
    "disease",
    "diseases",
    "study",
    "trial",
    "phase",
    "program",
    "drug",
    "drugs",
    "same",
    "target",
    "indication",
    "competitor",
    "partner",
    "collaboration",
    "expected",
    "readthrough",
    "receptor",
    "receptors",
    "protein",
    "proteins",
    "hormone",
    "agonist",
    "agonists",
    "antibody",
    "antibodies",
    "modality",
    "mechanism",
    "pathway",
    "biology",
    "agent",
    "asset",
    "candidate",
    "investigational",
    "oral",
    "subcutaneous",
    "intravenous",
    "selective",
    "novel",
    "approved",
    "commercial",
    "metabolic",
}


_TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "thr-beta": (
        "thr beta",
        "thr-beta",
        "thrβ",
        "tr beta",
        "tr-beta",
        "trβ",
        "thrb",
        "thrbeta",
        "thyroid hormone receptor beta",
        "thyroid hormone receptor-beta",
        "thyroid receptor beta",
    ),
    "fcrn": (
        "fcrn",
        "fc-rn",
        "neonatal fc receptor",
        "neonatal-fc-receptor",
        "fc receptor neonatal",
        "anti fcrn",
        "anti-fcrn",
        "fc neonatal receptor",
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
        "mash",
        "nash",
        "metabolic dysfunction associated steatohepatitis",
        "metabolic dysfunction-associated steatohepatitis",
        "nonalcoholic steatohepatitis",
        "non-alcoholic steatohepatitis",
    ),
    "sickle-cell-disease": ("sickle cell disease", "scd", "sickle-cell disease"),
    "tdt": (
        "transfusion dependent beta thalassemia",
        "transfusion-dependent beta thalassemia",
        "tdt",
        "beta thalassemia",
    ),
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


@lru_cache(maxsize=8192)
def _normalize_text(text: str) -> str:
    text = text.lower().replace("β", " beta ").replace("α", " alpha ").replace("γ", " gamma ")
    return " ".join(_NON_ALNUM_RE.sub(" ", text).split())


def _normalize(value: Any) -> str:
    return _normalize_text(" ".join(_listify(value)))


def _tokens(value: Any) -> set[str]:
    return set(tokens(_normalize(value), _GENERIC, pattern=_TOKEN_RE))


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


_COMPANY_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "se",
    "ag",
    "sa",
    "nv",
    "bv",
    "llc",
    "the",
}
_SYNONYM_MAPS = {"target": _TARGET_SYNONYMS, "indication": _INDICATION_SYNONYMS}


def canonical_key(term: Any, kind: str) -> str:
    matches = _canonical_matches([term], _SYNONYM_MAPS[kind])
    return "+".join(sorted(matches)) if matches else _normalize(term)


def canonical_terms(terms: Any, kind: str, exclude: Any = ()) -> list[str]:
    excluded = {_normalize(x) for x in _listify(exclude)}
    seen: set[str] = set()
    out: list[str] = []
    for term in _listify(terms):
        if _normalize(term) in excluded:
            continue
        key = canonical_key(term, kind)
        if key and key not in seen:
            seen.add(key)
            out.append(str(term).strip())
    return out


def company_key(name: Any) -> str:
    return " ".join(w for w in _normalize(name).split() if w not in _COMPANY_SUFFIXES)


def _identity(entry: dict[str, Any]) -> tuple[str, str]:
    ticker = str(entry.get("ticker") or "").strip().upper()
    return ticker, company_key(entry.get("company") or entry.get("name") or "")


def same_company(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ticker, left_name = _identity(left)
    right_ticker, right_name = _identity(right)
    if left_ticker and right_ticker and left_ticker == right_ticker:
        return True
    return bool(left_name) and left_name == right_name


def dedupe_peers(peers: list[Any], company: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for peer in peers:
        if not isinstance(peer, dict) or not (peer.get("company") or peer.get("name")):
            continue
        if company and same_company(peer, company):
            continue
        if any(same_company(peer, kept) for kept in merged):
            continue
        merged.append(peer)
    return merged


def _peer_asset_rows_for_company(company: dict[str, Any]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    company_name = company.get("name") or company.get("company") or "Unknown"
    ticker = company.get("ticker") or ""
    source_urls = _listify(company.get("source_urls"))
    base_url = source_urls[0] if source_urls else ""
    aliases = _listify(company.get("aliases"))
    targets = canonical_terms(company.get("targets"), "target", exclude=aliases) or [""]
    indications = canonical_terms(company.get("indications"), "indication") or [""]
    asset_name = aliases[0] if aliases else company_name
    for target in targets[:8]:
        for indication in indications[:8]:
            rows_out.append(
                {
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
                }
            )

    for peer in company.get("peers") or []:
        if not isinstance(peer, dict):
            continue
        peer_name = peer.get("company") or peer.get("name") or "Unknown peer"
        rows_out.append(
            {
                "company_id": int(company.get("id") or 0),
                "company_name": peer_name,
                "ticker": peer.get("ticker") or "",
                "asset_name": peer.get("asset") or peer.get("program") or peer.get("target") or peer_name,
                "target": peer.get("target") or "",
                "indication": peer.get("indication") or "",
                "modality": peer.get("modality") or peer.get("relationship") or "manual peer",
                "development_stage": peer.get("stage") or "manual peer evidence",
                "source_url": peer.get("evidence_url") or peer.get("source_url") or "",
                "source_title": peer.get("evidence_title") or f"{peer_name} peer evidence",
                "evidence_excerpt": peer.get("evidence_excerpt")
                or peer.get("expected_readthrough")
                or f"Watchlist entry for {peer_name}.",
                "confidence": float(peer.get("confidence") or 0.66),
            }
        )
    return rows_out


def _active_companies() -> list[dict[str, Any]]:
    return [company_from_row(r) for r in rows("SELECT * FROM companies WHERE active = 1")]


def _asset_rows(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for company in companies:
        for asset in _peer_asset_rows_for_company(company):
            key = (
                asset["company_id"],
                asset["company_name"],
                asset["asset_name"],
                asset["target"],
                asset["indication"],
            )
            if key not in seen:
                seen.add(key)
                out.append(asset)
    return out


def refresh_peer_asset_index(companies: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    assets = _asset_rows(_active_companies() if companies is None else companies)
    with transaction():
        execute("DELETE FROM peer_asset_index")
        for asset in assets:
            execute(
                """
                INSERT OR REPLACE INTO peer_asset_index (
                    company_id, company_name, ticker, asset_name, target, indication, modality,
                    development_stage, source_url, source_title, evidence_excerpt, confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    asset["company_id"],
                    asset["company_name"],
                    asset["ticker"],
                    asset["asset_name"],
                    asset["target"],
                    asset["indication"],
                    asset["modality"],
                    asset["development_stage"],
                    asset["source_url"],
                    asset["source_title"],
                    asset["evidence_excerpt"],
                    asset["confidence"],
                ),
            )
    return assets


def _relationship(
    peer_name: str, ticker: str, target_overlap: list[str], indication_overlap: list[str], base_confidence: float
) -> dict[str, Any] | None:
    if target_overlap and indication_overlap:
        relationship_type, confidence = "mixed_target_and_indication", min(base_confidence + 0.08, 0.82)
    elif target_overlap:
        relationship_type, confidence = "same_target_validation", min(base_confidence + 0.04, 0.76)
    elif indication_overlap:
        relationship_type, confidence = "same_indication_competition", min(base_confidence, 0.68)
    else:
        return None
    return {
        "company": peer_name,
        "ticker": ticker,
        "relationship": relationship_type,
        "relationship_type": relationship_type,
        "target": ", ".join(target_overlap) or "not overlapping",
        "indication": ", ".join(indication_overlap) or "not overlapping",
        "target_overlap": target_overlap,
        "indication_overlap": indication_overlap,
        "confidence": round(confidence, 3),
    }


def _asset_index_relationship(company: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any] | None:
    peer_name = asset.get("company_name") or "Unknown peer"
    if same_company({"company": peer_name, "ticker": asset.get("ticker")}, company):
        return None
    target_overlap = target_overlap_terms(
        _listify(company.get("targets")) + _listify(company.get("aliases")),
        [asset.get("target"), asset.get("asset_name"), asset.get("modality")],
    )
    indication_overlap = indication_overlap_terms(company.get("indications"), [asset.get("indication")])
    rel = _relationship(
        peer_name, asset.get("ticker") or "", target_overlap, indication_overlap, float(asset.get("confidence") or 0.6)
    )
    if rel:
        rel.update(
            {
                "source": "dynamic_peer_asset_index",
                "evidence_url": asset.get("source_url") or "",
                "evidence_title": asset.get("source_title") or f"{peer_name} watchlist entry",
            }
        )
    return rel


def infer_watchlist_relationship(company: dict[str, Any], peer: dict[str, Any]) -> dict[str, Any] | None:
    if company.get("id") == peer.get("id") or same_company(peer, company):
        return None
    target_overlap = target_overlap_terms(
        _listify(company.get("targets")) + _listify(company.get("aliases")),
        _listify(peer.get("targets")) + _listify(peer.get("aliases")),
    )
    indication_overlap = indication_overlap_terms(company.get("indications"), peer.get("indications"))
    peer_name = peer.get("name") or peer.get("company") or "Unknown peer"
    rel = _relationship(peer_name, peer.get("ticker") or "", target_overlap, indication_overlap, 0.62)
    if rel:
        urls = _listify(peer.get("source_urls"))
        rel.update(
            {
                "source": "conservative_watchlist_overlap",
                "evidence_url": urls[0] if urls else "",
                "evidence_title": f"{peer_name} watchlist entry",
            }
        )
    return rel


def infer_relationships(
    company: dict[str, Any], companies: list[dict[str, Any]], assets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    found = [infer_watchlist_relationship(company, peer) for peer in companies]
    found += [_asset_index_relationship(company, asset) for asset in assets]
    return dedupe_peers([rel for rel in found if rel], company)


def enrich_company_peers(company: dict[str, Any]) -> dict[str, Any]:
    manual = company.get("peers") or []
    inferred: list[dict[str, Any]] = []
    if company.get("id"):
        companies = _active_companies()
        inferred = infer_relationships(company, companies, _asset_rows(companies))
    merged = dedupe_peers([*manual, *inferred], company)
    out = dict(company)
    out["peers"] = merged
    out["inferred_peer_count"] = sum(1 for peer in merged if peer.get("source"))
    return out


def sync_all_peer_relationships() -> int:
    companies = _active_companies()
    count = 0
    with transaction():
        assets = refresh_peer_asset_index(companies)
        execute("DELETE FROM peer_relationships")
        for company in companies:
            for rel in infer_relationships(company, companies, assets):
                execute(
                    """
                    INSERT OR REPLACE INTO peer_relationships (
                        company_id, peer_name, peer_ticker, relationship_type,
                        target_overlap_json, indication_overlap_json, confidence, source, rationale, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', CURRENT_TIMESTAMP)
                    """,
                    (
                        company["id"],
                        rel["company"],
                        rel.get("ticker", ""),
                        rel["relationship_type"],
                        json_dumps(rel.get("target_overlap", [])),
                        json_dumps(rel.get("indication_overlap", [])),
                        float(rel.get("confidence", 0.0)),
                        rel.get("source", "dynamic_peer_asset_index"),
                    ),
                )
                count += 1
    return count
