from __future__ import annotations

import json
from typing import Any, Optional

from app.db import company_from_row, execute, row, rows
from app.services.analyzer import analyze_catalyst, save_report
from app.services.sources import SourceDocumentCandidate, SourceFetchError, all_sources
from app.services.utils import classify_catalyst, clean_text, normalize_url, source_key
from app.services.peer_graph import enrich_company_peers, sync_all_peer_relationships
from app.services.source_verifier import verify_sources_for_catalyst
from app.services.reliability import canonical_event_fingerprint


def get_companies(active_only: bool = True) -> list[dict[str, Any]]:
    sql = "SELECT * FROM companies"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY name"
    return [company_from_row(r) for r in rows(sql)]


def get_company(company_id: int) -> Optional[dict[str, Any]]:
    result = row("SELECT * FROM companies WHERE id = ?", (company_id,))
    return company_from_row(result) if result else None


def create_company(data: dict[str, Any]) -> int:
    company_id = execute(
        """
        INSERT INTO companies (
            ticker, name, aliases_json, targets_json, indications_json,
            peers_json, source_urls_json, notes, active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (data.get("ticker") or "").upper().strip(),
            data.get("name", "").strip(),
            json.dumps(data.get("aliases") or [], ensure_ascii=False),
            json.dumps(data.get("targets") or [], ensure_ascii=False),
            json.dumps(data.get("indications") or [], ensure_ascii=False),
            json.dumps(data.get("peers") or [], ensure_ascii=False),
            json.dumps(data.get("source_urls") or [], ensure_ascii=False),
            clean_text(data.get("notes", ""), 1000),
            1 if data.get("active", True) else 0,
        ),
    )
    sync_all_peer_relationships()
    return company_id


def update_company(company_id: int, data: dict[str, Any]) -> None:
    execute(
        """
        UPDATE companies
        SET ticker = ?, name = ?, aliases_json = ?, targets_json = ?, indications_json = ?,
            peers_json = ?, source_urls_json = ?, notes = ?, active = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            (data.get("ticker") or "").upper().strip(),
            data.get("name", "").strip(),
            json.dumps(data.get("aliases") or [], ensure_ascii=False),
            json.dumps(data.get("targets") or [], ensure_ascii=False),
            json.dumps(data.get("indications") or [], ensure_ascii=False),
            json.dumps(data.get("peers") or [], ensure_ascii=False),
            json.dumps(data.get("source_urls") or [], ensure_ascii=False),
            clean_text(data.get("notes", ""), 1000),
            1 if data.get("active", True) else 0,
            company_id,
        ),
    )
    sync_all_peer_relationships()


def delete_company(company_id: int) -> None:
    execute("DELETE FROM companies WHERE id = ?", (company_id,))
    sync_all_peer_relationships()


def _event_cluster_key(company_id: int, event_type: str, title: str, event_date: str | None, excerpt: str = "") -> str:
    """Normalize likely duplicate catalyst events across SEC/PR/FDA/news sources."""
    return canonical_event_fingerprint(company_id, event_type, title, event_date, excerpt=excerpt)


def _save_candidate_as_catalyst(company_id: int, candidate: SourceDocumentCandidate) -> Optional[int]:
    
    
    key = source_key(str(company_id), candidate.key)
    existing = row("SELECT id FROM catalysts WHERE source_key = ?", (key,))
    if existing:
        return None
    event_type = classify_catalyst(candidate.title, candidate.excerpt)
    cluster_key = _event_cluster_key(company_id, event_type, candidate.title, candidate.published_at, candidate.excerpt)
    existing_cluster = row("SELECT id FROM catalysts WHERE company_id = ? AND event_cluster_key = ?", (company_id, cluster_key))
    if existing_cluster:
        return None
    catalyst_id = execute(
        """
        INSERT INTO catalysts (company_id, event_type, title, summary, event_date, source_key, event_cluster_key, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id,
            event_type,
            candidate.title,
            candidate.excerpt,
            candidate.published_at,
            key,
            cluster_key,
            json.dumps(candidate.raw, ensure_ascii=False),
        ),
    )
    execute(
        """
        INSERT INTO source_docs (catalyst_id, title, url, publisher, published_at, excerpt, source_type, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            catalyst_id,
            candidate.title,
            candidate.url,
            candidate.publisher,
            candidate.published_at,
            candidate.excerpt,
            candidate.source_type,
            json.dumps(candidate.raw, ensure_ascii=False),
        ),
    )
    return catalyst_id


def add_manual_catalyst(company_id: int, data: dict[str, Any]) -> int:
    title = clean_text(data.get("title", "Manual catalyst"), 250) or "Manual catalyst"
    url = normalize_url(data.get("url", "")) or f"manual://{source_key(str(company_id), title, data.get('excerpt', ''))}"
    excerpt = clean_text(data.get("excerpt", ""), 1600) or title
    publisher = clean_text(data.get("publisher", "Manual"), 120) or "Manual"
    event_type = data.get("event_type") or classify_catalyst(title, excerpt)
    key = source_key(str(company_id), "manual", url, title)
    existing = row("SELECT id FROM catalysts WHERE source_key = ?", (key,))
    if existing:
        return int(existing["id"])
    catalyst_id = execute(
        """
        INSERT INTO catalysts (company_id, event_type, title, summary, event_date, source_key, event_cluster_key, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id,
            event_type,
            title,
            excerpt,
            data.get("event_date"),
            key,
            _event_cluster_key(company_id, event_type, title, data.get("event_date"), excerpt),
            json.dumps({"manual": True}, ensure_ascii=False),
        ),
    )
    execute(
        """
        INSERT INTO source_docs (catalyst_id, title, url, publisher, published_at, excerpt, source_type, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            catalyst_id,
            title,
            url,
            publisher,
            data.get("event_date"),
            excerpt,
            "manual",
            json.dumps({"manual": True}, ensure_ascii=False),
        ),
    )
    try:
        from app.services.outcome_learning import CandidateReview, record_prediction
        direction, direction_conf = ("unknown", 0.0)
        try:
            from app.services.utils import infer_direction
            direction, direction_conf = infer_direction(title, excerpt)
        except Exception:
            pass
        review = CandidateReview(
            decision="promote_to_catalyst",
            event_type=event_type,
            confidence=max(0.72, direction_conf),
            direction=direction,
            company_detected=True,
            asset_detected=True,
            required_fields_present=True,
            risk_flags=["manual_seed_or_user_supplied_catalyst"],
            evidence_needed_later=["source verification and outcome check after analysis"],
            reasoning_summary="Manual catalyst was inserted by the user and recorded in the prediction ledger for outcome learning.",
            model_name="manual_user_submission",
        )
        record_prediction(company_id, "manual", key, url, title, review, gate_score=0.90, catalyst_id=catalyst_id)
    except Exception:
        pass
    return catalyst_id



def _peer_evidence_url(peer: dict[str, Any]) -> str:
    for key in ["evidence_url", "source_url", "relationship_url", "url"]:
        value = normalize_url(peer.get(key, ""))
        if value:
            return value
    return ""


def _attach_peer_relationship_source_docs(catalyst_id: int, company: dict[str, Any]) -> dict[str, Any]:
    """Persist peer-relationship evidence as source_docs for this catalyst.

    A peer read-through needs two proofs: the catalyst happened, and the peer
    really overlaps through target, indication, or partnership. This function
    turns explicit peer evidence URLs into citable source docs so reports do not
    rely only on the catalyst press release for peer claims.
    """
    peers = company.get("peers") or []
    for peer in peers:
        if not isinstance(peer, dict):
            continue
        url = _peer_evidence_url(peer)
        if not url:
            continue
        existing = row("SELECT id FROM source_docs WHERE catalyst_id = ? AND url = ?", (catalyst_id, url))
        if existing:
            continue
        peer_name = peer.get("company") or peer.get("name") or "Peer company"
        title = clean_text(peer.get("evidence_title") or f"Peer relationship evidence: {peer_name}", 250)
        relationship = peer.get("relationship") or peer.get("relationship_type") or "peer relationship"
        excerpt = clean_text(
            peer.get("evidence_excerpt")
            or (
                f"{peer_name} relationship evidence for catalyst propagation. "
                f"Relationship: {relationship}. Target/program: {peer.get('target') or peer.get('target_overlap') or 'not specified'}. "
                f"Indication: {peer.get('indication') or peer.get('indication_overlap') or 'not specified'}. "
                f"Expected read-through: {peer.get('expected_readthrough') or 'not specified'}."
            ),
            1400,
        )
        execute(
            """
            INSERT INTO source_docs (catalyst_id, title, url, publisher, published_at, excerpt, source_type, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalyst_id,
                title,
                url,
                clean_text(peer.get("evidence_publisher") or "Peer relationship evidence", 120),
                None,
                excerpt or title,
                "peer_relationship",
                json.dumps({"peer": peer_name, "relationship": relationship, "auto_added": True}, ensure_ascii=False),
            ),
        )

    source_docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    url_to_index = {normalize_url(doc.get("url", "")): idx for idx, doc in enumerate(source_docs, start=1)}
    updated_peers: list[dict[str, Any]] = []
    for peer in peers:
        if not isinstance(peer, dict):
            updated_peers.append(peer)
            continue
        peer_copy = dict(peer)
        url = _peer_evidence_url(peer_copy)
        if url and url in url_to_index:
            peer_copy["_relationship_source_index"] = url_to_index[url]
        updated_peers.append(peer_copy)
    out = dict(company)
    out["peers"] = updated_peers
    return out

def analyze_and_save(catalyst_id: int) -> int:
    catalyst = row(
        """
        SELECT c.*, co.name AS company_name, co.ticker, co.aliases_json, co.targets_json,
               co.indications_json, co.peers_json, co.source_urls_json, co.notes
        FROM catalysts c
        JOIN companies co ON c.company_id = co.id
        WHERE c.id = ?
        """,
        (catalyst_id,),
    )
    if not catalyst:
        raise ValueError(f"Unknown catalyst_id {catalyst_id}")
    company = {
        "id": catalyst["company_id"],
        "name": catalyst["company_name"],
        "ticker": catalyst["ticker"],
        "aliases": json.loads(catalyst["aliases_json"] or "[]"),
        "targets": json.loads(catalyst["targets_json"] or "[]"),
        "indications": json.loads(catalyst["indications_json"] or "[]"),
        "peers": json.loads(catalyst["peers_json"] or "[]"),
        "source_urls": json.loads(catalyst["source_urls_json"] or "[]"),
        "notes": catalyst.get("notes", ""),
    }
    company = enrich_company_peers(company)
    company = _attach_peer_relationship_source_docs(catalyst_id, company)
    
    
    verify_sources_for_catalyst(catalyst_id)
    source_docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    analysis = analyze_catalyst(company, catalyst, source_docs)
    return save_report(catalyst_id, analysis)


def run_monitor(auto_analyze: bool = True, per_source_limit: int = 6) -> dict[str, Any]:
    """Run the advanced cost-aware adaptive monitor on demand.

    The old implementation checked every company/source every time. This wrapper
    now delegates to the adaptive scheduler with due_only=False so the dashboard
    button still forces an immediate pass, but the order, budget, LLM gating,
    backoff, scoring, and reward tracking all come from the scheduler.
    """
    from app.services.scheduler import run_adaptive_monitor

    return run_adaptive_monitor(auto_analyze=auto_analyze, per_source_limit=per_source_limit, due_only=False)


def list_catalysts(limit: int = 50, company_id: Optional[int] = None) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ""
    if company_id is not None:
        where = "WHERE c.company_id = ?"
        params.append(company_id)
    params.append(limit)
    return rows(
        f"""
        SELECT c.*, co.name AS company_name, co.ticker,
               r.id AS report_id, r.overall_direction, r.overall_confidence
        FROM catalysts c
        JOIN companies co ON c.company_id = co.id
        LEFT JOIN reports r ON r.catalyst_id = c.id
        {where}
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        tuple(params),
    )


def get_report(report_id: int) -> Optional[dict[str, Any]]:
    report = row(
        """
        SELECT r.*, c.title AS catalyst_title, c.event_type, c.event_date, c.summary,
               co.name AS company_name, co.ticker
        FROM reports r
        JOIN catalysts c ON r.catalyst_id = c.id
        JOIN companies co ON c.company_id = co.id
        WHERE r.id = ?
        """,
        (report_id,),
    )
    if not report:
        return None
    effects = rows("SELECT * FROM effects WHERE report_id = ? ORDER BY layer, confidence DESC", (report_id,))
    for effect in effects:
        effect["claims"] = rows(
            """
            SELECT cl.*, sd.title AS source_title, sd.publisher, sd.url
            FROM claims cl
            LEFT JOIN source_docs sd ON sd.id = cl.source_doc_id
            WHERE cl.effect_id = ?
            ORDER BY cl.id
            """,
            (effect["id"],),
        )
    report["effects"] = effects
    report["source_docs"] = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (report["catalyst_id"],))
    audit = row("SELECT * FROM analysis_audits WHERE report_id = ?", (report_id,))
    if audit:
        try:
            audit["audit"] = json.loads(audit.get("audit_json") or "{}")
        except Exception:
            audit["audit"] = {}
    report["evidence_audit"] = audit
    try:
        report["analysis_raw"] = json.loads(report.get("raw_json") or "{}")
    except Exception:
        report["analysis_raw"] = {}
    report["biomedical_context_audit"] = (report.get("analysis_raw") or {}).get("biomedical_context_audit") or {}
    report["review"] = row("SELECT * FROM report_reviews WHERE report_id = ?", (report_id,))
    report["alert_deliveries"] = rows("SELECT * FROM alert_deliveries WHERE report_id = ? ORDER BY id DESC LIMIT 10", (report_id,))
    report["llm_usage"] = rows("SELECT * FROM llm_usage WHERE report_id = ? OR catalyst_id = ? ORDER BY id DESC LIMIT 10", (report_id, report["catalyst_id"]))
    report["source_correlations"] = rows("SELECT * FROM source_correlations WHERE catalyst_id = ? ORDER BY layer, entity", (report["catalyst_id"],))
    try:
        from app.services.verified_evidence_graph import evidence_graph_for_report
        report["evidence_graph"] = evidence_graph_for_report(report_id)
    except Exception:
        report["evidence_graph"] = {"summary": {}, "claims": []}
    return report


def seed_examples(force: bool = False) -> None:
    if rows("SELECT id FROM companies LIMIT 1") and not force:
        return
    examples = [
        {
            "ticker": "MDGL",
            "name": "Madrigal Pharmaceuticals",
            "aliases": ["Madrigal", "Madrigal Pharma", "Rezdiffra", "resmetirom"],
            "targets": ["THR-beta", "thyroid hormone receptor beta", "resmetirom"],
            "indications": ["MASH", "NASH", "metabolic dysfunction-associated steatohepatitis"],
            "source_urls": ["https://ir.madrigalpharma.com/news-releases"],
            "notes": "Track MASH/NASH readouts, label expansions, FDA/regulatory updates, and competitive oral metabolic therapies.",
            "peers": [
                {"company": "Viking Therapeutics", "ticker": "VKTX", "relationship": "same target and same indication competitor", "target": "VK2809 / thyroid hormone receptor beta (THR-beta)", "indication": "MASH/NASH", "evidence_url": "https://vikingtherapeutics.com/pipeline/vk2809/", "evidence_title": "Viking VK2809 pipeline evidence", "evidence_excerpt": "Viking describes VK2809 as a thyroid hormone receptor beta agonist being developed for metabolic disorders including NASH/MASH.", "expected_readthrough": "Mixed read-through: THR-beta biology validation can help the class, but same-indication competition can raise the efficacy/safety bar."},
                {"company": "89bio", "ticker": "ETNB", "relationship": "same indication competitor", "target": "pegozafermin / FGF21 analog", "indication": "MASH", "evidence_url": "https://89bio.com/pipeline/", "evidence_title": "89bio pegozafermin pipeline evidence", "evidence_excerpt": "89bio describes pegozafermin as an investigational FGF21 analog in clinical development for MASH.", "expected_readthrough": "Indication-level competitive bar; not same-target THR-beta biology."},
                {"company": "Akero Therapeutics", "ticker": "AKRO", "relationship": "same indication competitor", "target": "efruxifermin / FGF21 analog", "indication": "MASH", "evidence_url": "https://www.akerotx.com/pipeline/", "evidence_title": "Akero efruxifermin pipeline evidence", "evidence_excerpt": "Akero describes efruxifermin as an FGF21 analog program for MASH populations.", "expected_readthrough": "Indication-level competitive bar; not same-target THR-beta biology."},
            ],
            "active": True,
        },
        {
            "ticker": "CRSP",
            "name": "CRISPR Therapeutics",
            "aliases": ["CRISPR Therapeutics AG", "Casgevy", "exa-cel", "exagamglogene autotemcel"],
            "targets": ["BCL11A enhancer", "CRISPR/Cas9 ex vivo editing"],
            "indications": ["sickle cell disease", "transfusion-dependent beta thalassemia"],
            "source_urls": ["https://crisprtx.gcs-web.com/news-releases"],
            "notes": "Track ex vivo gene-editing catalysts, FDA decisions, safety updates, and partner economics with Vertex.",
            "peers": [
                {"company": "Vertex Pharmaceuticals", "ticker": "VRTX", "relationship": "development/commercial partner", "target": "Casgevy / BCL11A enhancer CRISPR-Cas9 editing", "indication": "SCD/TDT", "evidence_url": "https://www.casgevy.com/", "evidence_title": "CASGEVY official therapy evidence", "evidence_excerpt": "CASGEVY is presented as an ex vivo CRISPR/Cas9 edited cell therapy for sickle cell disease and transfusion-dependent beta thalassemia, commercialized with Vertex/CRISPR Therapeutics context.", "expected_readthrough": "Partner read-through when Casgevy economics, approvals, safety, or adoption change."},
                {"company": "bluebird bio", "ticker": "BLUE", "relationship": "same indication competitor", "target": "LYFGENIA / lentiviral gene therapy", "indication": "sickle cell disease", "evidence_url": "https://investor.bluebirdbio.com/news-releases/news-release-details/bluebird-bio-announces-fda-approval-lyfgeniatm-lovotibeglogene", "evidence_title": "bluebird LYFGENIA FDA approval evidence", "evidence_excerpt": "bluebird describes LYFGENIA as a one-time gene therapy approved for sickle cell disease patients with vaso-occlusive events.", "expected_readthrough": "Competitive read-through in curative SCD therapies, but not same-target BCL11A validation."},
                {"company": "Beam Therapeutics", "ticker": "BEAM", "relationship": "same indication and adjacent editing modality competitor", "target": "BEAM-101 / base editing", "indication": "sickle cell disease", "evidence_url": "https://beamtx.com/pipeline/", "evidence_title": "Beam pipeline evidence", "evidence_excerpt": "Beam lists base-editing programs including sickle-cell-disease programs, creating modality/indication overlap rather than exact BCL11A target overlap.", "expected_readthrough": "Editing-modality validation if safety/efficacy is strong; competitive if the curative therapy bar rises."},
            ],
            "active": True,
        },
        {
            "ticker": "ARGX",
            "name": "argenx SE",
            "aliases": ["argenx", "Vyvgart", "efgartigimod"],
            "targets": ["FcRn", "neonatal Fc receptor", "efgartigimod"],
            "indications": ["myasthenia gravis", "CIDP", "autoimmune diseases"],
            "source_urls": ["https://www.argenx.com/news"],
            "notes": "Track FcRn readouts, label expansions, autoimmune indication wins/failures, and competitor FcRn programs.",
            "peers": [
                {"company": "Immunovant", "ticker": "IMVT", "relationship": "same target competitor", "target": "IMVT-1402 / Batoclimab anti-FcRn", "indication": "autoimmune diseases including myasthenia gravis and CIDP", "evidence_url": "https://www.immunovant.com/pipeline", "evidence_title": "Immunovant anti-FcRn pipeline evidence", "evidence_excerpt": "Immunovant lists anti-FcRn programs including IMVT-1402 and batoclimab across autoimmune indications such as myasthenia gravis and CIDP.", "expected_readthrough": "Same-target validation if FcRn works; competitive if Vyvgart raises efficacy/safety/convenience bar."},
                {"company": "UCB", "ticker": "UCB.BR", "relationship": "same target and same indication competitor", "target": "Rystiggo / rozanolixizumab anti-FcRn", "indication": "generalized myasthenia gravis", "evidence_url": "https://www.ucb.com/newsroom/press-releases/article/ucb-announces-us-fda-approval-of-rystiggor-rozanolixizumab-noli-for-the-treatment-of-adults-with-generalized-myasthenia-gravis", "evidence_title": "UCB Rystiggo FcRn approval evidence", "evidence_excerpt": "UCB describes Rystiggo/rozanolixizumab as an FcRn-binding antibody approved for generalized myasthenia gravis.", "expected_readthrough": "Class validation plus direct competitive read-through in generalized myasthenia gravis."},
                {"company": "Johnson & Johnson", "ticker": "JNJ", "relationship": "same target and same indication competitor", "target": "nipocalimab / FcRn blocker", "indication": "generalized myasthenia gravis and autoimmune diseases", "evidence_url": "https://www.investor.jnj.com/investor-news/news-details/2026/IMAAVY-nipocalimab-aahu-shows-over-two-years-of-sustained-disease-control-in-a-broad-population-with-generalized-myasthenia-gravis-gMG/default.aspx", "evidence_title": "Johnson & Johnson IMAAVY nipocalimab FcRn evidence", "evidence_excerpt": "Johnson & Johnson describes IMAAVY/nipocalimab-aahu as an FcRn blocker in generalized myasthenia gravis with Phase 3 data and sustained IgG reduction, creating same-target and same-indication overlap with FcRn competitors.", "expected_readthrough": "Class validation for FcRn biology plus competitive read-through in generalized myasthenia gravis; magnitude depends on product profile and label."},
            ],
            "active": True,
        },
    ]
    for example in examples:
        existing = row("SELECT id FROM companies WHERE upper(ticker) = ? OR lower(name) = lower(?)", (example["ticker"], example["name"]))
        if existing and force:
            update_company(int(existing["id"]), example)
        elif not existing:
            create_company(example)
    sync_all_peer_relationships()
