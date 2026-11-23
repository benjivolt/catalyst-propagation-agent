from __future__ import annotations

import json
from typing import Any

from app.db import execute, row, rows
from app.services.monitor import analyze_and_save, get_companies, seed_examples
from app.services.peer_graph import sync_all_peer_relationships
from app.services.source_verifier import verify_sources_for_catalyst
from app.services.utils import clean_text, source_key


def _company_by_ticker(ticker: str) -> dict[str, Any] | None:
    item = row("SELECT * FROM companies WHERE upper(ticker) = ?", (ticker.upper(),))
    if not item:
        return None
    from app.db import company_from_row

    return company_from_row(item)


def _ensure_seed_company(ticker: str) -> int:
    seed_examples(force=True)
    company = _company_by_ticker(ticker)
    if not company:
        raise RuntimeError(f"Seed company {ticker} was not created")
    return int(company["id"])


def _insert_curated_catalyst(
    company_id: int, event_type: str, title: str, summary: str, event_date: str, source_docs: list[dict[str, Any]]
) -> int:
    cluster = source_key(str(company_id), event_type, event_date, title[:180])
    key = source_key(str(company_id), "demo", title, event_date)
    existing = row("SELECT id FROM catalysts WHERE source_key = ?", (key,))
    if existing:
        catalyst_id = int(existing["id"])
    else:
        catalyst_id = execute(
            """
            INSERT INTO catalysts (company_id, event_type, title, summary, event_date, source_key, event_cluster_key,
                raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                event_type,
                clean_text(title, 250),
                clean_text(summary, 1800),
                event_date,
                key,
                cluster,
                json.dumps({"curated_demo": True, "purpose": "demo_reports"}, ensure_ascii=False),
            ),
        )
    for doc in source_docs:
        url = doc.get("url", "")
        if row("SELECT id FROM source_docs WHERE catalyst_id = ? AND url = ?", (catalyst_id, url)):
            continue
        execute(
            """
            INSERT INTO source_docs (catalyst_id, title, url, publisher, published_at, excerpt, source_type, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalyst_id,
                clean_text(doc.get("title", title), 300),
                url,
                clean_text(doc.get("publisher", "Curated source"), 160),
                doc.get("published_at"),
                clean_text(doc.get("excerpt", summary), 1600),
                doc.get("source_type", "company_news"),
                json.dumps({"curated_demo": True, **(doc.get("raw") or {})}, ensure_ascii=False),
            ),
        )
    return catalyst_id


def seed_demo_reports(auto_analyze: bool = True) -> dict[str, Any]:
    seed_examples(force=True)
    sync_all_peer_relationships()
    created: list[dict[str, Any]] = []

    mdgl_id = _ensure_seed_company("MDGL")
    mdgl_catalyst = _insert_curated_catalyst(
        mdgl_id,
        "fda_decision",
        "FDA approves Rezdiffra/resmetirom as the first treatment for adults with noncirrhotic "
        "NASH/MASH with moderate to advanced fibrosis",
        "FDA approval of Rezdiffra/resmetirom is a well-documented regulatory catalyst for THR-beta "
        "biology, the MASH treatment landscape, and same-indication peers.",
        "2024-03-14",
        [
            {
                "title": "FDA approval evidence for Rezdiffra/resmetirom in NASH/MASH",
                "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-first-treatment-patients-liver-scarring-due-fatty-liver-disease",
                "publisher": "U.S. Food and Drug Administration",
                "published_at": "2024-03-14",
                "source_type": "fda_press_announcements",
                "excerpt": (
                    "FDA approved Rezdiffra (resmetirom) for adults with noncirrhotic NASH with moderate to "
                    "advanced liver fibrosis, to be used with diet and exercise. The FDA page also describes "
                    "Rezdiffra as a partial activator of the thyroid hormone receptor and notes accelerated "
                    "approval based on surrogate endpoints from a randomized placebo-controlled trial."
                ),
            }
        ],
    )
    created.append({"ticker": "MDGL", "catalyst_id": mdgl_catalyst})

    crsp_id = _ensure_seed_company("CRSP")
    crsp_catalyst = _insert_curated_catalyst(
        crsp_id,
        "fda_decision",
        "FDA approves CASGEVY/exa-cel, a CRISPR/Cas9 cell therapy, for sickle cell disease",
        "FDA approval of CASGEVY/exa-cel is a high-impact regulatory catalyst for CRISPR/Cas9 "
        "editing, sickle cell disease, and curative gene-therapy competitors/partners.",
        "2023-12-08",
        [
            {
                "title": "FDA approval evidence for first gene therapies in sickle cell disease",
                "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-first-gene-therapies-treat-patients-sickle-cell-disease",
                "publisher": "U.S. Food and Drug Administration",
                "published_at": "2023-12-08",
                "source_type": "fda_press_announcements",
                "excerpt": (
                    "FDA approved Casgevy and Lyfgenia as cell-based gene therapies for sickle cell disease "
                    "patients age 12 and older. FDA describes Casgevy as the first FDA-approved therapy using "
                    "CRISPR/Cas9 genome-editing technology, with modified blood stem cells intended to raise "
                    "fetal hemoglobin."
                ),
            },
            {
                "title": "FDA CASGEVY product page evidence",
                "url": "https://www.fda.gov/vaccines-blood-biologics/casgevy",
                "publisher": "U.S. Food and Drug Administration",
                "published_at": None,
                "source_type": "fda_press_announcements",
                "excerpt": (
                    "FDA lists CASGEVY, proper name exagamglogene autotemcel (exa-cel), manufactured by Vertex "
                    "Pharmaceuticals, with indications for sickle cell disease and transfusion-dependent beta "
                    "thalassemia in patients 12 years of age and older."
                ),
            },
            {
                "title": "FDA LYFGENIA product page evidence for sickle-cell competitive context",
                "url": "https://www.fda.gov/vaccines-blood-biologics/lyfgenia",
                "publisher": "U.S. Food and Drug Administration",
                "published_at": None,
                "source_type": "fda_press_announcements",
                "excerpt": (
                    "FDA lists LYFGENIA/lovotibeglogene autotemcel from bluebird bio as a gene therapy for "
                    "sickle cell disease patients 12 years of age or older with a history of vaso-occlusive "
                    "events, creating same-indication competitive context for CASGEVY."
                ),
            },
        ],
    )
    created.append({"ticker": "CRSP", "catalyst_id": crsp_catalyst})

    argx_id = _ensure_seed_company("ARGX")
    argx_catalyst = _insert_curated_catalyst(
        argx_id,
        "fda_decision",
        "UCB receives U.S. FDA approval for RYSTIGGO/rozanolixizumab-noli in generalized myasthenia gravis",
        "The RYSTIGGO approval provides a same-target/same-indication FcRn comparator catalyst for "
        "argenx/Vyvgart and other FcRn peers, with both class-validation and competition effects.",
        "2023-06-27",
        [
            {
                "title": "UCB RYSTIGGO FDA approval evidence",
                "url": "https://www.ucb.com/newsroom/press-releases/article/ucb-announces-us-fda-approval-of-rystiggor-rozanolixizumab-noli-for-the-treatment-of-adults-with-generalized-myasthenia-gravis",
                "publisher": "UCB",
                "published_at": "2023-06-27",
                "source_type": "company_news",
                "excerpt": (
                    "UCB announced U.S. FDA approval of RYSTIGGO (rozanolixizumab-noli) for adults with "
                    "generalized myasthenia gravis who are anti-AChR or anti-MuSK antibody positive. UCB "
                    "describes rozanolixizumab-noli as an antibody that binds the neonatal Fc receptor (FcRn), "
                    "reducing circulating IgG."
                ),
            }
        ],
    )
    created.append({"ticker": "ARGX", "catalyst_id": argx_catalyst})

    reports: list[int] = []
    for item in created:
        catalyst_id = int(item["catalyst_id"])
        verify_sources_for_catalyst(catalyst_id, fetch=False)
        if auto_analyze:
            reports.append(analyze_and_save(catalyst_id))

    return {
        "status": "seeded",
        "companies": len(get_companies(active_only=False)),
        "curated_catalysts": len(created),
        "catalyst_ids": [int(x["catalyst_id"]) for x in created],
        "report_ids": reports,
        "message": "Created official-source showcase catalysts and analyzed reports."
        if auto_analyze
        else "Created official-source showcase catalysts.",
    }


def demo_readiness_summary() -> dict[str, Any]:
    audits = rows(
        """
        SELECT aa.evidence_grade, COUNT(*) AS count
        FROM analysis_audits aa
        GROUP BY aa.evidence_grade
        ORDER BY aa.evidence_grade
        """
    )
    review = rows("SELECT status, COUNT(*) AS count FROM report_reviews GROUP BY status ORDER BY status")
    source_status = rows(
        "SELECT verification_status, COUNT(*) AS count FROM source_docs GROUP BY "
        "verification_status ORDER BY count DESC"
    )
    llm = rows(
        "SELECT status, model, COUNT(*) AS count, SUM(total_tokens) AS tokens FROM llm_usage GROUP "
        "BY status, model ORDER BY id DESC LIMIT 10"
    )
    return {
        "companies": len(rows("SELECT id FROM companies")),
        "catalysts": len(rows("SELECT id FROM catalysts")),
        "reports": len(rows("SELECT id FROM reports")),
        "sources": len(rows("SELECT id FROM source_docs")),
        "evidence_grades": audits,
        "review_statuses": review,
        "source_verification_statuses": source_status,
        "llm_usage": llm,
    }
