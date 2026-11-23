from __future__ import annotations

from app.db import row, rows
from app.services.verified_evidence_graph import build_verified_evidence_graph, evidence_graph_for_report


def test_evidence_graph_is_built_for_report(demo):
    report_id = demo["report_ids"][0]

    audit = build_verified_evidence_graph(report_id)
    graph = evidence_graph_for_report(report_id)

    assert audit["grade"] in {"A", "B", "C", "D"}
    assert int(graph["summary"].get("total") or 0) > 0
    for table in ("source_snapshots", "source_paragraphs", "claim_evidence_spans", "claim_verification_summary"):
        assert rows(f"SELECT id FROM {table} LIMIT 1"), table


def test_effect_confidence_never_exceeds_report_cap(demo):
    report_id = demo["report_ids"][0]
    audit = build_verified_evidence_graph(report_id)
    report = row("SELECT overall_confidence FROM reports WHERE id = ?", (report_id,))
    assert report["overall_confidence"] <= audit["report_confidence_cap"] + 1e-9


def test_evidence_graph_route(client, demo):
    report_id = demo["report_ids"][0]
    assert client.post(f"/api/reports/{report_id}/verify-evidence-graph").status_code == 200
    assert "summary" in client.get(f"/api/reports/{report_id}/evidence-graph").json()
