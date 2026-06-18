from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, row, rows
from app.services.verified_evidence_graph import build_verified_evidence_graph, evidence_graph_for_report


def main() -> None:
    init_db()
    report = row("SELECT id FROM reports ORDER BY id DESC LIMIT 1")
    if not report:
        raise SystemExit("No reports exist yet. Create/analyze a catalyst first.")
    audit = build_verified_evidence_graph(int(report["id"]))
    graph = evidence_graph_for_report(int(report["id"]))
    summary = graph["summary"]
    assert int(summary.get("total") or 0) > 0, "expected atomic claims"
    assert rows("SELECT id FROM source_snapshots LIMIT 1"), "expected source snapshots"
    assert rows("SELECT id FROM source_paragraphs LIMIT 1"), "expected source paragraphs"
    assert rows("SELECT id FROM claim_evidence_spans LIMIT 1"), "expected evidence spans"
    assert rows("SELECT id FROM claim_verification_summary LIMIT 1"), "expected claim summaries"
    print(json.dumps({"status": "ok", "report_id": report["id"], "audit": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
