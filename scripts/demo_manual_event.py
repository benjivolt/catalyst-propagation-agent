"""Create one offline demo catalyst and report (no network, no API key)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db
from app.services.monitor import add_manual_catalyst, analyze_and_save, get_companies, seed_examples


def main() -> None:
    init_db()
    seed_examples()
    company_id = get_companies(active_only=False)[0]["id"]
    catalyst_id = add_manual_catalyst(
        company_id,
        {
            "title": "Company announces positive topline Phase 2 trial results",
            "url": "https://example.com/demo-catalyst",
            "publisher": "Demo press release",
            "event_date": "2026-01-01",
            "event_type": "trial_readout",
            "excerpt": (
                "The study met the primary endpoint with statistically significant improvement versus placebo. "
                "This is demo text for local testing only."
            ),
        },
    )
    report_id = analyze_and_save(catalyst_id)
    print(f"Created demo report: http://127.0.0.1:8000/report/{report_id}")


if __name__ == "__main__":
    main()
