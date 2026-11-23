#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db
from app.services.demo_scenarios import demo_readiness_summary, seed_demo_reports


def main() -> int:
    init_db()
    result = seed_demo_reports(auto_analyze=True)
    readiness = demo_readiness_summary()
    print(json.dumps({"created": result, "readiness": readiness}, indent=2, ensure_ascii=False, default=str))
    report_ids = result.get("report_ids") or []
    if report_ids:
        print("\nOpen these reports in the dashboard:")
        for rid in report_ids:
            print(f"  http://127.0.0.1:8000/report/{rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
