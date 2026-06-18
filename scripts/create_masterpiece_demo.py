#!/usr/bin/env python3
"""Seed official-source demo reports for a reviewer.

Usage:
    python scripts/create_masterpiece_demo.py

Then open http://127.0.0.1:8000 and go to the created report IDs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import init_db  
from app.services.demo_scenarios import seed_masterpiece_showcase, demo_readiness_summary  


def main() -> int:
    init_db()
    result = seed_masterpiece_showcase(auto_analyze=True)
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
