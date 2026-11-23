#!/usr/bin/env python3
"""Run one adaptive scheduler tick and exit, for cron or an external job runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db
from app.services.scheduler import run_adaptive_monitor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one adaptive scheduler tick.")
    parser.add_argument("--force", action="store_true", help="Check top-priority sources even when not due yet.")
    parser.add_argument("--no-analyze", action="store_true", help="Store detected catalysts without writing reports.")
    parser.add_argument("--limit", type=int, default=4, help="Max candidates to fetch per source.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Max company-source jobs this tick.")
    args = parser.parse_args()
    init_db()
    result = run_adaptive_monitor(
        auto_analyze=not args.no_analyze,
        per_source_limit=args.limit,
        due_only=not args.force,
        max_jobs=args.max_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"finished", "finished_with_errors"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
