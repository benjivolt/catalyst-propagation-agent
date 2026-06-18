#!/usr/bin/env python3
"""External scheduler worker entry point.

Use this from cron, GitHub Actions, Cloud Run Jobs, Render Cron Jobs, or any
managed scheduler. It runs exactly one adaptive due tick and exits, which is
safer in production than relying only on the web process background thread.
"""
from __future__ import annotations

import argparse
import json

from app.db import init_db
from app.services.scheduler import run_adaptive_monitor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Catalyst Propagation adaptive scheduler tick.")
    parser.add_argument("--force", action="store_true", help="Check top-priority sources even when not due yet.")
    parser.add_argument("--no-analyze", action="store_true", help="Store detected catalysts but do not create LLM/fallback reports.")
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
