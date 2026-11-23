#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db
from app.services.openai_health import openai_status


def main() -> int:
    init_db()
    status = openai_status(run_live_test=True)
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
    if status.get("status") == "ok":
        print("\nOpenAI check passed. New reports will use the OpenAI analyzer.")
        return 0
    print("\nOpenAI check failed. Check OPENAI_API_KEY, OPENAI_MODEL, billing or rate limits, and network access.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
