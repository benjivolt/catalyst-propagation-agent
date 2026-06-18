#!/usr/bin/env python3
"""Live OpenAI key/model smoke test for the Catalyst Propagation Agent.

Run after editing .env:
    python scripts/check_openai_key.py

It uses the same OpenAI SDK + Responses API structured-output path as the
analyzer, but sends only a tiny health-check prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import init_db  
from app.services.openai_health import openai_status  


def main() -> int:
    init_db()
    status = openai_status(run_live_test=True)
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
    if status.get("status") == "ok":
        print("\n✅ OpenAI is working for this project. Reports will use the real OpenAI analyzer when OPENAI_API_KEY is set.")
        return 0
    print("\n❌ OpenAI is not verified yet. Check OPENAI_API_KEY, OPENAI_MODEL, billing/rate limits, and your internet connection.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
