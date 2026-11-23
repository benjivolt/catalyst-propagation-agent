#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
fi

echo "Starting Catalyst Propagation at http://127.0.0.1:8000"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
