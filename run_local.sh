#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
  elif [ -f "env.example" ]; then
    cp env.example .env
  fi
fi

echo "Starting Catalyst Propagation at http://127.0.0.1:8000"
echo "OpenAI check: after adding OPENAI_API_KEY to .env, run: python scripts/check_openai_key.py"
uvicorn app.main:app --reload
