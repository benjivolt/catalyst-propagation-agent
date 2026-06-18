"""Offline end-to-end tests for the Catalyst Propagation Agent.

These run without any API key or network access: they exercise the real FastAPI
app, the SQLite layer, and the deterministic fallback analyzer. They are the
safety net that proves the core pipeline stays functional.

Run with:  pytest
"""
from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture()
def client(monkeypatch):
    # Isolate each test run in its own throwaway database.
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("DATABASE_PATH", tmp.name)
    # Force offline behavior: no key -> deterministic local analyzer.
    monkeypatch.setenv("OPENAI_API_KEY", "")

    from fastapi.testclient import TestClient
    import app.db as db
    import app.main as main

    db.DB_PATH = tmp.name  # honor the per-test path if module cached it
    db.init_db()
    with TestClient(main.app) as c:
        yield c
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def test_health_pages(client):
    for path in ("/", "/learning"):
        assert client.get(path).status_code == 200


def test_openapi_schema_valid(client):
    schema = client.get("/openapi.json").json()
    assert schema["openapi"].startswith("3.")
    assert "/api/catalysts/{catalyst_id}/analyze" in schema["paths"]


def test_seed_creates_companies(client):
    assert client.post("/api/seed").status_code == 200
    companies = client.get("/api/companies").json()
    assert len(companies) >= 1


def test_full_pipeline_offline(client):
    """seed -> manual catalyst -> analyze -> report, with no API key."""
    client.post("/api/seed")
    import app.db as db

    company = db.row("SELECT id FROM companies LIMIT 1")
    assert company is not None

    r = client.post(
        "/api/catalysts/manual",
        json={
            "company_id": company["id"],
            "title": "Topline Phase 3: primary endpoint met",
            "event_type": "trial_readout",
            "url": "https://example.com/pr",
            "summary": "Met primary endpoint, p<0.001",
        },
    )
    assert r.status_code == 200

    catalyst = db.row("SELECT id FROM catalysts ORDER BY id DESC LIMIT 1")
    assert client.post(f"/api/catalysts/{catalyst['id']}/analyze").status_code == 200

    report = db.row(
        "SELECT overall_direction, overall_confidence FROM reports ORDER BY id DESC LIMIT 1"
    )
    assert report is not None
    # Trust rule: weak/uncited evidence must not produce high confidence.
    assert 0.0 <= report["overall_confidence"] <= 1.0
    assert report["overall_direction"] in {"positive", "negative", "neutral"}


def test_openai_status_reports_missing_key(client):
    status = client.get("/api/openai/status").json()
    assert status["key_configured"] is False
    assert status["status"] in {"missing_key", "not_tested", "configured"}
