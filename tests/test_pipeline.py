from __future__ import annotations

from app import db


def test_openapi_schema_lists_analyze_route(client):
    schema = client.get("/openapi.json").json()
    assert schema["openapi"].startswith("3.")
    assert "/api/catalysts/{catalyst_id}/analyze" in schema["paths"]


def test_seed_creates_companies(client):
    assert client.post("/api/seed").status_code == 200
    assert len(client.get("/api/companies").json()) >= 1


def test_full_pipeline_offline(client):
    client.post("/api/seed")
    company = db.row("SELECT id FROM companies LIMIT 1")
    response = client.post(
        "/api/catalysts/manual",
        json={
            "company_id": company["id"],
            "title": "Topline Phase 3: primary endpoint met",
            "event_type": "trial_readout",
            "url": "https://example.com/pr",
            "excerpt": "Met primary endpoint, p<0.001",
        },
    )
    assert response.status_code == 200
    catalyst_id = response.json()["catalyst_id"]

    assert client.post(f"/api/catalysts/{catalyst_id}/analyze").status_code == 200
    report = db.row(
        "SELECT id, overall_direction, overall_confidence, raw_json FROM reports WHERE catalyst_id = ?",
        (catalyst_id,),
    )
    assert report is not None
    assert 0.0 <= report["overall_confidence"] <= 1.0
    assert report["overall_direction"] in {"positive", "negative", "neutral"}
    assert db.json_loads(report["raw_json"])["analysis_metadata"]["provider_used"] == "fallback"


def test_fallback_summary_reads_as_prose(demo):
    summary = db.row("SELECT plain_language_summary FROM reports ORDER BY id LIMIT 1")["plain_language_summary"]
    assert "fda_decision" not in summary
    assert "FDA decision" in summary


def test_openai_status_reports_missing_key(client):
    status = client.get("/api/openai/status").json()
    assert status["key_configured"] is False
    assert status["status"] in {"missing_key", "not_tested", "configured"}
