from __future__ import annotations

from app.db import execute, row
from app.services.outcome_learning import (
    CandidateReview,
    learning_dashboard,
    record_prediction,
    run_due_outcome_checks,
    run_policy_update,
)


def _company() -> int:
    return execute(
        """
        INSERT INTO companies (
            ticker, name, aliases_json, targets_json, indications_json, peers_json, source_urls_json, notes, active
        )
        VALUES ('TEST', 'Outcome Learning Test Bio', '["TestDrug"]', '["TargetX"]', '["DiseaseY"]', '[]', '[]', '', 1)
        """
    )


def test_prediction_outcome_check_and_policy_update(db):
    company_id = _company()
    review = CandidateReview(
        decision="store_as_context",
        event_type="trial_readout",
        confidence=0.67,
        direction="unknown",
        company_detected=True,
        asset_detected=True,
        required_fields_present=False,
        missing_fields=["explicit endpoint result"],
        risk_flags=["test_context_candidate"],
        evidence_needed_later=["official readout confirmation"],
        reasoning_summary="Test prediction.",
        model_name="test_fixture",
    )
    prediction_id = record_prediction(
        company_id=company_id,
        source_type="pubmed",
        source_key_value=f"test-learning-{company_id}",
        source_url="https://example.invalid/test-learning",
        source_title="Test context item",
        review=review,
        gate_score=0.55,
    )
    execute(
        "UPDATE ai_prediction_outcome_checks SET due_at = CURRENT_TIMESTAMP WHERE prediction_id = ?", (prediction_id,)
    )

    outcome = run_due_outcome_checks(limit=10)
    policy = run_policy_update()
    dashboard = learning_dashboard()

    assert outcome["processed"] >= 1
    assert row("SELECT id FROM ai_predictions WHERE id = ?", (prediction_id,))
    prediction = row("SELECT outcome_status FROM ai_predictions WHERE id = ?", (prediction_id,))
    assert prediction["outcome_status"] != "pending"
    assert isinstance(policy, dict)
    assert dashboard["prediction_counts"]["total"] >= 1


def test_learning_routes(client):
    assert client.post("/api/learning/check-outcomes").status_code == 200
    assert client.post("/api/learning/policy-update").status_code == 200
    assert "prediction_counts" in client.get("/api/learning/dashboard").json()
