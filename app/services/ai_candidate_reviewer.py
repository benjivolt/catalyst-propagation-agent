from __future__ import annotations

from typing import Any

from app.services.outcome_learning import CandidateReview, ai_candidate_review, deterministic_candidate_review, record_scheduler_candidate_prediction


def review_candidate(company: dict[str, Any], candidate: Any, gate_score: float = 0.0) -> CandidateReview:
    return ai_candidate_review(company, candidate, gate_score)


def review_candidate_locally(company: dict[str, Any], candidate: Any, gate_score: float = 0.0) -> CandidateReview:
    return deterministic_candidate_review(company, candidate, gate_score)


def save_candidate_review(company: dict[str, Any], candidate: Any, review: CandidateReview, gate_score: float = 0.0, catalyst_id: int | None = None) -> int:
    return record_scheduler_candidate_prediction(company, candidate, gate_score, review=review, catalyst_id=catalyst_id)
