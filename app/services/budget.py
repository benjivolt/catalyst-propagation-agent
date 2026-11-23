from __future__ import annotations

from typing import Any

from app.config import settings
from app.db import execute, row
from app.services.openai_utils import response_usage
from app.services.utils import json_dumps, redact_secrets

LLM_CALL_FLOOR_USD = 0.012


def daily_spend() -> float:
    result = row(
        "SELECT COALESCE(SUM(estimated_cost), 0) AS spent FROM budget_ledger "
        "WHERE created_at >= datetime('now', 'start of day')"
    )
    return float((result or {}).get("spent") or 0.0)


def budget_allows(estimated_cost: float) -> bool:
    budget = float(settings.scheduler_daily_budget_usd)
    if budget <= 0:
        return True
    return daily_spend() + float(estimated_cost) <= budget


def record_spend(
    amount: float,
    reason: str,
    *,
    source_name: str = "",
    model: str = "none",
    input_tokens: int = 0,
    output_tokens: int = 0,
    run_id: int | None = None,
    job_id: int | None = None,
    company_id: int | None = None,
) -> None:
    execute(
        """
        INSERT INTO budget_ledger (run_id, job_id, company_id, source_name, model, input_tokens, output_tokens,
            estimated_cost, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, job_id, company_id, source_name, model, input_tokens, output_tokens, float(amount), reason),
    )


def token_cost(usage: dict[str, int]) -> float:
    return (usage["input_tokens"] / 1_000_000 * float(settings.openai_input_cost_per_1m)) + (
        usage["output_tokens"] / 1_000_000 * float(settings.openai_output_cost_per_1m)
    )


def record_llm_usage(
    purpose: str,
    response: Any | None,
    *,
    model: str,
    status: str = "success",
    catalyst_id: int | None = None,
    error: str = "",
    attempts: list[dict[str, str]] | None = None,
) -> float:
    usage = response_usage(response)
    cost = token_cost(usage)
    if status == "success" and cost <= 0:
        cost = LLM_CALL_FLOOR_USD
    raw: dict[str, Any] = {"purpose": purpose, "status": status, "model": model}
    if attempts:
        raw["model_attempts"] = attempts[-8:]
    if error:
        raw["error"] = redact_secrets(error)[:1000]
    execute(
        """
        INSERT INTO llm_usage (
            catalyst_id, model, provider, input_tokens, output_tokens, total_tokens, estimated_cost, status, raw_json
        )
        VALUES (?, ?, 'openai', ?, ?, ?, ?, ?, ?)
        """,
        (
            catalyst_id,
            model,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
            float(cost),
            status,
            json_dumps(raw),
        ),
    )
    record_spend(
        cost,
        f"{purpose}:{status}",
        source_name=f"openai_{purpose}",
        model=model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
    )
    return cost
