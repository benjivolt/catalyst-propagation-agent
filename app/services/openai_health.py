from __future__ import annotations

import json
from typing import Any, Optional

try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  

from app.config import settings
from app.db import execute, row
from app.services.openai_utils import openai_model_candidates, output_text as openai_output_text, response_usage as openai_response_usage

HEALTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "service": {"type": "string"},
        "purpose": {"type": "string"},
    },
    "required": ["ok", "service", "purpose"],
}


def _usage(response: Any) -> dict[str, int]:
    return openai_response_usage(response)


def _output_text(response: Any) -> str:
    return openai_output_text(response)


def _record(status: dict[str, Any]) -> None:
    try:
        execute(
            """
            INSERT INTO openai_health_checks (
                status, model, sdk_installed, key_configured, responses_api_available,
                input_tokens, output_tokens, total_tokens, message, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(status.get("status") or "unknown"),
                str(status.get("model") or settings.openai_healthcheck_model),
                1 if status.get("sdk_installed") else 0,
                1 if status.get("key_configured") else 0,
                1 if status.get("responses_api_available") else 0,
                int(status.get("input_tokens") or 0),
                int(status.get("output_tokens") or 0),
                int(status.get("total_tokens") or 0),
                str(status.get("message") or "")[:1000],
                json.dumps(status, ensure_ascii=False),
            ),
        )
    except Exception:
        
        pass


def latest_openai_health() -> Optional[dict[str, Any]]:
    item = row("SELECT * FROM openai_health_checks ORDER BY id DESC LIMIT 1")
    if not item:
        return None
    try:
        item["raw"] = json.loads(item.get("raw_json") or "{}")
    except Exception:
        item["raw"] = {}
    return item


def openai_status(run_live_test: bool = False) -> dict[str, Any]:
    """Return configuration status and optionally perform a real low-token API check.

    The live check sends a tiny structured-output request through the same OpenAI
    SDK/Responses path the analyzer uses, proving that the key, model, SDK, and
    response parsing work before the user spends tokens on a full biotech report.
    """
    key_configured = bool(settings.openai_api_key)
    sdk_installed = OpenAI is not None
    base = {
        "provider": "openai",
        "model": settings.openai_healthcheck_model,
        "analysis_model": settings.openai_model,
        "key_configured": key_configured,
        "sdk_installed": sdk_installed,
        "strict_analysis": bool(settings.openai_strict_analysis),
        "responses_api_available": False,
        "status": "not_tested",
        "message": "OpenAI live test has not been run from this process.",
        "model_candidates": openai_model_candidates(settings.openai_healthcheck_model),
        "model_attempts": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "last_check": latest_openai_health(),
    }
    if not key_configured:
        base.update({"status": "missing_key", "message": "OPENAI_API_KEY is not set. Add it to .env, then restart the app."})
        return base
    if not sdk_installed:
        base.update({"status": "sdk_missing", "message": "The openai Python package is not installed. Run pip install -r requirements.txt."})
        return base
    if not run_live_test:
        base.update({"status": "configured", "message": "OPENAI_API_KEY is present. Run the live OpenAI test to verify the key/model. If the primary model is unavailable, the configured fallback list will be tried."})
        return base

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        responses_api_available = hasattr(client, "responses")
        base["responses_api_available"] = responses_api_available
        if not responses_api_available:
            raise RuntimeError("Installed OpenAI SDK does not expose client.responses; upgrade openai>=2.0.")

        attempts: list[dict[str, str]] = []
        last_exc: Exception | None = None
        for model_name in openai_model_candidates(settings.openai_healthcheck_model):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": "Return only the requested JSON schema."},
                        {"role": "user", "content": "OpenAI health check for the Catalyst Propagation Agent."},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "openai_health_check",
                            "schema": HEALTH_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                usage = _usage(response)
                raw_text = _output_text(response)
                parsed = json.loads(raw_text or "{}")
                ok = bool(parsed.get("ok")) and str(parsed.get("service", "")).lower() == "openai"
                attempts.append({"model": model_name, "status": "ok" if ok else "unexpected_response"})
                base.update(
                    {
                        "status": "ok" if ok else "unexpected_response",
                        "model": model_name,
                        "configured_primary_model": settings.openai_healthcheck_model,
                        "model_candidates": openai_model_candidates(settings.openai_healthcheck_model),
                        "model_attempts": attempts,
                        "message": "OpenAI API key, model, Responses API, and structured JSON parsing worked." if ok else "OpenAI responded, but not with the expected health-check JSON.",
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "response_preview": parsed,
                    }
                )
                _record(base)
                return base
            except Exception as model_exc:
                last_exc = model_exc
                attempts.append({"model": model_name, "status": "failed", "error": str(model_exc)[:400]})

        raise last_exc or RuntimeError("OpenAI live test failed for all configured model candidates")
    except Exception as exc:
        base.update({
            "status": "failed",
            "message": f"OpenAI live test failed: {exc}",
            "model_candidates": openai_model_candidates(settings.openai_healthcheck_model),
            "model_attempts": locals().get("attempts", []),
        })
        _record(base)
        return base
