from __future__ import annotations

from typing import Any

from app.config import settings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def sdk_installed() -> bool:
    return OpenAI is not None


def openai_client(timeout: float = 60) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    return OpenAI(api_key=settings.openai_api_key, timeout=timeout, max_retries=1)


def structured_response(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> Any:
    return client.responses.create(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        max_output_tokens=max_output_tokens,
    )


def openai_model_candidates(primary: str | None = None) -> list[str]:
    raw = [primary or settings.openai_model]
    raw.extend(x.strip() for x in str(settings.openai_model_fallbacks or "").split(","))
    out: list[str] = []
    for model in raw:
        model = (model or "").strip()
        if model and model not in out:
            out.append(model)
    return out


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def get(*names: str) -> int:
        for key in names:
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = get("input_tokens", "prompt_tokens")
    output_tokens = get("output_tokens", "completion_tokens")
    total_tokens = get("total_tokens") or input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                chunks.append(str(getattr(content, "text", "")))
    return "".join(chunks)
