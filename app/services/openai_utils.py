from __future__ import annotations

from typing import Any

from app.config import settings


def openai_model_candidates(primary: str | None = None) -> list[str]:
    """Return ordered, de-duplicated model candidates for resilient demos.

    The primary model remains the user's configured choice. Fallbacks are only
    tried when the OpenAI request itself fails, which keeps normal production
    behavior deterministic while making local grading less fragile.
    """
    raw = [primary or settings.openai_model]
    raw.extend(x.strip() for x in str(settings.openai_model_fallbacks or "").split(","))
    out: list[str] = []
    seen: set[str] = set()
    for model in raw:
        model = (model or "").strip()
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def get(name: str, *fallbacks: str) -> int:
        for key in (name, *fallbacks):
            value = getattr(usage, key, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return 0

    input_tokens = get("input_tokens", "prompt_tokens")
    output_tokens = get("output_tokens", "completion_tokens")
    total_tokens = get("total_tokens") or input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    try:
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") in {"output_text", "text"}:
                    chunks.append(str(getattr(content, "text", "")))
        return "".join(chunks)
    except Exception:
        return ""
