from __future__ import annotations

"""Production reliability helpers for source monitoring.

These helpers make the monitor less fragile by using deterministic event
fingerprints, retry/backoff wrappers, source-health records, and freshness
checks. They are intentionally small and dependency-light so tests/local demos
remain fast.
"""

import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from app.db import execute, row
from app.services.utils import clean_text, normalize_url, source_key

T = TypeVar("T")


def canonical_event_fingerprint(company_id: int, event_type: str, title: str, event_date: str | None = None, url: str = "", excerpt: str = "") -> str:
    title_norm = " ".join(clean_text(title or excerpt, 360).lower().split())
    
    title_norm = title_norm.replace("announces", "announce").replace("announced", "announce")[:220]
    date_bucket = (event_date or "undated")[:10]
    url_norm = normalize_url(url)
    
    return source_key("event_cluster_v2", str(company_id), event_type or "other_catalyst", date_bucket, url_norm or title_norm)


def content_fingerprint(candidates: list[Any]) -> str:
    parts: list[str] = []
    for candidate in candidates:
        if hasattr(candidate, "to_dict"):
            data = candidate.to_dict()
        elif isinstance(candidate, dict):
            data = candidate
        else:
            data = {"value": str(candidate)}
        parts.append("|".join(str(data.get(k) or "") for k in ["source_type", "url", "title", "published_at"]))
    return source_key("snapshot_v2", "\n".join(sorted(parts)))


def run_with_retries(label: str, fn: Callable[[], T], *, attempts: int = 3, base_sleep: float = 0.25) -> tuple[T | None, dict[str, Any]]:
    errors: list[str] = []
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            result = fn()
            return result, {"status": "ok", "attempts": attempt, "started_at": started, "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "errors": errors}
        except Exception as exc:  
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < attempts:
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.05)
    return None, {"status": "failed", "attempts": attempts, "started_at": started, "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "errors": errors}


def record_source_health(company_id: int, source_name: str, status: str, *, latency_ms: int = 0, candidate_count: int = 0, error: str = "") -> None:
    existing = row("SELECT id, consecutive_failures FROM source_health WHERE company_id = ? AND source_name = ?", (company_id, source_name))
    failures = 0
    if status != "ok":
        failures = int((existing or {}).get("consecutive_failures") or 0) + 1
    execute(
        """
        INSERT INTO source_health (
            company_id, source_name, last_status, consecutive_failures, last_latency_ms,
            last_candidate_count, last_error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(company_id, source_name) DO UPDATE SET
            last_status = excluded.last_status,
            consecutive_failures = excluded.consecutive_failures,
            last_latency_ms = excluded.last_latency_ms,
            last_candidate_count = excluded.last_candidate_count,
            last_error = excluded.last_error,
            updated_at = CURRENT_TIMESTAMP
        """,
        (company_id, source_name, status, failures, int(latency_ms), int(candidate_count), clean_text(error, 600)),
    )


def should_skip_stale_candidate(candidate: Any, state: dict[str, Any]) -> bool:
    """Skip candidates older than the last successful check when a source provides dates.

    This is a conservative helper: if dates are unparsable, do not skip because
    biotech sources often publish free-form timestamps.
    """
    published = getattr(candidate, "published_at", None) if not isinstance(candidate, dict) else candidate.get("published_at")
    last_checked = state.get("last_checked_at")
    if not published or not last_checked:
        return False
    try:
        p = str(published).replace("Z", "+00:00").replace(" ", "T")
        l = str(last_checked).replace("Z", "+00:00").replace(" ", "T")
        if len(p) == 10:
            p += "T23:59:59+00:00"
        if len(l) == 10:
            l += "T00:00:00+00:00"
        pub_dt = datetime.fromisoformat(p)
        last_dt = datetime.fromisoformat(l)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return pub_dt.astimezone(timezone.utc) < last_dt.astimezone(timezone.utc)
    except Exception:
        return False
