from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

CATALYST_KEYWORDS = {
    "trial_readout": [
        "topline", "readout", "trial results", "phase 1", "phase 2", "phase 3",
        "endpoint", "met the primary endpoint", "failed to meet", "clinical data",
        "interim analysis", "study results", "pivotal trial", "late-breaking", "abstract",
    ],
    "fda_decision": [
        "fda approves", "approved by the fda", "complete response letter", " crl ",
        "pdufa", "priority review", "accelerated approval", "advisory committee",
        "fda rejects", "label expansion", "biologics license application", "new drug application",
        "bla", "nda", "snda", "approval letter", "fda grants",
    ],
    "licensing_deal": [
        "licensing", "license agreement", "collaboration", "strategic partnership",
        "upfront payment", "milestone payments", "royalties", "exclusive rights", "option agreement",
    ],
    "acquisition": [
        "acquires", "to acquire", "acquisition", "merger", "buyout", "tender offer",
        "definitive agreement", "all-cash transaction",
    ],
}

DIRECTION_POSITIVE = [
    "positive", "met the primary endpoint", "statistically significant", "approved",
    "breakthrough", "priority review", "acquired", "upfront payment", "validated",
    "improved", "superior", "first treatment", "granted", "acceptance",
]
DIRECTION_NEGATIVE = [
    "failed", "did not meet", "complete response letter", "crl", "rejected",
    "terminated", "discontinued", "safety signal", "serious adverse", "clinical hold",
    "inferior", "missed", "delay",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(text: str, max_chars: int = 900) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return text


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    parsed = parsed._replace(fragment="", query="")
    return urlunparse(parsed).rstrip("/")


def source_key(*parts: str) -> str:
    normalized = "|".join((p or "").strip().lower() for p in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


# Detection delegates to the weighted, calibrated engine in detection.py. These
# thin wrappers preserve the long-standing API used across the codebase while the
# real logic (word-boundary matching, signal weighting, forward-looking handling,
# negation-aware direction) lives in one auditable, unit-tested place.
from app.services.detection import (  # noqa: E402
    classify_catalyst,
    detect as detect_catalyst,
    detection_confidence,
    infer_direction,
    is_likely_catalyst,
)

__all__ = [
    "CATALYST_KEYWORDS",
    "classify_catalyst",
    "clean_text",
    "detect_catalyst",
    "detection_confidence",
    "infer_direction",
    "is_likely_catalyst",
    "normalize_url",
    "now_iso",
    "parse_list_field",
    "safe_json",
    "source_key",
]


def parse_list_field(value: Union[str, list[str], None]) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,;\n]", value) if v.strip()]
    return []


def safe_json(value: Any, default: Any) -> Any:
    return value if value is not None else default
