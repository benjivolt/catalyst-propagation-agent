from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from app.services.detection import classify_catalyst, detection_confidence, infer_direction, is_likely_catalyst
from app.services.detection import detect as detect_catalyst

__all__ = [
    "classify_catalyst",
    "clamp",
    "clean_text",
    "detect_catalyst",
    "detection_confidence",
    "humanize",
    "infer_direction",
    "is_http_url",
    "is_likely_catalyst",
    "iso",
    "json_dumps",
    "json_loads",
    "normalize_url",
    "parse_dt",
    "parse_list_field",
    "plain_text",
    "redact_secrets",
    "source_key",
    "token_set",
    "tokens",
    "truncate_sentences",
    "utc_now",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-/]{1,}")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_TRACKING_PARAMS = {"fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "igshid", "_hsenc", "_hsmi", "ref_src"}

LABELS = {
    "fda_decision": "FDA decision",
    "trial_readout": "trial readout",
    "licensing_deal": "licensing deal",
    "acquisition": "acquisition",
    "other_catalyst": "other catalyst",
    "all": "all events",
    "none": "no event",
    "needs_review": "needs review",
    "approved": "approved",
    "rejected": "rejected",
    "sec_edgar": "SEC EDGAR",
    "fda_press_announcements": "FDA press releases",
    "fda_press": "FDA press releases",
    "openfda_drugs": "openFDA Drugs@FDA",
    "openfda": "openFDA Drugs@FDA",
    "clinicaltrials_gov": "ClinicalTrials.gov",
    "pubmed": "PubMed",
    "company_news": "company news",
    "gdelt_news": "GDELT news",
    "manual": "manual entry",
    "peer_relationship": "watchlist note",
    "watchlist_peer_evidence": "watchlist note",
    "watchlist_note": "watchlist note",
    "official_metadata_trusted": "official source",
    "company_metadata_trusted": "company source, not fetched",
    "url_verified": "page checked",
    "url_verified_weak_match": "page checked, weak match",
    "fetch_failed": "fetch failed",
    "manual_unverified": "no source link",
    "manual_url_unchecked": "link not checked",
    "discovery_url_unchecked": "link not checked",
    "relationship_url_unchecked": "link not checked",
    "not_checked": "not checked",
    "not_verified": "not checked",
    "verified": "verified",
    "weakly_verified": "partly verified",
    "needs_human_review": "needs review",
    "unsupported": "unsupported",
    "contradicted": "contradicted",
    "strong": "strong",
    "weak": "weak",
    "validation": "validation",
    "competitive": "competitive",
    "partner": "partner",
    "neutral": "neutral",
    "same_target_validation": "same target",
    "same_indication_competition": "same disease",
    "mixed_target_and_indication": "same target and disease",
    "promote_to_catalyst": "promote to catalyst",
    "store_as_context": "keep as context",
    "reject_irrelevant": "reject",
    "needs_ai_review": "needs review",
    "pending": "pending",
    "true_positive": "confirmed",
    "partial_positive": "partly confirmed",
    "unsupported_prediction": "unsupported",
    "false_positive": "false alarm",
    "false_negative": "missed",
    "true_negative": "correctly rejected",
    "true_context": "context only",
    "finished": "finished",
    "finished_with_errors": "finished with errors",
    "failed": "failed",
    "running": "running",
    "skipped": "skipped",
    "sent": "sent",
    "unsupported_promotion": "unsupported promotion",
    "missed_catalyst": "missed catalyst",
    "partial_evidence": "partial evidence",
    "promote_threshold": "promote threshold",
}


def humanize(value: Any) -> str:
    text = str(value or "")
    return LABELS.get(text, text.replace("_", " "))


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def parse_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = raw.replace("Z", "+00:00")
    if len(candidate) > 10 and candidate[10] == " ":
        candidate = candidate[:10] + "T" + candidate[11:]
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def json_loads(value: Any, default: Any = None) -> Any:
    fallback = [] if default is None else default
    if value is None or value == "":
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def tokens(
    text: Any, stopwords: Iterable[str] = (), *, pattern: re.Pattern[str] = TOKEN_RE, min_len: int = 2
) -> list[str]:
    raw = " ".join(str(x) for x in text) if isinstance(text, (list, tuple, set)) else str(text or "")
    stop = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
    out: list[str] = []
    for token in pattern.findall(raw.lower()):
        token = token.strip("-/")
        if len(token) >= min_len and token not in stop:
            out.append(token)
    return out


def token_set(
    text: Any, stopwords: Iterable[str] = (), *, pattern: re.Pattern[str] = TOKEN_RE, min_len: int = 2
) -> set[str]:
    return set(tokens(text, stopwords, pattern=pattern, min_len=min_len))


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def clean_text(text: Any, max_chars: int = 900) -> str:
    if not text:
        return ""
    value = html.unescape(str(text))
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "html.parser").get_text(" ")
    return _shorten(re.sub(r"\s+", " ", value).strip(), max_chars)


def plain_text(text: Any, max_chars: int = 4000) -> str:
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text).replace("\r\n", "\n").split("\n")]
    value = "\n".join(line for line in lines if line)
    return value if len(value) <= max_chars else value[:max_chars].rstrip()


def truncate_sentences(text: Any, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    kept = ""
    for sentence in _SENTENCE_END_RE.split(value):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    return kept or _shorten(value, max_chars)


def is_http_url(value: Any) -> bool:
    parts = urlparse(str(value or "").strip())
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def normalize_url(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    return urlunparse(parsed._replace(path=parsed.path.rstrip("/"), query=urlencode(query), fragment=""))


def source_key(*parts: str) -> str:
    normalized = "|".join((p or "").strip().lower() for p in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def parse_list_field(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not value or not isinstance(value, str):
        return []
    return [v.strip() for v in re.split(r"[,;\n]", value) if v.strip()]


SECRET_PATTERNS = (
    (re.compile(r"(api_key=)[^&\s'\"]+", re.I), r"\1[redacted]"),
    (re.compile(r"https://hooks\.slack\.com/[^\s'\"]+"), "[slack webhook]"),
    (re.compile(r"xox[abposr]-[A-Za-z0-9-]{8,}"), "[redacted slack token]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "[redacted key]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}", re.I), r"\1[redacted]"),
)


def redact_secrets(text: Any) -> str:
    value = str(text or "")
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value
