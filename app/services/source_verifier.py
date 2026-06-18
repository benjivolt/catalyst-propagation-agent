from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from app.config import settings
from app.db import execute, rows
from app.services.sources import fetch_url_text
from app.services.utils import clean_text, normalize_url, source_key

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-/]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "there", "their", "company", "source",
    "evidence", "relationship", "program", "trial", "study", "results", "press", "release", "pipeline",
    "therapy", "therapeutics", "pharmaceuticals", "biopharmaceutical", "announces", "announced",
}

OFFICIAL_DOMAIN_HINTS = {
    "sec_edgar": ["sec.gov"],
    "fda_press_announcements": ["fda.gov"],
    "openfda_drugs": ["open.fda.gov", "accessdata.fda.gov", "fda.gov"],
    "clinicaltrials_gov": ["clinicaltrials.gov"],
    "pubmed": ["pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"],
}


def _tokens(text: str) -> set[str]:
    return {
        t.strip("-/").lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t.strip("-/")) > 2 and t.lower() not in _STOPWORDS
    }


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _raw(doc: dict[str, Any]) -> dict[str, Any]:
    raw = doc.get("raw_json") or {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def _source_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "), 12000)


def _offline_verification(doc: dict[str, Any], expected_terms: Optional[list[str]] = None) -> dict[str, Any]:
    """Non-network source integrity audit.

    This is deliberately conservative. It does not claim a URL proves an excerpt;
    it only separates official metadata, manual placeholders, and relationship
    evidence that still needs URL-level verification.
    """
    source_type = str(doc.get("source_type") or "").lower()
    url = normalize_url(doc.get("url", ""))
    domain = _domain(url)
    raw = _raw(doc)
    expected_terms = expected_terms or []

    if not url or url.startswith("manual://"):
        return {
            "verification_status": "manual_unverified",
            "verification_score": 0.15,
            "verification_reason": "Manual/no external URL. Use for demos, but treat as analyst-entered evidence until replaced with a real URL.",
            "canonical_url": url,
            "content_hash": "",
        }

    for official_type, hints in OFFICIAL_DOMAIN_HINTS.items():
        if source_type == official_type and any(domain.endswith(h) for h in hints):
            return {
                "verification_status": "official_metadata_trusted",
                "verification_score": 0.82,
                "verification_reason": f"Official-source URL/domain matched {official_type}. Full page fetch was not required for this audit pass.",
                "canonical_url": url,
                "content_hash": source_key(doc.get("title", ""), doc.get("excerpt", ""))[:24],
            }

    if source_type in {"peer_relationship", "watchlist_peer_evidence"}:
        excerpt = str(doc.get("excerpt") or "")
        auto_added = bool(raw.get("auto_added"))
        placeholder = "placeholder" in excerpt.lower() or "verify" in excerpt.lower()
        excerpt_terms = _tokens(excerpt)
        title_terms = _tokens(str(doc.get("title") or ""))
        
        
        
        generic_demo_domain = domain in {"example.com", "example.org", "example.net"}
        if not placeholder and url and not generic_demo_domain and len(excerpt_terms | title_terms) >= 6:
            return {
                "verification_status": "relationship_metadata_trusted",
                "verification_score": 0.74,
                "verification_reason": "Peer relationship evidence has a real URL plus a specific target/indication/program excerpt. Offline metadata gate passed; fetch verification can still be run for final approval.",
                "canonical_url": url,
                "content_hash": source_key(doc.get("title", ""), excerpt)[:24],
            }
        return {
            "verification_status": "relationship_url_unchecked",
            "verification_score": 0.42 if not placeholder else 0.28,
            "verification_reason": "Peer relationship has a URL, but the page text was not fetched/verified. Confidence is capped until source verification is run." if not auto_added else "Watchlist peer evidence was auto-added from stored metadata; run URL verification or analyst review before approving.",
            "canonical_url": url,
            "content_hash": source_key(doc.get("title", ""), excerpt)[:24],
        }

    if source_type == "manual":
        return {
            "verification_status": "manual_url_unchecked",
            "verification_score": 0.32,
            "verification_reason": "Manual catalyst has a URL but page text was not fetched. Treat as unverified until source verification is run.",
            "canonical_url": url,
            "content_hash": source_key(doc.get("title", ""), doc.get("excerpt", ""))[:24],
        }

    
    expected = _tokens(" ".join(expected_terms))
    doc_terms = _tokens(" ".join([str(doc.get("title") or ""), str(doc.get("excerpt") or "")]))
    expected_overlap = len(expected & doc_terms) / max(len(expected), 1) if expected else 0.0
    generic_demo_domain = domain in {"example.com", "example.org", "example.net"}
    if source_type == "company_news" and url and not generic_demo_domain and len(doc_terms) >= 8 and not any(x in str(doc.get("excerpt") or "").lower() for x in ["placeholder", "verify before approving"]):
        return {
            "verification_status": "company_metadata_trusted",
            "verification_score": round(min(0.76, 0.72 + expected_overlap * 0.04), 3),
            "verification_reason": "Company-source URL has a specific title/excerpt and passed offline metadata review. Fetch verification can still be run before investment-grade approval.",
            "canonical_url": url,
            "content_hash": source_key(doc.get("title", ""), doc.get("excerpt", ""))[:24],
        }
    return {
        "verification_status": "discovery_url_unchecked",
        "verification_score": round(min(0.62, 0.36 + expected_overlap * 0.20), 3),
        "verification_reason": "Source has an external URL but was not fetched in this pass; confidence remains capped until checked.",
        "canonical_url": url,
        "content_hash": source_key(doc.get("title", ""), doc.get("excerpt", ""))[:24],
    }


def verify_source_doc(
    doc: dict[str, Any],
    expected_terms: Optional[list[str]] = None,
    *,
    fetch: bool = False,
    fetcher: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    """Verify that a stored source document is traceable to its URL.

    When fetch=False, this performs a safe offline audit that never touches the
    network. When fetch=True, it downloads the URL text and checks whether the
    stored title/excerpt/key terms are actually supported by the page.
    """
    offline = _offline_verification(doc, expected_terms)
    url = offline.get("canonical_url") or normalize_url(doc.get("url", ""))
    if not fetch or not url or str(url).startswith("manual://"):
        return offline

    try:
        if fetcher:
            html = fetcher(url)
        else:
            html = fetch_url_text(url, timeout=min(float(settings.source_verification_timeout_seconds), float(settings.request_timeout_seconds)), use_scraperapi=bool(settings.scraperapi_api_key))
        page_text = _source_text_from_html(html)
        if not page_text:
            raise ValueError("No readable page text extracted")
        title_text = str(doc.get("title") or "")
        excerpt_text = str(doc.get("excerpt") or "")
        title_terms = _tokens(title_text)
        excerpt_terms = _tokens(excerpt_text)
        expected = _tokens(" ".join(expected_terms or []))
        page_terms = _tokens(page_text)
        title_score = len(title_terms & page_terms) / max(len(title_terms), 1) if title_terms else 0.0
        excerpt_score = len(excerpt_terms & page_terms) / max(len(excerpt_terms), 1) if excerpt_terms else 0.0
        expected_score = len(expected & page_terms) / max(len(expected), 1) if expected else 0.0
        
        page_low = page_text.lower()
        title_phrase = 1.0 if title_text and clean_text(title_text, 180).lower() in page_low else 0.0
        excerpt_clean = clean_text(excerpt_text, 260).lower()
        excerpt_phrase = 1.0 if excerpt_clean and excerpt_clean[:180] in page_low else 0.0
        score = round(min(1.0, (title_score * 0.25) + (excerpt_score * 0.35) + (expected_score * 0.15) + (title_phrase * 0.10) + (excerpt_phrase * 0.15)), 3)
        status = "url_verified" if score >= 0.42 else "url_verified_weak_match"
        reason = (
            f"Fetched source page and compared stored title/excerpt/key terms plus exact snippet phrases. "
            f"title={title_score:.2f}, excerpt={excerpt_score:.2f}, expected_terms={expected_score:.2f}, "
            f"title_phrase={title_phrase:.2f}, excerpt_phrase={excerpt_phrase:.2f}."
        )
        return {
            "verification_status": status,
            "verification_score": score,
            "verification_reason": reason,
            "canonical_url": url,
            "content_hash": sha256(page_text.encode("utf-8", errors="ignore")).hexdigest(),
        }
    except Exception as exc:
        fallback = dict(offline)
        fallback["verification_status"] = "fetch_failed"
        fallback["verification_score"] = min(float(fallback.get("verification_score") or 0.0), 0.25)
        fallback["verification_reason"] = f"Could not verify URL text: {exc}. Confidence remains capped."
        return fallback


def verify_sources_for_catalyst(catalyst_id: int, *, fetch: Optional[bool] = None) -> list[dict[str, Any]]:
    """Run source verification for every source document attached to a catalyst."""
    if fetch is None:
        fetch = bool(settings.source_verification_fetch_on_analyze)
    catalyst_rows = rows(
        """
        SELECT c.*, co.name AS company_name, co.ticker, co.aliases_json, co.targets_json, co.indications_json
        FROM catalysts c
        JOIN companies co ON c.company_id = co.id
        WHERE c.id = ?
        """,
        (catalyst_id,),
    )
    expected_terms: list[str] = []
    if catalyst_rows:
        cat = catalyst_rows[0]
        expected_terms.extend([str(cat.get("title") or ""), str(cat.get("summary") or ""), str(cat.get("company_name") or ""), str(cat.get("ticker") or "")])
        for key in ["aliases_json", "targets_json", "indications_json"]:
            try:
                expected_terms.extend(json.loads(cat.get(key) or "[]"))
            except Exception:
                pass

    docs = rows("SELECT * FROM source_docs WHERE catalyst_id = ? ORDER BY id", (catalyst_id,))
    verified: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for doc in docs:
        result = verify_source_doc(doc, expected_terms, fetch=fetch)
        execute(
            """
            UPDATE source_docs
            SET verification_status = ?, verification_score = ?, verification_reason = ?,
                verified_at = ?, content_hash = ?, canonical_url = ?
            WHERE id = ?
            """,
            (
                result.get("verification_status", "not_checked"),
                float(result.get("verification_score") or 0.0),
                clean_text(result.get("verification_reason", ""), 900),
                now,
                str(result.get("content_hash") or "")[:128],
                str(result.get("canonical_url") or doc.get("url") or "")[:1000],
                int(doc["id"]),
            ),
        )
        verified.append({**doc, **result})
    return verified
