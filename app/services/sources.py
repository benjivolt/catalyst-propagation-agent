from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.db import execute, row
from app.services.utils import clean_text, classify_catalyst, is_likely_catalyst, normalize_url, source_key


@dataclass
class SourceDocumentCandidate:
    title: str
    url: str
    publisher: str
    published_at: Optional[str]
    excerpt: str
    source_type: str
    company_name: str
    raw: dict[str, Any]

    @property
    def key(self) -> str:
        
        
        return source_key(self.source_type, self.url or self.title, self.published_at or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceFetchError(Exception):
    pass


def _headers(accept_html: bool = True) -> dict[str, str]:
    accept = "application/json,text/html;q=0.9,*/*;q=0.8" if accept_html else "application/json"
    return {
        "User-Agent": settings.sec_user_agent,
        "Accept": accept,
    }


def _parse_cache_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        raw = value.replace("Z", "+00:00").replace(" ", "T")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _cache_key(url: str) -> str:
    return source_key("url_text", normalize_url(url) or url)


def _cache_get(url: str) -> Optional[str]:
    try:
        item = row("SELECT payload_json, expires_at FROM source_cache WHERE cache_key = ?", (_cache_key(url),))
        if not item:
            return None
        expires = _parse_cache_time(item.get("expires_at"))
        if expires and expires < datetime.now(timezone.utc):
            return None
        payload = json.loads(item.get("payload_json") or "{}")
        return payload.get("text")
    except Exception:
        return None


def _cache_put(url: str, source_name: str, text: str) -> None:
    try:
        ttl = max(int(settings.source_cache_ttl_seconds), 0)
        if ttl <= 0:
            return
        now = datetime.now(timezone.utc).replace(microsecond=0)
        expires = now + timedelta(seconds=ttl)
        execute(
            """
            INSERT OR REPLACE INTO source_cache (cache_key, source_name, url, content_hash, payload_json, fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _cache_key(url),
                source_name,
                normalize_url(url) or url,
                source_key(text[:2000]),
                json.dumps({"text": text}, ensure_ascii=False),
                now.isoformat(),
                expires.isoformat(),
            ),
        )
    except Exception:
        pass


def fetch_url_text(url: str, timeout: Optional[float] = None, use_scraperapi: bool = True) -> str:
    """Fetch a page directly, or through ScraperAPI when configured.

    A short SQLite cache saves repeated dashboard/manual monitor runs from
    re-fetching the same official pages over and over. Cache failure is ignored
    so network fetching still works in clean installs or tests.
    """
    cached = _cache_get(url)
    if cached is not None:
        return cached
    target = url
    if use_scraperapi and settings.scraperapi_api_key:
        target = "https://api.scraperapi.com/" + f"?api_key={settings.scraperapi_api_key}&url={quote_plus(url)}"
    with httpx.Client(timeout=timeout or settings.request_timeout_seconds, follow_redirects=True, headers=_headers()) as client:
        resp = client.get(target)
        resp.raise_for_status()
        text = resp.text
        _cache_put(url, "url_text", text)
        return text


def company_search_terms(company: dict[str, Any]) -> list[str]:
    aliases = company.get("aliases") or []
    ticker = company.get("ticker") or ""
    terms = [company.get("name", ""), *aliases]
    if ticker:
        terms.append(ticker)
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out[:6]


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "source"
    except Exception:
        return "source"


def _meta_or_title(html_text: str, fallback: str) -> tuple[str, str]:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = clean_text(soup.title.get_text(" ") if soup.title else fallback, 250)
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    desc = clean_text(meta.get("content", "") if meta else "", 800)
    if desc:
        return title, desc
    paragraphs = [clean_text(p.get_text(" "), 300) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    return title, clean_text(" ".join(paragraphs[:3]) or fallback, 900)


def _relevant_excerpt(text: str, fallback: str, window: int = 420) -> str:
    cleaned = clean_text(text, 5000)
    low = cleaned.lower()
    keywords = [
        "phase", "trial", "topline", "endpoint", "fda", "approval", "complete response letter",
        "pdufa", "license", "collaboration", "upfront", "milestone", "royalt", "acquire",
        "acquisition", "merger", "tender offer", "clinical hold", "terminated", "discontinued",
    ]
    positions = [low.find(k) for k in keywords if low.find(k) >= 0]
    if not positions:
        return clean_text(fallback or cleaned, 900)
    pos = min(positions)
    start = max(0, pos - window // 2)
    end = min(len(cleaned), pos + window)
    return clean_text(cleaned[start:end], 900)


class CompanyNewsSource:
    """User-defined company IR/news pages or RSS feeds."""

    name = "company_news"

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        urls = [u for u in company.get("source_urls", []) if str(u).startswith(("http://", "https://"))]
        docs: list[SourceDocumentCandidate] = []
        for source_url in urls[:4]:
            try:
                html_text = fetch_url_text(source_url)
            except Exception as exc:  
                raise SourceFetchError(f"Company news fetch failed for {source_url}: {exc}") from exc
            soup = BeautifulSoup(html_text, "xml" if "<rss" in html_text[:500].lower() or "<feed" in html_text[:500].lower() else "html.parser")
            candidates: list[tuple[str, str, Optional[str]]] = []

            
            for item in soup.find_all(["item", "entry"]):
                title = clean_text(item.find("title").get_text(" ") if item.find("title") else "", 250)
                link_tag = item.find("link")
                href = ""
                if link_tag:
                    href = link_tag.get("href") or link_tag.get_text(" ")
                date_tag = item.find(["pubDate", "published", "updated"])
                published = clean_text(date_tag.get_text(" "), 80) if date_tag else None
                if title and href:
                    candidates.append((title, urljoin(source_url, href), published))

            
            if not candidates:
                for a in soup.find_all("a", href=True):
                    title = clean_text(a.get_text(" "), 250)
                    href = urljoin(source_url, a["href"])
                    if title and len(title) > 12:
                        candidates.append((title, href, None))

            seen: set[str] = set()
            for title, href, published in candidates:
                normalized = normalize_url(href)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                if not is_likely_catalyst(title):
                    continue
                excerpt = title
                final_title = title
                try:
                    article_html = fetch_url_text(normalized, timeout=10)
                    final_title, excerpt = _meta_or_title(article_html, title)
                    if not is_likely_catalyst(final_title, excerpt):
                        continue
                except Exception:
                    pass
                docs.append(SourceDocumentCandidate(
                    title=final_title,
                    url=normalized,
                    publisher=_domain(normalized),
                    published_at=published,
                    excerpt=excerpt,
                    source_type=self.name,
                    company_name=company.get("name", ""),
                    raw={"source_url": source_url},
                ))
                if len(docs) >= limit:
                    return docs
        return docs


class GDELTSource:
    name = "gdelt_news"

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        terms = company_search_terms(company)
        if not terms:
            return []
        company_part = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
        catalyst_part = " OR ".join([
            '"trial results"', '"topline"', '"FDA approves"', '"complete response letter"',
            '"PDUFA"', '"license agreement"', '"strategic collaboration"', '"to acquire"', '"acquisition"',
            '"clinical hold"', '"phase 3"', '"phase 2"',
        ])
        query = f"({company_part}) ({catalyst_part}) sourcelang:English"
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={quote_plus(query)}&mode=ArtList&format=json&maxrecords={limit * 2}&sort=HybridRel"
        )
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers()) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  
            raise SourceFetchError(f"GDELT fetch failed: {exc}") from exc

        docs: list[SourceDocumentCandidate] = []
        for article in data.get("articles", [])[: limit * 2]:
            title = clean_text(article.get("title", ""), 250)
            excerpt = clean_text(f"{article.get('seendate', '')} {article.get('sourceCountry', '')}", 500)
            if not title or not is_likely_catalyst(title, excerpt):
                continue
            docs.append(SourceDocumentCandidate(
                title=title,
                url=normalize_url(article.get("url", "")),
                publisher=article.get("domain", "GDELT"),
                published_at=article.get("seendate"),
                excerpt=excerpt or title,
                source_type=self.name,
                company_name=company.get("name", ""),
                raw=article,
            ))
            if len(docs) >= limit:
                break
        return docs


class ClinicalTrialsSource:
    name = "clinicaltrials_gov"

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        terms = company_search_terms(company)
        if not terms:
            return []
        query = " OR ".join(terms)
        url = (
            "https://clinicaltrials.gov/api/v2/studies"
            f"?query.term={quote_plus(query)}&pageSize={min(limit * 3, 100)}&format=json"
        )
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers()) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  
            raise SourceFetchError(f"ClinicalTrials.gov fetch failed: {exc}") from exc

        docs: list[SourceDocumentCandidate] = []
        for study in data.get("studies", [])[: limit * 3]:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            arms = protocol.get("armsInterventionsModule", {})
            sponsor = protocol.get("sponsorCollaboratorsModule", {})
            nct_id = ident.get("nctId", "")
            brief_title = ident.get("briefTitle") or ident.get("officialTitle") or nct_id
            overall_status = status.get("overallStatus", "")
            last_update = status.get("lastUpdateSubmitDate") or status.get("lastUpdatePostDateStruct", {}).get("date")
            has_results = bool(study.get("resultsSection"))
            phases = ", ".join(design.get("phases", []) or [])
            interventions = ", ".join(item.get("name", "") for item in arms.get("interventions", []) if item.get("name"))
            org = sponsor.get("leadSponsor", {}).get("name", "")
            if not has_results and overall_status.lower() not in {"completed", "terminated", "withdrawn", "suspended"}:
                continue
            title = f"{brief_title} ({overall_status})"
            excerpt = clean_text(
                f"ClinicalTrials.gov record {nct_id}. Lead sponsor: {org}. Phase: {phases or 'not listed'}. "
                f"Interventions: {interventions or 'not listed'}. Last update: {last_update or 'not listed'}. "
                f"Results posted: {has_results}.",
                900,
            )
            docs.append(SourceDocumentCandidate(
                title=title,
                url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "https://clinicaltrials.gov/",
                publisher="ClinicalTrials.gov",
                published_at=last_update,
                excerpt=excerpt,
                source_type=self.name,
                company_name=company.get("name", ""),
                raw=study,
            ))
            if len(docs) >= limit:
                break
        return docs


class FDAPressSource:
    name = "fda_press_announcements"
    index_url = "https://www.fda.gov/news-events/fda-newsroom/press-announcements"

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        terms = [t.lower() for t in company_search_terms(company)]
        target_terms = [str(x).lower() for x in company.get("targets", []) + company.get("indications", [])]
        try:
            html_text = fetch_url_text(self.index_url, use_scraperapi=False)
        except Exception as exc:  
            raise SourceFetchError(f"FDA page fetch failed: {exc}") from exc
        soup = BeautifulSoup(html_text, "html.parser")
        docs: list[SourceDocumentCandidate] = []
        for a in soup.find_all("a", href=True):
            text = clean_text(a.get_text(" "), 300)
            if not text:
                continue
            low = text.lower()
            if not any(term and term in low for term in terms + target_terms):
                continue
            if classify_catalyst(text) not in {"fda_decision", "other_catalyst"} and "fda" not in low:
                continue
            href = urljoin("https://www.fda.gov", a["href"])
            docs.append(SourceDocumentCandidate(
                title=text,
                url=normalize_url(href),
                publisher="FDA",
                published_at=None,
                excerpt=text,
                source_type=self.name,
                company_name=company.get("name", ""),
                raw={"index_url": self.index_url},
            ))
            if len(docs) >= limit:
                break
        return docs


class OpenFDADrugsSource:
    """Drugs@FDA approval/submission discovery through the official openFDA endpoint.

    This is used as a high-precision regulatory source. It is intentionally
    conservative: it only emits candidate catalysts when the application record
    has an approval-looking submission status/date tied to a sponsor or product
    alias from the watchlist.
    """

    name = "openfda_drugs"
    endpoint = "https://api.fda.gov/drug/drugsfda.json"

    def _queries(self, company: dict[str, Any]) -> list[str]:
        terms = company_search_terms(company)
        
        product_terms = [str(x) for x in (company.get("aliases") or [])[:5]]
        queries: list[str] = []
        for term in terms[:4]:
            safe = str(term).replace('"', "").strip()
            if safe:
                queries.append(f'sponsor_name:"{safe}"')
        for term in product_terms[:4]:
            safe = str(term).replace('"', "").strip()
            if safe:
                queries.append(f'products.brand_name:"{safe}"')
        
        out: list[str] = []
        seen: set[str] = set()
        for q in queries:
            if q.lower() not in seen:
                out.append(q)
                seen.add(q.lower())
        return out

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        docs: list[SourceDocumentCandidate] = []
        for query in self._queries(company):
            url = f"{self.endpoint}?search={quote_plus(query)}&limit={min(max(limit, 1) * 2, 20)}"
            try:
                with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers(False)) as client:
                    resp = client.get(url)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:  
                raise SourceFetchError(f"openFDA Drugs@FDA fetch failed: {exc}") from exc

            for record in data.get("results", [])[: limit * 2]:
                sponsor = clean_text(record.get("sponsor_name", ""), 160)
                application = clean_text(record.get("application_number", ""), 80)
                products = record.get("products") or []
                product_names = sorted({clean_text(p.get("brand_name") or p.get("active_ingredients", [{}])[0].get("name", ""), 80) for p in products if p})
                submissions = record.get("submissions") or []
                
                submissions = sorted(submissions, key=lambda x: str(x.get("submission_status_date") or ""), reverse=True)
                for sub in submissions[:4]:
                    status = clean_text(str(sub.get("submission_status") or ""), 80)
                    status_date = clean_text(str(sub.get("submission_status_date") or ""), 30) or None
                    submission_type = clean_text(str(sub.get("submission_type") or ""), 60)
                    review_priority = clean_text(str(sub.get("review_priority") or ""), 80)
                    label = " ".join([status, submission_type, review_priority]).lower()
                    
                    if not any(x in label for x in ["ap", "approval", "approved", "priority"]):
                        continue
                    product = ", ".join([p for p in product_names if p]) or application or company.get("name", "Drug")
                    title = f"Drugs@FDA {status or 'submission'} for {product} ({application})"
                    excerpt = clean_text(
                        f"openFDA Drugs@FDA record for sponsor {sponsor or 'not listed'}, application {application or 'not listed'}, "
                        f"product {product}. Submission type {submission_type or 'not listed'}; status {status or 'not listed'}; "
                        f"status date {status_date or 'not listed'}; review priority {review_priority or 'not listed'}.",
                        900,
                    )
                    docs.append(SourceDocumentCandidate(
                        title=title,
                        url=f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={application.replace('NDA', '').replace('BLA', '')}" if application else "https://open.fda.gov/apis/drug/drugsfda/",
                        publisher="openFDA / Drugs@FDA",
                        published_at=status_date,
                        excerpt=excerpt,
                        source_type=self.name,
                        company_name=company.get("name", ""),
                        raw={"query": query, "application_number": application, "submission": sub, "sponsor_name": sponsor},
                    ))
                    if len(docs) >= limit:
                        return docs
        return docs


class PubMedSource:
    """PubMed literature context for target/indication validation.

    PubMed is not a company-news feed, so the scheduler score usually gates it
    unless a paper title/excerpt includes catalyst-like terms. When it passes,
    it gives the analysis a citable biomedical source for target or indication
    reasoning instead of relying only on press-release language.
    """

    name = "pubmed"
    search_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        terms = [*(company.get("targets") or [])[:3], *(company.get("indications") or [])[:3], *(company.get("aliases") or [])[:3]]
        terms = [clean_text(t, 120) for t in terms if clean_text(t, 120)]
        if not terms:
            return []
        catalyst_terms = '(clinical trial OR phase 2 OR phase 3 OR endpoint OR efficacy OR safety OR FDA OR approval)'
        query = " OR ".join(f'"{t}"' if " " in t else t for t in terms[:6])
        params = {
            "db": "pubmed",
            "term": f"({query}) AND {catalyst_terms}",
            "retmode": "json",
            "retmax": str(min(max(limit * 2, 2), 20)),
            "sort": "pub+date",
        }
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers(False)) as client:
                search = client.get(self.search_endpoint, params=params)
                search.raise_for_status()
                ids = search.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                summary = client.get(self.summary_endpoint, params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
                summary.raise_for_status()
                data = summary.json().get("result", {})
        except Exception as exc:  
            raise SourceFetchError(f"PubMed fetch failed: {exc}") from exc

        docs: list[SourceDocumentCandidate] = []
        for pmid in ids:
            item = data.get(str(pmid)) or {}
            title = clean_text(item.get("title", ""), 260)
            if not title:
                continue
            pubdate = clean_text(item.get("pubdate", ""), 80) or None
            journal = clean_text(item.get("fulljournalname", "PubMed"), 140) or "PubMed"
            authors = item.get("authors") or []
            first_author = clean_text(authors[0].get("name", ""), 80) if authors else ""
            excerpt = clean_text(
                f"PubMed record PMID {pmid}. Journal: {journal}. Publication date: {pubdate or 'not listed'}. First author: {first_author or 'not listed'}. Title: {title}",
                900,
            )
            docs.append(SourceDocumentCandidate(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                publisher=f"PubMed / {journal}",
                published_at=pubdate,
                excerpt=excerpt,
                source_type=self.name,
                company_name=company.get("name", ""),
                raw={"pmid": pmid, "query": params["term"]},
            ))
            if len(docs) >= limit:
                break
        return docs


class SECSource:
    name = "sec_edgar"
    tickers_url = "https://www.sec.gov/files/company_tickers.json"

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        if not ticker:
            return None
        with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers(False)) as client:
            resp = client.get(self.tickers_url)
            resp.raise_for_status()
            data = resp.json()
        for value in data.values():
            if str(value.get("ticker", "")).upper() == ticker.upper():
                return str(value.get("cik_str", "")).zfill(10)
        return None

    def fetch(self, company: dict[str, Any], limit: int = 6) -> list[SourceDocumentCandidate]:
        ticker = company.get("ticker") or ""
        try:
            cik = self._lookup_cik(ticker)
            if not cik:
                return []
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=_headers(False)) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  
            raise SourceFetchError(f"SEC fetch failed: {exc}") from exc

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [])
        docs: list[SourceDocumentCandidate] = []
        for form, filing_date, accession, primary_doc, item in zip(forms, dates, accessions, primary_docs, items):
            if form not in {"8-K", "425", "SC TO-T", "S-4"}:
                continue
            compact_accession = accession.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{compact_accession}/{primary_doc}"
            title = f"{company.get('name', 'Company')} {form} filing {filing_date}"
            excerpt = clean_text(
                f"SEC filing form {form}; item(s): {item or 'not listed'}. Material event filing from EDGAR.",
                500,
            )
            try:
                filing_html = fetch_url_text(filing_url, timeout=12, use_scraperapi=False)
                filing_text = BeautifulSoup(filing_html, "html.parser").get_text(" ")
                relevant = _relevant_excerpt(filing_text, excerpt)
                if not is_likely_catalyst(title, relevant) and form == "8-K":
                    continue
                excerpt = relevant
            except Exception:
                if form == "8-K" and not re.search(r"1\.01|2\.01|8\.01", str(item or "")):
                    continue
            docs.append(SourceDocumentCandidate(
                title=title,
                url=filing_url,
                publisher="SEC EDGAR",
                published_at=filing_date,
                excerpt=excerpt,
                source_type=self.name,
                company_name=company.get("name", ""),
                raw={"cik": cik, "accession": accession, "form": form, "item": item},
            ))
            if len(docs) >= limit:
                break
        return docs


def all_sources() -> list[Any]:
    
    return [CompanyNewsSource(), SECSource(), FDAPressSource(), OpenFDADrugsSource(), ClinicalTrialsSource(), PubMedSource(), GDELTSource()]
