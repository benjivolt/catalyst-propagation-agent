from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.config import settings
from app.db import execute, row, rows
from app.services.sources import SourceDocumentCandidate, SourceFetchError, all_sources
from app.services.utils import CATALYST_KEYWORDS, classify_catalyst, clean_text, detect_catalyst, is_likely_catalyst, source_key
from app.services.reliability import content_fingerprint, record_source_health, run_with_retries, should_skip_stale_candidate
from app.services.ai_candidate_reviewer import review_candidate, save_candidate_review
from app.services.outcome_learning import learned_threshold, reputation_score


@dataclass(frozen=True)
class SourceProfile:
    reliability: float
    cost: float
    expected_impact: float
    usd_per_check: float
    label: str


SOURCE_PROFILES: dict[str, SourceProfile] = {
    "sec_edgar": SourceProfile(0.96, 0.08, 0.90, 0.001, "SEC official filing trigger"),
    "fda_press_announcements": SourceProfile(0.95, 0.10, 0.88, 0.001, "FDA official regulatory trigger"),
    "clinicaltrials_gov": SourceProfile(0.84, 0.14, 0.76, 0.002, "ClinicalTrials.gov study-status trigger"),
    "openfda_drugs": SourceProfile(0.94, 0.10, 0.84, 0.001, "openFDA Drugs@FDA approval/submission trigger"),
    "pubmed": SourceProfile(0.88, 0.16, 0.70, 0.002, "PubMed target/indication evidence source"),
    "company_news": SourceProfile(0.86, 0.18, 0.78, 0.004, "Company IR / press-release source"),
    "gdelt_news": SourceProfile(0.62, 0.30, 0.56, 0.006, "Broad news discovery source"),
}

HIGH_IMPACT_TERMS = (
    "pdufa", "fda", "approval", "complete response letter", "crl", "phase 3",
    "pivotal", "topline", "readout", "primary endpoint", "clinical hold",
    "acquisition", "merger", "licensing", "license agreement", "data expected",
)
OFFICIAL_SOURCE_NAMES = {"sec_edgar", "fda_press_announcements", "clinicaltrials_gov", "openfda_drugs", "pubmed", "company_news"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    
    candidates = [raw, raw.replace("Z", "+00:00"), raw.replace(" ", "T")]
    for candidate in candidates:
        try:
            if len(candidate) == 10 and candidate.count("-") == 2:
                return datetime.fromisoformat(candidate).replace(tzinfo=timezone.utc)
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _source_instances() -> dict[str, Any]:
    return {getattr(source, "name", source.__class__.__name__): source for source in all_sources()}


def _profile_for(source_name: str) -> SourceProfile:
    return SOURCE_PROFILES.get(source_name, SourceProfile(0.70, 0.22, 0.60, 0.004, "Generic monitored source"))


def company_context_score(company: dict[str, Any]) -> float:
    """Estimate how important/near-term the company appears from watchlist context."""
    text = " ".join(
        [
            str(company.get("name", "")),
            str(company.get("ticker", "")),
            str(company.get("notes", "")),
            " ".join(str(x) for x in company.get("targets", []) or []),
            " ".join(str(x) for x in company.get("indications", []) or []),
        ]
    ).lower()
    score = 0.42
    hits = sum(1 for term in HIGH_IMPACT_TERMS if term in text)
    score += min(hits * 0.055, 0.30)
    if company.get("peers"):
        score += min(len(company.get("peers") or []) * 0.025, 0.10)
    if company.get("source_urls"):
        score += 0.05
    if company.get("ticker"):
        score += 0.04
    return _clip(score, 0.15, 0.95)


def catalyst_window_score(company: dict[str, Any], today: Optional[datetime] = None) -> float:
    """Lightweight hazard/window score from explicit dates and biotech guidance language."""
    today = today or _now()
    text = " ".join(
        [
            str(company.get("notes", "")),
            " ".join(str(x) for x in company.get("targets", []) or []),
            " ".join(str(x) for x in company.get("indications", []) or []),
        ]
    ).lower()
    score = 0.12
    if any(term in text for term in ("pdufa", "adcom", "fda decision", "approval decision")):
        score += 0.30
    if any(term in text for term in ("readout", "topline", "data expected", "phase 3", "pivotal")):
        score += 0.24
    if any(term in text for term in ("acquisition", "merger", "license agreement", "collaboration")):
        score += 0.12

    
    for token in set(__import__("re").findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)):
        dt = _parse_dt(token)
        if not dt:
            continue
        days = abs((dt.date() - today.date()).days)
        if days <= 1:
            score += 0.45
        elif days <= 7:
            score += 0.30
        elif days <= 30:
            score += 0.18
        elif days <= 90:
            score += 0.08

    
    for q, year in __import__("re").findall(r"\bq([1-4])\s*(20\d{2})\b", text):
        qn = int(q)
        yr = int(year)
        start_month = 1 + (qn - 1) * 3
        start = datetime(yr, start_month, 1, tzinfo=timezone.utc)
        end_month = start_month + 2
        
        next_month = datetime(yr + (1 if end_month == 12 else 0), 1 if end_month == 12 else end_month + 1, 1, tzinfo=timezone.utc)
        end = next_month - timedelta(days=1)
        if start <= today <= end:
            elapsed = max((today - start).days, 0)
            total = max((end - start).days, 1)
            score += 0.18 + 0.20 * (elapsed / total)
        else:
            days_to_start = (start.date() - today.date()).days
            if 0 < days_to_start <= 60:
                score += 0.10
    return _clip(score, 0.05, 1.0)


def _company_from_state_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": r["company_id"],
        "name": r.get("company_name") or "",
        "ticker": r.get("ticker") or "",
        "aliases": _json_loads(r.get("aliases_json"), []),
        "targets": _json_loads(r.get("targets_json"), []),
        "indications": _json_loads(r.get("indications_json"), []),
        "peers": _json_loads(r.get("peers_json"), []),
        "source_urls": _json_loads(r.get("source_urls_json"), []),
        "notes": r.get("notes") or "",
        "active": True,
    }


def sync_scheduler_state() -> None:
    """Create one adaptive state row per active company/source pair."""
    companies = rows("SELECT * FROM companies WHERE active = 1")
    source_names = list(_source_instances().keys())
    now = _iso(_now())
    for company in companies:
        c = {
            "id": company["id"],
            "name": company.get("name", ""),
            "ticker": company.get("ticker", ""),
            "aliases": _json_loads(company.get("aliases_json"), []),
            "targets": _json_loads(company.get("targets_json"), []),
            "indications": _json_loads(company.get("indications_json"), []),
            "peers": _json_loads(company.get("peers_json"), []),
            "source_urls": _json_loads(company.get("source_urls_json"), []),
            "notes": company.get("notes", ""),
        }
        company_score = company_context_score(c)
        for source_name in source_names:
            profile = _profile_for(source_name)
            execute(
                """
                INSERT OR IGNORE INTO scheduler_source_state (
                    company_id, source_name, reliability_score, check_cost,
                    expected_impact, company_priority, next_check_at, priority_score, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company["id"],
                    source_name,
                    profile.reliability,
                    profile.cost,
                    profile.expected_impact,
                    company_score,
                    now,
                    0.50,
                    f"Initialized from {profile.label}; waiting for adaptive scoring.",
                ),
            )


def _scheduler_rows() -> list[dict[str, Any]]:
    return rows(
        """
        SELECT s.*, c.name AS company_name, c.ticker, c.aliases_json, c.targets_json,
               c.indications_json, c.peers_json, c.source_urls_json, c.notes
        FROM scheduler_source_state s
        JOIN companies c ON c.id = s.company_id
        WHERE c.active = 1 AND s.enabled = 1
        ORDER BY s.priority_score DESC, s.next_check_at ASC
        """
    )


def calculate_priority(company: dict[str, Any], state: dict[str, Any], total_checks: Optional[int] = None) -> tuple[float, str]:
    source_name = str(state.get("source_name") or "")
    profile = _profile_for(source_name)
    learned_rep = reputation_score(source_name, "all", int(state.get("company_id") or 0) or None)
    reliability = (0.55 * float(state.get("reliability_score") or profile.reliability)) + (0.45 * learned_rep)
    cost = max(float(state.get("check_cost") or profile.cost), 0.03)
    expected_impact = float(state.get("expected_impact") or profile.expected_impact)
    company_priority = company_context_score(company)
    window = catalyst_window_score(company)

    last_changed = _parse_dt(state.get("last_changed_at"))
    last_catalyst = _parse_dt(state.get("last_catalyst_at"))
    activity = 0.10
    now = _now()
    if last_catalyst and (now - last_catalyst) <= timedelta(days=3):
        activity = 0.90
    elif last_changed and (now - last_changed) <= timedelta(days=2):
        activity = 0.65
    elif last_changed and (now - last_changed) <= timedelta(days=14):
        activity = 0.35

    checks_count = int(state.get("checks_count") or 0)
    avg_reward = float(state.get("average_reward") or 0.0)
    reward_score = _clip((avg_reward + 4.0) / 14.0)
    total = max(total_checks or 1, 1)
    exploration = settings.scheduler_exploration_factor * math.sqrt(math.log(total + 2.0) / (checks_count + 1.0))
    exploration = _clip(exploration, 0.0, 0.25)

    official_boost = 0.06 if state.get("source_name") in OFFICIAL_SOURCE_NAMES else 0.0
    raw = (
        0.27 * window
        + 0.18 * expected_impact
        + 0.16 * reliability
        + 0.15 * company_priority
        + 0.10 * activity
        + 0.08 * reward_score
        + 0.06 * exploration
        + official_boost
    )
    backoff = int(state.get("backoff_level") or 0)
    backoff_multiplier = 1.0 / (1.0 + 0.10 * backoff)
    cost_penalty = 0.72 + (0.42 * cost)
    priority = _clip((raw * backoff_multiplier) / cost_penalty)

    reasons = [profile.label, f"learned source reputation {learned_rep:.2f}"]
    if window >= 0.65:
        reasons.append("near-term catalyst window / hazard boost")
    elif window >= 0.38:
        reasons.append("moderate biotech catalyst-window signal")
    if reliability >= 0.90:
        reasons.append("high-reliability official source")
    if activity >= 0.65:
        reasons.append("recent source activity")
    if avg_reward > 1.5:
        reasons.append("historically useful source")
    if exploration > 0.12:
        reasons.append("bandit exploration bonus")
    if backoff > 0:
        reasons.append(f"backoff level {backoff} after quiet/noisy checks")
    reasons.append(f"cost penalty {cost:.2f}")
    return round(priority, 4), "; ".join(reasons)


def calculate_next_check_at(priority: float, state: dict[str, Any]) -> str:
    min_minutes = max(settings.adaptive_min_check_minutes, 1)
    max_minutes = max(settings.adaptive_max_check_hours * 60, min_minutes + 1)
    urgency = _clip(priority) ** 1.25
    minutes = max_minutes - (max_minutes - min_minutes) * urgency
    source_name = str(state.get("source_name") or "")
    if source_name in {"sec_edgar", "fda_press_announcements", "openfda_drugs", "pubmed"}:
        minutes *= 0.82
    backoff = int(state.get("backoff_level") or 0)
    minutes *= min(1.0 + 0.18 * backoff, 3.5)
    minutes = max(min_minutes, min(max_minutes, minutes))
    
    jitter = int(_hash_text(f"{state.get('id')}:{source_name}")[:2], 16) % 7
    return _iso(_now() + timedelta(minutes=minutes + jitter))


def refresh_priorities() -> list[dict[str, Any]]:
    sync_scheduler_state()
    states = _scheduler_rows()
    total_checks = int((row("SELECT COALESCE(SUM(checks_count), 0) AS n FROM scheduler_source_state") or {"n": 1})["n"] or 1)
    refreshed: list[dict[str, Any]] = []
    for state in states:
        company = _company_from_state_row(state)
        priority, reason = calculate_priority(company, state, total_checks=total_checks)
        
        execute(
            """
            UPDATE scheduler_source_state
            SET priority_score = ?, reason = ?, company_priority = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (priority, reason, company_context_score(company), state["id"]),
        )
        state = dict(state)
        state["priority_score"] = priority
        state["reason"] = reason
        refreshed.append(state)
    return refreshed


def _candidate_text(candidate: SourceDocumentCandidate) -> str:
    return f"{candidate.title} {candidate.publisher} {candidate.excerpt} {candidate.url}".lower()


def score_candidate(company: dict[str, Any], candidate: SourceDocumentCandidate, state: dict[str, Any]) -> tuple[float, list[str]]:
    text = _candidate_text(candidate)
    source_name = str(state.get("source_name") or candidate.source_type)
    profile = _profile_for(source_name)
    detection = detect_catalyst(candidate.title, candidate.excerpt)
    event_type = detection.event_type
    # Calibrated detection confidence drives the gate instead of a flat 0.44, so a
    # decisive readout outscores a marginal hint, and a forward-looking
    # announcement ("plans to report topline in Q4") is penalized rather than
    # treated as a fired catalyst.
    event_score = 0.12 + 0.46 * detection.confidence
    if detection.is_forward_looking:
        event_score -= 0.12
    keyword_hits = 0
    for words in CATALYST_KEYWORDS.values():
        keyword_hits += sum(1 for word in words if word in text)
    event_score += min(keyword_hits * 0.02, 0.12)

    terms = [company.get("name", ""), company.get("ticker", ""), *(company.get("aliases") or [])]
    company_matches = sum(1 for term in terms if term and str(term).lower() in text)
    company_score = 0.16 + min(company_matches * 0.11, 0.34)
    target_terms = [*(company.get("targets") or []), *(company.get("indications") or [])]
    target_hits = sum(1 for term in target_terms if term and str(term).lower() in text)
    target_score = min(target_hits * 0.06, 0.18)

    candidate_key = source_key(str(company.get("id")), candidate.key)
    try:
        duplicate = row("SELECT id FROM catalysts WHERE source_key = ?", (candidate_key,)) is not None
    except Exception:
        
        
        
        duplicate = False
    novelty = 0.0 if duplicate else 0.12
    learned_reputation = reputation_score(source_name, event_type, int(company.get("id") or 0) or None, candidate.publisher, candidate.url)
    blended_reliability = (0.45 * float(state.get("reliability_score") or profile.reliability)) + (0.55 * learned_reputation)
    source_score = 0.20 * blended_reliability

    raw_score = event_score + company_score + target_score + novelty + source_score - (0.18 if duplicate else 0.0)
    # Event-presence cap: company/target relevance must not, on its own, push a
    # non-event past the promote threshold. A candidate can only score high when a
    # catalyst was actually detected. Forward-looking items (already discounted to
    # low confidence) are capped just under the default promote threshold so they
    # are watched, not promoted.
    event_cap = 0.45 + 0.55 * detection.confidence
    score = _clip(min(raw_score, event_cap))
    reasons: list[str] = []
    if event_type != "other_catalyst":
        reasons.append(f"{event_type} (detection conf {detection.confidence:.2f})")
    if detection.is_forward_looking:
        reasons.append("forward-looking announcement (penalized)")
    if raw_score > event_cap + 1e-9:
        reasons.append(f"capped by event presence at {event_cap:.2f}")
    if company_matches:
        reasons.append(f"company/alias match x{company_matches}")
    if target_hits:
        reasons.append(f"target/indication match x{target_hits}")
    if duplicate:
        reasons.append("duplicate already stored")
    else:
        reasons.append("novel source key")
    reasons.append(f"source reliability {blended_reliability:.2f} (learned {learned_reputation:.2f})")
    policy = learned_threshold(source_name, event_type, int(company.get("id") or 0) or None)
    reasons.append(f"learned promote threshold {float(policy['promote_threshold']):.2f}")
    return round(score, 4), reasons


def _fingerprint_candidates(candidates: list[SourceDocumentCandidate]) -> str:
    
    
    return content_fingerprint(candidates)


def _daily_spend() -> float:
    start = _iso(_now().replace(hour=0, minute=0, second=0, microsecond=0))
    result = row("SELECT COALESCE(SUM(estimated_cost), 0) AS spent FROM budget_ledger WHERE created_at >= ?", (start,))
    return float(result.get("spent", 0.0) if result else 0.0)


def _budget_allows(estimated_cost: float) -> bool:
    budget = float(settings.scheduler_daily_budget_usd)
    if budget <= 0:
        return True
    return (_daily_spend() + estimated_cost) <= budget


def _record_budget(run_id: Optional[int], job_id: Optional[int], company_id: Optional[int], source_name: str, amount: float, reason: str, model: str = "none", input_tokens: int = 0, output_tokens: int = 0) -> None:
    execute(
        """
        INSERT INTO budget_ledger (run_id, job_id, company_id, source_name, model, input_tokens, output_tokens, estimated_cost, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, job_id, company_id, source_name, model, input_tokens, output_tokens, float(amount), reason),
    )


def _update_state_after_job(state: dict[str, Any], reward: float, changed: bool, found_catalyst: bool, duplicate_count: int, false_positive_count: int, cost: float, reason: str) -> None:
    checks = int(state.get("checks_count") or 0)
    current_avg = float(state.get("average_reward") or 0.0)
    new_avg = ((current_avg * checks) + reward) / (checks + 1)
    backoff = int(state.get("backoff_level") or 0)
    if found_catalyst or reward >= 3:
        backoff = 0
    elif changed:
        backoff = min(backoff + 1, 8)
    else:
        backoff = min(backoff + 1, 12)
    next_state = dict(state)
    next_state["average_reward"] = new_avg
    next_state["checks_count"] = checks + 1
    next_state["backoff_level"] = backoff
    next_state["last_changed_at"] = _iso(_now()) if changed else state.get("last_changed_at")
    next_state["last_catalyst_at"] = _iso(_now()) if found_catalyst else state.get("last_catalyst_at")
    priority, priority_reason = calculate_priority(_company_from_state_row(state), next_state)
    next_check_at = calculate_next_check_at(priority, next_state)
    execute(
        """
        UPDATE scheduler_source_state
        SET last_checked_at = ?, next_check_at = ?, last_changed_at = ?, last_catalyst_at = ?,
            backoff_level = ?, priority_score = ?, average_reward = ?, checks_count = checks_count + 1,
            useful_detection_count = useful_detection_count + ?, false_positive_count = false_positive_count + ?,
            duplicate_count = duplicate_count + ?, estimated_cost_total = estimated_cost_total + ?,
            reason = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            _iso(_now()),
            next_check_at,
            next_state.get("last_changed_at"),
            next_state.get("last_catalyst_at"),
            backoff,
            priority,
            new_avg,
            1 if found_catalyst else 0,
            int(false_positive_count),
            int(duplicate_count),
            float(cost),
            f"{reason} Next: {priority_reason}",
            state["id"],
        ),
    )


def run_adaptive_monitor(auto_analyze: bool = True, per_source_limit: int = 4, due_only: bool = False, max_jobs: Optional[int] = None) -> dict[str, Any]:
    """Run the cost-aware adaptive scheduler once.

    due_only=False is used by the dashboard button so the user can force an immediate pass.
    due_only=True is used by the background scheduler so quiet sources wait for next_check_at.
    """
    run_id = execute("INSERT INTO monitor_runs (status, message) VALUES ('running', 'adaptive scheduler started')")
    errors: list[str] = []
    new_catalyst_ids: list[int] = []
    jobs_started = 0
    jobs_skipped_budget = 0
    source_map = _source_instances()
    try:
        states = refresh_priorities()
        now = _now()
        if due_only:
            states = [s for s in states if (_parse_dt(s.get("next_check_at")) or now) <= now]
        states.sort(key=lambda s: float(s.get("priority_score") or 0.0), reverse=True)
        if max_jobs is None:
            max_jobs = settings.adaptive_max_jobs_per_tick
        states = states[:max_jobs]

        for state in states:
            source_name = str(state.get("source_name") or "")
            source = source_map.get(source_name)
            profile = _profile_for(source_name)
            company = _company_from_state_row(state)
            estimated_cost = profile.usd_per_check
            if not source:
                errors.append(f"Unknown source {source_name}")
                continue
            if not _budget_allows(estimated_cost):
                jobs_skipped_budget += 1
                execute(
                    """
                    UPDATE scheduler_source_state
                    SET next_check_at = ?, reason = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (_iso(_now() + timedelta(hours=1)), "Skipped: daily adaptive scheduler budget would be exceeded.", state["id"]),
                )
                continue

            job_id = execute(
                """
                INSERT INTO scheduler_jobs (run_id, state_id, company_id, source_name, scheduled_for, priority_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, state["id"], company["id"], source_name, state.get("next_check_at"), float(state.get("priority_score") or 0.0)),
            )
            jobs_started += 1
            _record_budget(run_id, job_id, company["id"], source_name, estimated_cost, "source_check")

            candidates: list[SourceDocumentCandidate] = []
            new_count = 0
            duplicate_count = 0
            low_score_count = 0
            false_positive_count = 0
            changed = False
            reward = -1.0
            message = "no candidates"
            try:
                started_perf = datetime.now(timezone.utc)
                fetched, health = run_with_retries(
                    source_name,
                    lambda: source.fetch(company, limit=per_source_limit),
                    attempts=max(int(settings.source_fetch_retries), 1),
                    base_sleep=max(float(settings.source_retry_base_sleep_seconds), 0.0),
                )
                latency_ms = int((datetime.now(timezone.utc) - started_perf).total_seconds() * 1000)
                if fetched is None:
                    error_text = "; ".join(health.get("errors") or []) or "source fetch failed"
                    record_source_health(company["id"], source_name, "failed", latency_ms=latency_ms, candidate_count=0, error=error_text)
                    raise SourceFetchError(f"{source_name} failed after {health.get('attempts')} attempts: {error_text}")
                candidates = [c for c in fetched if not should_skip_stale_candidate(c, state)]
                record_source_health(company["id"], source_name, "ok", latency_ms=latency_ms, candidate_count=len(candidates))
                fingerprint = _fingerprint_candidates(candidates)
                previous_hash = state.get("last_hash")
                changed = bool(candidates) and fingerprint != previous_hash
                execute(
                    """
                    INSERT INTO scheduler_snapshots (state_id, content_hash, candidate_count, candidate_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        state["id"],
                        fingerprint,
                        len(candidates),
                        json.dumps([c.to_dict() for c in candidates[:10]], ensure_ascii=False),
                    ),
                )
                execute("UPDATE scheduler_source_state SET last_hash = ? WHERE id = ?", (fingerprint, state["id"]))

                for candidate in candidates:
                    candidate_score, score_reasons = score_candidate(company, candidate, state)
                    candidate_key = source_key(str(company["id"]), candidate.key)
                    if row("SELECT id FROM catalysts WHERE source_key = ?", (candidate_key,)):
                        duplicate_count += 1
                        continue
                    review = review_candidate(company, candidate, candidate_score)
                    prediction_id = save_candidate_review(company, candidate, review, gate_score=candidate_score)
                    policy = learned_threshold(source_name, review.event_type, int(company.get("id") or 0) or None)
                    promote_threshold = float(policy.get("promote_threshold") or settings.scheduler_candidate_threshold)
                    if review.decision != "promote_to_catalyst" or candidate_score < min(promote_threshold, float(settings.scheduler_candidate_threshold)):
                        low_score_count += 1
                        if review.decision in {"reject_irrelevant", "needs_ai_review"}:
                            false_positive_count += 1
                        message = f"candidate stored as {review.decision}; prediction {prediction_id}; score {candidate_score:.2f}; threshold {promote_threshold:.2f}; {review.reasoning_summary[:240]}"
                        continue
                    from app.services.monitor import _save_candidate_as_catalyst, analyze_and_save  

                    catalyst_id = _save_candidate_as_catalyst(company["id"], candidate)
                    if catalyst_id:
                        save_candidate_review(company, candidate, review, gate_score=candidate_score, catalyst_id=catalyst_id)
                        new_count += 1
                        new_catalyst_ids.append(catalyst_id)
                        should_analyze = auto_analyze and (
                            candidate_score >= float(settings.scheduler_llm_threshold) or not settings.openai_api_key
                        )
                        if should_analyze:
                            report_id = analyze_and_save(catalyst_id)
                            llm_cost = 0.012 if settings.openai_api_key else 0.0
                            _record_budget(
                                run_id,
                                job_id,
                                company["id"],
                                source_name,
                                llm_cost,
                                "llm_analysis" if settings.openai_api_key else "fallback_analysis",
                                settings.openai_model if settings.openai_api_key else "fallback_rules",
                                1800 if settings.openai_api_key else 0,
                                900 if settings.openai_api_key else 0,
                            )
                            message = f"created catalyst {catalyst_id} and report {report_id}; prediction {prediction_id}; score {candidate_score:.2f}; {', '.join(score_reasons)}"
                        else:
                            message = f"created catalyst {catalyst_id}; LLM gated to save tokens; prediction {prediction_id}; score {candidate_score:.2f}; {', '.join(score_reasons)}"

                if new_count:
                    reward = 10.0 + min(new_count - 1, 3)
                    message = message or f"{new_count} new catalysts"
                elif candidates and duplicate_count == len(candidates):
                    reward = -3.0
                    message = "all candidates were duplicates"
                elif candidates and low_score_count:
                    reward = -4.0
                    message = f"{low_score_count} low-score candidates blocked before LLM"
                elif candidates:
                    reward = 1.0
                    message = f"{len(candidates)} changed candidates but no new catalyst"

                execute(
                    """
                    UPDATE scheduler_jobs
                    SET status = 'finished', finished_at = CURRENT_TIMESTAMP, candidate_count = ?,
                        new_catalysts = ?, duplicates = ?, skipped_low_score = ?, reward_value = ?, message = ?
                    WHERE id = ?
                    """,
                    (len(candidates), new_count, duplicate_count, low_score_count, reward, clean_text(message, 1000), job_id),
                )
                execute(
                    "INSERT INTO scheduler_rewards (state_id, job_id, reward_value, reward_reason) VALUES (?, ?, ?, ?)",
                    (state["id"], job_id, reward, clean_text(message, 500)),
                )
                _update_state_after_job(state, reward, changed, new_count > 0, duplicate_count, false_positive_count, estimated_cost, clean_text(message, 500))
            except SourceFetchError as exc:
                reward = -2.0
                errors.append(str(exc))
                execute(
                    "UPDATE scheduler_jobs SET status = 'failed', finished_at = CURRENT_TIMESTAMP, reward_value = ?, message = ? WHERE id = ?",
                    (reward, str(exc), job_id),
                )
                execute(
                    "INSERT INTO scheduler_rewards (state_id, job_id, reward_value, reward_reason) VALUES (?, ?, ?, ?)",
                    (state["id"], job_id, reward, str(exc)),
                )
                _update_state_after_job(state, reward, False, False, 0, 0, estimated_cost, f"Fetch error: {exc}")
            except Exception as exc:  
                reward = -4.0
                errors.append(f"{source_name} unexpected error: {exc}")
                execute(
                    "UPDATE scheduler_jobs SET status = 'failed', finished_at = CURRENT_TIMESTAMP, reward_value = ?, message = ? WHERE id = ?",
                    (reward, str(exc), job_id),
                )
                _update_state_after_job(state, reward, False, False, 0, 1, estimated_cost, f"Unexpected error: {exc}")

        status = "finished" if not errors else "finished_with_errors"
        message = (
            f"adaptive_jobs={jobs_started}; skipped_budget={jobs_skipped_budget}; due_only={due_only}; "
            f"new_catalysts={len(new_catalyst_ids)}"
        )
        if errors:
            message += "; errors=" + "; ".join(errors[:4])
        execute(
            "UPDATE monitor_runs SET status = ?, message = ?, new_catalysts = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, clean_text(message, 1500), len(new_catalyst_ids), run_id),
        )
        return {
            "run_id": run_id,
            "status": status,
            "mode": "cost_aware_adaptive",
            "jobs_started": jobs_started,
            "jobs_skipped_budget": jobs_skipped_budget,
            "new_catalysts": len(new_catalyst_ids),
            "errors": errors,
            "daily_budget_used": round(_daily_spend(), 4),
        }
    except Exception as exc:
        execute(
            "UPDATE monitor_runs SET status = 'failed', message = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
            (str(exc), run_id),
        )
        raise


def scheduler_dashboard() -> dict[str, Any]:
    states = refresh_priorities()
    now = _now()
    due_count = sum(1 for s in states if (_parse_dt(s.get("next_check_at")) or now) <= now)
    high_priority = sum(1 for s in states if float(s.get("priority_score") or 0.0) >= 0.70)
    rows_out = []
    for state in sorted(states, key=lambda s: float(s.get("priority_score") or 0.0), reverse=True)[:12]:
        rows_out.append(
            {
                "id": state["id"],
                "company_id": state["company_id"],
                "company_name": state.get("company_name"),
                "ticker": state.get("ticker"),
                "source_name": state.get("source_name"),
                "priority_score": round(float(state.get("priority_score") or 0.0), 3),
                "next_check_at": state.get("next_check_at"),
                "last_checked_at": state.get("last_checked_at"),
                "backoff_level": state.get("backoff_level"),
                "average_reward": round(float(state.get("average_reward") or 0.0), 2),
                "check_cost": round(float(state.get("check_cost") or 0.0), 2),
                "reliability_score": round(float(state.get("reliability_score") or 0.0), 2),
                "reason": state.get("reason") or "",
            }
        )
    recent_jobs = rows(
        """
        SELECT j.*, c.name AS company_name, c.ticker
        FROM scheduler_jobs j
        JOIN companies c ON c.id = j.company_id
        ORDER BY j.id DESC
        LIMIT 8
        """
    )
    return {
        "enabled": settings.adaptive_scheduler_autostart and not settings.external_scheduler_worker,
        "external_worker": settings.external_scheduler_worker,
        "tick_seconds": settings.scheduler_tick_seconds,
        "daily_budget": settings.scheduler_daily_budget_usd,
        "daily_budget_used": round(_daily_spend(), 4),
        "sources": len(states),
        "due_count": due_count,
        "high_priority": high_priority,
        "rows": rows_out,
        "recent_jobs": recent_jobs,
    }


class AdaptiveSchedulerDaemon:
    """Small in-process scheduler loop for local demos.

    It does not blindly monitor on a fixed interval. The loop only wakes up to ask
    SQLite which sources are due according to their own dynamic next_check_at.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="adaptive-catalyst-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def _run(self) -> None:
        
        if self._stop.wait(5):
            return
        while not self._stop.is_set():
            try:
                run_adaptive_monitor(
                    auto_analyze=True,
                    per_source_limit=3,
                    due_only=True,
                    max_jobs=settings.adaptive_max_jobs_per_tick,
                )
            except Exception:
                
                pass
            if self._stop.wait(max(settings.scheduler_tick_seconds, 10)):
                break
