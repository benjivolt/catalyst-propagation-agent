from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Iterable

from app.config import settings

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    targets_json TEXT NOT NULL DEFAULT '[]',
    indications_json TEXT NOT NULL DEFAULT '[]',
    peers_json TEXT NOT NULL DEFAULT '[]',
    source_urls_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS catalysts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    event_date TEXT,
    source_key TEXT NOT NULL UNIQUE,
    event_cluster_key TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalyst_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    publisher TEXT,
    published_at TEXT,
    excerpt TEXT NOT NULL,
    source_type TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}',
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    verification_status TEXT NOT NULL DEFAULT 'not_checked',
    verification_score REAL NOT NULL DEFAULT 0.0,
    verification_reason TEXT NOT NULL DEFAULT '',
    verified_at TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(catalyst_id) REFERENCES catalysts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalyst_id INTEGER NOT NULL UNIQUE,
    overall_direction TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    plain_language_summary TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(catalyst_id) REFERENCES catalysts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS effects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL,
    layer TEXT NOT NULL,
    entity TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    readthrough_type TEXT NOT NULL DEFAULT 'none',
    rationale TEXT NOT NULL,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    effect_id INTEGER NOT NULL,
    claim TEXT NOT NULL,
    source_doc_id INTEGER,
    citation_url TEXT,
    verification_status TEXT NOT NULL DEFAULT 'not_verified',
    verification_score REAL NOT NULL DEFAULT 0.0,
    confidence_cap REAL NOT NULL DEFAULT 0.45,
    exact_quote TEXT NOT NULL DEFAULT '',
    source_paragraph_id INTEGER,
    verifier_reason TEXT NOT NULL DEFAULT '',
    verified_at TEXT,
    FOREIGN KEY(effect_id) REFERENCES effects(id) ON DELETE CASCADE,
    FOREIGN KEY(source_doc_id) REFERENCES source_docs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id INTEGER NOT NULL,
    url TEXT,
    source_type TEXT,
    title TEXT,
    publisher TEXT,
    fetched_at TEXT NOT NULL,
    raw_html TEXT,
    raw_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    archive_path TEXT,
    http_status INTEGER,
    final_url TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_doc_id) REFERENCES source_docs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    paragraph_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    embedding_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(snapshot_id) REFERENCES source_snapshots(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS atomic_claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL,
    effect_id INTEGER,
    source_claim_id INTEGER,
    original_sentence TEXT NOT NULL,
    atomic_claim TEXT NOT NULL,
    claim_type TEXT,
    entity_company TEXT,
    entity_drug TEXT,
    entity_target TEXT,
    entity_indication TEXT,
    event_type TEXT,
    direction TEXT,
    importance TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE,
    FOREIGN KEY(effect_id) REFERENCES effects(id) ON DELETE CASCADE,
    FOREIGN KEY(source_claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_evidence_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    source_doc_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    paragraph_id INTEGER NOT NULL,
    quote_text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    retrieval_score REAL,
    entity_match_score REAL,
    entailment_label TEXT,
    entailment_score REAL,
    contradiction_score REAL,
    coverage_score REAL,
    verifier_model TEXT,
    verification_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(claim_id) REFERENCES atomic_claims(id) ON DELETE CASCADE,
    FOREIGN KEY(source_doc_id) REFERENCES source_docs(id) ON DELETE CASCADE,
    FOREIGN KEY(snapshot_id) REFERENCES source_snapshots(id) ON DELETE CASCADE,
    FOREIGN KEY(paragraph_id) REFERENCES source_paragraphs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_verification_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    final_status TEXT NOT NULL,
    best_span_id INTEGER,
    best_support_score REAL,
    best_contradiction_score REAL,
    required_source_type TEXT,
    has_required_source_type INTEGER,
    confidence_cap REAL,
    publishable INTEGER,
    failure_reason TEXT,
    verified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(claim_id) REFERENCES atomic_claims(id) ON DELETE CASCADE,
    FOREIGN KEY(best_span_id) REFERENCES claim_evidence_spans(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS analysis_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL UNIQUE,
    evidence_grade TEXT NOT NULL DEFAULT 'D',
    citation_coverage REAL NOT NULL DEFAULT 0.0,
    unsupported_effect_count INTEGER NOT NULL DEFAULT 0,
    weak_claim_count INTEGER NOT NULL DEFAULT 0,
    audit_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_cache (
    cache_key TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS monitor_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    status TEXT NOT NULL,
    message TEXT,
    new_catalysts INTEGER NOT NULL DEFAULT 0
);


CREATE TABLE IF NOT EXISTS scheduler_source_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    reliability_score REAL NOT NULL DEFAULT 0.70,
    check_cost REAL NOT NULL DEFAULT 0.20,
    expected_impact REAL NOT NULL DEFAULT 0.60,
    company_priority REAL NOT NULL DEFAULT 0.60,
    last_checked_at TEXT,
    next_check_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_changed_at TEXT,
    last_catalyst_at TEXT,
    last_hash TEXT,
    backoff_level INTEGER NOT NULL DEFAULT 0,
    priority_score REAL NOT NULL DEFAULT 0.50,
    average_reward REAL NOT NULL DEFAULT 0.0,
    checks_count INTEGER NOT NULL DEFAULT 0,
    useful_detection_count INTEGER NOT NULL DEFAULT 0,
    false_positive_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    estimated_cost_total REAL NOT NULL DEFAULT 0.0,
    reason TEXT NOT NULL DEFAULT 'Awaiting first adaptive score.',
    locked_until TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, source_name),
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scheduler_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id INTEGER NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    content_hash TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    candidate_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(state_id) REFERENCES scheduler_source_state(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scheduler_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    state_id INTEGER NOT NULL,
    company_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    scheduled_for TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    priority_score REAL NOT NULL DEFAULT 0.0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    new_catalysts INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    skipped_low_score INTEGER NOT NULL DEFAULT 0,
    reward_value REAL NOT NULL DEFAULT 0.0,
    message TEXT,
    FOREIGN KEY(run_id) REFERENCES monitor_runs(id) ON DELETE SET NULL,
    FOREIGN KEY(state_id) REFERENCES scheduler_source_state(id) ON DELETE CASCADE,
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scheduler_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id INTEGER NOT NULL,
    job_id INTEGER,
    reward_value REAL NOT NULL,
    reward_reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(state_id) REFERENCES scheduler_source_state(id) ON DELETE CASCADE,
    FOREIGN KEY(job_id) REFERENCES scheduler_jobs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS budget_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    job_id INTEGER,
    company_id INTEGER,
    source_name TEXT,
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES monitor_runs(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES scheduler_jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE SET NULL
);


CREATE TABLE IF NOT EXISTS peer_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    peer_name TEXT NOT NULL,
    peer_ticker TEXT,
    relationship_type TEXT NOT NULL DEFAULT 'unknown',
    target_overlap_json TEXT NOT NULL DEFAULT '[]',
    indication_overlap_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    source TEXT NOT NULL DEFAULT 'watchlist',
    rationale TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, peer_name, relationship_type, source),
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS report_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'needs_review',
    reviewer TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scheduler_due ON scheduler_source_state(enabled, next_check_at, priority_score);
CREATE INDEX IF NOT EXISTS idx_analysis_grade ON analysis_audits(evidence_grade, citation_coverage);
CREATE INDEX IF NOT EXISTS idx_source_cache_expiry ON source_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_source_docs_verification ON source_docs(verification_status, verification_score);
CREATE INDEX IF NOT EXISTS idx_scheduler_jobs_started ON scheduler_jobs(started_at);
CREATE INDEX IF NOT EXISTS idx_budget_created ON budget_ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_catalysts_cluster ON catalysts(event_cluster_key);
CREATE INDEX IF NOT EXISTS idx_peer_relationships_company ON peer_relationships(company_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_source_snapshots_doc_hash ON source_snapshots(source_doc_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_source_paragraphs_snapshot ON source_paragraphs(snapshot_id, paragraph_index);
CREATE INDEX IF NOT EXISTS idx_atomic_claims_report ON atomic_claims(report_id, effect_id, claim_type);
CREATE INDEX IF NOT EXISTS idx_atomic_claims_source_claim ON atomic_claims(source_claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim ON claim_evidence_spans(claim_id, entailment_label);
CREATE INDEX IF NOT EXISTS idx_claim_verification_status ON claim_verification_summary(final_status, publishable);


CREATE TABLE IF NOT EXISTS alert_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    channel TEXT NOT NULL,
    target TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    catalyst_id INTEGER,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'openai',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'recorded',
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE SET NULL,
    FOREIGN KEY(catalyst_id) REFERENCES catalysts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS openai_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    sdk_installed INTEGER NOT NULL DEFAULT 0,
    key_configured INTEGER NOT NULL DEFAULT 0,
    responses_api_available INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalyst_id INTEGER NOT NULL,
    entity TEXT NOT NULL,
    layer TEXT NOT NULL,
    distinct_source_count INTEGER NOT NULL DEFAULT 0,
    official_source_count INTEGER NOT NULL DEFAULT 0,
    verified_source_count INTEGER NOT NULL DEFAULT 0,
    corroboration_status TEXT NOT NULL DEFAULT 'not_checked',
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(catalyst_id) REFERENCES catalysts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    last_status TEXT NOT NULL DEFAULT 'unknown',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_latency_ms INTEGER NOT NULL DEFAULT 0,
    last_candidate_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, source_name),
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS peer_asset_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    company_name TEXT NOT NULL,
    ticker TEXT NOT NULL DEFAULT '',
    asset_name TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    indication TEXT NOT NULL DEFAULT '',
    modality TEXT NOT NULL DEFAULT '',
    development_stage TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL DEFAULT '',
    evidence_excerpt TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, company_name, asset_name, target, indication)
);



CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    catalyst_id INTEGER,
    report_id INTEGER,
    company_id INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'unknown',
    source_key TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL DEFAULT '',
    predicted_decision TEXT NOT NULL,
    predicted_event_type TEXT NOT NULL DEFAULT 'none',
    predicted_direction TEXT NOT NULL DEFAULT 'unknown',
    predicted_confidence REAL NOT NULL DEFAULT 0.0,
    predicted_peer_effects_json TEXT NOT NULL DEFAULT '{}',
    model_name TEXT NOT NULL DEFAULT 'deterministic_policy',
    gate_score REAL NOT NULL DEFAULT 0.0,
    ai_reasoning_summary TEXT NOT NULL DEFAULT '',
    risk_flags_json TEXT NOT NULL DEFAULT '[]',
    evidence_needed_later_json TEXT NOT NULL DEFAULT '[]',
    outcome_status TEXT NOT NULL DEFAULT 'pending',
    current_outcome_label TEXT NOT NULL DEFAULT 'pending',
    last_outcome_checked_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE,
    FOREIGN KEY(catalyst_id) REFERENCES catalysts(id) ON DELETE SET NULL,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ai_prediction_outcome_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    due_at TEXT NOT NULL,
    checked_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    check_type TEXT NOT NULL DEFAULT 'delayed_verification',
    result_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(prediction_id) REFERENCES ai_predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    outcome_label TEXT NOT NULL,
    outcome_confidence REAL NOT NULL DEFAULT 0.0,
    confirmation_status TEXT NOT NULL DEFAULT 'unknown',
    official_confirmation_count INTEGER NOT NULL DEFAULT 0,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    verifier_model TEXT NOT NULL DEFAULT 'local_outcome_verifier',
    verifier_reason TEXT NOT NULL DEFAULT '',
    checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(prediction_id) REFERENCES ai_predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_reputation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL,
    source_type TEXT NOT NULL,
    publisher TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL DEFAULT 'all',
    company_id INTEGER,
    total_candidates INTEGER NOT NULL DEFAULT 0,
    true_positives INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    partial_positives INTEGER NOT NULL DEFAULT 0,
    contradicted_count INTEGER NOT NULL DEFAULT 0,
    unsupported_count INTEGER NOT NULL DEFAULT 0,
    false_negative_count INTEGER NOT NULL DEFAULT 0,
    avg_confirmation_delay_hours REAL NOT NULL DEFAULT 0.0,
    reliability_score REAL NOT NULL DEFAULT 0.50,
    catalyst_conversion_rate REAL NOT NULL DEFAULT 0.0,
    noise_score REAL NOT NULL DEFAULT 0.50,
    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_key, source_type, event_type, company_id)
);

CREATE TABLE IF NOT EXISTS gate_policy_thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    company_id INTEGER,
    promote_threshold REAL NOT NULL DEFAULT 0.80,
    review_threshold REAL NOT NULL DEFAULT 0.60,
    context_threshold REAL NOT NULL DEFAULT 0.40,
    false_positive_rate REAL NOT NULL DEFAULT 0.0,
    false_negative_rate REAL NOT NULL DEFAULT 0.0,
    precision_score REAL NOT NULL DEFAULT 0.0,
    recall_score REAL NOT NULL DEFAULT 0.0,
    evidence_multiplier REAL NOT NULL DEFAULT 1.0,
    policy_reason TEXT NOT NULL DEFAULT 'Default adaptive policy.',
    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, event_type, company_id)
);

CREATE TABLE IF NOT EXISTS confidence_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'all',
    source_type TEXT NOT NULL DEFAULT 'all',
    confidence_bucket TEXT NOT NULL,
    prediction_count INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    partial_count INTEGER NOT NULL DEFAULT 0,
    observed_accuracy REAL NOT NULL DEFAULT 0.0,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    recommended_multiplier REAL NOT NULL DEFAULT 1.0,
    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_type, source_type, confidence_bucket)
);

CREATE TABLE IF NOT EXISTS ai_error_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_type TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'unknown',
    source_type TEXT NOT NULL DEFAULT 'unknown',
    pattern_description TEXT NOT NULL,
    example_prediction_id INTEGER,
    corrective_rule TEXT NOT NULL,
    severity REAL NOT NULL DEFAULT 0.5,
    active INTEGER NOT NULL DEFAULT 1,
    seen_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(example_prediction_id) REFERENCES ai_predictions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS policy_update_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    predictions_reviewed INTEGER NOT NULL DEFAULT 0,
    thresholds_updated INTEGER NOT NULL DEFAULT 0,
    reputations_updated INTEGER NOT NULL DEFAULT 0,
    calibration_rows_updated INTEGER NOT NULL DEFAULT 0,
    error_memories_updated INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS policy_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    source_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    company_id INTEGER,
    update_type TEXT NOT NULL,
    old_value REAL,
    new_value REAL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES policy_update_runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS peer_effect_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL,
    effect_id INTEGER,
    peer_company_id INTEGER,
    peer_name TEXT NOT NULL DEFAULT '',
    peer_ticker TEXT NOT NULL DEFAULT '',
    predicted_direction TEXT NOT NULL DEFAULT 'neutral',
    predicted_readthrough_type TEXT NOT NULL DEFAULT 'none',
    predicted_confidence REAL NOT NULL DEFAULT 0.0,
    outcome_label TEXT NOT NULL DEFAULT 'pending',
    outcome_confidence REAL NOT NULL DEFAULT 0.0,
    outcome_evidence_json TEXT NOT NULL DEFAULT '[]',
    observed_market_move REAL,
    observed_news_confirmation INTEGER NOT NULL DEFAULT 0,
    observed_program_update INTEGER NOT NULL DEFAULT 0,
    verifier_reason TEXT NOT NULL DEFAULT '',
    checked_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE,
    FOREIGN KEY(effect_id) REFERENCES effects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai_learning_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_type TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    severity REAL NOT NULL DEFAULT 0.5,
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_report_reviews_status ON report_reviews(status);
CREATE INDEX IF NOT EXISTS idx_alert_deliveries_report ON alert_deliveries(report_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_catalyst ON llm_usage(catalyst_id, created_at);
CREATE INDEX IF NOT EXISTS idx_openai_health_created ON openai_health_checks(created_at);
CREATE INDEX IF NOT EXISTS idx_source_correlations_catalyst ON source_correlations(catalyst_id, entity);
CREATE INDEX IF NOT EXISTS idx_source_health_company ON source_health(company_id, source_name, updated_at);
CREATE INDEX IF NOT EXISTS idx_peer_asset_index_target ON peer_asset_index(target, indication, company_name);

CREATE INDEX IF NOT EXISTS idx_ai_predictions_status ON ai_predictions(outcome_status, current_outcome_label, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_source_event ON ai_predictions(source_type, predicted_event_type, company_id);
CREATE INDEX IF NOT EXISTS idx_ai_outcome_checks_due ON ai_prediction_outcome_checks(status, due_at);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_prediction ON prediction_outcomes(prediction_id, checked_at);
CREATE INDEX IF NOT EXISTS idx_source_reputation_lookup ON source_reputation(source_type, event_type, company_id, reliability_score);
CREATE INDEX IF NOT EXISTS idx_gate_policy_lookup ON gate_policy_thresholds(source_type, event_type, company_id);
CREATE INDEX IF NOT EXISTS idx_confidence_calibration_lookup ON confidence_calibration(event_type, source_type, confidence_bucket);
CREATE INDEX IF NOT EXISTS idx_ai_error_memory_active ON ai_error_memory(active, source_type, event_type, severity);
CREATE INDEX IF NOT EXISTS idx_peer_effect_outcomes_report ON peer_effect_outcomes(report_id, outcome_label);
CREATE INDEX IF NOT EXISTS idx_policy_updates_created ON policy_updates(created_at, source_type, event_type);


"""


@contextmanager
def connect() -> Iterable[sqlite3.Connection]:
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.database_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _column_names(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate(con: sqlite3.Connection) -> None:
    """Add fields safely when a user already has an older local SQLite DB."""
    company_cols = _column_names(con, "companies")
    if "source_urls_json" not in company_cols:
        con.execute("ALTER TABLE companies ADD COLUMN source_urls_json TEXT NOT NULL DEFAULT '[]'")
    if "notes" not in company_cols:
        con.execute("ALTER TABLE companies ADD COLUMN notes TEXT NOT NULL DEFAULT ''")
    catalyst_cols = _column_names(con, "catalysts")
    if "event_cluster_key" not in catalyst_cols:
        con.execute("ALTER TABLE catalysts ADD COLUMN event_cluster_key TEXT")
    source_doc_cols = _column_names(con, "source_docs")
    source_doc_defaults = {
        "verification_status": "TEXT NOT NULL DEFAULT 'not_checked'",
        "verification_score": "REAL NOT NULL DEFAULT 0.0",
        "verification_reason": "TEXT NOT NULL DEFAULT ''",
        "verified_at": "TEXT",
        "content_hash": "TEXT NOT NULL DEFAULT ''",
        "canonical_url": "TEXT NOT NULL DEFAULT ''",
    }
    for col, ddl in source_doc_defaults.items():
        if col not in source_doc_cols:
            con.execute(f"ALTER TABLE source_docs ADD COLUMN {col} {ddl}")
    claim_cols = _column_names(con, "claims")
    claim_defaults = {
        "verification_status": "TEXT NOT NULL DEFAULT 'not_verified'",
        "verification_score": "REAL NOT NULL DEFAULT 0.0",
        "confidence_cap": "REAL NOT NULL DEFAULT 0.45",
        "exact_quote": "TEXT NOT NULL DEFAULT ''",
        "source_paragraph_id": "INTEGER",
        "verifier_reason": "TEXT NOT NULL DEFAULT ''",
        "verified_at": "TEXT",
    }
    for col, ddl in claim_defaults.items():
        if col not in claim_cols:
            con.execute(f"ALTER TABLE claims ADD COLUMN {col} {ddl}")


def init_db() -> None:
    try:
        with connect() as con:
            con.executescript(SCHEMA)
            _migrate(con)
    except sqlite3.DatabaseError as exc:
        
        
        db_path = Path(settings.database_path)
        if db_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            backup = db_path.with_suffix(db_path.suffix + f".corrupt-{stamp}")
            db_path.replace(backup)
        with connect() as con:
            con.executescript(SCHEMA)
            _migrate(con)


def rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def row(sql: str, params: tuple[Any, ...] = ()) -> Optional[dict[str, Any]]:
    with connect() as con:
        r = con.execute(sql, params).fetchone()
        return dict(r) if r else None


def execute(sql: str, params: tuple[Any, ...] = ()) -> int:
    with connect() as con:
        cur = con.execute(sql, params)
        return int(cur.lastrowid)


def execute_many(sql: str, params_seq: list[tuple[Any, ...]]) -> None:
    with connect() as con:
        con.executemany(sql, params_seq)


def json_dumps(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def json_loads(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return [] if default is None else default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return [] if default is None else default


def company_from_row(r: dict[str, Any]) -> dict[str, Any]:
    item = dict(r)
    item["aliases"] = json_loads(item.pop("aliases_json", "[]"))
    item["targets"] = json_loads(item.pop("targets_json", "[]"))
    item["indications"] = json_loads(item.pop("indications_json", "[]"))
    item["peers"] = json_loads(item.pop("peers_json", "[]"))
    item["source_urls"] = json_loads(item.pop("source_urls_json", "[]"))
    item["active"] = bool(item.get("active"))
    return item
