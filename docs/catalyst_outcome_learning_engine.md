# Catalyst Outcome Learning Engine Implementation Specification

## Purpose

This document is an implementation specification for adding a **Catalyst Outcome Learning Engine** to the existing **Catalyst Propagation Agent** project.

The current project has the shape of an AI-learning system, but the learning layer is mostly scaffolded. For example, the existing `ai_candidate_reviewer.py` can fall back to re-running deterministic gate logic, and `ai_gate_learning.py` mostly labels outcomes using local rules and adjusts thresholds. That is useful for a prototype, but it is not enough to claim that the system truly learns from real-world outcomes.

The goal of this implementation is to convert the current learning layer into a real **outcome-based self-improving intelligence system**.

The system should learn from:

- Whether promoted candidates became real catalysts.
- Whether rejected/context candidates were later proven to be catalysts.
- Whether sources historically produced useful catalysts or noise.
- Whether later official evidence confirmed, contradicted, or weakened the original prediction.
- Whether the AI overstated confidence.
- Whether specific event types need stricter or looser thresholds.
- Whether certain companies or source environments are unusually noisy.
- Whether peer-company read-through predictions were later supported.
- Whether target/indication/peer claims were later contradicted or needed revision.

The new system should make every candidate decision measurable, verifiable, and learnable over time.

---

## Executive Summary

Create a new system called:

```text
Catalyst Outcome Learning Engine
```

This engine should:

1. Store every AI or gate decision as a prediction.
2. Re-check predictions after time passes.
3. Verify whether later official sources confirmed, contradicted, or failed to support the decision.
4. Label outcomes such as `true_positive`, `false_positive`, `partial_positive`, `misclassified_event_type`, `premature_prediction`, `unsupported_prediction`, and `false_negative`.
5. Update source reputation based on historical performance.
6. Calibrate confidence scores based on observed accuracy.
7. Adjust event-gate thresholds by source type, event type, company, and historical error rates.
8. Track repeated AI mistakes in an AI error memory.
9. Evaluate whether peer-company read-through predictions were later supported.
10. Feed all learned policies back into the scheduler, event gate, AI reviewer, analyzer, and report verifier.

This turns the system from:

```text
rules + AI-shaped scaffolding
```

into:

```text
self-improving outcome-based biotech intelligence system
```

---

## High-Level Problem Being Solved

### Current Weakness

The project currently describes AI review and AI learning, but much of the implementation is deterministic or local-rule based.

The weakness is not that deterministic rules exist. Deterministic rules are necessary. The weakness is that the system currently has limited ability to answer:

```text
Did the prediction become true later?
Was the source historically reliable?
Did the model overstate confidence?
Did the event receive official confirmation?
Was a rejected candidate actually a missed catalyst?
Did peer read-through predictions hold up?
Which source/event/company combinations need stricter gates?
```

### Desired Fix

Implement a real learning loop:

```text
candidate detected
→ event gate and AI reviewer make a prediction
→ prediction is stored
→ outcome checks are scheduled
→ later official and high-quality sources are checked
→ prediction is labeled
→ source reputation is updated
→ confidence calibration is updated
→ thresholds are updated
→ AI error memory is updated
→ future scheduling and gating decisions improve
```

The engine should not claim to retrain a neural network unless the project actually implements a training dataset, evaluation split, fine-tuning pipeline, model registry, and versioned deployment. This implementation should focus first on **system-level policy learning**, which is more practical, auditable, and immediately useful.

---

## Core Design Principles

### 1. Every Decision Becomes a Prediction

Every candidate review, promotion, rejection, context decision, report confidence, and peer-effect claim should be saved as a prediction.

If the system does not store the original prediction, it cannot learn whether it was right.

### 2. Learning Requires Outside Evidence

The system should not say it learned just because its own rule says it was right.

Correct learning requires later evidence from sources such as:

- Company press releases.
- SEC filings.
- FDA announcements or structured FDA/openFDA records.
- ClinicalTrials.gov changes/results.
- Official conference abstracts.
- PubMed publications.
- Reputable financial/biotech news.
- Analyst summaries if available.
- Follow-up company filings or presentations.
- Market reaction as a weak supporting signal, not primary proof.

### 3. Source Reputation Must Be Granular

Do not assign one global source score only.

Source quality varies by event type.

Example:

```text
SEC EDGAR is strong for licensing/acquisition events.
FDA sources are strong for regulatory events.
PubMed is useful for science context but weak for catalyst triggering.
GDELT/general news may be useful for discovery but noisy for confirmation.
Company press releases are strong for event facts but biased for interpretation.
```

Therefore, source reliability should be tracked by:

- source type
- publisher/domain
- company
- event type
- confirmation rate
- false-positive rate
- contradiction rate
- average confirmation delay
- average evidence strength

### 4. Confidence Must Be Calibrated

The model should not freely invent confidence values.

A confidence score of `0.90` should mean that predictions of that type are actually correct roughly 90% of the time.

If historical outcomes show that 0.90 predictions are only correct 65% of the time, future confidence should be reduced.

### 5. Learning Should Be Auditable

Every policy update must explain:

- what changed
- why it changed
- what data supported the change
- which source/event/company combination was affected
- whether the change increases or decreases strictness

No silent threshold changes.

### 6. AI Can Review, But Rules Still Enforce Safety

AI should review borderline candidates and evaluate outcomes, but deterministic validators should enforce required fields.

The AI reviewer should not be allowed to promote an event that lacks required evidence fields.

---

## Required New Database Tables

Add these tables to the existing database initialization/migration logic.

The current project likely uses SQLite. These schemas should work with SQLite. If migrating to PostgreSQL later, preserve the same logical structure.

---

### 1. `ai_predictions`

Stores every candidate decision as a prediction that can later be evaluated.

```sql
CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    context_id INTEGER,
    catalyst_id INTEGER,
    report_id INTEGER,
    company_id INTEGER NOT NULL,

    source_type TEXT NOT NULL,
    source_key TEXT,
    source_url TEXT,
    source_title TEXT,
    publisher TEXT,

    predicted_decision TEXT NOT NULL,
    predicted_event_type TEXT,
    predicted_direction TEXT,
    predicted_confidence REAL NOT NULL,

    gate_score REAL,
    gate_decision TEXT,
    ai_review_used INTEGER NOT NULL DEFAULT 0,
    model_name TEXT,
    prompt_version TEXT,

    required_fields_json TEXT NOT NULL DEFAULT '{}',
    missing_fields_json TEXT NOT NULL DEFAULT '[]',
    evidence_span_ids_json TEXT NOT NULL DEFAULT '[]',
    risk_flags_json TEXT NOT NULL DEFAULT '[]',
    predicted_peer_effects_json TEXT NOT NULL DEFAULT '{}',

    ai_reasoning_summary TEXT,
    validation_notes TEXT,

    outcome_status TEXT NOT NULL DEFAULT 'pending',
    outcome_label TEXT,
    outcome_confidence REAL,
    outcome_verified_at TEXT,
    outcome_verifier_model TEXT,
    outcome_reason TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Important Fields

- `predicted_decision`: `promote_to_catalyst`, `store_as_context`, `reject_irrelevant`, `needs_review`.
- `predicted_event_type`: `trial_readout`, `fda_decision`, `licensing_deal`, `acquisition`, `none`.
- `outcome_label`: label assigned later after verification.
- `risk_flags_json`: stores warnings such as `rumor_source`, `company_source_bias`, `missing_official_confirmation`, `weak_asset_match`.

---

### 2. `prediction_outcome_checks`

Schedules and records delayed outcome checks.

```sql
CREATE TABLE IF NOT EXISTS prediction_outcome_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,

    check_type TEXT NOT NULL,
    scheduled_for TEXT NOT NULL,
    checked_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',

    sources_checked_json TEXT NOT NULL DEFAULT '[]',
    new_evidence_json TEXT NOT NULL DEFAULT '[]',

    outcome_label TEXT,
    outcome_confidence REAL,
    verifier_reason TEXT,
    verifier_model TEXT,

    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Check Types

Use these values:

```text
initial_24h
followup_7d
followup_30d
followup_90d
manual_backtest
peer_effect_check
false_negative_scan
```

#### Purpose

After the system predicts something, the engine should check later whether new official evidence appeared.

Example:

```text
Prediction: possible acquisition
24h later: no SEC filing, still rumor
7d later: company confirms definitive agreement
Outcome: early_true_positive
```

---

### 3. `source_reputation`

Tracks historical reliability of source types, publishers, and domains.

```sql
CREATE TABLE IF NOT EXISTS source_reputation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_key TEXT NOT NULL,
    source_type TEXT NOT NULL,
    publisher TEXT,
    domain TEXT,
    event_type TEXT,
    company_id INTEGER,

    total_candidates INTEGER NOT NULL DEFAULT 0,
    promoted_count INTEGER NOT NULL DEFAULT 0,
    context_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,

    true_positives INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    partial_positives INTEGER NOT NULL DEFAULT 0,
    false_negatives INTEGER NOT NULL DEFAULT 0,
    unsupported_count INTEGER NOT NULL DEFAULT 0,
    contradicted_count INTEGER NOT NULL DEFAULT 0,
    premature_count INTEGER NOT NULL DEFAULT 0,
    misclassified_count INTEGER NOT NULL DEFAULT 0,

    catalyst_conversion_rate REAL NOT NULL DEFAULT 0.0,
    false_positive_rate REAL NOT NULL DEFAULT 0.0,
    precision_score REAL NOT NULL DEFAULT 0.0,
    recall_score REAL NOT NULL DEFAULT 0.0,
    contradiction_rate REAL NOT NULL DEFAULT 0.0,

    avg_confirmation_delay_hours REAL,
    avg_evidence_support_score REAL NOT NULL DEFAULT 0.0,
    reliability_score REAL NOT NULL DEFAULT 0.50,

    scheduler_priority_multiplier REAL NOT NULL DEFAULT 1.0,
    gate_strictness_multiplier REAL NOT NULL DEFAULT 1.0,

    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_key, source_type, event_type, company_id)
);
```

#### Source Key Examples

```text
sec_edgar
company_news:madrigalpharma.com
pubmed
clinicaltrials_gov
gdelt:reuters.com
fda_press
openfda
```

#### Purpose

This lets the scheduler and event gate learn:

```text
Which sources create real catalysts?
Which sources create noise?
Which sources should be checked more often?
Which sources require official confirmation before promotion?
```

---

### 4. `gate_policy_thresholds`

Stores adaptive thresholds by source, event, and company.

```sql
CREATE TABLE IF NOT EXISTS gate_policy_thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    company_id INTEGER,

    promote_threshold REAL NOT NULL DEFAULT 0.80,
    review_threshold REAL NOT NULL DEFAULT 0.60,
    context_threshold REAL NOT NULL DEFAULT 0.40,

    min_official_confirmation_required INTEGER NOT NULL DEFAULT 0,
    require_asset_match INTEGER NOT NULL DEFAULT 1,
    require_company_match INTEGER NOT NULL DEFAULT 1,
    require_event_action INTEGER NOT NULL DEFAULT 1,

    false_positive_rate REAL NOT NULL DEFAULT 0.0,
    false_negative_rate REAL NOT NULL DEFAULT 0.0,
    precision_score REAL NOT NULL DEFAULT 0.0,
    recall_score REAL NOT NULL DEFAULT 0.0,

    policy_reason TEXT,
    last_policy_update_id INTEGER,
    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_type, event_type, company_id)
);
```

#### Purpose

Different source/event combinations need different strictness.

Example:

```text
PubMed + trial_readout:
promote_threshold = 0.93

SEC + acquisition:
promote_threshold = 0.68

Company news + trial_readout:
promote_threshold = 0.74

GDELT/news + acquisition:
promote_threshold = 0.90 and require official confirmation
```

---

### 5. `confidence_calibration`

Tracks whether confidence scores are accurate.

```sql
CREATE TABLE IF NOT EXISTS confidence_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    model_name TEXT,
    event_type TEXT,
    source_type TEXT,
    confidence_bucket TEXT NOT NULL,

    prediction_count INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    partial_count INTEGER NOT NULL DEFAULT 0,
    incorrect_count INTEGER NOT NULL DEFAULT 0,

    observed_accuracy REAL NOT NULL DEFAULT 0.0,
    expected_confidence REAL NOT NULL DEFAULT 0.0,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    recommended_multiplier REAL NOT NULL DEFAULT 1.0,

    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_name, event_type, source_type, confidence_bucket)
);
```

#### Confidence Buckets

```text
0.00-0.10
0.10-0.20
...
0.90-1.00
```

#### Purpose

If the model says `0.90` but outcomes show those predictions are correct only 65% of the time, apply a multiplier to future confidence.

---

### 6. `ai_error_memory`

Stores repeated AI/system failure patterns.

```sql
CREATE TABLE IF NOT EXISTS ai_error_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    error_type TEXT NOT NULL,
    event_type TEXT,
    source_type TEXT,
    company_id INTEGER,

    pattern_description TEXT NOT NULL,
    example_prediction_id INTEGER,
    example_claim_id INTEGER,

    corrective_rule TEXT NOT NULL,
    severity REAL NOT NULL DEFAULT 0.50,
    active INTEGER NOT NULL DEFAULT 1,

    seen_count INTEGER NOT NULL DEFAULT 1,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Example Entries

```text
Error type:
pubmed_context_promoted_as_catalyst

Pattern:
PubMed mechanism papers about target biology were promoted as trial readouts.

Corrective rule:
For PubMed trial_readout promotion, require company + asset + explicit clinical result language.
```

```text
Error type:
rumor_promoted_as_acquisition

Pattern:
General news rumors were classified as acquisitions without SEC or company confirmation.

Corrective rule:
Acquisition promotion from news sources requires official filing, definitive agreement language, or company confirmation.
```

```text
Error type:
safety_superiority_overstatement

Pattern:
The model claimed superior safety based only on "generally well tolerated" wording.

Corrective rule:
Safety superiority claims require comparative safety evidence.
```

---

### 7. `ai_policy_updates`

Stores auditable policy changes.

```sql
CREATE TABLE IF NOT EXISTS ai_policy_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    update_type TEXT NOT NULL,
    source_type TEXT,
    event_type TEXT,
    company_id INTEGER,

    old_value_json TEXT NOT NULL DEFAULT '{}',
    new_value_json TEXT NOT NULL DEFAULT '{}',

    reason TEXT NOT NULL,
    evidence_summary TEXT,
    affected_predictions_json TEXT NOT NULL DEFAULT '[]',

    applied INTEGER NOT NULL DEFAULT 0,
    applied_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Purpose

Every threshold, reputation, scheduler, or confidence policy update must be logged.

Example:

```json
{
  "update_type": "raise_promote_threshold",
  "source_type": "pubmed",
  "event_type": "trial_readout",
  "old_value": 0.82,
  "new_value": 0.93,
  "reason": "PubMed produced 12 false-positive trial_readout candidates and only 1 true catalyst."
}
```

---

### 8. `peer_effect_outcomes`

Tracks whether peer read-through predictions were later supported.

```sql
CREATE TABLE IF NOT EXISTS peer_effect_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    prediction_id INTEGER,
    report_id INTEGER NOT NULL,
    peer_company_id INTEGER,
    peer_ticker TEXT,

    predicted_direction TEXT NOT NULL,
    predicted_readthrough_type TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    predicted_reason TEXT,

    outcome_label TEXT NOT NULL DEFAULT 'pending',
    outcome_confidence REAL,
    outcome_evidence_json TEXT NOT NULL DEFAULT '[]',

    observed_market_move REAL,
    observed_news_confirmation INTEGER NOT NULL DEFAULT 0,
    observed_program_update INTEGER NOT NULL DEFAULT 0,
    observed_official_confirmation INTEGER NOT NULL DEFAULT 0,

    verifier_reason TEXT,
    checked_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### Purpose

The original assignment cares deeply about peer effects. This table lets the system evaluate whether peer-company read-throughs were plausible and later supported.

Important: Market movement alone should not be treated as proof. Stock moves are noisy. Use market reaction only as one weak signal.

---

### 9. `false_negative_candidates`

Tracks missed catalysts found later.

```sql
CREATE TABLE IF NOT EXISTS false_negative_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    company_id INTEGER NOT NULL,
    source_type TEXT,
    source_url TEXT,
    source_title TEXT,
    published_at TEXT,

    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    true_event_type TEXT NOT NULL,
    original_context_id INTEGER,
    original_candidate_id INTEGER,

    missed_reason TEXT,
    missed_gate_score REAL,
    missed_gate_decision TEXT,
    missed_source_reputation REAL,

    corrective_action TEXT,
    policy_update_id INTEGER
);
```

#### Purpose

Learning should not only reduce false positives. It must also detect missed true catalysts.

---

## New Services to Add

Add the following files under `app/services/`.

---

### 1. `app/services/prediction_ledger.py`

Responsible for creating and updating `ai_predictions`.

#### Required Functions

```python
def record_candidate_prediction(
    company_id: int,
    candidate: Any,
    gate_result: Any,
    ai_review: dict | None = None,
    catalyst_id: int | None = None,
    context_id: int | None = None,
    report_id: int | None = None,
) -> int:
    """Create an ai_predictions row and return prediction_id."""
```

```python
def mark_prediction_outcome(
    prediction_id: int,
    outcome_label: str,
    outcome_confidence: float,
    reason: str,
    verifier_model: str | None = None,
) -> None:
    """Update an ai_predictions row with verified outcome results."""
```

```python
def get_pending_predictions(limit: int = 100) -> list[dict]:
    """Return predictions whose outcome_status is pending or due for checks."""
```

```python
def attach_report_to_prediction(prediction_id: int, report_id: int) -> None:
    """Attach final report ID to a prediction after analysis is created."""
```

#### Implementation Requirements

- Do not overwrite prediction history.
- Store model names and prompt versions.
- Store risk flags and missing fields.
- Keep logic simple and database-safe.

---

### 2. `app/services/ai_candidate_reviewer.py`

Upgrade this file so it does not merely rerun deterministic gates.

It should use a real structured AI review when `OPENAI_API_KEY` is available, and only use deterministic fallback when the key is missing or AI review is disabled.

#### Required Behavior

Input:

```text
company
candidate
deterministic_gate_result
source_reputation
active_error_memory_rules
current_threshold_policy
```

Output structured JSON:

```json
{
  "decision": "promote_to_catalyst",
  "event_type": "trial_readout",
  "confidence": 0.87,
  "company_detected": true,
  "asset_detected": true,
  "event_action_detected": true,
  "required_fields_present": true,
  "missing_fields": [],
  "risk_flags": ["company_source_bias"],
  "reasoning_summary": "Official company source reports Phase 3 topline data with endpoint result language.",
  "evidence_needed_later": ["independent confirmation", "trial registry update"]
}
```

#### Allowed Decisions

```text
promote_to_catalyst
store_as_context
reject_irrelevant
needs_review
```

#### Allowed Event Types

```text
trial_readout
fda_decision
licensing_deal
acquisition
none
```

#### Safety Requirements

The AI reviewer cannot override missing required fields.

Example:

```text
If event_type = acquisition and there is no acquirer + target + transaction action,
do not promote.
```

The final validator must still block bad AI outputs.

---

### 3. `app/services/outcome_verifier.py`

Responsible for checking whether predictions were later confirmed or contradicted.

#### Required Functions

```python
def schedule_outcome_checks(prediction_id: int) -> None:
    """Create 24h, 7d, 30d, and 90d checks for a prediction."""
```

```python
def run_due_outcome_checks(limit: int = 50) -> dict:
    """Run scheduled checks whose scheduled_for <= now."""
```

```python
def verify_prediction_outcome(prediction: dict, check_type: str) -> dict:
    """Return structured outcome label and evidence."""
```

#### Outcome Labels

Use these labels:

```text
true_positive
false_positive
partial_positive
misclassified_event_type
premature_prediction
unsupported_prediction
contradicted_prediction
true_context
false_negative
still_pending
insufficient_evidence
```

#### Outcome Verification Logic

For each prediction, check:

1. Was the same event later confirmed by official source?
2. Was the event contradicted?
3. Was the event real but event type was wrong?
4. Was the prediction premature but eventually confirmed?
5. Was the item only context, not a catalyst?
6. Did later official sources validate or invalidate the original read-through?

#### Source Search Priority

For each event type:

##### Trial Readout

Check:

- company press releases
- ClinicalTrials.gov
- conference abstracts if available
- SEC filings
- reputable news
- PubMed later if clinical data published

##### FDA Decision

Check:

- FDA press pages
- Drugs@FDA/openFDA
- company press releases
- SEC filings
- reputable news

##### Licensing Deal

Check:

- SEC 8-K
- company press release
- partner company press release
- merger/deal filings
- reputable news

##### Acquisition

Check:

- SEC 8-K
- S-4
- SC TO-T
- company press releases
- definitive agreement announcements
- reputable news

---

### 4. `app/services/source_reputation.py`

Responsible for updating and reading source reliability.

#### Required Functions

```python
def get_source_reputation(
    source_type: str,
    source_key: str,
    event_type: str | None = None,
    company_id: int | None = None,
) -> dict:
    """Return source reputation record or default score."""
```

```python
def update_source_reputation_from_outcome(prediction: dict, outcome: dict) -> None:
    """Update source stats based on verified prediction outcome."""
```

```python
def compute_reliability_score(record: dict) -> float:
    """Compute a 0-1 reliability score."""
```

```python
def source_priority_multiplier(record: dict) -> float:
    """Return multiplier used by scheduler."""
```

#### Recommended Reliability Formula

Use a transparent formula:

```text
reliability_score =
  0.35 × precision_score
+ 0.20 × catalyst_conversion_rate
+ 0.15 × avg_evidence_support_score
+ 0.15 × official_confirmation_rate
- 0.20 × false_positive_rate
- 0.15 × contradiction_rate
```

Clamp the result between `0.05` and `1.00`.

For low sample sizes, use conservative smoothing:

```text
if total_candidates < 10:
    blend 70% default reliability + 30% observed reliability
```

---

### 5. `app/services/confidence_calibrator.py`

Responsible for calibrating AI confidence against actual outcome data.

#### Required Functions

```python
def bucket_confidence(confidence: float) -> str:
    """Return confidence bucket like '0.80-0.90'."""
```

```python
def update_confidence_calibration(prediction: dict, outcome: dict) -> None:
    """Update observed accuracy for the relevant bucket."""
```

```python
def get_confidence_multiplier(
    model_name: str | None,
    event_type: str | None,
    source_type: str | None,
    confidence: float,
) -> float:
    """Return multiplier for future confidence scores."""
```

```python
def calibrate_confidence(
    raw_confidence: float,
    source_reputation: float,
    evidence_support_score: float,
    model_name: str | None,
    event_type: str | None,
    source_type: str | None,
) -> float:
    """Return evidence- and history-adjusted confidence."""
```

#### Formula

```text
calibrated_confidence =
raw_confidence
× source_reputation_multiplier
× evidence_support_multiplier
× historical_calibration_multiplier
```

Clamp between `0.0` and `0.99`.

---

### 6. `app/services/threshold_optimizer.py`

Responsible for updating event-gate thresholds.

#### Required Functions

```python
def get_gate_policy(
    source_type: str,
    event_type: str,
    company_id: int | None = None,
) -> dict:
    """Return thresholds and required-field policies."""
```

```python
def optimize_thresholds_for_policy(source_type: str, event_type: str, company_id: int | None = None) -> dict:
    """Use historical outcomes to recommend threshold changes."""
```

```python
def apply_policy_update(policy_update_id: int) -> None:
    """Apply a logged policy update."""
```

```python
def nightly_threshold_optimization() -> dict:
    """Run threshold optimization across all source/event combinations."""
```

#### Threshold Update Rules

Use conservative adjustments.

Examples:

```text
If false_positive_rate > 0.30 and total_candidates >= 20:
    raise promote_threshold by 0.05 to 0.10

If false_positive_rate > 0.50:
    require official confirmation if source is news/GDELT/blog-like

If false_negative_rate > 0.20 and precision is still acceptable:
    lower review_threshold by 0.05

If source/event precision > 0.85 and total_candidates >= 20:
    lower promote_threshold by 0.03 max

If contradiction_rate > 0.20:
    raise promote_threshold and require stronger evidence
```

Never move a threshold by more than `0.10` in one optimization cycle.

---

### 7. `app/services/ai_error_memory.py`

Stores repeated AI/system errors and retrieves corrective rules.

#### Required Functions

```python
def record_error_pattern(
    error_type: str,
    pattern_description: str,
    corrective_rule: str,
    event_type: str | None = None,
    source_type: str | None = None,
    company_id: int | None = None,
    example_prediction_id: int | None = None,
    severity: float = 0.5,
) -> int:
    """Create or increment an error memory record."""
```

```python
def get_active_error_rules(
    event_type: str | None = None,
    source_type: str | None = None,
    company_id: int | None = None,
) -> list[dict]:
    """Return active corrective rules relevant to a candidate."""
```

```python
def apply_error_memory_to_gate(candidate: Any, gate_result: Any) -> Any:
    """Modify gate strictness or add risk flags using error memories."""
```

#### Error Types to Support

```text
pubmed_context_promoted_as_catalyst
clinicaltrials_status_promoted_without_results
rumor_promoted_as_acquisition
weak_asset_match_promoted
safety_superiority_overstatement
soc_claim_without_guideline_evidence
peer_effect_without_asset_overlap
company_press_release_overinterpretation
unsupported_target_validation
unsupported_competitive_readthrough
```

---

### 8. `app/services/policy_updater.py`

Runs scheduled learning jobs and logs policy changes.

#### Required Functions

```python
def run_daily_learning_cycle() -> dict:
    """Run outcome checks, update reputation, calibrate confidence, optimize thresholds, and write summary."""
```

```python
def generate_learning_summary(days: int = 7) -> dict:
    """Summarize recent learning results."""
```

```python
def create_policy_update(
    update_type: str,
    source_type: str | None,
    event_type: str | None,
    company_id: int | None,
    old_value: dict,
    new_value: dict,
    reason: str,
    evidence_summary: str,
    affected_prediction_ids: list[int],
) -> int:
    """Create auditable policy update."""
```

#### Daily Learning Cycle

Run these steps:

```text
1. Run due outcome checks.
2. Update source reputation.
3. Update confidence calibration.
4. Detect repeated error patterns.
5. Optimize thresholds.
6. Update scheduler priority multipliers.
7. Generate learning summary.
8. Save policy updates.
```

---

### 9. `app/services/peer_outcome_tracker.py`

Tracks whether peer read-through predictions were supported.

#### Required Functions

```python
def record_peer_effect_predictions(report_id: int, prediction_id: int | None = None) -> None:
    """Save peer effect predictions from a generated report."""
```

```python
def run_peer_effect_outcome_checks(limit: int = 50) -> dict:
    """Check pending peer effect predictions."""
```

```python
def verify_peer_effect_outcome(peer_effect: dict) -> dict:
    """Return support label and evidence."""
```

#### Peer Outcome Labels

```text
supported
partially_supported
unsupported
contradicted
still_pending
insufficient_evidence
```

#### Peer Evidence Signals

Strong evidence:

```text
official peer company update
competitor program update
analyst/news discussion specifically naming the read-through
same-target/same-indication program data update
partner/licensor response
regulatory precedent affecting peer asset
```

Weak evidence:

```text
short-term stock movement
broad sector movement
social media discussion
unconfirmed rumor
```

---

### 10. `app/services/learning_metrics.py`

Provides dashboard-ready metrics.

#### Required Functions

```python
def get_learning_dashboard_summary(days: int = 30) -> dict:
    """Return overall learning metrics for UI."""
```

```python
def get_source_reputation_table() -> list[dict]:
    """Return source reliability table."""
```

```python
def get_confidence_calibration_table() -> list[dict]:
    """Return calibration by bucket/source/event."""
```

```python
def get_recent_policy_updates(limit: int = 25) -> list[dict]:
    """Return recent policy changes."""
```

```python
def get_top_error_patterns(limit: int = 20) -> list[dict]:
    """Return active AI error memories."""
```

---

## Existing Files to Modify

### 1. `app/services/scheduler.py`

Modify the scheduler to use learned reputation and threshold policies.

#### Current Concept

```text
Scheduler checks companies at fixed/adaptive intervals and uses candidate scores.
```

#### New Concept

```text
Scheduler ranks company/source jobs by expected value using learned source performance.
```

#### Add Priority Formula

```text
priority_score =
  0.25 × company_importance
+ 0.20 × source_reliability
+ 0.20 × catalyst_window_score
+ 0.15 × historical_conversion_rate
+ 0.10 × source_freshness
- 0.15 × false_positive_rate
- 0.10 × estimated_cost
```

Clamp between `0.0` and `1.0`.

#### Scheduler Should Use

```python
source_rep = get_source_reputation(source_type, source_key, event_type=None, company_id=company_id)
priority_multiplier = source_rep["scheduler_priority_multiplier"]
```

#### Job Selection Should Prefer

- official sources with high true-positive rates
- sources near expected catalyst windows
- companies with high importance scores
- source/event pairs with high precision and low false positives
- cheaper sources when expected value is similar

#### Job Selection Should Deprioritize

- noisy sources
- sources with high contradiction rate
- repeated duplicate sources
- PubMed as a catalyst trigger unless strict requirements are met
- GDELT/news unless official confirmation exists

---

### 2. `app/services/event_gate.py`

Modify event gate thresholds to be dynamic.

#### Add

```python
policy = get_gate_policy(candidate.source_type, predicted_event_type, company_id)
promote_threshold = policy["promote_threshold"]
review_threshold = policy["review_threshold"]
```

#### Add Error Memory Rules

```python
error_rules = get_active_error_rules(event_type, source_type, company_id)
```

Apply them before promotion.

Example:

```text
If active error rule says "rumor promoted as acquisition",
then news acquisition candidates require official confirmation.
```

#### Required Gate Rule

The AI or dynamic threshold system must never bypass required fields.

```text
No company/asset/event-action proof = no promotion.
```

---

### 3. `app/services/ai_candidate_reviewer.py`

Replace local-rule-only behavior with structured AI review.

#### Required

- Use deterministic gate result as input.
- Use source reputation as input.
- Use active error memories as input.
- Return structured JSON.
- Validate JSON before using it.
- Fall back to deterministic gate only when AI is unavailable.
- Record every review in `ai_predictions`.

#### Prompt Requirements

The reviewer prompt should explicitly tell the model:

```text
You are not deciding whether the item is biotech-related.
You are deciding whether it is one of four catalyst events:
trial_readout, fda_decision, licensing_deal, acquisition.

If required fields are missing, do not promote.
If item is useful but not a catalyst, store_as_context.
If item is irrelevant, reject_irrelevant.
If item is borderline, needs_review.
```

---

### 4. `app/services/ai_gate_learning.py`

Replace the simple local-rule learning with outcome-based learning.

This file should become either:

- a wrapper around the new learning services, or
- be renamed/deprecated in favor of `policy_updater.py`.

#### New Responsibilities

```text
- consume verified prediction outcomes
- update source reputation
- update confidence calibration
- optimize thresholds
- record AI error memories
- generate daily learning summaries
```

Do not label outcomes only from local candidate rules.

---

### 5. `app/services/analyzer.py`

When a catalyst report is generated:

1. Attach report ID to the original prediction.
2. Record peer-effect predictions.
3. Store model confidence and evidence support.
4. Send claims through the evidence/citation verifier.
5. Schedule outcome checks.

Pseudo-flow:

```python
report = analyze_catalyst(catalyst_id)
attach_report_to_prediction(prediction_id, report.id)
record_peer_effect_predictions(report.id, prediction_id)
schedule_outcome_checks(prediction_id)
```

---

### 6. `app/db.py` or Migration System

Add all new tables.

If the current project has a function like `init_db()` or `ensure_schema()`, add the new SQL there.

If the project uses migration files, add a migration:

```text
migrations/XXXX_add_outcome_learning_engine.sql
```

---

### 7. Frontend

Add a new page:

```text
/learning
```

or a dashboard tab:

```text
AI Learning
```

The UI should show:

1. Source reliability table.
2. False-positive trends.
3. Catalyst detection precision.
4. Event-type accuracy.
5. Confidence calibration.
6. Recent policy updates.
7. Active AI error patterns.
8. Peer read-through outcome accuracy.
9. Pending outcome checks.
10. Sources downgraded/upgraded by the AI.

---

## API Endpoints to Add

Add endpoints similar to these:

```text
GET /api/learning/summary
GET /api/learning/source-reputation
GET /api/learning/confidence-calibration
GET /api/learning/policy-updates
GET /api/learning/error-memory
GET /api/learning/pending-outcome-checks
POST /api/learning/run-cycle
POST /api/learning/run-outcome-checks
POST /api/learning/recalculate-thresholds
```

### Response Example: `GET /api/learning/summary`

```json
{
  "window_days": 30,
  "total_predictions": 245,
  "true_positives": 38,
  "false_positives": 21,
  "partial_positives": 14,
  "pending": 172,
  "precision": 0.64,
  "false_positive_rate": 0.19,
  "top_reliable_sources": [
    {"source": "sec_edgar", "reliability_score": 0.94},
    {"source": "company_news", "reliability_score": 0.88}
  ],
  "noisy_sources": [
    {"source": "gdelt_news", "reliability_score": 0.22},
    {"source": "pubmed", "reliability_score": 0.31}
  ],
  "recent_policy_updates": 6,
  "active_error_patterns": 9
}
```

---

## AI Reviewer Prompt Specification

Create a prompt template for candidate review.

### System Message

```text
You are an expert biotech catalyst review model inside a catalyst monitoring system.

Your job is not to decide whether an item is generally biotech-related.
Your job is to decide whether the item is one of the four allowed catalyst event types:

1. trial_readout
2. fda_decision
3. licensing_deal
4. acquisition

If the item is useful background context but not a catalyst, classify it as store_as_context.
If the item is irrelevant, classify it as reject_irrelevant.
If it is borderline and needs later confirmation, classify it as needs_review.
Only classify as promote_to_catalyst when the required fields for the event type are present.

You must be strict. Avoid false positives. Never promote broad scientific context as a catalyst.
```

### User Message Should Include

```text
Company:
- name
- ticker
- known assets
- target
- indication
- aliases

Candidate:
- title
- url
- source_type
- publisher
- published_at
- excerpt
- fetched text if available

Deterministic gate result:
- score
- decision
- detected event type
- missing fields
- reasons

Source reputation:
- reliability score
- false-positive rate
- historical conversion rate

Active error-memory rules:
- relevant corrective rules

Task:
Return strict JSON only.
```

### Required JSON Output

```json
{
  "decision": "store_as_context",
  "event_type": "none",
  "confidence": 0.72,
  "company_detected": true,
  "asset_detected": false,
  "event_action_detected": false,
  "required_fields_present": false,
  "missing_fields": ["asset", "event action", "official confirmation"],
  "risk_flags": ["target_or_indication_only_match"],
  "reasoning_summary": "The item discusses the disease area but does not report a specific catalyst event for the watched company.",
  "evidence_needed_later": []
}
```

---

## AI Outcome Verifier Prompt Specification

### System Message

```text
You are an outcome verification model for a biotech catalyst monitoring system.

You are given a previous prediction and later evidence.
Your job is to decide whether the original prediction was correct.

You must classify the outcome strictly as:
true_positive, false_positive, partial_positive, misclassified_event_type,
premature_prediction, unsupported_prediction, contradicted_prediction,
true_context, false_negative, still_pending, or insufficient_evidence.

Do not reward a prediction just because it sounded plausible.
Reward it only if later evidence supports it.
Official sources are strongest. Market movement alone is not proof.
```

### Input

```text
Original prediction:
- predicted decision
- event type
- source
- confidence
- reasoning
- evidence spans

Later evidence:
- official sources found
- source titles
- exact evidence spans
- dates
- contradiction evidence

Task:
Return strict JSON only.
```

### Required JSON Output

```json
{
  "outcome_label": "false_positive",
  "outcome_confidence": 0.91,
  "supporting_evidence_ids": [],
  "contradicting_evidence_ids": [42],
  "reason": "The original item was a rumor and no company, SEC, or FDA source confirmed the event after 30 days.",
  "recommended_learning_actions": [
    {
      "type": "raise_threshold",
      "source_type": "gdelt_news",
      "event_type": "acquisition",
      "reason": "Rumor-like acquisition items from general news created a false positive."
    }
  ]
}
```

---

## Event-Type Outcome Verification Criteria

### Trial Readout

#### True Positive Criteria

A prediction is a true positive if later evidence confirms:

```text
company/sponsor
asset/intervention
trial/readout
result or outcome
date
```

#### False Positive Criteria

Label false positive if:

```text
no actual result/readout occurred
article was background science only
trial status changed but no result was reported
the candidate described a different company/asset
```

#### Partial Positive Criteria

Use partial positive if:

```text
the candidate detected a real trial update but event type or confidence was overstated
the candidate was early but lacked endpoint detail
```

---

### FDA Decision

#### True Positive Criteria

Requires later official or company-confirmed evidence of:

```text
approval
CRL
label expansion
clinical hold
advisory committee decision/recommendation
PDUFA action
regulatory rejection/delay
```

#### False Positive Criteria

```text
old FDA database record treated as new event
regulatory speculation with no action
FDA mention unrelated to company/asset
```

---

### Licensing Deal

#### True Positive Criteria

Requires:

```text
licensor
licensee
asset/platform
rights/deal language
official company source, SEC filing, or reputable confirmation
```

#### False Positive Criteria

```text
collaboration rumor
old partnership
general scientific collaboration
no financial/rights/deal terms
```

---

### Acquisition

#### True Positive Criteria

Requires:

```text
acquirer
target
definitive transaction language
official company source, SEC filing, or reputable confirmation
```

#### False Positive Criteria

```text
takeover rumor
analyst speculation
market chatter
generic M&A article
```

---

## Learning Policy Logic

### Source Reputation Update

When an outcome is verified, update source reputation.

Pseudo-code:

```python
def update_source_reputation_from_outcome(prediction, outcome):
    record = get_or_create_source_reputation(
        source_type=prediction["source_type"],
        source_key=prediction["source_key"],
        event_type=prediction["predicted_event_type"],
        company_id=prediction["company_id"],
    )

    record.total_candidates += 1

    if prediction["predicted_decision"] == "promote_to_catalyst":
        record.promoted_count += 1
    elif prediction["predicted_decision"] == "store_as_context":
        record.context_count += 1
    elif prediction["predicted_decision"] == "reject_irrelevant":
        record.rejected_count += 1

    if outcome["outcome_label"] == "true_positive":
        record.true_positives += 1
    elif outcome["outcome_label"] == "false_positive":
        record.false_positives += 1
    elif outcome["outcome_label"] == "partial_positive":
        record.partial_positives += 1
    elif outcome["outcome_label"] == "contradicted_prediction":
        record.contradicted_count += 1
    elif outcome["outcome_label"] == "unsupported_prediction":
        record.unsupported_count += 1

    recompute_rates(record)
    record.reliability_score = compute_reliability_score(record)
    record.scheduler_priority_multiplier = compute_scheduler_multiplier(record)
    record.gate_strictness_multiplier = compute_gate_strictness(record)
    save(record)
```

---

### Threshold Optimization

Pseudo-code:

```python
def optimize_thresholds_for_policy(source_type, event_type, company_id=None):
    stats = load_outcome_stats(source_type, event_type, company_id)
    policy = get_gate_policy(source_type, event_type, company_id)

    old_policy = policy.copy()

    if stats.total_candidates < 10:
        return no_change("Insufficient data")

    if stats.false_positive_rate > 0.30:
        policy.promote_threshold = min(0.95, policy.promote_threshold + 0.05)

    if stats.false_positive_rate > 0.50:
        if source_type in {"gdelt_news", "general_news", "blog", "pubmed"}:
            policy.min_official_confirmation_required = 1

    if stats.false_negative_rate > 0.20 and stats.precision_score > 0.70:
        policy.review_threshold = max(0.35, policy.review_threshold - 0.05)

    if stats.precision_score > 0.85 and stats.total_candidates >= 20:
        policy.promote_threshold = max(0.55, policy.promote_threshold - 0.03)

    if abs(policy.promote_threshold - old_policy.promote_threshold) > 0.10:
        cap_change_to_0_10(policy, old_policy)

    create_policy_update(old_policy, policy, stats)
    save(policy)
```

---

### Confidence Calibration

Pseudo-code:

```python
def update_confidence_calibration(prediction, outcome):
    bucket = bucket_confidence(prediction["predicted_confidence"])
    record = get_or_create_calibration_record(
        model_name=prediction["model_name"],
        event_type=prediction["predicted_event_type"],
        source_type=prediction["source_type"],
        confidence_bucket=bucket,
    )

    record.prediction_count += 1

    if outcome["outcome_label"] == "true_positive":
        record.correct_count += 1
    elif outcome["outcome_label"] == "partial_positive":
        record.partial_count += 1
    else:
        record.incorrect_count += 1

    record.observed_accuracy = (
        record.correct_count + 0.5 * record.partial_count
    ) / max(1, record.prediction_count)

    record.expected_confidence = bucket_midpoint(bucket)
    record.calibration_error = record.expected_confidence - record.observed_accuracy

    if record.calibration_error > 0.15:
        record.recommended_multiplier = max(0.65, record.observed_accuracy / record.expected_confidence)
    else:
        record.recommended_multiplier = 1.0

    save(record)
```

---

### Error Memory Creation

Pseudo-code:

```python
def detect_error_patterns(prediction, outcome):
    if (
        prediction["source_type"] == "pubmed"
        and prediction["predicted_event_type"] == "trial_readout"
        and outcome["outcome_label"] == "false_positive"
    ):
        record_error_pattern(
            error_type="pubmed_context_promoted_as_catalyst",
            event_type="trial_readout",
            source_type="pubmed",
            pattern_description="PubMed context/science article was promoted as a trial readout.",
            corrective_rule="PubMed trial_readout promotion requires company + asset + explicit result language + recent clinical event evidence.",
            example_prediction_id=prediction["id"],
            severity=0.85,
        )
```

---

## Integration With Cost-Aware Scheduler

The Catalyst Outcome Learning Engine must feed into the scheduler.

### Scheduler Inputs From Learning Engine

```text
source_reliability
false_positive_rate
historical_conversion_rate
scheduler_priority_multiplier
gate_strictness_multiplier
active error memory
confidence calibration
event-type threshold policy
```

### Updated Scheduler Formula

```text
expected_value =
(company_importance × source_reliability × catalyst_window_score × historical_conversion_rate)
÷ max(estimated_cost, 0.01)
```

Then adjust:

```text
priority_score =
expected_value
× scheduler_priority_multiplier
× freshness_multiplier
× deduplication_multiplier
```

If a source is noisy:

```text
scheduler_priority_multiplier < 1.0
```

If a source is high-signal:

```text
scheduler_priority_multiplier > 1.0
```

### Example

```text
SEC EDGAR + acquisition:
high reliability, low false positives
→ check more often
→ lower threshold slightly

PubMed + trial_readout:
low catalyst conversion, many context-only items
→ check less often for catalyst detection
→ still store as context
→ require stricter gate for promotion

GDELT/news + acquisition:
high rumor/noise rate
→ require official confirmation
→ route to needs_review/context, not immediate catalyst
```

---

## Integration With False-Positive Reduction Pipeline

The Outcome Learning Engine should improve the false-positive pipeline over time.

### Existing False-Positive Pipeline

```text
source item
→ context quarantine
→ deterministic event gate
→ AI review for borderline items
→ promote/store/reject
```

### With Learning Engine

```text
source item
→ context quarantine
→ deterministic event gate using learned thresholds
→ AI review using source reputation and active error memory
→ prediction ledger stores decision
→ promoted catalysts analyzed
→ outcome checks scheduled
→ outcomes update future thresholds
```

This creates a full feedback loop.

---

## Integration With Evidence/Citation Verification

The learning engine should also consume evidence-verification outcomes.

If the evidence verifier finds:

```text
unsupported claim
contradicted claim
weak citation
missing source span
```

then the learning engine should:

1. Lower report confidence.
2. Record an error pattern.
3. Update confidence calibration.
4. If source-related, update source reputation.
5. Add corrective rules for future analysis.

Example:

```text
The analyzer repeatedly claims "superior safety" when evidence only says "generally well tolerated."
```

Create error memory:

```text
safety_superiority_overstatement
```

Corrective rule:

```text
Safety superiority claims require comparative safety evidence against competitors.
```

---

## AI Learning Dashboard

Add a frontend page or section called:

```text
AI Learning
```

### Required Sections

#### 1. Summary Cards

Show:

```text
Total predictions
True positives
False positives
Partial positives
Pending checks
Precision
False-positive rate
Average confidence calibration error
Sources downgraded
Sources upgraded
Active error patterns
```

#### 2. Source Reputation Table

Columns:

```text
Source
Event Type
Company
Total Candidates
True Positives
False Positives
Precision
False Positive Rate
Reliability Score
Scheduler Multiplier
Gate Strictness
```

#### 3. Confidence Calibration Chart/Table

Columns:

```text
Model
Event Type
Source Type
Confidence Bucket
Prediction Count
Observed Accuracy
Calibration Error
Recommended Multiplier
```

#### 4. Recent Policy Updates

Columns:

```text
Date
Update Type
Source Type
Event Type
Old Value
New Value
Reason
Applied
```

#### 5. AI Error Memory

Columns:

```text
Error Type
Source Type
Event Type
Seen Count
Severity
Corrective Rule
Active
```

#### 6. Peer Effect Accuracy

Columns:

```text
Peer Company
Predicted Direction
Read-Through Type
Outcome
Evidence
Confidence
```

#### 7. Pending Outcome Checks

Columns:

```text
Prediction
Company
Event Type
Check Type
Scheduled For
Status
```

---

## Backtesting Mode

Add a backtesting mode so the learning system can be seeded with historical events.

### Purpose

The system will not have enough live outcomes at first. Backtesting gives it initial data.

### Recommended Historical Dataset Format

Create a file:

```text
data/historical_catalysts.json
```

Example:

```json
[
  {
    "company": "Example Bio",
    "ticker": "EXBIO",
    "event_type": "trial_readout",
    "event_date": "2024-09-12",
    "asset": "examplemab",
    "indication": "NSCLC",
    "outcome_label": "true_positive",
    "official_source_url": "https://...",
    "source_type": "company_news"
  }
]
```

### Backtest Flow

```text
1. Load historical catalysts.
2. Replay source candidates from before each event if available.
3. Run event gate and AI reviewer.
4. Compare predictions against known outcome.
5. Populate ai_predictions and outcome labels.
6. Update source reputation and calibration.
```

### Minimum Data Before Claims

Do not claim strong learning until the system has at least:

```text
500+ labeled candidate outcomes
100+ true catalyst examples
100+ false-positive examples
coverage across all four event types
coverage across multiple source types
```

Until then, call it:

```text
policy learning and outcome calibration
```

not:

```text
fully trained AI model
```

---

## Implementation Phases

### Phase 1: Prediction Ledger and Outcome Tables

Add:

```text
ai_predictions
prediction_outcome_checks
source_reputation
gate_policy_thresholds
confidence_calibration
ai_error_memory
ai_policy_updates
peer_effect_outcomes
false_negative_candidates
```

Add `prediction_ledger.py`.

Acceptance:

```text
Every candidate decision creates an ai_predictions row.
Every promoted catalyst schedules outcome checks.
```

---

### Phase 2: Real AI Candidate Reviewer

Upgrade `ai_candidate_reviewer.py`.

Acceptance:

```text
Borderline candidates use structured AI review when OPENAI_API_KEY exists.
AI output is validated.
Deterministic fallback remains for offline mode.
Every AI review is stored.
```

---

### Phase 3: Outcome Verifier

Add `outcome_verifier.py`.

Acceptance:

```text
System creates 24h, 7d, 30d, and 90d checks.
Due checks can be run from API or scheduler.
Outcome labels are saved.
```

---

### Phase 4: Source Reputation and Confidence Calibration

Add:

```text
source_reputation.py
confidence_calibrator.py
```

Acceptance:

```text
Verified outcomes update source reliability.
Confidence buckets update observed accuracy.
Future confidence scores use calibration multiplier.
```

---

### Phase 5: Threshold Optimizer and AI Error Memory

Add:

```text
threshold_optimizer.py
ai_error_memory.py
policy_updater.py
```

Acceptance:

```text
False-positive-heavy sources become stricter.
High-precision official sources become slightly less strict.
Repeated error patterns become active corrective rules.
All policy updates are logged.
```

---

### Phase 6: Scheduler/Event Gate Integration

Modify:

```text
scheduler.py
event_gate.py
```

Acceptance:

```text
Scheduler uses source reputation and priority multipliers.
Event gate uses dynamic thresholds.
Active error memory affects gate decisions.
```

---

### Phase 7: Peer Outcome Tracking

Add:

```text
peer_outcome_tracker.py
```

Acceptance:

```text
Peer effect predictions are stored.
Pending peer outcomes are checked.
Peer read-through accuracy appears in dashboard.
```

---

### Phase 8: Learning Dashboard

Add backend endpoints and frontend page.

Acceptance:

```text
User can see source reliability, false positives, confidence calibration, error memory, policy updates, and pending checks.
```

---

### Phase 9: Backtesting Mode

Add historical replay.

Acceptance:

```text
System can ingest historical outcomes and seed learning tables.
Backtest metrics appear in learning dashboard.
```

---

## Acceptance Tests

### Test 1: Prediction Ledger

Given a candidate is reviewed:

```text
When event gate returns a decision
Then ai_predictions row is created
And source_type, event_type, confidence, and risk flags are saved
```

### Test 2: Outcome Check Scheduling

Given a candidate is promoted:

```text
When catalyst is saved
Then outcome checks are scheduled for 24h, 7d, 30d, and 90d
```

### Test 3: Source Reputation Update

Given a prediction is labeled false positive:

```text
When update_source_reputation_from_outcome runs
Then false_positive_count increases
And reliability_score decreases
And gate_strictness_multiplier increases
```

### Test 4: Threshold Optimization

Given PubMed trial readout predictions have high false-positive rate:

```text
When nightly_threshold_optimization runs
Then PubMed trial_readout promote_threshold increases
And a policy update is logged
```

### Test 5: Confidence Calibration

Given 0.90 confidence predictions are only correct 60% of the time:

```text
When confidence calibration updates
Then recommended_multiplier becomes less than 1.0
And future confidence is reduced
```

### Test 6: Error Memory

Given a PubMed context article was falsely promoted:

```text
When detect_error_patterns runs
Then ai_error_memory records pubmed_context_promoted_as_catalyst
And future PubMed trial_readout candidates require stricter evidence
```

### Test 7: Scheduler Integration

Given SEC has high reliability and GDELT has low reliability:

```text
When scheduler ranks jobs
Then SEC jobs receive higher priority
And GDELT jobs receive lower priority or require confirmation
```

### Test 8: Peer Outcome

Given a report predicted positive validation read-through for a peer:

```text
When peer_outcome_tracker runs
Then peer_effect_outcomes is updated with supported/unsupported/partial label
```

### Test 9: No Silent Policy Changes

Given any threshold changes:

```text
When policy update is applied
Then ai_policy_updates contains old value, new value, reason, and affected predictions
```

---

## Success Metrics

The project should track these metrics before and after implementation.

### Detection Metrics

```text
candidate precision
candidate recall
false-positive rate
false-negative rate
event-type accuracy
promotion accuracy
```

### Evidence Metrics

```text
percentage of predictions with official confirmation
percentage of predictions contradicted
average evidence support score
unsupported claim rate
```

### Scheduler Metrics

```text
cost per true catalyst
tokens per promoted catalyst
source checks per true catalyst
duplicate processing rate
```

### Learning Metrics

```text
source reliability improvement
confidence calibration error
threshold changes over time
AI error recurrence rate
peer-effect support rate
```

### Goal Targets

Initial targets:

```text
false-positive rate reduced by 40%+
cost per true catalyst reduced by 25%+
unsupported claim rate below 5%
confidence calibration error below 0.15
official-source confirmation for promoted catalysts above 80%
```

---

## Important Safety and Accuracy Rules

1. Do not call broad biotech context a catalyst.
2. Do not promote rumors as acquisitions.
3. Do not treat ClinicalTrials.gov status changes as readouts unless results or event action exist.
4. Do not treat PubMed mechanism articles as trial readouts.
5. Do not allow AI reviewer to override missing required fields.
6. Do not silently adjust thresholds.
7. Do not claim model retraining unless actual training exists.
8. Do not use market movement alone as proof of peer read-through.
9. Do not let confidence exceed evidence support.
10. Do not publish unsupported predictions as final reports.

---

## Recommended Naming

Use these names in code and UI:

```text
Catalyst Outcome Learning Engine
Prediction Ledger
Outcome Verifier
Source Reputation
Confidence Calibration
Threshold Optimizer
AI Error Memory
Policy Updater
Peer Outcome Tracker
AI Learning Dashboard
```

---

## Final Expected Behavior

After implementation, the project should work like this:

```text
1. Scheduler fetches source item.
2. Item is stored as context/candidate.
3. Event gate evaluates item using learned thresholds.
4. AI reviewer reviews borderline/high-value candidates.
5. Decision is stored in ai_predictions.
6. True catalyst candidates are promoted.
7. Catalyst analysis runs.
8. Evidence verifier checks report claims.
9. Outcome checks are scheduled.
10. Later sources are checked.
11. Outcome is labeled.
12. Source reputation updates.
13. Confidence calibration updates.
14. Thresholds update.
15. Error memory records repeated mistakes.
16. Scheduler uses new learning data.
17. Future false positives decrease and confidence becomes more accurate.
```

This is the desired final transformation:

```text
Before:
The system looked like it could learn, but mostly used deterministic scaffolding.

After:
The system records predictions, verifies delayed outcomes, learns source reliability,
calibrates confidence, adjusts thresholds, tracks repeated errors, and improves future decisions.
```

---

## Implementation Instruction for the AI Coding Agent

When implementing this specification:

1. Preserve existing project behavior unless directly changing it for learning integration.
2. Do not remove the deterministic event gate; make it stronger with learned policies.
3. Do not remove the cost-aware scheduler; make it use source reputation and expected-value scheduling.
4. Do not remove the false-positive reduction pipeline; feed learning outcomes into it.
5. Keep deterministic fallback behavior for local/offline runs, but do not present it as full AI learning.
6. Use structured JSON outputs for AI reviewer and AI outcome verifier.
7. Validate all AI outputs before saving or using them.
8. Add all new database tables safely with `CREATE TABLE IF NOT EXISTS`.
9. Make the UI show what the AI learned and why policies changed.
10. Add tests or test scripts that prove learning changes thresholds/reputation/confidence based on outcomes.

The final implementation should make the Catalyst Propagation Agent significantly more credible to a biotech company or investor because the system can now demonstrate not only what it predicted, but whether it was right, why it was right or wrong, and how it changed future behavior based on evidence.
