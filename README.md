# Catalyst Propagation Agent

Monitors a watchlist of biotech companies and, when a catalyst hits one of them —
a trial readout, FDA decision, licensing deal, or acquisition — analyzes the event
and reports its **propagated effects** across three layers:

1. **Drug / target** — does the event validate or undermine the underlying biology?
2. **Indication** — how does it shift the standard of care and the competitive bar
   for that disease?
3. **Peer companies** — read-throughs to competitors and partners, distinguishing
   *validation* read-throughs (a win that helps same-target rivals) from
   *competitive* read-throughs (a win that hurts them). No blanket contagion.

Every reported effect carries a **direction** (positive / negative / neutral), a
**calibrated confidence**, and **claims cited to real sources**.

---

## Quickstart

```bash
git clone <repo-url> && cd catalyst-propagation-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your API keys (see below)
./run_local.sh                # serves http://127.0.0.1:8000
```

The app runs fully offline with no keys: it falls back to a deterministic local
analyzer so the UI, scheduler, and evidence pipeline all work. Adding an
`OPENAI_API_KEY` upgrades analysis to the live model.

---

## Configuration

Set these in `.env` (the file is git-ignored; never commit real keys):

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enables live model analysis. Without it, the local fallback is used. | _(empty)_ |
| `OPENAI_MODEL` | Primary analysis model. | `gpt-5.5` |
| `OPENAI_MODEL_FALLBACKS` | Tried in order only if the primary call fails. | `gpt-5.2,gpt-5.4-mini,gpt-4.1-mini` |
| `SCRAPERAPI_API_KEY` | Proxy for fetching pages that block direct requests. SEC fetches bypass it. | _(empty)_ |
| `OPENAI_STRICT_ANALYSIS` | If `true`, an OpenAI failure raises instead of falling back. | `false` |
| `SCHEDULER_DAILY_BUDGET_USD` | Caps model spend per day for the adaptive scheduler. | `2.50` |

Verify your keys after editing `.env`:

```bash
python scripts/check_openai_key.py   # live OpenAI Responses smoke test
```

---

## Architecture

```
app/
  main.py            FastAPI routes (HTML pages + JSON API + form handlers)
  db.py              SQLite schema, migrations, query helpers
  config.py          Typed settings loaded from .env
  services/
    monitor.py             Watchlist ingestion + catalyst detection orchestration
    sources.py             Source adapters: ClinicalTrials.gov, openFDA, FDA press,
                           SEC/EDGAR, company IR/RSS  (+ ScraperAPI fetch helper)
    clinical_extractor.py  Pulls endpoints / stats / deal terms into a fact checklist
    analyzer.py            OpenAI Responses (strict JSON schema) -> 3-layer report
    evidence.py            Evidence guardrails + confidence caps
    verified_evidence_graph.py  Claim-to-span verification, source snapshots, hashes
    peer_graph.py          Validation vs. competitive read-through classification
    scheduler.py           Budget-aware adaptive check cadence per company
    outcome_learning.py    Predictions ledger, delayed outcome checks, calibration
  templates/ , static/     Jinja pages + CSS
scripts/             Operational + verification scripts
docs/                Design specs and the audit report
tests/               Offline pytest suite
```

### How ingestion saves tokens

Sources are fetched and de-duplicated *before* any model call. A short-lived
SQLite cache prevents re-fetching the same official page across monitor runs.
Only catalyst-classified, company-relevant documents are passed to the analyzer,
and the analyzer receives a compact extracted **fact checklist** rather than raw
HTML. The adaptive scheduler raises check frequency for companies with imminent
catalysts and backs off otherwise, all under a daily USD budget.

### Accuracy & trust design

The pipeline is **generate → verify → score → render**, never generate → display:

- **Claim-to-span verification** — a citation counts only when an exact supporting
  span is found in a content-hashed source snapshot; keyword overlap is not enough.
- **Confidence caps** — weak, partial, unsupported, or contradicted evidence
  automatically lowers report and per-effect confidence.
- **Contradiction surfacing** — disagreeing sources are flagged, not averaged away.
- **Source-type requirements** — regulatory claims require FDA/official/company
  sources; peer read-through claims require peer evidence.
- **Outcome learning** — gate decisions become auditable predictions that are
  checked later to recalibrate source reputation and thresholds.

Verified behavior: with weak or uncited evidence the analyzer returns *low*
confidence and a *neutral* direction rather than inventing certainty.

---

## Data storage

Everything persists in a single SQLite database (`DATABASE_PATH`, default
`./catalyst_propagation.db`): companies/watchlist, catalysts, source documents and
snapshots, reports and per-layer effects, the verified-evidence graph, the
predictions ledger, source reputation, and OpenAI usage/health logs. The schema is
created and migrated automatically on startup. The database is git-ignored.

---

## Key endpoints

| Page | |
|---|---|
| `GET /` | Dashboard: watchlist, catalyst feed, reports |
| `GET /learning` | Outcome-learning dashboard |
| `GET /report/{id}` | Report with the evidence panel |

| API | |
|---|---|
| `POST /api/seed` | Load demo companies |
| `POST /api/monitor/run` | Run a monitoring pass over the watchlist |
| `POST /api/catalysts/manual` | Add a catalyst manually |
| `POST /api/catalysts/{id}/analyze` | Analyze a catalyst → report |
| `GET  /api/reports/{id}/evidence-graph` | Verified evidence graph for a report |
| `GET  /api/openai/status` | OpenAI configuration / health |
| `GET  /api/learning/dashboard` | Learning-engine metrics |
| `GET  /openapi.json`, `/docs` | OpenAPI schema + Swagger UI |

---

## Testing

```bash
pytest                       # offline suite (no network/keys required)
python scripts/test_outcome_learning.py
python scripts/test_verified_evidence_graph.py
```

---

## License

Proprietary — internal company use.
