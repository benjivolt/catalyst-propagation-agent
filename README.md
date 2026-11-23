# Catalyst Propagation Agent

A biotech news monitor that asks what a single event means for everything connected to it. When a company reports a trial readout, an FDA decision or a deal, the agent writes a report on how that event reads through to the drug target, the disease indication and the peer companies on your watchlist, and it ties every claim to a quoted source passage. Read-through is the part most news feeds skip: a Phase 3 win for one FcRn antibody says something about every other FcRn program, and something different about competitors in the same disease.

The app runs locally with FastAPI and SQLite. An OpenAI key is optional; without one, reports come from a rule-based analyzer.

## How it works

1. **Sources.** For each watched company the app polls SEC EDGAR, FDA press announcements, openFDA Drugs@FDA, ClinicalTrials.gov, PubMed, GDELT news, and any IR pages or RSS feeds you add. Outbound fetches go through an SSRF guard with size and time limits.
2. **Cheap detector.** A weighted keyword model (`app/services/detection.py`) labels each item as a trial readout, FDA decision, licensing deal, acquisition or other catalyst, discounts forward-looking announcements ("plans to report topline data in Q4"), and handles negation ("did not meet the primary endpoint"). It costs nothing to run, so it filters noise before any model call.
3. **Adaptive scheduler.** Each company and source pair gets a priority from catalyst timing hints, source reliability, recent activity, past usefulness, an exploration bonus for rarely checked sources, and check cost. Next checks land between 15 minutes and 24 hours away, and quiet sources back off.
4. **LLM analysis.** Candidates that clear the score threshold become catalysts. With a key, the analyzer calls the OpenAI Responses API with a strict JSON schema that requires a direction, confidence and cited claims for each target, indication and peer effect, and it falls back through a configurable model list. Without a key, or after an API error, the rule-based analyzer produces the same report shape.
5. **Evidence checks.** Guardrails cap confidence when effects lack citations, when a peer read-through has no relationship source, or when fewer than two sources back it. A second pass splits sources into paragraphs and scores each claim against them (see [docs/verified_evidence_graph.md](docs/verified_evidence_graph.md)). Each report gets an evidence grade from A to D.
6. **Human review.** Reports start as "needs review" unless every check passes with grade A. A reviewer approves or rejects each one on its report page.
7. **Alerts.** Slack and email alerts go out only for approved reports.
8. **Outcome learning.** Every candidate decision is stored as a prediction and checked again after 1, 7, 30 and 90 days against the sources and verification results stored by then. The results update per-source reliability, confidence calibration and the promote thresholds the scheduler uses.

## Quick start

You need Python 3.12 or newer. The run scripts create `.venv`, install dependencies, copy `.env.example` to `.env` if it is missing, and start the server on `127.0.0.1:8000`.

Windows (PowerShell):

```powershell
.\run_local.ps1
```

macOS or Linux:

```bash
./run_local.sh
```

Open http://127.0.0.1:8000. On an empty database the home page offers "Load sample watchlist" and "Create demo reports". The demo reports cover three real FDA approvals (Rezdiffra for MASH, Casgevy for sickle cell disease, Rystiggo for myasthenia gravis) with official source excerpts, so you can see full reports without an API key or network access.

To use OpenAI, set `OPENAI_API_KEY` in `.env` and check it with:

```bash
python scripts/check_openai_key.py
```

### Docker

```bash
docker build -t catalyst-propagation .
docker run --rm -p 127.0.0.1:8080:8080 -v catalyst-data:/data \
  -e ALLOWED_HOSTS=localhost,127.0.0.1 \
  catalyst-propagation
```

This command publishes the port on localhost only. Before enabling remote access, set a strong, unique `APP_PASSWORD` and put the app behind HTTPS.

## Configuration

Settings come from environment variables or `.env`. Variables already set in the environment take precedence over `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Enables LLM analysis. Empty means rule-based reports. |
| `OPENAI_MODEL` | `gpt-5.5` | Primary model. |
| `OPENAI_MODEL_FALLBACKS` | `gpt-5.2,gpt-5.4-mini,gpt-4.1-mini` | Models tried in order if the primary call fails. |
| `OPENAI_STRICT_ANALYSIS` | `false` | Fail instead of falling back to rule-based analysis. |
| `OPENAI_INPUT_COST_PER_1M`, `OPENAI_OUTPUT_COST_PER_1M` | `0` | Token prices for spend tracking. At 0, each call is counted as $0.012. |
| `SCHEDULER_DAILY_BUDGET_USD` | `2.50` | Daily cap on estimated source-check and LLM spend. Set 0 to disable. |
| `SCHEDULER_CANDIDATE_THRESHOLD` | `0.62` | Score a candidate needs to become a catalyst. A lower learned threshold for that source can override it. |
| `SCHEDULER_LLM_THRESHOLD` | `0.72` | Minimum score to spend an LLM call on a new catalyst. |
| `ADAPTIVE_SCHEDULER_AUTOSTART` | `true` | Run the scheduler inside the web process. |
| `EXTERNAL_SCHEDULER_WORKER` | `false` | Set to `true` when cron runs `scripts/scheduler_worker.py` instead. |
| `APP_PASSWORD` | empty | Turns on HTTP basic auth (any username). |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1,testserver` | Accepted `Host` headers. |
| `DATABASE_PATH` | `./catalyst_propagation.db` | SQLite file. |
| `SEC_USER_AGENT` | placeholder | SEC asks for a contact email in the user agent. |
| `SCRAPERAPI_API_KEY` | empty | Optional proxy for company pages. |
| `SOURCE_VERIFICATION_FETCH_ON_ANALYZE` | `false` | Fetch each cited URL during analysis to confirm it. |
| `ALERTS_ENABLED` | `false` | Send alerts automatically when an approved report is saved. |
| `SLACK_WEBHOOK_URL`, `SMTP_*`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` | empty | Alert channels. |

`.env.example` lists the rest, including scheduler intervals, retries and timeouts.

## Security notes

- Without `APP_PASSWORD`, anyone who can reach the port can use the app, and it logs a warning at startup. The run scripts bind to `127.0.0.1`.
- `TrustedHostMiddleware` rejects requests whose `Host` is not in `ALLOWED_HOSTS`. POST, PUT, PATCH and DELETE requests with a cross-site `Origin` or `Referer` get a 403.
- Pages send a strict Content-Security-Policy, `X-Frame-Options: DENY` and `nosniff`. Links from source data render only if they are http or https.
- Outbound fetches allow only http and https on ports 80 and 443, refuse hosts that resolve to private, loopback or link-local addresses (including cloud metadata at 169.254.169.254), re-check every redirect, and stop at 2 MB.
- The daily budget applies to OpenAI calls as well as source checks. Once it is reached, new reports use the rule-based analyzer.
- Alerts never go out for a report that a reviewer has not approved, even when sent by hand.
- API keys and Slack webhook URLs are redacted from stored errors.

## Project layout

```text
app/
  main.py                      routes, auth, security headers
  config.py                    settings from the environment
  db.py                        SQLite schema and helpers
  services/
    sources.py                 source adapters and the guarded fetcher
    detection.py               rule-based catalyst detector
    scheduler.py               adaptive scheduler and budget
    analyzer.py                OpenAI analysis, schema, fallback routing
    expert_fallback.py         rule-based read-through analysis
    evidence.py                citation and corroboration checks
    verified_evidence_graph.py claim-to-passage verification
    peer_graph.py              target and indication overlap between companies
    outcome_learning.py        prediction ledger and threshold updates
    alerting.py                Slack and email alerts
  templates/, static/          server-rendered UI
scripts/                       demo seeding, OpenAI check, scheduler worker
tests/                         pytest suite
docs/                          claim verification notes
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app scripts tests
ruff format --check app scripts tests
```

The suite runs offline. It uses a temporary database for each test, blocks DNS lookups, and leaves the OpenAI key empty. It covers the detector's accuracy on a labeled set of headlines, the full seed-to-report pipeline, every HTML page, the SSRF guard, host and origin checks, basic auth, secret redaction, the alert approval rule, outcome learning and claim verification.

## Tech stack

Python 3.12, FastAPI, Jinja2, SQLite, httpx, Beautiful Soup, the OpenAI Python SDK, pytest and Ruff.

## License

MIT. See [LICENSE](LICENSE).
