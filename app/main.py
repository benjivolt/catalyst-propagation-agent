from __future__ import annotations

import base64
import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import settings
from app.db import execute, init_db, row, rows
from app.services.alerting import send_report_alert
from app.services.demo_scenarios import demo_readiness_summary, seed_demo_reports
from app.services.monitor import (
    add_manual_catalyst,
    analyze_and_save,
    create_company,
    delete_company,
    get_companies,
    get_company,
    get_report,
    list_catalysts,
    run_monitor,
    seed_examples,
    update_company,
)
from app.services.openai_health import openai_status
from app.services.outcome_learning import learning_dashboard, run_due_outcome_checks, run_policy_update
from app.services.peer_graph import sync_all_peer_relationships
from app.services.scheduler import AdaptiveSchedulerDaemon, run_adaptive_monitor, scheduler_dashboard
from app.services.source_verifier import verify_sources_for_catalyst
from app.services.utils import humanize, parse_list_field
from app.services.verified_evidence_graph import build_verified_evidence_graph, evidence_graph_for_report

REVIEW_STATUSES = {"needs_review", "approved", "rejected"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.app_password:
        logging.getLogger("uvicorn.error").warning(
            "APP_PASSWORD is not set. Anyone who can reach this port can use the app; keep it on localhost."
        )
    init_db()
    app.state.adaptive_scheduler = AdaptiveSchedulerDaemon()
    sync_all_peer_relationships()
    if settings.adaptive_scheduler_autostart and not settings.external_scheduler_worker:
        app.state.adaptive_scheduler.start()
    try:
        yield
    finally:
        app.state.adaptive_scheduler.stop()


app = FastAPI(title="Catalyst Propagation Agent", version="3.1.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))


def safe_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if urlsplit(url).scheme in {"http", "https"} else "#"


templates.env.filters["safe_url"] = safe_url


def humanize_filter(value: Any, capitalize: bool = False) -> str:
    text = humanize(value)
    return text[:1].upper() + text[1:] if capitalize else text


templates.env.filters["humanize"] = humanize_filter


def _password_ok(header: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        _, _, password = base64.b64decode(encoded).decode("utf-8").partition(":")
    except ValueError:
        return False
    return secrets.compare_digest(password.encode(), settings.app_password.encode())


def _same_origin(request: Request) -> bool:
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return True
    return urlsplit(source).netloc == request.headers.get("host", "")


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path == "/healthz":
        return PlainTextResponse("ok")
    if settings.app_password and not _password_ok(request.headers.get("authorization", "")):
        return PlainTextResponse(
            "Authentication required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="catalyst"'}
        )
    if request.method in UNSAFE_METHODS and not _same_origin(request):
        return PlainTextResponse("Cross-site request blocked", status_code=403)
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


def render(request: Request, template: str, section: str = "", **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {"section": section, **context})


def require(query: str, item_id: int, label: str) -> None:
    if not row(query, (item_id,)):
        raise HTTPException(status_code=404, detail=f"{label} not found")


def _scheduler_state(request: Request) -> dict[str, Any]:
    scheduler = scheduler_dashboard()
    daemon = getattr(request.app.state, "adaptive_scheduler", None)
    if scheduler.get("external_worker"):
        scheduler["runtime_label"] = "External worker"
    elif daemon and daemon.is_running():
        scheduler["runtime_label"] = "Running"
    else:
        scheduler["runtime_label"] = "Idle"
    return scheduler


def _count(table: str) -> int:
    return int(row(f"SELECT COUNT(*) AS n FROM {table}")["n"])


def _save_review(report_id: int, status: str, reviewer: str, notes: str) -> None:
    execute(
        """
        INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (report_id, status, reviewer[:120], notes[:1000]),
    )


class CompanyIn(BaseModel):
    ticker: str = Field("", max_length=12)
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=50)
    targets: list[str] = Field(default_factory=list, max_length=50)
    indications: list[str] = Field(default_factory=list, max_length=50)
    peers: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    source_urls: list[str] = Field(default_factory=list, max_length=20)
    notes: str = Field("", max_length=4000)
    active: bool = True


class ManualCatalystIn(BaseModel):
    company_id: int
    title: str = Field(min_length=1, max_length=300)
    url: str = Field("", max_length=2000)
    publisher: str = Field("Manual", max_length=120)
    event_date: str | None = Field(None, max_length=20)
    event_type: str | None = Field(None, max_length=40)
    excerpt: str = Field("", max_length=20000)
    auto_analyze: bool = True


class ReviewIn(BaseModel):
    status: str = Field(pattern="^(needs_review|approved|rejected)$")
    reviewer: str = Field("", max_length=120)
    notes: str = Field("", max_length=1000)


@app.get("/", response_class=HTMLResponse)
def catalysts_page(request: Request) -> HTMLResponse:
    companies = get_companies(active_only=False)
    stats = {
        "companies": len(companies),
        "active_companies": sum(1 for c in companies if c.get("active")),
        "catalysts": _count("catalysts"),
        "reports": _count("reports"),
    }
    return render(
        request,
        "index.html",
        "catalysts",
        catalysts=list_catalysts(limit=50),
        stats=stats,
        scheduler=scheduler_dashboard(),
        openai=openai_status(run_live_test=False),
    )


@app.get("/watchlist", response_class=HTMLResponse)
def watchlist_page(request: Request) -> HTMLResponse:
    return render(request, "watchlist.html", "watchlist", companies=get_companies(active_only=False))


@app.get("/watchlist/new", response_class=HTMLResponse)
def new_company_page(request: Request) -> HTMLResponse:
    return render(request, "company_form.html", "watchlist")


@app.get("/catalysts/new", response_class=HTMLResponse)
def new_catalyst_page(request: Request) -> HTMLResponse:
    return render(request, "catalyst_form.html", "catalysts", companies=get_companies(active_only=False))


@app.get("/company/{company_id}", response_class=HTMLResponse)
def company_page(request: Request, company_id: int) -> HTMLResponse:
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    catalysts = list_catalysts(limit=100, company_id=company_id)
    return render(request, "company.html", "watchlist", company=company, catalysts=catalysts)


@app.get("/report/{report_id}", response_class=HTMLResponse)
def report_page(request: Request, report_id: int) -> HTMLResponse:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return render(request, "report.html", "catalysts", report=report)


@app.get("/scheduler", response_class=HTMLResponse)
def scheduler_page(request: Request) -> HTMLResponse:
    runs = rows("SELECT * FROM monitor_runs ORDER BY id DESC LIMIT 6")
    return render(request, "scheduler.html", "scheduler", scheduler=_scheduler_state(request), runs=runs)


@app.get("/learning", response_class=HTMLResponse)
def learning_page(request: Request) -> HTMLResponse:
    return render(request, "learning.html", "learning", learning=learning_dashboard())


@app.get("/api/openai/status")
def api_openai_status() -> dict[str, Any]:
    return openai_status(run_live_test=False)


@app.post("/api/openai/test")
def api_openai_test() -> dict[str, Any]:
    return openai_status(run_live_test=True)


@app.post("/api/demo")
def api_demo() -> dict[str, Any]:
    return seed_demo_reports(auto_analyze=True)


@app.get("/api/demo/readiness")
def api_demo_readiness() -> dict[str, Any]:
    return demo_readiness_summary()


@app.post("/api/seed")
def api_seed() -> dict[str, str]:
    seed_examples()
    return {"status": "ok"}


@app.get("/api/companies")
def api_companies() -> list[dict[str, Any]]:
    return get_companies(active_only=False)


@app.post("/api/companies")
def api_create_company(company: CompanyIn) -> dict[str, Any]:
    return {"id": create_company(company.model_dump()), "status": "created"}


@app.put("/api/companies/{company_id}")
def api_update_company(company_id: int, company: CompanyIn) -> dict[str, str]:
    require("SELECT id FROM companies WHERE id = ?", company_id, "Company")
    update_company(company_id, company.model_dump())
    return {"status": "updated"}


@app.delete("/api/companies/{company_id}")
def api_delete_company(company_id: int) -> dict[str, str]:
    delete_company(company_id)
    return {"status": "deleted"}


@app.post("/api/catalysts/manual")
def api_manual_catalyst(payload: ManualCatalystIn) -> dict[str, Any]:
    require("SELECT id FROM companies WHERE id = ?", payload.company_id, "Company")
    catalyst_id = add_manual_catalyst(payload.company_id, payload.model_dump())
    report_id = analyze_and_save(catalyst_id) if payload.auto_analyze else None
    return {"status": "created", "catalyst_id": catalyst_id, "report_id": report_id}


@app.get("/api/catalysts")
def api_catalysts() -> list[dict[str, Any]]:
    return list_catalysts(limit=100)


@app.post("/api/catalysts/{catalyst_id}/verify-sources")
def api_verify_sources(catalyst_id: int, fetch: bool = True) -> dict[str, Any]:
    require("SELECT id FROM catalysts WHERE id = ?", catalyst_id, "Catalyst")
    verified = verify_sources_for_catalyst(catalyst_id, fetch=fetch)
    return {"status": "verified", "catalyst_id": catalyst_id, "sources_checked": len(verified), "fetch": fetch}


@app.post("/api/catalysts/{catalyst_id}/analyze")
def api_analyze(catalyst_id: int) -> dict[str, Any]:
    require("SELECT id FROM catalysts WHERE id = ?", catalyst_id, "Catalyst")
    return {"status": "analyzed", "report_id": analyze_and_save(catalyst_id)}


@app.post("/api/monitor/run")
def api_run_monitor() -> dict[str, Any]:
    return run_monitor(auto_analyze=True, per_source_limit=4)


@app.get("/api/scheduler/status")
def api_scheduler_status() -> dict[str, Any]:
    return scheduler_dashboard()


@app.post("/api/scheduler/tick")
def api_scheduler_tick() -> dict[str, Any]:
    return run_adaptive_monitor(auto_analyze=True, per_source_limit=4, due_only=True)


@app.get("/api/learning/dashboard")
def api_learning_dashboard() -> dict[str, Any]:
    return learning_dashboard()


@app.post("/api/learning/check-outcomes")
def api_learning_check_outcomes(include_not_due: bool = False) -> dict[str, Any]:
    return run_due_outcome_checks(limit=100, include_not_due=include_not_due)


@app.post("/api/learning/policy-update")
def api_learning_policy_update() -> dict[str, Any]:
    return run_policy_update()


@app.get("/api/reports/{report_id}")
def api_report(report_id: int) -> dict[str, Any]:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/api/reports/{report_id}/evidence-graph")
def api_report_evidence_graph(report_id: int) -> dict[str, Any]:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    return evidence_graph_for_report(report_id)


@app.post("/api/reports/{report_id}/verify-evidence-graph")
def api_verify_evidence_graph(report_id: int) -> dict[str, Any]:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    return build_verified_evidence_graph(report_id)


@app.post("/api/reports/{report_id}/send-alert")
def api_send_report_alert(report_id: int) -> dict[str, Any]:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    return send_report_alert(report_id, force=True)


@app.post("/api/reports/{report_id}/review")
def api_review_report(report_id: int, payload: ReviewIn) -> dict[str, Any]:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    _save_review(report_id, payload.status, payload.reviewer, payload.notes)
    return {"status": "updated", "report_id": report_id, "review_status": payload.status}


@app.post("/api/peer-graph/sync")
def api_peer_graph_sync() -> dict[str, Any]:
    return {"status": "ok", "relationships": sync_all_peer_relationships()}


def _parse_peers_json(peers_text: str) -> list[dict[str, Any]]:
    try:
        peers = json.loads(peers_text) if peers_text.strip() else []
    except json.JSONDecodeError:
        return []
    return [p for p in peers if isinstance(p, dict)][:50] if isinstance(peers, list) else []


def _field(form: Any, name: str, limit: int, default: str = "") -> str:
    return str(form.get(name) or default).strip()[:limit]


def redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=303)


@app.post("/forms/company")
async def form_company(request: Request) -> RedirectResponse:
    form = await request.form()
    data = {
        "ticker": _field(form, "ticker", 12),
        "name": _field(form, "name", 200),
        "aliases": parse_list_field(_field(form, "aliases", 4000)),
        "targets": parse_list_field(_field(form, "targets", 4000)),
        "indications": parse_list_field(_field(form, "indications", 4000)),
        "source_urls": [u for u in parse_list_field(_field(form, "source_urls", 8000)) if safe_url(u) != "#"][:20],
        "peers": _parse_peers_json(_field(form, "peers_json", 20000, "[]")),
        "notes": _field(form, "notes", 4000),
        "active": True,
    }
    if not data["name"]:
        return redirect("/watchlist/new")
    return redirect(f"/company/{create_company(data)}")


@app.post("/forms/company/{company_id}/delete")
def form_delete_company(company_id: int) -> RedirectResponse:
    delete_company(company_id)
    return redirect("/watchlist")


@app.post("/forms/manual-catalyst")
async def form_manual_catalyst(request: Request) -> RedirectResponse:
    form = await request.form()
    try:
        company_id = int(form.get("company_id") or 0)
    except ValueError:
        company_id = 0
    require("SELECT id FROM companies WHERE id = ?", company_id, "Company")
    payload = {
        "company_id": company_id,
        "title": _field(form, "title", 300, "Manual catalyst"),
        "url": _field(form, "url", 2000),
        "publisher": _field(form, "publisher", 120, "Manual"),
        "event_date": _field(form, "event_date", 20) or None,
        "event_type": _field(form, "event_type", 40) or None,
        "excerpt": _field(form, "excerpt", 20000),
    }
    catalyst_id = add_manual_catalyst(company_id, payload)
    return redirect(f"/report/{analyze_and_save(catalyst_id)}")


@app.post("/forms/demo")
def form_demo() -> RedirectResponse:
    report_ids = seed_demo_reports(auto_analyze=True).get("report_ids") or []
    return redirect(f"/report/{report_ids[0]}" if report_ids else "/")


@app.post("/forms/seed")
def form_seed() -> RedirectResponse:
    seed_examples()
    return redirect("/watchlist")


@app.post("/forms/monitor")
def form_monitor() -> RedirectResponse:
    run_monitor(auto_analyze=True, per_source_limit=4)
    return redirect("/scheduler")


@app.post("/forms/scheduler/tick")
def form_scheduler_tick() -> RedirectResponse:
    run_adaptive_monitor(auto_analyze=True, per_source_limit=4, due_only=True)
    return redirect("/scheduler")


@app.post("/forms/learning/check-outcomes")
def form_learning_check_outcomes() -> RedirectResponse:
    run_due_outcome_checks(limit=100, include_not_due=True)
    return redirect("/learning")


@app.post("/forms/learning/policy-update")
def form_learning_policy_update() -> RedirectResponse:
    run_policy_update()
    return redirect("/learning")


@app.post("/forms/catalysts/{catalyst_id}/verify-sources")
def form_verify_sources(catalyst_id: int) -> RedirectResponse:
    require("SELECT id FROM catalysts WHERE id = ?", catalyst_id, "Catalyst")
    verify_sources_for_catalyst(catalyst_id, fetch=True)
    report = row("SELECT id FROM reports WHERE catalyst_id = ?", (catalyst_id,))
    return redirect(f"/report/{report['id']}" if report else "/")


@app.post("/forms/catalysts/{catalyst_id}/analyze")
def form_analyze(catalyst_id: int) -> RedirectResponse:
    require("SELECT id FROM catalysts WHERE id = ?", catalyst_id, "Catalyst")
    return redirect(f"/report/{analyze_and_save(catalyst_id)}")


@app.post("/forms/reports/{report_id}/verify-evidence-graph")
def form_verify_evidence_graph(report_id: int) -> RedirectResponse:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    build_verified_evidence_graph(report_id)
    return redirect(f"/report/{report_id}")


@app.post("/forms/reports/{report_id}/send-alert")
def form_send_report_alert(report_id: int) -> RedirectResponse:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    send_report_alert(report_id, force=True)
    return redirect(f"/report/{report_id}")


@app.post("/forms/reports/{report_id}/review")
async def form_review_report(report_id: int, request: Request) -> RedirectResponse:
    require("SELECT id FROM reports WHERE id = ?", report_id, "Report")
    form = await request.form()
    status = _field(form, "status", 20, "needs_review")
    if status not in REVIEW_STATUSES:
        status = "needs_review"
    _save_review(report_id, status, _field(form, "reviewer", 120), _field(form, "notes", 1000))
    return redirect(f"/report/{report_id}")
