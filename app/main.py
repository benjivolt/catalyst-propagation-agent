from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import settings
from app.db import execute, init_db, row, rows
from app.services.scheduler import AdaptiveSchedulerDaemon, run_adaptive_monitor, scheduler_dashboard
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
from app.services.utils import parse_list_field
from app.services.peer_graph import sync_all_peer_relationships
from app.services.source_verifier import verify_sources_for_catalyst
from app.services.alerting import send_report_alert
from app.services.openai_health import openai_status, latest_openai_health
from app.services.demo_scenarios import seed_masterpiece_showcase, demo_readiness_summary
from app.services.outcome_learning import learning_dashboard, run_due_outcome_checks, run_policy_update
from app.services.verified_evidence_graph import build_verified_evidence_graph, evidence_graph_for_report

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.adaptive_scheduler = AdaptiveSchedulerDaemon()
    sync_all_peer_relationships()
    if settings.adaptive_scheduler_autostart and not settings.external_scheduler_worker:
        app.state.adaptive_scheduler.start()
    try:
        yield
    finally:
        app.state.adaptive_scheduler.stop()


app = FastAPI(title="Catalyst Propagation Agent", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))


class CompanyIn(BaseModel):
    ticker: str = ""
    name: str
    aliases: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)
    peers: list[dict[str, Any]] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    notes: str = ""
    active: bool = True


class ManualCatalystIn(BaseModel):
    company_id: int
    title: str
    url: str = ""
    publisher: str = "Manual"
    event_date: Optional[str] = None
    event_type: Optional[str] = None
    excerpt: str = ""
    auto_analyze: bool = True


class ReviewIn(BaseModel):
    status: str = Field(pattern="^(needs_review|approved|rejected)$")
    reviewer: str = ""
    notes: str = ""



@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    init_db()
    companies = get_companies(active_only=False)
    catalysts = list_catalysts(limit=40)
    runs = rows("SELECT * FROM monitor_runs ORDER BY id DESC LIMIT 6")
    scheduler = scheduler_dashboard()
    daemon = getattr(request.app.state, "adaptive_scheduler", None)
    scheduler["runtime_running"] = bool(daemon and daemon.is_running())
    scheduler["runtime_label"] = "Scheduler running" if scheduler["runtime_running"] else "Scheduler idle"
    if scheduler.get("external_worker"):
        scheduler["runtime_label"] = "External scheduler worker"
        scheduler["runtime_running"] = True
    stats = {
        "companies": len(companies),
        "active_companies": len([c for c in companies if c.get("active")]),
        "catalysts": len(rows("SELECT id FROM catalysts")),
        "reports": len(rows("SELECT id FROM reports")),
        "sources": len(rows("SELECT id FROM source_docs")),
        "adaptive_sources": scheduler.get("sources", 0),
        "due_sources": scheduler.get("due_count", 0),
        "ai_predictions": len(rows("SELECT id FROM ai_predictions")),
        "learning_errors": len(rows("SELECT id FROM ai_error_memory WHERE active = 1")),
    }
    openai = openai_status(run_live_test=False)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "companies": companies,
            "catalysts": catalysts,
            "runs": runs,
            "stats": stats,
            "scheduler": scheduler,
            "openai": openai,
            "readiness": demo_readiness_summary(),
        },
    )


@app.get("/company/{company_id}", response_class=HTMLResponse)
def company_page(request: Request, company_id: int) -> HTMLResponse:
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    catalysts = list_catalysts(limit=100, company_id=company_id)
    return templates.TemplateResponse(request, "company.html", {"company": company, "catalysts": catalysts})


@app.get("/report/{report_id}", response_class=HTMLResponse)
def report_page(request: Request, report_id: int) -> HTMLResponse:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return templates.TemplateResponse(request, "report.html", {"report": report})


@app.get("/learning", response_class=HTMLResponse)
def learning_page(request: Request) -> HTMLResponse:
    init_db()
    return templates.TemplateResponse(request, "learning.html", {"learning": learning_dashboard()})


@app.get("/api/openai/status")
def api_openai_status(test: bool = False) -> dict[str, Any]:
    return openai_status(run_live_test=test)


@app.post("/api/openai/test")
def api_openai_test() -> dict[str, Any]:
    return openai_status(run_live_test=True)


@app.post("/api/demo/masterpiece")
def api_demo_masterpiece() -> dict[str, Any]:
    return seed_masterpiece_showcase(auto_analyze=True)


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
    company_id = create_company(company.model_dump())
    return {"id": company_id, "status": "created"}


@app.put("/api/companies/{company_id}")
def api_update_company(company_id: int, company: CompanyIn) -> dict[str, str]:
    if not row("SELECT id FROM companies WHERE id = ?", (company_id,)):
        raise HTTPException(status_code=404, detail="Company not found")
    update_company(company_id, company.model_dump())
    return {"status": "updated"}


@app.delete("/api/companies/{company_id}")
def api_delete_company(company_id: int) -> dict[str, str]:
    delete_company(company_id)
    return {"status": "deleted"}


@app.post("/api/catalysts/manual")
def api_manual_catalyst(payload: ManualCatalystIn) -> dict[str, Any]:
    if not row("SELECT id FROM companies WHERE id = ?", (payload.company_id,)):
        raise HTTPException(status_code=404, detail="Company not found")
    catalyst_id = add_manual_catalyst(payload.company_id, payload.model_dump())
    report_id = None
    if payload.auto_analyze:
        report_id = analyze_and_save(catalyst_id)
    return {"status": "created", "catalyst_id": catalyst_id, "report_id": report_id}


@app.get("/api/catalysts")
def api_catalysts() -> list[dict[str, Any]]:
    return list_catalysts(limit=100)




@app.post("/api/catalysts/{catalyst_id}/verify-sources")
def api_verify_sources(catalyst_id: int, fetch: bool = True) -> dict[str, Any]:
    if not row("SELECT id FROM catalysts WHERE id = ?", (catalyst_id,)):
        raise HTTPException(status_code=404, detail="Catalyst not found")
    verified = verify_sources_for_catalyst(catalyst_id, fetch=fetch)
    return {"status": "verified", "catalyst_id": catalyst_id, "sources_checked": len(verified), "fetch": fetch}


@app.post("/api/catalysts/{catalyst_id}/analyze")
def api_analyze(catalyst_id: int) -> dict[str, Any]:
    if not row("SELECT id FROM catalysts WHERE id = ?", (catalyst_id,)):
        raise HTTPException(status_code=404, detail="Catalyst not found")
    report_id = analyze_and_save(catalyst_id)
    return {"status": "analyzed", "report_id": report_id}


@app.post("/api/monitor/run")
def api_run_monitor() -> dict[str, Any]:
    return run_monitor(auto_analyze=True, per_source_limit=4)


@app.get("/api/scheduler/status")
def api_scheduler_status() -> dict[str, Any]:
    return scheduler_dashboard()


@app.get("/api/learning/dashboard")
def api_learning_dashboard() -> dict[str, Any]:
    return learning_dashboard()


@app.post("/api/learning/check-outcomes")
def api_learning_check_outcomes(include_not_due: bool = False) -> dict[str, Any]:
    return run_due_outcome_checks(limit=100, include_not_due=include_not_due)


@app.post("/api/learning/policy-update")
def api_learning_policy_update() -> dict[str, Any]:
    return run_policy_update()


@app.post("/api/scheduler/tick")
def api_scheduler_tick() -> dict[str, Any]:
    return run_adaptive_monitor(auto_analyze=True, per_source_limit=4, due_only=True)


@app.get("/api/reports/{report_id}")
def api_report(report_id: int) -> dict[str, Any]:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/api/reports/{report_id}/evidence-graph")
def api_report_evidence_graph(report_id: int) -> dict[str, Any]:
    if not row("SELECT id FROM reports WHERE id = ?", (report_id,)):
        raise HTTPException(status_code=404, detail="Report not found")
    return evidence_graph_for_report(report_id)


@app.post("/api/reports/{report_id}/verify-evidence-graph")
def api_verify_evidence_graph(report_id: int) -> dict[str, Any]:
    if not row("SELECT id FROM reports WHERE id = ?", (report_id,)):
        raise HTTPException(status_code=404, detail="Report not found")
    return build_verified_evidence_graph(report_id)


@app.post("/api/peer-graph/sync")
def api_peer_graph_sync() -> dict[str, Any]:
    count = sync_all_peer_relationships()
    return {"status": "ok", "relationships": count}




@app.post("/api/reports/{report_id}/send-alert")
def api_send_report_alert(report_id: int) -> dict[str, Any]:
    if not row("SELECT id FROM reports WHERE id = ?", (report_id,)):
        raise HTTPException(status_code=404, detail="Report not found")
    return send_report_alert(report_id, force=True)


@app.post("/api/reports/{report_id}/review")
def api_review_report(report_id: int, payload: ReviewIn) -> dict[str, Any]:
    if not row("SELECT id FROM reports WHERE id = ?", (report_id,)):
        raise HTTPException(status_code=404, detail="Report not found")
    execute(
        """
        INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (report_id, payload.status, payload.reviewer[:120], payload.notes[:1000]),
    )
    return {"status": "updated", "report_id": report_id, "review_status": payload.status}


def _parse_peers_json(peers_text: str) -> list[dict[str, Any]]:
    try:
        peers = json.loads(peers_text) if peers_text.strip() else []
        return peers if isinstance(peers, list) else []
    except json.JSONDecodeError:
        return []


@app.post("/forms/openai/test")
def form_openai_test() -> RedirectResponse:
    openai_status(run_live_test=True)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/company")
async def form_company(request: Request) -> RedirectResponse:
    form = await request.form()
    data = {
        "ticker": str(form.get("ticker") or ""),
        "name": str(form.get("name") or ""),
        "aliases": parse_list_field(str(form.get("aliases") or "")),
        "targets": parse_list_field(str(form.get("targets") or "")),
        "indications": parse_list_field(str(form.get("indications") or "")),
        "source_urls": parse_list_field(str(form.get("source_urls") or "")),
        "peers": _parse_peers_json(str(form.get("peers_json") or "[]")),
        "notes": str(form.get("notes") or ""),
        "active": bool(form.get("active", "on")),
    }
    if data["name"]:
        create_company(data)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/company/{company_id}/delete")
def form_delete_company(company_id: int) -> RedirectResponse:
    delete_company(company_id)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/manual-catalyst")
async def form_manual_catalyst(request: Request) -> RedirectResponse:
    form = await request.form()
    payload = {
        "company_id": int(form.get("company_id") or 0),
        "title": str(form.get("title") or "Manual catalyst"),
        "url": str(form.get("url") or ""),
        "publisher": str(form.get("publisher") or "Manual"),
        "event_date": str(form.get("event_date") or "") or None,
        "event_type": str(form.get("event_type") or "") or None,
        "excerpt": str(form.get("excerpt") or ""),
    }
    catalyst_id = add_manual_catalyst(payload["company_id"], payload)
    report_id = analyze_and_save(catalyst_id)
    return RedirectResponse(f"/report/{report_id}", status_code=303)


@app.post("/forms/demo/masterpiece")
def form_demo_masterpiece() -> RedirectResponse:
    result = seed_masterpiece_showcase(auto_analyze=True)
    report_ids = result.get("report_ids") or []
    if report_ids:
        return RedirectResponse(f"/report/{report_ids[0]}", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/seed")
def form_seed() -> RedirectResponse:
    seed_examples()
    return RedirectResponse("/", status_code=303)


@app.post("/forms/monitor")
def form_monitor() -> RedirectResponse:
    run_monitor(auto_analyze=True, per_source_limit=4)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/scheduler/tick")
def form_scheduler_tick() -> RedirectResponse:
    run_adaptive_monitor(auto_analyze=True, per_source_limit=4, due_only=True)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/learning/check-outcomes")
def form_learning_check_outcomes() -> RedirectResponse:
    run_due_outcome_checks(limit=100, include_not_due=True)
    return RedirectResponse("/learning", status_code=303)


@app.post("/forms/learning/policy-update")
def form_learning_policy_update() -> RedirectResponse:
    run_policy_update()
    return RedirectResponse("/learning", status_code=303)




@app.post("/forms/catalysts/{catalyst_id}/verify-sources")
def form_verify_sources(catalyst_id: int) -> RedirectResponse:
    verify_sources_for_catalyst(catalyst_id, fetch=True)
    report = row("SELECT id FROM reports WHERE catalyst_id = ?", (catalyst_id,))
    if report:
        return RedirectResponse(f"/report/{report['id']}", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.post("/forms/catalysts/{catalyst_id}/analyze")
def form_analyze(catalyst_id: int) -> RedirectResponse:
    report_id = analyze_and_save(catalyst_id)
    return RedirectResponse(f"/report/{report_id}", status_code=303)




@app.post("/forms/reports/{report_id}/verify-evidence-graph")
def form_verify_evidence_graph(report_id: int) -> RedirectResponse:
    build_verified_evidence_graph(report_id)
    return RedirectResponse(f"/report/{report_id}", status_code=303)


@app.post("/forms/reports/{report_id}/send-alert")
def form_send_report_alert(report_id: int) -> RedirectResponse:
    send_report_alert(report_id, force=True)
    return RedirectResponse(f"/report/{report_id}", status_code=303)


@app.post("/forms/reports/{report_id}/review")
async def form_review_report(report_id: int, request: Request) -> RedirectResponse:
    form = await request.form()
    status = str(form.get("status") or "needs_review")
    if status not in {"needs_review", "approved", "rejected"}:
        status = "needs_review"
    execute(
        """
        INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (report_id, status, str(form.get("reviewer") or "")[:120], str(form.get("notes") or "")[:1000]),
    )
    return RedirectResponse(f"/report/{report_id}", status_code=303)


@app.post("/forms/peer-graph/sync")
def form_peer_graph_sync() -> RedirectResponse:
    sync_all_peer_relationships()
    return RedirectResponse("/", status_code=303)
