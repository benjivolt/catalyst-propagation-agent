from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from typing import Any

import httpx

from app.config import settings
from app.db import execute, row, rows
from app.services.utils import clean_text, redact_secrets


def _report_payload(report_id: int) -> dict[str, Any]:
    report = row(
        """
        SELECT r.*, c.title AS catalyst_title, c.event_type, co.name AS company_name, co.ticker
        FROM reports r
        JOIN catalysts c ON c.id = r.catalyst_id
        JOIN companies co ON co.id = c.company_id
        WHERE r.id = ?
        """,
        (report_id,),
    )
    if not report:
        raise ValueError(f"Report {report_id} not found")
    audit = row("SELECT * FROM analysis_audits WHERE report_id = ?", (report_id,)) or {}
    effects = rows(
        "SELECT layer, entity, direction, confidence, readthrough_type FROM effects "
        "WHERE report_id = ? ORDER BY layer, confidence DESC LIMIT 8",
        (report_id,),
    )
    return {
        "report_id": report_id,
        "company": f"{report['company_name']} ({report['ticker']})" if report.get("ticker") else report["company_name"],
        "event_type": report.get("event_type"),
        "catalyst_title": report.get("catalyst_title"),
        "overall_direction": report.get("overall_direction"),
        "overall_confidence": report.get("overall_confidence"),
        "evidence_grade": audit.get("evidence_grade", "not_audited"),
        "citation_coverage": audit.get("citation_coverage", 0),
        "summary": report.get("plain_language_summary"),
        "effects": effects,
    }


def _slack_escape(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _message_text(payload: dict[str, Any]) -> str:
    payload = {k: (_slack_escape(clean_text(v, 600)) if isinstance(v, str) else v) for k, v in payload.items()}
    confidence = round(float(payload.get("overall_confidence") or 0) * 100)
    lines = [
        f"Catalyst Propagation Alert: {payload.get('company')}",
        f"Event: {payload.get('catalyst_title')}",
        f"Direction/confidence: {payload.get('overall_direction')} / {confidence}%",
        f"Evidence grade: {payload.get('evidence_grade')} | Citation coverage: "
        f"{round(float(payload.get('citation_coverage') or 0) * 100)}%",
        f"Summary: {payload.get('summary')}",
        "Top effects:",
    ]
    for effect in payload.get("effects") or []:
        effect = {k: (_slack_escape(clean_text(v, 200)) if isinstance(v, str) else v) for k, v in effect.items()}
        lines.append(
            f"- {effect.get('layer')}: {effect.get('entity')} → {effect.get('direction')} "
            f"({round(float(effect.get('confidence') or 0) * 100)}%, {effect.get('readthrough_type')})"
        )
    return "\n".join(lines)


def _record(report_id: int, channel: str, target: str, status: str, payload: dict[str, Any], error: str = "") -> None:
    execute(
        """
        INSERT INTO alert_deliveries (report_id, channel, target, status, payload_json, error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            report_id,
            channel,
            target[:500],
            status,
            json.dumps(payload, ensure_ascii=False),
            clean_text(redact_secrets(error), 1000),
        ),
    )


def send_report_alert(report_id: int, *, force: bool = False) -> dict[str, Any]:
    payload = _report_payload(report_id)
    review = row("SELECT status FROM report_reviews WHERE report_id = ?", (report_id,))
    if not review or review.get("status") != "approved":
        _record(report_id, "none", "not_approved", "skipped", payload, "Report has not been approved by a reviewer")
        return {"status": "skipped", "reason": "not_approved", "deliveries": 0}
    if not force and not settings.alerts_enabled:
        _record(report_id, "none", "alerts_disabled", "skipped", payload, "ALERTS_ENABLED=false")
        return {"status": "skipped", "reason": "alerts_disabled", "deliveries": 0}

    deliveries = 0
    failures: list[str] = []
    text = _message_text(payload)

    if settings.slack_webhook_url:
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds) as client:
                resp = client.post(settings.slack_webhook_url, json={"text": text})
                resp.raise_for_status()
            _record(report_id, "slack", "slack_webhook", "sent", payload)
            deliveries += 1
        except Exception as exc:
            failures.append(f"slack: {redact_secrets(str(exc))}")
            _record(report_id, "slack", "slack_webhook", "failed", payload, str(exc))

    if settings.smtp_host and settings.alert_email_to:
        try:
            msg = EmailMessage()
            msg["Subject"] = f"Catalyst alert: {payload.get('company')} — {payload.get('overall_direction')}"
            msg["From"] = settings.alert_email_from or settings.smtp_username or "alerts@localhost"
            msg["To"] = settings.alert_email_to
            msg.set_content(text)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.request_timeout_seconds) as smtp:
                if settings.smtp_starttls:
                    smtp.starttls()
                if settings.smtp_username:
                    smtp.login(settings.smtp_username, settings.smtp_password)
                smtp.send_message(msg)
            _record(report_id, "email", settings.alert_email_to, "sent", payload)
            deliveries += 1
        except Exception as exc:
            failures.append(f"email: {exc}")
            _record(report_id, "email", settings.alert_email_to, "failed", payload, str(exc))

    if deliveries == 0 and not failures:
        _record(report_id, "none", "not_configured", "skipped", payload, "No Slack webhook or SMTP target configured")
        return {"status": "skipped", "reason": "not_configured", "deliveries": 0}

    return {"status": "sent" if deliveries else "failed", "deliveries": deliveries, "errors": failures}


def list_alert_deliveries(report_id: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if report_id is not None:
        return rows("SELECT * FROM alert_deliveries WHERE report_id = ? ORDER BY id DESC LIMIT ?", (report_id, limit))
    return rows("SELECT * FROM alert_deliveries ORDER BY id DESC LIMIT ?", (limit,))
