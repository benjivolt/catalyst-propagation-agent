from __future__ import annotations

import httpx
import pytest

from app import db
from app.services.alerting import send_report_alert


@pytest.fixture
def report_id(demo):
    return demo["report_ids"][0]


def _set_review(report_id: int, status: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO report_reviews (report_id, status, reviewer, notes) VALUES (?, ?, 'test', '')",
        (report_id, status),
    )


class ExplodingClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("alerting must not contact Slack for an unapproved report")


@pytest.mark.parametrize("status", ["needs_review", "rejected"])
def test_unapproved_report_is_not_sent(report_id, set_setting, monkeypatch, status):
    set_setting(alerts_enabled=True, slack_webhook_url="https://hooks.slack.com/services/T/B/X")
    monkeypatch.setattr(httpx, "Client", ExplodingClient)
    _set_review(report_id, status)

    result = send_report_alert(report_id, force=True)

    assert result == {"status": "skipped", "reason": "not_approved", "deliveries": 0}
    delivery = db.row("SELECT channel, status FROM alert_deliveries WHERE report_id = ? ORDER BY id DESC", (report_id,))
    assert delivery == {"channel": "none", "status": "skipped"}


def test_approved_report_respects_alerts_disabled(report_id, set_setting):
    set_setting(alerts_enabled=False)
    _set_review(report_id, "approved")
    assert send_report_alert(report_id)["reason"] == "alerts_disabled"


def test_approved_report_without_channels_is_skipped(report_id, set_setting):
    set_setting(alerts_enabled=True, slack_webhook_url="", smtp_host="")
    _set_review(report_id, "approved")
    assert send_report_alert(report_id)["reason"] == "not_configured"


def test_send_alert_route_refuses_unapproved_report(client, report_id):
    _set_review(report_id, "needs_review")
    assert client.post(f"/api/reports/{report_id}/send-alert").json()["reason"] == "not_approved"


@pytest.mark.parametrize("channel", ["slack", "email"])
def test_transport_exceptions_are_not_exposed(client, report_id, set_setting, monkeypatch, channel):
    from app.services import alerting

    class UnavailableTransport:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("smtp.private.invalid failed while reading /srv/private/settings.ini")

    _set_review(report_id, "approved")
    set_setting(
        alerts_enabled=True,
        slack_webhook_url="https://hooks.slack.com/services/T/B/X" if channel == "slack" else "",
        smtp_host="smtp.private.invalid" if channel == "email" else "",
        alert_email_to="reviewer@example.invalid" if channel == "email" else "",
    )
    monkeypatch.setattr(alerting.httpx, "Client", UnavailableTransport)
    monkeypatch.setattr(alerting.smtplib, "SMTP", UnavailableTransport)

    response = client.post(f"/api/reports/{report_id}/send-alert")
    assert response.json()["errors"] == [f"{channel}: delivery failed"]
    assert "smtp.private.invalid" not in response.text
    assert "/srv/private" not in response.text
    delivery = db.row("SELECT error FROM alert_deliveries WHERE report_id = ? ORDER BY id DESC", (report_id,))
    assert "smtp.private.invalid" not in delivery["error"]
    assert "/srv/private" not in delivery["error"]
