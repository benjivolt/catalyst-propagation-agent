from __future__ import annotations

import base64
import socket

import pytest
from fastapi import HTTPException

from app.main import redirect, safe_url
from app.services.sources import SourceFetchError, assert_public_url
from app.services.utils import redact_secrets


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.fda.gov/news", "https://www.fda.gov/news"),
        ("  http://example.com/a  ", "http://example.com/a"),
        ("javascript:alert(1)", "#"),
        ("JaVaScRiPt:alert(1)", "#"),
        ("data:text/html,<script>alert(1)</script>", "#"),
        ("file:///etc/passwd", "#"),
        ("//evil.example/path", "#"),
        ("", "#"),
        (None, "#"),
    ],
)
def test_safe_url(value, expected):
    assert safe_url(value) == expected


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example",
        "//evil.example",
        "///evil.example",
        "/\\evil.example",
        "/%2fevil.example",
        "/%5cevil.example",
        "/%0d%0aevil",
        "",
    ],
)
def test_redirect_refuses_external_or_ambiguous_targets(path):
    with pytest.raises(HTTPException) as error:
        redirect(path)
    assert error.value.status_code == 400


@pytest.mark.parametrize("path", ["/", "/report/1", "/watchlist?tab=active"])
def test_redirect_preserves_local_targets(path):
    response = redirect(path)
    assert response.status_code == 303
    assert response.headers["location"] == path


def _resolve_to(address: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]

    return fake_getaddrinfo


@pytest.mark.parametrize(
    ("url", "address"),
    [
        ("http://127.0.0.1/admin", "127.0.0.1"),
        ("http://localhost/", "127.0.0.1"),
        ("http://169.254.169.254/latest/meta-data/", "169.254.169.254"),
        ("http://internal.example/", "10.0.0.8"),
        ("http://[::1]/", "::1"),
    ],
)
def test_assert_public_url_blocks_internal_addresses(monkeypatch, url, address):
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_to(address))
    with pytest.raises(SourceFetchError):
        assert_public_url(url)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/", "https://example.com:8443/"]
)
def test_assert_public_url_blocks_bad_scheme_and_port(monkeypatch, url):
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    with pytest.raises(SourceFetchError):
        assert_public_url(url)


def test_assert_public_url_allows_public_host(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    assert_public_url("https://www.fda.gov/news-events")


def test_redact_secrets():
    text = (
        "GET https://api.scraperapi.com/?api_key=abc123SECRET&url=x failed; "
        "webhook https://hooks.slack.com/services/T000/B000/XXXX; key sk-proj-ABCDEFGHIJKLMNOP"
    )
    redacted = redact_secrets(text)
    assert "abc123SECRET" not in redacted
    assert "hooks.slack.com/services" not in redacted
    assert "sk-proj-ABCDEFGHIJKLMNOP" not in redacted
    assert "api_key=[redacted]" in redacted
    assert redact_secrets("nothing secret here") == "nothing secret here"


def test_cross_origin_post_is_blocked(client):
    response = client.post("/api/seed", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_cross_origin_referer_is_blocked(client):
    response = client.post("/forms/seed", headers={"Referer": "https://evil.example/page"}, follow_redirects=False)
    assert response.status_code == 403


def test_same_origin_post_is_allowed(client):
    response = client.post("/api/seed", headers={"Origin": "http://testserver"})
    assert response.status_code == 200


def test_cross_origin_get_is_allowed(client):
    assert client.get("/api/companies", headers={"Origin": "https://evil.example"}).status_code == 200


def test_untrusted_host_header_is_rejected(client):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def _basic(password: str) -> dict[str, str]:
    token = base64.b64encode(f"analyst:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_app_password_required(client, set_setting):
    set_setting(app_password="s3cret-test-pass")
    assert client.get("/").status_code == 401
    assert client.get("/", headers=_basic("wrong")).status_code == 401
    assert client.get("/", headers=_basic("s3cret-test-pass")).status_code == 200
    assert client.get("/healthz").status_code == 200


def test_app_password_rejects_malformed_header(client, set_setting):
    set_setting(app_password="s3cret-test-pass")
    assert client.get("/", headers={"Authorization": "Basic not-base64!!"}).status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer s3cret-test-pass"}).status_code == 401


def test_daily_budget_counts_todays_spend(db, set_setting):
    from app.services import scheduler
    from app.services.budget import record_spend

    set_setting(scheduler_daily_budget_usd=0.05)
    record_spend(0.04, "test")
    assert scheduler._daily_spend() >= 0.04
    assert not scheduler._budget_allows(0.02)
