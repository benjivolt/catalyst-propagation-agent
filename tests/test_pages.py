from __future__ import annotations

import pytest

PAGES = ["/", "/watchlist", "/watchlist/new", "/catalysts/new", "/scheduler", "/learning", "/company/1", "/report/1"]


@pytest.mark.parametrize("path", PAGES)
def test_page_renders(client, demo, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Content-Security-Policy" in response.headers


@pytest.mark.parametrize("path", ["/", "/watchlist", "/learning", "/scheduler"])
def test_page_renders_on_empty_database(client, path):
    assert client.get(path).status_code == 200


def test_unknown_report_is_404(client):
    assert client.get("/report/9999").status_code == 404


def test_catalyst_table_shows_event_date(client, demo):
    assert "2024-03-14" in client.get("/").text
