from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

os.environ.update(
    {
        "DATABASE_PATH": str(Path(tempfile.mkdtemp(prefix="catalyst-tests-")) / "import.db"),
        "OPENAI_API_KEY": "",
        "OPENAPI_API_KEY": "",
        "SCRAPERAPI_API_KEY": "",
        "SOURCE_VERIFICATION_FETCH_ON_ANALYZE": "false",
        "ADAPTIVE_SCHEDULER_AUTOSTART": "false",
        "ALERTS_ENABLED": "false",
        "SLACK_WEBHOOK_URL": "",
        "SMTP_HOST": "",
        "APP_PASSWORD": "",
        "ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
    }
)


def _no_network(*args, **kwargs):
    raise OSError("network access is disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)


@pytest.fixture
def set_setting():
    from app.config import settings

    saved: dict[str, object] = {}

    def _set(**values: object) -> None:
        for name, value in values.items():
            saved.setdefault(name, getattr(settings, name))
            object.__setattr__(settings, name, value)

    yield _set
    for name, value in saved.items():
        object.__setattr__(settings, name, value)


@pytest.fixture
def db(tmp_path, set_setting):
    from app.db import init_db

    set_setting(database_path=str(tmp_path / "test.db"))
    init_db()
    return tmp_path / "test.db"


@pytest.fixture
def client(db):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def demo_template(tmp_path_factory):
    from app.config import settings
    from app.db import init_db
    from app.services.demo_scenarios import seed_demo_reports

    path = tmp_path_factory.mktemp("demo") / "demo.db"
    original = settings.database_path
    object.__setattr__(settings, "database_path", str(path))
    try:
        init_db()
        result = seed_demo_reports(auto_analyze=True)
    finally:
        object.__setattr__(settings, "database_path", original)
    assert result["report_ids"], result
    return path, result


@pytest.fixture
def demo(client, db, demo_template):
    path, result = demo_template
    shutil.copyfile(path, db)
    return result
