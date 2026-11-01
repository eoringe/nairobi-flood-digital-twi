"""
tests.test_resilience
=====================
Acceptance criterion 22 and specification §10.2: the dashboard stays usable
when an integrated source is unavailable. Every source is taken down in turn,
and then all at once; every operations page and the main API endpoints must
still render, saying what is missing rather than failing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.integrations import base, registry
from src.integrations.store import OpsStore, set_ops_store
from tests.test_ops_console import _text, _user

KEYS = ["forecast", "prediction", "outlook_alerts", "community", "news_floods", "satellite", "infrastructure",
        "iot", "alerts", "infrastructure_system", "supply", "beneficiaries"]
NOW = datetime.now(timezone.utc)


class _Down(base.DataSource):
    def __init__(self, key):
        self.key, self.name, self.system, self.provides = key, key, f"{key} system", ("x",)

    def fetch(self, what, **params):
        return base.SourceResult(self.key, base.ERROR, message="connection refused")


class _Empty(base.DataSource):
    def __init__(self, key):
        self.key, self.name, self.system, self.provides = key, key, f"{key} system", ("x",)

    def fetch(self, what, **params):
        return base.SourceResult(self.key, base.OK, [], observed_at=NOW, freshness=base.LIVE)


def _pages():
    from src.dashboard import ops_assets, ops_console, ops_monitoring, ops_predictions, ops_reports  # noqa: F401
    return {p: v[5] for p, v in ops_console.PAGES.items() if p != "/ops/predictions"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    monkeypatch.setattr(registry, "check_unknown", lambda: None)
    yield monkeypatch
    set_ops_store(None)


@pytest.mark.parametrize("down", KEYS + ["all"])
def test_every_page_renders_with_a_source_down(isolated, down):
    srcs = {k: (_Down(k) if down in (k, "all") else _Empty(k)) for k in KEYS}
    isolated.setattr(registry, "SOURCES", srcs)
    user = _user("admin")
    for path, build in _pages().items():
        text = _text(build(user, "dark", {"areas": [], "min_risk": 0}))
        assert text, f"{path} rendered nothing with {down} down"


def test_predictions_page_says_when_the_model_is_unavailable(isolated):
    from src.dashboard import ops_predictions
    isolated.setattr(registry, "SOURCES", {k: _Down(k) for k in KEYS})
    text = _text(ops_predictions.predictions_page(_user("admin"), "dark"))
    assert "unavailable" in text.lower()


def test_the_api_answers_with_every_source_down(isolated):
    from flask import Flask
    from src.api import v1
    isolated.setattr(registry, "SOURCES", {k: _Down(k) for k in KEYS})
    isolated.setattr(registry, "status_rows", lambda: [])
    isolated.setenv("TWIN_API_TOKENS", "t:admin")
    app = Flask("r"); app.secret_key = "x"; v1.init_api(app)
    c = app.test_client()
    for path in ("/api/v1/dashboard/summary", "/api/v1/visualization/stations", "/api/v1/visualization/alerts",
                 "/api/v1/visualization/predictions", "/api/v1/visualization/rainfall"):
        r = c.get(path, headers={"Authorization": "Bearer t"})
        assert r.status_code == 200, path
        assert all(s["state"] == "error" for s in r.get_json()["sources"]), path
