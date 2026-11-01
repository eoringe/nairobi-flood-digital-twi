"""
tests.test_ops_predictions
==========================
Prediction runs, prediction-versus-observation matching, and the alerts
screen (docs/capstone/TASKLIST.md Phase 5).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.integrations import base, registry
from src.integrations.store import OpsStore, set_ops_store
from tests.test_ops_console import _Fake, _text, _user

ISSUED = datetime(2026, 4, 24, 7, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = OpsStore(tmp_path / "ops.db")
    set_ops_store(s)
    yield s
    set_ops_store(None)


def test_a_stored_run_is_compared_with_observations_by_ward(store, monkeypatch):
    from src.dashboard import ops_predictions as op
    from src.geo import boundaries
    rid = store.save_prediction_run({"issued_at": ISSUED, "source": "live", "model_version": "v2", "horizon_h": 12,
                                     "rain_72h_mm": 60, "chance": 0.8, "chance_call": "likely", "city_level": 3,
                                     "place_levels": {"Mabatini": "CRITICAL", "Kibera": "MODERATE"}, "headline": "x"})
    mabatini = boundaries.place_areas()["Mabatini"]["ward"]
    other = boundaries.ids_by_name(["Karen"])[0]
    monkeypatch.setattr(op, "observations", lambda issued, areas=None: [
        {"kind": "News report", "ward": mabatini, "when": "2026-04-24", "label": "Mathare"},
        {"kind": "Community report", "ward": other, "when": "2026-04-24", "label": "Karen: water"}])
    res = op.compare(f"run:{rid}")
    results = {boundaries.get(r["ward"])["name"]: r["result"] for r in res["rows"]}
    assert results["Mabatini"] == "Confirmed" and results["Karen"] == "Missed"
    assert list(results.values()).count("Not confirmed") == 1, "Kibera was predicted but not observed"
    assert (res["hits"], res["predicted"], res["observed"]) == (1, 2, 2)


def test_an_area_filter_limits_the_comparison(store, monkeypatch):
    from src.dashboard import ops_predictions as op
    from src.geo import boundaries
    rid = store.save_prediction_run({"issued_at": ISSUED, "source": "live", "model_version": "v2", "horizon_h": 12,
                                     "rain_72h_mm": 60, "chance": None, "chance_call": None, "city_level": 2,
                                     "place_levels": {"Mabatini": "HIGH", "Kibera": "HIGH"}, "headline": "x"})
    monkeypatch.setattr(op, "observations", lambda issued, areas=None: [])
    res = op.compare(f"run:{rid}", boundaries.ids_by_name(["Mathare"]))
    assert res["predicted"] == 1


def test_model_warnings_are_never_presented_as_published_alerts(store, monkeypatch):
    from src.dashboard import ops_predictions as op
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(registry, "SOURCES", {
        "alerts": _Fake("alerts", state=base.NOT_CONNECTED),
        "outlook_alerts": _Fake("outlook_alerts", observed=now, records=[
            {"id": "outlook-1", "title": "High flooding expected from 14:00", "severity": "high",
             "status": "model_warning", "areas": ["Mabatini"], "created_at": now.isoformat(), "published_at": None,
             "recommended_actions": ["Move valuables above floor level"], "acknowledged": None}]),
    })
    page = _text(op.alerts_page(_user("emergency_officer"), "dark"))
    assert "Automated Alert System not connected" in page and "not published alerts" in page
    assert "not published" in page and "available when the alert system is connected" in page
    assert "Move valuables above floor level" in page and "Full record" in page
    assert "Full record" not in _text(op.alert_cards(
        registry.SOURCES["outlook_alerts"].fetch("active").records, _user("data_analyst"), "all", "all")), \
        "the full alert record is for operations roles"


def test_new_pages_are_registered_in_order():
    from src.dashboard import ops_predictions  # noqa: F401
    from src.dashboard.ops_console import PAGES
    keys = list(PAGES)
    assert {"/ops/predictions", "/ops/compare", "/ops/alerts"} <= set(keys) and keys[-1] == "/ops/status"
