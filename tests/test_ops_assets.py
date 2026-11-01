"""
tests.test_ops_assets
=====================
Infrastructure exposure, restricted supply and beneficiary panels, incidents
and period comparison (docs/capstone/TASKLIST.md Phase 6).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.integrations import base, registry
from src.integrations.store import OpsStore, set_ops_store
from tests.test_ops_console import _Fake, _text, _user

NOW = datetime.now(timezone.utc)


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    f = {
        "infrastructure_system": _Fake("infrastructure_system", state=base.NOT_CONNECTED),
        "infrastructure": _Fake("infrastructure", observed=NOW, records=[
            {"id": "a", "name": "Mathare North Health Centre", "category": "health_centre", "level": 2,
             "lat": -1.255, "lon": 36.86, "ward": None},
            {"id": "b", "name": "Karen School", "category": "school", "level": 0, "lat": -1.32, "lon": 36.71,
             "ward": None},
            {"id": "c", "name": "Kibera Bridge", "category": "bridge", "level": 0, "lat": -1.3125, "lon": 36.7880,
             "ward": None}]),
        "community": _Fake("community", observed=NOW, records=[
            {"lat": -1.3126, "lon": 36.7881, "kind": "impassable", "place": "Kibera", "created_at": NOW.isoformat(),
             "status": "active", "verified": False, "confirmations": 2}]),
        "supply": _Fake("supply", state=base.NOT_CONNECTED),
        "beneficiaries": _Fake("beneficiaries", observed=NOW, records=[
            {"area": "Mathare", "level": "sub_county", "beneficiaries": 12}]),
        "news_floods": _Fake("news_floods", observed=NOW, records=[
            {"id": "n1", "start": "2024-04-24", "end": "2024-04-25", "area_km2": 1.0, "name": "Mathare",
             "lat": -1.262, "lon": 36.858}]),
    }
    monkeypatch.setattr(registry, "SOURCES", f)
    yield f
    set_ops_store(None)


def test_exposed_facilities_include_predicted_and_reported_flooding(fakes):
    from src.dashboard import ops_assets
    page = _text(ops_assets.infrastructure_page(_user("infrastructure_officer"), "dark"))
    assert "Mathare North Health Centre" in page, "in a predicted flood area"
    assert "Kibera Bridge" in page and "Report impassable at Kibera" in page, "within 300 m of a report"
    assert "Karen School" not in page.split("Exposed facilities")[1]
    assert "OpenStreetMap (interim" in page


def test_supplies_and_beneficiaries_are_restricted(fakes):
    from src.dashboard import ops_assets
    officer = _text(ops_assets.infrastructure_page(_user("infrastructure_officer"), "dark"))
    assert "Vulnerable communities" not in officer and "Emergency supplies" not in officer
    emergency = _text(ops_assets.infrastructure_page(_user("emergency_officer"), "dark"))
    assert "Vulnerable communities" in emergency and "Aggregated counts only" in emergency
    assert "supply system not connected" in emergency, "a missing partner is named, not hidden"


def test_incidents_combine_sources_and_filter_by_type(fakes):
    from src.dashboard import ops_assets
    rows, _ = ops_assets.incidents(date(2024, 1, 1), date(2026, 12, 31), None)
    assert {r["source"] for r in rows} == {"News report", "Community report"}
    assert "user_id" not in str(rows), "reporters are never shown"
    rows, _ = ops_assets.incidents(date(2024, 1, 1), date(2026, 12, 31), None, kinds=["impassable"])
    assert [r["place"] for r in rows] == ["Kibera"]
    from src.geo import boundaries
    rows, _ = ops_assets.incidents(date(2024, 1, 1), date(2026, 12, 31), boundaries.ids_by_name(["Mathare"]))
    assert [r["source"] for r in rows] == ["News report"], "the area filter keeps Mathare only"
