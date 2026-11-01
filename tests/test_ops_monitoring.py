"""
tests.test_ops_monitoring
=========================
Sensor, rainfall and water-level screens (docs/capstone/TASKLIST.md Phase 4).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.integrations import base
from src.integrations.store import OpsStore, set_ops_store

T0 = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)


def _rows(values):
    return [{"observed_at": (T0 + timedelta(hours=i)).isoformat(), "value": v} for i, v in enumerate(values)]


@pytest.mark.parametrize("values, expected", [
    ([1.0, 1.0, 1.1, 1.3], "Rising"), ([1.5, 1.4, 1.3, 1.2], "Falling"), ([1.0, 1.01, 1.0, 1.02], "Stable"),
    ([1.0, None, None, 1.4], "Rising"), ([None, None, 1.0], "Not enough data")])
def test_water_level_trend_over_three_hours_ignores_gaps(values, expected):
    from src.dashboard.ops_monitoring import trend
    assert trend(_rows(values)) == expected


def test_unusual_days_are_above_the_months_95th_percentile():
    from src.dashboard.ops_monitoring import unusual_days
    daily = {date(2015 + y, 4, d): float(d % 10) for y in range(5) for d in range(1, 31)}
    daily[date(2024, 4, 20)] = 80.0
    out = unusual_days(daily, date(2024, 4, 1), date(2024, 4, 30))
    assert [(d, v) for d, v, _t in out] == [(date(2024, 4, 20), 80.0)]


def test_the_newest_observation_is_kept_whichever_call_brought_it(tmp_path):
    store = OpsStore(tmp_path / "ops.db")
    store.record_call("iot", ok=True, observed_at=T0)
    store.record_call("iot", ok=True, observed_at=T0 - timedelta(hours=9))     # an offline station's history
    assert store.status("iot")["last_observed"] == T0.isoformat(timespec="seconds")


def test_an_old_reading_is_shown_as_stale_not_as_a_current_level(tmp_path, monkeypatch):
    monkeypatch.setenv("INTEGRATION_DEMO", "1")
    monkeypatch.delenv("IOT_API_URL", raising=False)
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    try:
        from src.dashboard import ops_monitoring
        from tests.test_ops_console import _text
        page = _text(ops_monitoring.water_levels_page({"role": "hydrologist", "active": True, "user_id": 1}, "dark"))
        assert "DEMO Nairobi River, Kariobangi" in page and "Stale reading" in page and "Not current" in page
    finally:
        set_ops_store(None)


def test_sensor_screens_say_which_system_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("INTEGRATION_DEMO", raising=False)
    monkeypatch.delenv("IOT_API_URL", raising=False)
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    try:
        from src.dashboard import ops_monitoring
        from tests.test_ops_console import _text
        for page in (ops_monitoring.sensors_page, ops_monitoring.water_levels_page):
            text = _text(page({"role": "monitoring_officer", "active": True, "user_id": 1}, "dark"))
            assert "IoT Sensor Data Platform not connected" in text and "IOT_API_URL" in text
    finally:
        set_ops_store(None)


def test_monitoring_pages_are_in_the_console_before_system_status():
    from src.dashboard import ops_monitoring  # noqa: F401  registers the pages
    from src.dashboard.ops_console import PAGES
    keys = list(PAGES)
    assert keys.index("/ops/sensors") < keys.index("/ops/status") and keys[-1] == "/ops/status"
