"""
tests.test_integrations
=======================
The integration layer (docs/capstone/TASKLIST.md Phase 1): freshness rules,
partner systems that are not connected yet, demo mode, HTTP retries, status
recording, and the operations store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.integrations import base, registry
from src.integrations.partners import AutomatedAlertSystem, BeneficiarySystem, IoTSensorPlatform
from src.integrations.store import OpsStore, set_ops_store

NOW = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def ops(tmp_path, monkeypatch):
    for var in ("IOT_API_URL", "ALERTS_API_URL", "BENEFICIARY_API_URL", "INTEGRATION_DEMO"):
        monkeypatch.delenv(var, raising=False)
    store = OpsStore(tmp_path / "ops.db")
    set_ops_store(store)
    yield store
    set_ops_store(None)


# -------------------------------------------------------------- freshness --
@pytest.mark.parametrize("age, expected", [
    (timedelta(minutes=10), base.LIVE), (timedelta(hours=2), base.DELAYED), (timedelta(days=1), base.STALE)])
def test_freshness_classes(age, expected):
    assert base.freshness(NOW - age, timedelta(hours=1), timedelta(hours=6), now=NOW) == expected


def test_no_observation_is_unavailable_and_naive_times_are_nairobi_local():
    assert base.freshness(None, timedelta(hours=1), timedelta(hours=6)) == base.UNAVAILABLE
    assert base.as_utc(datetime(2026, 11, 1, 12, 0)) == NOW


# ------------------------------------------------------- partner systems --
def test_an_unconfigured_partner_is_not_connected_and_makes_no_request(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("no request expected"))
    r = IoTSensorPlatform().fetch("stations")
    assert r.state == base.NOT_CONNECTED and r.records == []
    assert "IOT_API_URL" in r.message and "IoT Sensor Data Platform" in r.message


def test_demo_mode_is_labelled_on_every_record(monkeypatch):
    monkeypatch.setenv("INTEGRATION_DEMO", "1")
    r = IoTSensorPlatform().fetch("stations")
    assert r.ok and r.demo and r.records
    assert all(s["name"].startswith("DEMO") for s in r.records)
    assert {s["health"] for s in r.records} >= {"ok", "degraded", "offline"}, "every health state can be shown"


def test_demo_readings_keep_gaps_as_missing_not_zero(monkeypatch):
    monkeypatch.setenv("INTEGRATION_DEMO", "1")
    r = IoTSensorPlatform().fetch("readings", station_id="DEMO-WL-01", kind="water_level")
    missing = [x for x in r.records if x["quality"] == "missing"]
    assert missing and all(x["value"] is None for x in missing)


def test_a_real_url_overrides_demo_and_its_failures_are_recorded(monkeypatch, ops):
    import requests
    monkeypatch.setenv("INTEGRATION_DEMO", "1")
    monkeypatch.setenv("ALERTS_API_URL", "https://alerts.example.org/api")
    calls = []

    def down(url, **kw):
        calls.append(url)
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(requests, "get", down)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)

    src = AutomatedAlertSystem()
    r = src.fetch("active")
    assert r.state == base.ERROR and not r.demo
    assert len(calls) == src.retries + 1, "network errors are retried"
    st = ops.status("alerts")
    assert st["failures_24h"] == 1 and "connection refused" in st["last_error"]


def test_a_client_error_is_not_retried(monkeypatch):
    import requests

    class Resp:
        status_code, text = 404, "no such endpoint"
    calls = []
    monkeypatch.setattr(requests, "get", lambda url, **kw: calls.append(url) or Resp())
    monkeypatch.setenv("ALERTS_API_URL", "https://alerts.example.org/api")
    assert AutomatedAlertSystem().fetch("active").state == base.ERROR
    assert len(calls) == 1


def test_a_connected_partner_is_normalised(monkeypatch):
    import requests

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return [{"id": 7, "name": "Ruaraka gauge", "type": "water_level", "lat": "-1.24", "lon": "36.88",
                     "last_communication": "2026-11-01T11:50:00+03:00"}]
    monkeypatch.setattr(requests, "get", lambda url, **kw: Resp())
    monkeypatch.setenv("IOT_API_URL", "https://iot.example.org")
    r = IoTSensorPlatform().fetch("stations")
    assert r.ok and r.records[0]["id"] == "7" and r.records[0]["types"] == ["water_level"]
    assert r.records[0]["lat"] == -1.24


def test_beneficiary_data_is_reduced_to_aggregates(monkeypatch):
    import requests

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return [{"area": "Mathare", "level": "sub_county", "beneficiaries": 12, "names": ["A. Student"]}]
    monkeypatch.setattr(requests, "get", lambda url, **kw: Resp())
    monkeypatch.setenv("BENEFICIARY_API_URL", "https://b.example.org")
    rec = BeneficiarySystem().fetch("aggregate").records[0]
    assert "names" not in rec and rec["beneficiaries"] == 12


def test_unknown_request_is_an_error_not_an_exception():
    assert IoTSensorPlatform().fetch("weather").state == base.ERROR


# ------------------------------------------------------------- registry --
def test_archives_are_historical_never_live():
    from src.integrations.internal import NewsFloods
    assert NewsFloods().classify(datetime.now(timezone.utc)) == base.HISTORICAL


def test_status_rows_cover_every_source_and_flag_missing_partners():
    rows = {r["key"]: r for r in registry.status_rows()}
    assert {"forecast", "prediction", "community", "iot", "alerts", "supply", "beneficiaries"} <= set(rows)
    assert rows["iot"]["state"] == "not_connected" and rows["iot"]["kind"] == "partner"


# ---------------------------------------------------------- ops store --
def test_prediction_runs_are_stored_once_per_issue_time(ops):
    run = {"issued_at": datetime(2026, 11, 1, 9), "source": "live", "model_version": "v2", "horizon_h": 12,
           "rain_72h_mm": 31.0, "chance": 0.4, "chance_call": "possible", "city_level": 1,
           "place_levels": {"Mathare": "MODERATE"}, "headline": "Expect moderate flooding"}
    assert ops.save_prediction_run(run) is not None
    assert ops.save_prediction_run(run) is None
    runs = ops.prediction_runs()
    assert len(runs) == 1 and runs[0]["place_levels"] == {"Mathare": "MODERATE"}


def test_preferences_merge(ops):
    ops.set_preferences(5, {"refresh_s": 300})
    assert ops.set_preferences(5, {"layers": ["boundaries"]}) == {"refresh_s": 300, "layers": ["boundaries"]}


def test_report_history_hides_reporters(tmp_path):
    from src.reports.store import ReportStore
    store = ReportStore(tmp_path / "r.db")
    store.add(42, -1.2884, 36.8233, "water", "CBD")
    row = store.history()[0]
    assert "user_id" not in row and row["verified"] is False and row["status"] == "active"


def test_partner_infrastructure_is_normalised_with_ward_and_level(monkeypatch):
    import requests
    from src.integrations.partners import InfrastructureRiskSystem

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return [{"id": 91, "name": "Mathare North Health Centre", "category": "health_centre",
                     "lat": -1.2551, "lon": 36.8602, "risk_class": "High"}]
    monkeypatch.setattr(requests, "get", lambda url, **kw: Resp())
    monkeypatch.setenv("INFRA_API_URL", "https://infra.example.org")
    a = InfrastructureRiskSystem().fetch("assets").records[0]
    assert a["id"] == "91" and a["level"] == 2 and a["ward"] is not None
