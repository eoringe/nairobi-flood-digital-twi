"""
tests.test_api
==============
The /api/v1 REST API (specification §7): authentication, permissions,
response envelope, area filtering, honest empty data for partner systems that
are not connected, report formats and the OpenAPI document.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from flask import Flask

from src.api import v1
from src.integrations import base, registry
from src.integrations.store import OpsStore, set_ops_store
from tests.test_ops_console import _Fake

NOW = datetime.now(timezone.utc)
TOKENS = "mon-token:monitoring_officer,analyst-token:data_analyst,pub-token:citizen,kibra-token:county_officer"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TWIN_API_TOKENS", TOKENS)
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    nc = {"as_of": NOW.isoformat(), "city_level": [0, 2], "acc72_mm": [20, 45], "hours": ["09:00", "10:00"],
          "flooded_pct": [0, 3], "headline": "Expect high flooding in Mabatini", "detail": "", "is_replay": False,
          "warnings": [{"level": "HIGH", "places": ["Mabatini", "Kibera"], "text": "High flooding expected",
                        "hour": 1}]}
    monkeypatch.setattr(registry, "SOURCES", {
        "prediction": _Fake("prediction", observed=NOW, records={
            "outlook": [nc], "chance": [{"p": 0.6, "call": "likely", "call_text": "Flooding expected"}]}),
        "iot": _Fake("iot", state=base.NOT_CONNECTED),
        "alerts": _Fake("alerts", state=base.NOT_CONNECTED),
        "outlook_alerts": _Fake("outlook_alerts", observed=NOW, records=[
            {"id": "outlook-1", "title": "High flooding expected", "severity": "high", "status": "model_warning",
             "areas": ["Mabatini", "Kibera"]}]),
        "community": _Fake("community", observed=NOW),
        "forecast": _Fake("forecast", observed=NOW, records=[{"time": NOW.isoformat(), "rain_mm": 1.0}]),
    })
    monkeypatch.setattr(registry, "status_rows", lambda: [])
    app = Flask("t")
    app.secret_key = "x"
    v1.init_api(app)
    yield app.test_client()
    set_ops_store(None)


def _get(client, path, token="mon-token", **params):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get(path, headers=headers, query_string=params)


def test_requests_need_a_session_or_a_valid_token(client):
    assert _get(client, "/api/v1/dashboard/summary", token=None).status_code == 401
    assert _get(client, "/api/v1/dashboard/summary", token="wrong").status_code == 401
    r = _get(client, "/api/v1/dashboard/summary", token="pub-token")
    assert r.status_code == 403 and r.get_json()["error"]["code"] == "forbidden"


def test_the_envelope_carries_data_meta_and_sources(client):
    body = _get(client, "/api/v1/dashboard/summary").get_json()
    assert body["data"]["flood_chance"]["p"] == 0.6 and body["data"]["worst_level_12h"] == 2
    assert body["data"]["stations"] is None, "no invented station counts"
    states = {s["source"]: s["state"] for s in body["sources"]}
    assert states["iot"] == "not_connected" and body["data"]["alerts_are_published"] is False
    assert "generated_at" in body["meta"]


def test_a_partner_system_that_is_not_connected_returns_empty_data_not_an_error(client):
    r = _get(client, "/api/v1/visualization/stations")
    body = r.get_json()
    assert r.status_code == 200 and body["data"] == [] and body["sources"][0]["state"] == "not_connected"


def test_permissions_follow_the_role(client):
    assert _get(client, "/api/v1/system/integration-status", token="analyst-token").status_code == 403
    assert _get(client, "/api/v1/system/integration-status").status_code == 200
    assert _get(client, "/api/v1/visualization/alerts/outlook-1", token="analyst-token").status_code == 403


def test_area_filter_and_county_officer_restriction(client):
    from src.geo import boundaries
    mathare = boundaries.ids_by_name(["Mathare"])[0]
    alerts = _get(client, "/api/v1/visualization/alerts", area=mathare).get_json()
    assert alerts["data"][0]["areas"] == ["Mabatini"] and alerts["meta"]["published"] is False
    # A county officer token with no assigned areas sees no area data at all.
    r = _get(client, "/api/v1/visualization/alerts", token="kibra-token").get_json()
    assert r["data"] == []


def test_pagination_and_bad_requests(client):
    r = _get(client, "/api/v1/visualization/boundaries", level="ward").get_json()
    assert r["data"]["type"] == "FeatureCollection" and len(r["data"]["features"]) == 71
    assert _get(client, "/api/v1/visualization/boundaries", level="planet").status_code == 400
    assert _get(client, "/api/v1/visualization/readings").status_code == 400, "station_id is required"
    assert _get(client, "/api/v1/visualization/predictions/999").status_code == 404


def test_reports_in_three_formats(client, monkeypatch):
    from src.reporting import builder
    monkeypatch.setattr(builder, "BUILDERS", {**builder.BUILDERS, "rainfall": lambda **kw: builder.Report(
        "rainfall", "Rainfall report", NOW, {"area": "All of Nairobi"},
        [builder.Section("Daily", ["Date", "mm"], [["2024-04-24", "40.0"]])])})
    assert _get(client, "/api/v1/reports/rainfall").get_json()["data"]["sections"][0]["rows"] == [["2024-04-24", "40.0"]]
    csv_ = _get(client, "/api/v1/reports/rainfall", format="csv")
    assert csv_.mimetype == "text/csv" and b"2024-04-24" in csv_.data
    pdf = _get(client, "/api/v1/reports/rainfall", format="pdf")
    assert pdf.mimetype == "application/pdf" and pdf.data.startswith(b"%PDF")


def test_openapi_lists_every_specification_endpoint(client):
    spec = client.get("/api/v1/openapi.json").get_json()
    required = ["/dashboard/summary", "/dashboard/indicators", "/dashboard/recent-activities", "/visualization/map",
                "/visualization/layers", "/visualization/boundaries", "/visualization/locations",
                "/visualization/infrastructure", "/visualization/flood-zones", "/visualization/stations",
                "/visualization/stations/{station_id}", "/visualization/readings", "/visualization/rainfall",
                "/visualization/water-levels", "/visualization/sensor-health", "/visualization/predictions",
                "/visualization/predictions/{run_id}", "/visualization/predictions/{run_id}/map",
                "/visualization/predictions/history", "/visualization/predictions/comparison",
                "/visualization/alerts", "/visualization/alerts/{alert_id}", "/visualization/incidents",
                "/visualization/incidents/{incident_id}", "/reports/flood-summary", "/reports/rainfall",
                "/reports/water-levels", "/reports/flood-incidents", "/system/data-sources",
                "/system/integration-status"]
    missing = [p for p in required if "/api/v1" + p not in spec["paths"]]
    assert not missing, missing
    assert client.get("/api/docs").status_code == 200
