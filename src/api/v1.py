"""
src.api.v1
==========
The dashboard REST API (specification §7), mounted at /api/v1 on the Flask
server, documented by OpenAPI at /api/v1/openapi.json and /api/docs.

AUTHENTICATION
--------------
A signed-in browser session, or an API token for machine clients (partner
systems, scripts): `Authorization: Bearer <token>`, with tokens configured as
TWIN_API_TOKENS="token1:monitoring_officer,token2:data_analyst". A token acts
with its role's permissions; tokens are compared in constant time. Every
endpoint checks a permission, so a role sees through the API exactly what it
sees on screen.

RESPONSES
---------
    {"data": ..., "meta": {"generated_at", "count", "limit", "offset", ...},
     "sources": [{"source", "state", "freshness", "observed_at", "demo", "message"}]}

List endpoints take `limit` (max 500) and `offset`. Errors are
{"error": {"code", "message"}} with the matching HTTP status. Where a partner
system is not connected, the endpoint answers 200 with empty data and a
source entry saying "not_connected", never invented values.
"""

from __future__ import annotations

import hmac
import os
from datetime import date, datetime, timedelta, timezone
from functools import wraps

from flask import Blueprint, Response, jsonify, request

from src.auth.access import can, current_user
from src.geo import boundaries
from src.integrations import base, registry
from src.integrations.store import get_ops_store

bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")
EAT = timezone(timedelta(hours=3))
MAX_LIMIT = 500
#: path -> OpenAPI operation, filled by the @endpoint decorator.
SPEC_PATHS: dict[str, dict] = {}


# ------------------------------------------------------------------ auth --
def _token_user() -> dict | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    given = header[7:].strip()
    for pair in filter(None, (p.strip() for p in os.environ.get("TWIN_API_TOKENS", "").split(","))):
        token, _, role = pair.partition(":")
        if token and hmac.compare_digest(token, given):
            return {"user_id": 0, "role": role.strip(), "active": True, "email": f"api-token:{role.strip()}",
                    "assigned_areas": []}
    return None


def api_user() -> dict | None:
    return _token_user() or current_user()


def _error(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def endpoint(path: str, permission: str, summary: str, params: list[tuple[str, str, str]] | None = None,
             tag: str = "visualization"):
    """Register a GET endpoint with its permission check and its OpenAPI entry."""
    def wrap(fn):
        @wraps(fn)
        def inner(**kw):
            user = api_user()
            if user is None:
                return _error(401, "unauthenticated", "Sign in, or send 'Authorization: Bearer <token>'.")
            if not can(user, permission):
                return _error(403, "forbidden", f"Your role does not include '{permission}'.")
            try:
                return fn(user, **kw)
            except KeyError as exc:
                return _error(404, "not_found", f"Not found: {exc}")
            except ValueError as exc:
                return _error(400, "bad_request", str(exc))
        bp.add_url_rule(path, view_func=inner, methods=["GET"], endpoint=fn.__name__)
        oa_path = "/api/v1" + path.replace("<", "{").replace(">", "}")
        parameters = [{"name": n, "in": "path", "required": True, "schema": {"type": "string"}}
                      for n in __import__("re").findall(r"<(\w+)>", path)]
        parameters += [{"name": n, "in": "query", "required": False, "schema": {"type": t}, "description": d}
                       for n, t, d in (params or [])]
        SPEC_PATHS[oa_path] = {"get": {"summary": summary, "tags": [tag], "parameters": parameters,
                                       "x-permission": permission, "security": [{"bearer": []}, {"session": []}],
                                       "responses": {"200": {"description": "OK"}, "401": {"description": "Not signed in"},
                                                     "403": {"description": "Role lacks the permission"}}}}
        return inner
    return wrap


# --------------------------------------------------------------- helpers --
def _src(r: base.SourceResult) -> dict:
    return {"source": r.source, "state": r.state, "freshness": r.freshness, "demo": r.demo, "message": r.message,
            "observed_at": r.observed_at.isoformat() if r.observed_at else None}


def _page(rows: list) -> tuple[list, dict]:
    limit = min(int(request.args.get("limit", 100)), MAX_LIMIT)
    offset = max(int(request.args.get("offset", 0)), 0)
    return rows[offset:offset + limit], {"count": len(rows), "limit": limit, "offset": offset}


def _ok(data, sources: list[base.SourceResult] | None = None, **meta):
    return jsonify({"data": data, "meta": {"generated_at": datetime.now(timezone.utc).isoformat(), **meta},
                    "sources": [_src(s) for s in sources or []]})


def _areas(user: dict) -> list[str]:
    """?area=<id> narrowed to a county officer's assigned areas."""
    from src.dashboard.ops_console import effective_filter
    flt = effective_filter(user, request.args.get("area"), 0)
    return flt["areas"]


AREA = ("area", "string", "Sub-county or ward id (see /visualization/boundaries)")
RANGE = [("from", "string", "Start date, YYYY-MM-DD"), ("to", "string", "End date, YYYY-MM-DD")]
PAGE = [("limit", "integer", f"Rows per page (max {MAX_LIMIT})"), ("offset", "integer", "Rows to skip")]


# -------------------------------------------------------------- dashboard --
@endpoint("/dashboard/summary", "view_ops", "Key monitoring indicators", [AREA], tag="dashboard")
def dashboard_summary(user):
    areas = _areas(user)
    st = registry.get("iot").fetch("stations")
    if st.ok and areas:
        st.records = [s for s in st.records if boundaries.point_in(s["lat"], s["lon"], areas)]
    chance = registry.get("prediction").fetch("chance")
    pred = registry.get("prediction").fetch("outlook")
    alerts = registry.get("alerts").fetch("active")
    alerts_used = alerts if alerts.ok else registry.get("outlook_alerts").fetch("active")
    comm = registry.get("community").fetch("active")
    nc = pred.records[0] if pred.ok and pred.records else {}
    alert_places = {p for a in alerts_used.records for p in a.get("areas") or [] if boundaries.place_in(p, areas)}
    data = {
        "stations": {"total": len(st.records), "active": sum(1 for s in st.records if s.get("status") == "active")}
        if st.ok else None,
        "flood_chance": chance.records[0] if chance.ok and chance.records else None,
        "worst_level_12h": max(nc.get("city_level") or [0]),
        "areas_under_alert": sorted(alert_places),
        "alerts_are_published": alerts.ok,
        "active_incidents": sum(1 for r in comm.records if boundaries.point_in(r["lat"], r["lon"], areas)),
    }
    return _ok(data, [st, chance, pred, alerts_used, comm])


@endpoint("/dashboard/indicators", "view_ops", "Indicator definitions and current values", [AREA], tag="dashboard")
def dashboard_indicators(user):
    resp = dashboard_summary.__wrapped__(user).get_json()
    d = resp["data"]
    rows = [
        {"key": "stations_active", "label": "Active monitoring stations",
         "value": d["stations"]["active"] if d["stations"] else None},
        {"key": "flood_chance_3d", "label": "Flood chance, next 3 days",
         "value": d["flood_chance"]["p"] if d["flood_chance"] else None, "unit": "probability"},
        {"key": "worst_level_12h", "label": "Worst predicted level, next 12 h", "value": d["worst_level_12h"],
         "unit": "0 Low · 1 Moderate · 2 High · 3 Very High"},
        {"key": "areas_under_alert", "label": "Places under alert", "value": len(d["areas_under_alert"])},
        {"key": "active_incidents", "label": "Active community reports", "value": d["active_incidents"]},
    ]
    return jsonify({**resp, "data": rows})


@endpoint("/dashboard/recent-activities", "view_ops", "Recent prediction runs, reports and source failures",
          PAGE, tag="dashboard")
def dashboard_recent(user):
    store = get_ops_store()
    items = [{"type": "prediction_run", "at": r["issued_at"], "text": r["headline"]} for r in store.prediction_runs(limit=20)]
    comm = registry.get("community").fetch("history", start=(date.today() - timedelta(days=3)).isoformat())
    items += [{"type": "community_report", "at": r["created_at"], "text": f"{r['place']}: {r['kind']} ({r['status']})"}
              for r in comm.records[:20]] if comm.ok else []
    items += [{"type": "source_failure", "at": f["at"], "text": f"{f['source_key']}: {f['error']}"}
              for f in store.failed_calls(limit=20)]
    items.sort(key=lambda i: str(i["at"]), reverse=True)
    rows, meta = _page(items)
    return _ok(rows, [comm], **meta)


# ------------------------------------------------------------- geospatial --
@endpoint("/visualization/map", "view_ops", "Map configuration: centre, zoom limits and layers")
def vis_map(user):
    from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST
    return _ok({"center": [-1.2864, 36.8172], "bounds": [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH],
                "min_zoom": 10.3, "max_zoom": 20, "crs": "EPSG:4326", "layers": get_ops_store().layers()})


@endpoint("/visualization/layers", "view_ops", "Map layer catalogue")
def vis_layers(user):
    return _ok(get_ops_store().layers())


@endpoint("/visualization/boundaries", "view_ops", "Administrative boundaries as GeoJSON",
          [("level", "string", "county, sub_county or ward")])
def vis_boundaries(user):
    from shapely.geometry import mapping
    level = request.args.get("level")
    if level and level not in boundaries.LEVELS:
        raise ValueError(f"level must be one of {boundaries.LEVELS}")
    feats = [{"type": "Feature", "id": a["id"], "geometry": mapping(a["geom"]),
              "properties": {"name": a["name"], "level": a["level"], "parent": a["parent"]}}
             for lvl in ([level] if level else boundaries.LEVELS) for a in boundaries.load()[lvl]]
    return _ok({"type": "FeatureCollection", "features": feats}, source=boundaries.attribution())


@endpoint("/visualization/locations", "view_ops", "Search places", [("q", "string", "Text to search")])
def vis_locations(user):
    from src.dashboard.callbacks import location_search
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        raise ValueError("q must be at least 2 characters")
    hits = location_search.search(q, online=False)[:20]          # gazetteer only: no third-party call per request
    return _ok(hits)


@endpoint("/visualization/infrastructure", "view_infrastructure", "Facilities with predicted risk",
          [AREA, ("category", "string", "school, hospital, health_centre, bridge, public_building, "
                                        "emergency_service"), ("min_level", "integer", "0-3")] + PAGE)
def vis_infrastructure(user):
    partner = registry.get("infrastructure_system").fetch("assets")
    res = partner if partner.ok else registry.get("infrastructure").fetch(
        "assets", area_ids=_areas(user) or None, categories=[request.args["category"]] if request.args.get("category")
        else None, min_level=int(request.args.get("min_level", 0)))
    rows = [{k: v for k, v in a.items() if k != "cell"} for a in res.records]
    rows, meta = _page(rows)
    return _ok(rows, [res], **meta)


@endpoint("/visualization/flood-zones", "view_ops", "Predicted flood zones for the current outlook (GeoJSON)")
def vis_flood_zones(user):
    from src.dashboard.components.map_3d import generate_flood_contour_geojson
    from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST
    from src.integrations.internal import live_peak_probability
    import numpy as np
    prob, issued = live_peak_probability()
    if prob is None:
        return _ok({"type": "FeatureCollection", "features": []})
    gj, _halo = generate_flood_contour_geojson(depth_grid=prob, value_is_probability=True,
                                               lats=np.linspace(LAT_NORTH, LAT_SOUTH, GRID_H),
                                               lons=np.linspace(LON_WEST, LON_EAST, GRID_W),
                                               display_mode="PROBABILITY", theme="dark")
    for f in gj.get("features", []):
        f["properties"] = {k: v for k, v in f.get("properties", {}).items() if k not in ("fillColor", "lineColor")}
        f["properties"]["kind"] = "model_output"
    return _ok(gj, issued_at=issued.isoformat() if issued else None,
               note="Model output for the worst hour of the 12-hour outlook, not observed flooding.")


# ----------------------------------------------------------------- sensors --
def _stations(user):
    r = registry.get("iot").fetch("stations")
    areas = _areas(user)
    if r.ok and areas:
        r.records = [s for s in r.records if boundaries.point_in(s["lat"], s["lon"], areas)]
    return r


@endpoint("/visualization/stations", "view_sensors", "Monitoring stations", [AREA] + PAGE, tag="sensors")
def vis_stations(user):
    r = _stations(user)
    rows, meta = _page(r.records)
    return _ok(rows, [r], **meta)


@endpoint("/visualization/stations/<station_id>", "view_sensors", "One station", tag="sensors")
def vis_station(user, station_id):
    r = _stations(user)
    st = next((s for s in r.records if s["id"] == station_id), None)
    if st is None and r.ok:
        raise KeyError(station_id)
    return _ok(st, [r])


def _readings(kind_default: str | None):
    sid = request.args.get("station_id")
    if not sid:
        raise ValueError("station_id is required")
    kind = request.args.get("kind") or kind_default
    end = request.args.get("to") or datetime.now(EAT).date().isoformat()
    start = request.args.get("from") or (date.fromisoformat(end[:10]) - timedelta(days=3)).isoformat()
    r = registry.get("iot").fetch("readings", station_id=sid, kind=kind, start=start, end=end + "T23:59:59")
    rows, meta = _page(r.records)
    return _ok(rows, [r], **meta)


@endpoint("/visualization/readings", "view_sensors", "Station readings (missing values are null)",
          [("station_id", "string", "Station id"), ("kind", "string", "rainfall or water_level")] + RANGE + PAGE,
          tag="sensors")
def vis_readings(user):
    return _readings(None)


@endpoint("/visualization/rainfall", "view_sensors", "Hourly rainfall from the forecast feed, or a station's "
          "rainfall with station_id", [("station_id", "string", "Optional rain gauge")] + RANGE + PAGE, tag="sensors")
def vis_rainfall(user):
    if request.args.get("station_id"):
        return _readings("rainfall")
    r = registry.get("forecast").fetch("hourly")
    rows, meta = _page(r.records)
    return _ok(rows, [r], **meta)


@endpoint("/visualization/water-levels", "view_sensors", "Latest water level per station", [AREA], tag="sensors")
def vis_water_levels(user):
    r = _stations(user)
    rows = [{"station_id": s["id"], "name": s["name"], "warning_m": s.get("warning_m"), "critical_m": s.get("critical_m"),
             **(s.get("latest") or {}).get("water_level", {})} for s in r.records if "water_level" in s.get("types", [])]
    return _ok(rows, [r])


@endpoint("/visualization/sensor-health", "view_sensors", "Sensor health and last communication", [AREA],
          tag="sensors")
def vis_sensor_health(user):
    r = _stations(user)
    return _ok([{"station_id": s["id"], "name": s["name"], "health": s.get("health"), "status": s.get("status"),
                 "last_communication": s.get("last_communication")} for s in r.records], [r])


# ------------------------------------------------------------- predictions --
@endpoint("/visualization/predictions", "view_ops", "Current prediction: outlook and flood chance", tag="predictions")
def vis_predictions(user):
    pred = registry.get("prediction").fetch("outlook")
    chance = registry.get("prediction").fetch("chance")
    nc = pred.records[0] if pred.ok and pred.records else None
    data = None if nc is None else {k: nc.get(k) for k in ("as_of", "hours", "acc72_mm", "city_level", "flooded_pct",
                                                           "warnings", "headline", "detail", "is_replay")}
    if data is not None:
        data["flood_chance"] = chance.records[0] if chance.ok and chance.records else None
        data["kind"] = "model_output"
    return _ok(data, [pred, chance])


@endpoint("/visualization/predictions/history", "view_ops", "Stored prediction runs", RANGE + PAGE, tag="predictions")
def vis_prediction_history(user):
    runs = get_ops_store().prediction_runs(limit=5000, since=request.args.get("from"), until=request.args.get("to"))
    rows, meta = _page(runs)
    return _ok(rows, **meta)


@endpoint("/visualization/predictions/comparison", "view_ops", "A run against observations, by ward",
          [("run", "string", "run:<id> or replay:<source>"), AREA], tag="predictions")
def vis_prediction_comparison(user):
    from src.dashboard.ops_predictions import compare
    key = request.args.get("run")
    if not key:
        raise ValueError("run is required, e.g. run:12 or replay:replay-2024-04-24")
    res = compare(key, _areas(user) or None)
    if res is None:
        raise KeyError(key)
    run = res.pop("run")
    res["run"] = {"label": run["label"], "issued_at": run["issued"].isoformat(), "replay": run["replay"]}
    for r in res["rows"]:
        r["ward_name"] = boundaries.label(r["ward"])
    return _ok(res)


@endpoint("/visualization/predictions/<run_id>", "view_ops", "One stored prediction run", tag="predictions")
def vis_prediction(user, run_id):
    r = get_ops_store().prediction_run(int(run_id))
    if r is None:
        raise KeyError(run_id)
    return _ok(r)


@endpoint("/visualization/predictions/<run_id>/map", "view_ops", "A run's predicted places as GeoJSON points",
          tag="predictions")
def vis_prediction_map(user, run_id):
    import json
    from src.routing.flood_router import PLACES_FILE
    r = get_ops_store().prediction_run(int(run_id))
    if r is None:
        raise KeyError(run_id)
    places = {p["name"]: p for p in json.loads(open(PLACES_FILE, encoding="utf-8").read())["places"]}
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [places[n]["lon"], places[n]["lat"]]},
              "properties": {"place": n, "level": lvl, "kind": "model_output"}}
             for n, lvl in r["place_levels"].items() if n in places]
    return _ok({"type": "FeatureCollection", "features": feats}, issued_at=r["issued_at"])


# --------------------------------------------------------- alerts, incidents --
def _alert_source():
    partner = registry.get("alerts").fetch("active")
    return partner if partner.ok else registry.get("outlook_alerts").fetch("active")


@endpoint("/visualization/alerts", "view_ops", "Active alerts (model warnings while the alert system is not "
          "connected)", [AREA, ("severity", "string", "moderate, high, critical")] + PAGE, tag="alerts")
def vis_alerts(user):
    res, areas = _alert_source(), _areas(user)
    rows = [{**a, "areas": [p for p in a.get("areas") or [] if boundaries.place_in(p, areas)]} for a in res.records]
    rows = [a for a in rows if a["areas"] and (not request.args.get("severity")
                                               or (a.get("severity") or "").lower() == request.args["severity"])]
    rows, meta = _page(rows)
    return _ok(rows, [res], **meta, published=res.source == "alerts")


@endpoint("/visualization/alerts/<alert_id>", "view_alert_record", "Full alert record", tag="alerts")
def vis_alert(user, alert_id):
    partner = registry.get("alerts")
    if partner.configured() or partner.demo():
        r = partner.fetch("alert", alert_id=alert_id)
        if r.ok and r.records:
            return _ok(r.records[0], [r])
    res = registry.get("outlook_alerts").fetch("active")
    rec = next((a for a in res.records if a["id"] == alert_id), None)
    if rec is None:
        raise KeyError(alert_id)
    return _ok(rec, [res])


@endpoint("/visualization/incidents", "view_ops", "Community and news-reported incidents",
          [AREA, ("type", "string", "water, impassable, homes, news")] + RANGE + PAGE, tag="alerts")
def vis_incidents(user):
    from src.dashboard.ops_assets import incidents
    end = date.fromisoformat(request.args.get("to", datetime.now(EAT).date().isoformat())[:10])
    start = date.fromisoformat(request.args.get("from", (end - timedelta(days=30)).isoformat())[:10])
    rows, res = incidents(start, end, _areas(user) or None, [request.args["type"]] if request.args.get("type") else None)
    for i, r in enumerate(rows):
        r["id"] = f"{'c' if r['source'].startswith('Community') else 'n'}-{r['when']}-{i}"
    rows, meta = _page(rows)
    return _ok(rows, list(res.values()), **meta)


@endpoint("/visualization/incidents/<incident_id>", "view_ops", "One incident: a community report id", tag="alerts")
def vis_incident(user, incident_id):
    from src.reports.store import get_reports
    r = get_reports().get(int(incident_id))
    if r is None or r["status"] == "removed":
        raise KeyError(incident_id)
    r = {k: v for k, v in r.items() if k not in ("user_id", "verified_by")}
    return _ok(r)


# ----------------------------------------------------------------- reports --
def _report(key: str):
    from src.reporting import builder
    rep = builder.build(key, _areas(api_user()) or None, request.args.get("from"), request.args.get("to"))
    fmt = request.args.get("format", "json")
    user = api_user()
    get_ops_store().record_report(key, user.get("user_id", 0), fmt, rep.filters, rep.row_count)
    if fmt == "csv":
        return Response(builder.to_csv(rep), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={key}.csv"})
    if fmt == "pdf":
        return Response(builder.to_pdf(rep), mimetype="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename={key}.pdf"})
    return _ok({"title": rep.title, "filters": rep.filters, "demo": rep.demo,
                "generated_at": rep.generated_at.isoformat(),
                "sections": [{"heading": s.heading, "columns": s.columns, "rows": s.rows, "note": s.note,
                              "source": s.source} for s in rep.sections]})


FMT = ("format", "string", "json (default), csv or pdf")


@endpoint("/reports/flood-summary", "generate_reports", "Flood monitoring summary", [AREA, FMT] + RANGE, tag="reports")
def rep_flood_summary(user):
    return _report("flood_summary")


@endpoint("/reports/rainfall", "generate_reports", "Rainfall report", [FMT] + RANGE, tag="reports")
def rep_rainfall(user):
    return _report("rainfall")


@endpoint("/reports/water-levels", "generate_reports", "Station and water-level report", [AREA, FMT], tag="reports")
def rep_water_levels(user):
    return _report("station_report")


@endpoint("/reports/flood-incidents", "generate_reports", "Alert and incident summary", [AREA, FMT] + RANGE,
          tag="reports")
def rep_incidents(user):
    return _report("alerts_incidents")


# ------------------------------------------------------------------ system --
@endpoint("/system/data-sources", "view_ops", "Registered data sources", tag="system")
def sys_sources(user):
    return _ok([s.info() for s in registry.SOURCES.values()])


@endpoint("/system/integration-status", "view_system_status", "Each source's state, freshness and failures",
          tag="system")
def sys_status(user):
    return _ok(registry.status_rows(), failed_requests=get_ops_store().failed_calls(limit=20))


# ----------------------------------------------------------------- OpenAPI --
def openapi() -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Nairobi Flood Visualization Dashboard API", "version": "1.0.0",
                 "description": "Read API of the Flood Visualization Dashboard (Integrated Flood Monitoring System). "
                                "Every response lists its sources with state and freshness; partner systems that are "
                                "not connected return empty data with state 'not_connected'."},
        "servers": [{"url": "/"}],
        "components": {"securitySchemes": {
            "bearer": {"type": "http", "scheme": "bearer", "description": "Token from TWIN_API_TOKENS"},
            "session": {"type": "apiKey", "in": "cookie", "name": "session", "description": "Signed-in browser"}}},
        "paths": dict(sorted(SPEC_PATHS.items())),
    }


@bp.route("/openapi.json")
def openapi_json():
    return jsonify(openapi())


DOCS_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flood Dashboard API</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css"></head>
<body style="margin:0"><div id="ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url: "/api/v1/openapi.json", dom_id: "#ui", withCredentials: true});</script>
</body></html>"""


def init_api(server) -> None:
    server.register_blueprint(bp)

    @server.route("/api/docs")
    def api_docs():
        return Response(DOCS_HTML, mimetype="text/html")
