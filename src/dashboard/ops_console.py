"""
src.dashboard.ops_console
=========================
The operations console for staff (Flood Visualization Dashboard Specification
§6), at /ops. It shares the sidebar shell of the account and admin pages.

Every page has the same frame:
    refresh bar      last successful update, manual refresh, auto-refresh
                     interval saved per user (FR-016, FR-017)
    notices          data warnings: sources down, stale data, demo mode (FR-020)
    content          built by the page's function from the integration layer
                     (src.integrations), rebuilt on every refresh

Values are shown with their source, observation time and freshness. A source
that is not connected says so and names the system; nothing is filled in.
Permissions are checked again in the callback that builds the content.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.geo import boundaries
from src.integrations import base, registry
from src.integrations.store import get_ops_store

EAT = timezone(timedelta(hours=3))
REFRESH_CHOICES = [{"label": "Auto-refresh off", "value": 0}, {"label": "Every minute", "value": 60},
                   {"label": "Every 5 minutes", "value": 300}, {"label": "Every 15 minutes", "value": 900}]
DEFAULT_REFRESH_S = 300
FRESHNESS_LABELS = {base.LIVE: "Live", base.DELAYED: "Delayed", base.STALE: "Stale",
                    base.HISTORICAL: "Historical", base.UNAVAILABLE: "No data"}
LEVEL_RANK = {"CLEAR": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
#: The specification's risk classes (§2.7) for the model's bands.
RISK_CLASS = {"CLEAR": "Low", "MODERATE": "Moderate", "HIGH": "High", "CRITICAL": "Very High"}


def _icon(name: str, cls: str = "") -> html.I:
    return html.I(className=f"bi bi-{name} {cls}".strip())


def local_time(t) -> str:
    """'09:40, 1 Nov' in Nairobi time, or an em dash."""
    t = base.as_utc(t)
    return t.astimezone(EAT).strftime("%H:%M, %d %b").replace(" 0", " ") if t else "—"


def fresh_badge(freshness: str, observed=None) -> html.Span:
    title = f"Observed {local_time(observed)} (Nairobi time)" if observed else "No observation available"
    return html.Span(FRESHNESS_LABELS.get(freshness, freshness), className=f"ops-fresh is-{freshness}", title=title)


def demo_tag(result: base.SourceResult | None) -> html.Span | None:
    return html.Span("DEMO DATA", className="ops-demo-tag") if result is not None and result.demo else None


def tile(label: str, value, icon: str, tone: str, sub, result: base.SourceResult | None = None,
         source: str | None = None, href: str | None = None) -> html.Div:
    """A summary card with its source, observation time and freshness."""
    foot = []
    if result is not None:
        foot = [fresh_badge(result.freshness, result.observed_at),
                html.Span(source or registry.get(result.source).name, className="ops-tile-source"),
                demo_tag(result)]
    body = html.Div(className="dash-kpi ops-tile", children=[
        html.Div(_icon(icon), className=f"dash-kpi-icon tone-{tone}"),
        html.Div(className="dash-kpi-text", children=[
            html.Span(label, className="dash-kpi-label"),
            html.Span(value, className="dash-kpi-value"),
            html.Span(sub, className="dash-kpi-sub"),
            html.Div(foot, className="ops-tile-foot") if foot else None,
        ]),
    ])
    # Only link to pages that exist (later phases add them to PAGES).
    return dcc.Link(body, href=href, className="ops-tile-link") if href in PAGES else body


def not_connected(result: base.SourceResult) -> str:
    return "Not connected" if result.state == base.NOT_CONNECTED else "Unavailable"


def empty_state(result: base.SourceResult, what: str) -> html.Div:
    """Why there is nothing to show, in words a monitoring officer can act on."""
    src = registry.get(result.source)
    if result.state == base.NOT_CONNECTED:
        text = [html.Strong(f"{src.system} not connected. "),
                f"{what} will appear here once its API is available."]
    elif result.state == base.ERROR:
        text = [html.Strong(f"{src.name} unavailable. "), result.message]
    else:
        text = [f"No {what.lower()} at the moment."]
    return html.Div([_icon("plug" if result.state == base.NOT_CONNECTED else "info-circle", "me-2"), *text],
                    className="ops-empty")


def panel(title: str, *children, icon: str | None = None, actions=None, className: str = "") -> html.Section:
    from src.dashboard.admin_dashboard import _panel
    return _panel(title, *children, icon=icon, actions=actions, className=className)


# ------------------------------------------------------------- overview --
def _live_outlook():
    r = registry.get("prediction").fetch("outlook")
    return r, (r.records[0] if r.ok and r.records else None)


def overview(user: dict, theme: str, flt: dict | None = None) -> list:
    """Main flood dashboard (§2.1, §6.1, UC-FVD-001), for the filtered area and risk level."""
    flt = flt or {}
    areas, min_risk = flt.get("areas") or [], int(flt.get("min_risk") or 0)
    registry.check_unknown()
    pred, nc = _live_outlook()
    chance_r = registry.get("prediction").fetch("chance")
    stations = registry.get("iot").fetch("stations")
    forecast = registry.get("forecast").fetch("hourly")
    model_alerts = registry.get("outlook_alerts").fetch("active")
    partner_alerts = registry.get("alerts").fetch("active")
    community = registry.get("community").fetch("active")
    infra = registry.get("infrastructure_system").fetch("assets")
    if not infra.ok:                       # interim: OpenStreetMap facilities with the predicted risk
        infra = registry.get("infrastructure").fetch("assets", area_ids=areas or None, min_level=max(1, min_risk))
    if areas:
        stations.records = [s for s in stations.records if boundaries.point_in(s["lat"], s["lon"], areas)]
        community.records = [r for r in community.records if boundaries.point_in(r["lat"], r["lon"], areas)]
        for res in (model_alerts, partner_alerts):
            res.records = [{**a, "areas": [p for p in a.get("areas") or [] if boundaries.place_in(p, areas)]}
                           for a in res.records]
            res.records = [a for a in res.records if a["areas"]]

    # Stations and readings.
    if stations.ok:
        active = sum(1 for s in stations.records if s.get("status") == "active")
        st_tile = tile("Monitoring stations", len(stations.records), "broadcast-pin", "info",
                       f"{active} active · {len(stations.records) - active} inactive", stations, href="/ops/sensors")
        levels = [(s, s["latest"]["water_level"]) for s in stations.records
                  if "water_level" in s.get("latest", {}) and s["latest"]["water_level"].get("value") is not None]
        above = [s for s, l in levels if s.get("warning_m") and l["value"] >= s["warning_m"]]
        wl_tile = tile("Latest water levels", f"{len(levels)} reporting", "water", "warn" if above else "ok",
                       f"{len(above)} above warning level" if levels else "no current readings", stations,
                       href="/ops/water-levels")
    else:
        st_tile = tile("Monitoring stations", not_connected(stations), "broadcast-pin", "neutral",
                       registry.get("iot").system, stations)
        wl_tile = tile("Latest water levels", not_connected(stations), "water", "neutral",
                       "needs the IoT Sensor Data Platform", stations)

    # Rainfall: the forecast feed's last observed hour and the 72-hour total.
    if forecast.ok and forecast.records:
        past = [r for r in forecast.records if not r["forecast"]]
        last = past[-1] if past else forecast.records[0]
        rain_tile = tile("Rainfall, last hour", f"{last['rain_mm']:.1f} mm", "cloud-rain", "info",
                         f"{last['rain_72h_mm']:.0f} mm over 72 h (bias-corrected)", forecast,
                         source="Open-Meteo, Nairobi")
    else:
        rain_tile = tile("Rainfall, last hour", "Unavailable", "cloud-rain", "neutral", forecast.message[:60], forecast)

    # Prediction.
    if chance_r.ok and chance_r.records:
        c = chance_r.records[0]
        tone = {"likely": "danger", "possible": "warn", "unlikely": "ok"}[c["call"]]
        pred_tile = tile("Flood chance, next 3 days", f"{round(100 * c['p'])}%", "graph-up-arrow", tone,
                         f"{c['call_text']} (95% call)", chance_r, source="Flood chance model", href="/ops/predictions")
    else:
        pred_tile = tile("Flood chance, next 3 days", "Unavailable", "graph-up-arrow", "neutral",
                         chance_r.message[:60] or "needs 31 days of rainfall", chance_r)

    # Alerts: published ones when the alert system is connected, model warnings otherwise.
    alert_places = set()
    for a in (partner_alerts.records if partner_alerts.ok else []) + (model_alerts.records if model_alerts.ok else []):
        alert_places.update(a.get("areas") or [])
    using = partner_alerts if partner_alerts.ok else model_alerts
    alert_sub = (f"{len(partner_alerts.records)} published alerts" if partner_alerts.ok else
                 f"{len(model_alerts.records)} model warnings · alert system not connected")
    alert_tile = tile("Areas under alert", len(alert_places), "exclamation-triangle",
                      "danger" if alert_places else "ok", alert_sub, using, href="/ops/alerts")

    high = [(p, l) for p, l in _high_risk_places(nc) if boundaries.place_in(p, areas)
            and LEVEL_RANK[l] >= min_risk]
    risk_tile = tile("High-risk areas, next 12 h", len(high), "geo-alt", "danger" if high else "ok",
                     ", ".join(p for p, _l in high[:3]) or "none forecast", pred, source="12-hour outlook",
                     href="/ops/predictions")

    inc_tile = tile("Reported incidents", len(community.records) if community.ok else "Unavailable",
                    "megaphone", "warn" if community.ok and community.records else "ok",
                    "active community reports", community, href="/ops/incidents")
    if infra.ok:
        interim = infra.source == "infrastructure"
        infra_tile = tile("Exposed infrastructure", len(infra.records), "building-exclamation",
                          "warn" if infra.records else "ok",
                          "facilities in predicted flood areas, next 12 h" + (" (OpenStreetMap)" if interim else ""),
                          infra, href="/ops/infrastructure")
    else:
        infra_tile = tile("Exposed infrastructure", not_connected(infra), "building-exclamation", "neutral",
                          registry.get("infrastructure_system").system, infra)

    tiles = html.Div(className="dash-kpis cols-4 ops-tiles", children=[
        st_tile, rain_tile, wl_tile, pred_tile, alert_tile, risk_tile, inc_tile, infra_tile])
    return [
        tiles,
        html.Div(className="dash-grid cols-7-5", children=[
            panel("Predicted risk by area, next 12 hours", _risk_table(nc, areas, min_risk), icon="map",
                  actions=html.Span("Model output, not observed flooding", className="ops-model-label")),
            panel("Recent activity", _recent_activity(user), icon="clock-history"),
        ]),
        panel("Data sources", _source_strip(), icon="diagram-3",
              actions=dcc.Link("System status", href="/ops/status", className="dash-link")
              if can(user, "view_system_status") else None),
    ]


def _high_risk_places(nc: dict | None) -> list[tuple[str, str]]:
    """(place, peak level) at High or Critical in the 12-hour outlook, worst first."""
    if not nc:
        return []
    peak: dict[str, str] = {}
    for w in nc.get("warnings") or []:
        for p in w["places"]:
            if LEVEL_RANK[w["level"]] > LEVEL_RANK.get(peak.get(p, "CLEAR"), 0):
                peak[p] = w["level"]
    return sorted(((p, l) for p, l in peak.items() if LEVEL_RANK[l] >= 2), key=lambda x: -LEVEL_RANK[x[1]])


def _risk_table(nc: dict | None, areas: list[str] | None = None, min_risk: int = 0) -> html.Div:
    if not nc:
        return html.P("The outlook is not available.", className="dash-empty")
    rows = []
    for w in nc.get("warnings") or []:
        for p in w["places"]:
            if boundaries.place_in(p, areas) and LEVEL_RANK[w["level"]] >= min_risk:
                rows.append((p, w["level"], w["text"]))
    if not rows:
        where = " in the selected area" if areas else " anywhere"
        return html.P(f"No flooding expected{where} in the next 12 hours"
                      + (" at this risk level." if min_risk else "."), className="dash-empty")
    seen, uniq = set(), []
    for p, lvl, text in sorted(rows, key=lambda r: -LEVEL_RANK[r[1]]):
        if p not in seen:
            seen.add(p)
            uniq.append((p, lvl, text))
    return html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th("Area"), html.Th("Risk"), html.Th("Outlook")])),
        html.Tbody([html.Tr([
            html.Td(p, className="dash-cell-strong"),
            html.Td(html.Span(RISK_CLASS[lvl], className=f"ops-risk lvl-{LEVEL_RANK[lvl]}")),
            html.Td(text, className="dash-cell-muted"),
        ]) for p, lvl, text in uniq[:12]]),
    ]))


def _recent_activity(user: dict) -> html.Ul | html.P:
    store = get_ops_store()
    items = []
    for r in store.prediction_runs(limit=5):
        items.append((r["issued_at"], "graph-up", "kind-account",
                      f"Outlook issued: {r['headline'] or 'no headline'}"))
    community = registry.get("community").fetch("history", start=(datetime.now(timezone.utc) - timedelta(days=2))
                                                 .date().isoformat())
    for rep in (community.records if community.ok else [])[:5]:
        items.append((rep["created_at"], "megaphone", "kind-data",
                      f"Flooding reported at {rep['place']} ({rep['kind']}), {rep['status']}"))
    for f in store.failed_calls(limit=4):
        items.append((f["at"], "plug", "kind-failure",
                      f"{registry.get(f['source_key']).name if f['source_key'] in registry.SOURCES else f['source_key']}"
                      f" request failed"))
    if not items:
        return html.P("No recent activity.", className="dash-empty")
    items.sort(key=lambda i: base.as_utc(i[0]), reverse=True)
    return html.Ul(className="dash-feed", children=[
        html.Li(className="dash-feed-item", children=[
            html.Span(_icon(ic), className=f"dash-feed-icon {kind}"),
            html.Div([html.Div(text), html.Div(local_time(t), className="dash-feed-time")]),
        ]) for t, ic, kind, text in items[:8]])


STATE_LABELS = {"ok": "Connected", "down": "Down", "not_connected": "Not connected", "unknown": "Not checked yet"}
STATE_DOT = {"ok": "ok", "down": "down", "not_connected": "warn", "unknown": "warn"}


def _source_strip() -> html.Div:
    return html.Div(className="ops-sources", children=[
        html.Div(className="ops-source", title=r["description"], children=[
            html.Span(className=f"dash-dot state-{STATE_DOT[r['state']]}"),
            html.Span(r["name"], className="dash-cell-strong"),
            html.Span(STATE_LABELS[r["state"]] + (" · demo" if r["demo"] else ""), className="dash-cell-muted"),
        ]) for r in registry.status_rows()])


# ---------------------------------------------------------- system status --
def status_page(user: dict, theme: str, flt: dict | None = None) -> list:
    """Data source availability (§2.15, §6.12, UC-FVD-012)."""
    registry.check_unknown()
    rows = registry.status_rows()
    up = sum(r["state"] == "ok" for r in rows)
    down = [r for r in rows if r["state"] == "down"]
    waiting = [r for r in rows if r["state"] == "not_connected"]
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Sources connected", f"{up} of {len(rows)}", "diagram-3", "ok", "responding normally"),
        tile("Sources down", len(down), "x-octagon", "danger" if down else "ok",
             ", ".join(r["name"] for r in down) or "none"),
        tile("Awaiting partner APIs", len(waiting), "plug", "warn" if waiting else "ok",
             f"{len(waiting)} systems not connected yet"),
        tile("Failed requests, 24 h", sum(r["failures_24h"] for r in rows), "exclamation-octagon", "info",
             f"of {sum(r['calls_24h'] for r in rows)} requests"),
    ])
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Source", "Type", "State", "Data", "Last success",
                                                 "Last failure", "Failures 24 h", "Response")])),
        html.Tbody([html.Tr([
            html.Td([html.Div(r["name"], className="dash-cell-strong"),
                     html.Div(r["system"], className="dash-cell-muted")]),
            html.Td(r["kind"].title()),
            html.Td(html.Span([html.Span(className=f"dash-dot state-{STATE_DOT[r['state']]} me-2"),
                               STATE_LABELS[r["state"]] + (" (demo data)" if r["demo"] else "")],
                              className="ops-state")),
            html.Td(fresh_badge(r["freshness"], r["last_observed"])),
            html.Td(local_time(r["last_success"])),
            html.Td([local_time(r["last_failure"]),
                     html.Div(r["last_error"][:80], className="dash-cell-muted", title=r["last_error"])
                     if r["last_error"] and r["state"] == "down" else None]),
            html.Td(str(r["failures_24h"])),
            html.Td(f"{r['latency_ms']:.0f} ms" if r["latency_ms"] is not None else "—"),
        ]) for r in rows]),
    ]))
    failed = get_ops_store().failed_calls(limit=25)
    failed_view = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th("Time"), html.Th("Source"), html.Th("Error")])),
        html.Tbody([html.Tr([html.Td(local_time(f["at"])), html.Td(f["source_key"]),
                             html.Td(f["error"] or "", className="dash-cell-muted")]) for f in failed]),
    ])) if failed else html.P("No failed requests in the last 7 days.", className="dash-empty")
    partners = [r for r in rows if r["kind"] == "partner"]
    setup = html.Ul(className="ops-setup", children=[
        html.Li([html.Span(r["system"], className="dash-cell-strong"), " — ",
                 html.Code(registry.get(r["key"]).env_url), " and ", html.Code(registry.get(r["key"]).env_key),
                 html.Span(" (set)" if r["configured"] else " (not set)", className="dash-cell-muted")])
        for r in partners])
    layer_rows = get_ops_store().layers()
    catalogue = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th("Map layer"), html.Th("Type"), html.Th("What it shows"), html.Th("Source")])),
        html.Tbody([html.Tr([html.Td(l["name"], className="dash-cell-strong"), html.Td(l["category"]),
                             html.Td(l["description"], className="dash-cell-muted"),
                             html.Td(l["source"], className="dash-cell-muted")]) for l in layer_rows]),
    ])) if layer_rows else html.P("The layer catalogue is recorded when the app starts.", className="dash-empty")
    return [kpis, panel("Data sources", table, icon="hdd-network"),
            panel("Map layers", catalogue, icon="layers"),
            html.Div(className="dash-grid cols-7-5", children=[
                panel("Failed requests", failed_view, icon="exclamation-octagon"),
                panel("Connecting partner systems",
                      html.P("Each partner system is connected by setting its API address and key in the server's "
                             "environment. The contract each adapter expects is in docs/capstone/TASKLIST.md.",
                             className="dash-hint"), setup, icon="plug")])]


# ---------------------------------------------------------------- pages --
#: path -> (nav label, icon, title, subtitle, permission, builder, has area/risk filters)
PAGES: dict[str, tuple] = {
    "/ops": ("Overview", "speedometer2", "Flood monitoring overview",
             "Current conditions, predictions, alerts and incidents across Nairobi.", "view_ops", overview, True),
    "/ops/status": ("System status", "hdd-network", "System status",
                    "Every connected data source, its last update and failed requests.", "view_system_status",
                    status_page, False),
}


def nav_links(user: dict, path: str, link) -> list:
    """Sidebar entries for the pages this user may open."""
    out = [html.P("Operations", className="dash-nav-section")]
    for href, (label, icon, *_rest) in PAGES.items():
        if can(user, PAGES[href][4]):
            out.append(link(label, href, icon))
    return out if len(out) > 1 else []


def render_ops(user: dict | None, path: str, theme: str = "dark") -> html.Div:
    from src.dashboard.admin_dashboard import shell
    from src.dashboard.auth_pages import _denied
    if not can(user, "view_ops"):
        return _denied("The operations console is for monitoring and response staff. "
                       "Ask an administrator for a staff role.")
    path = path if path in PAGES else "/ops"
    label, icon, title, subtitle, permission, _builder, _filtered = PAGES[path]
    if not can(user, permission):
        return _denied(f"Your role does not include {label.lower()}.")
    prefs = get_ops_store().preferences(user["user_id"])
    every = prefs.get("refresh_s", DEFAULT_REFRESH_S)
    allowed = assigned_areas(user)
    area = prefs.get("area") if (not allowed or prefs.get("area") in allowed) else None
    filters = html.Div(className="ops-filter-bar", children=[
        _icon("funnel", "ops-filter-icon"),
        dcc.Dropdown(id="ops-area", options=area_options(user), value=area or (allowed[0] if allowed else None),
                     clearable=not allowed, placeholder="All of Nairobi", className="dash-select ops-filter-area"),
        dcc.Dropdown(id="ops-risk", options=RISK_CHOICES, value=prefs.get("min_risk", 0), clearable=False,
                     searchable=False, className="dash-select ops-filter-risk"),
        html.Button([_icon("x-circle", "me-1"), "Reset filters"], id="ops-reset", className="dash-btn-ghost btn",
                    n_clicks=0),
        html.Span("Limited to your assigned areas" if allowed else "", className="dash-cell-muted"),
    ], style=None if PAGES[path][6] else {"display": "none"})
    bar = html.Div(className="ops-refresh-bar", children=[
        html.Span([_icon("clock-history", "me-1"), "Last update ", html.Strong(id="ops-updated", children="…")],
                  className="ops-updated"),
        dcc.Dropdown(id="ops-every", options=REFRESH_CHOICES, value=every, clearable=False, searchable=False,
                     className="dash-select ops-every"),
        dbc.Button([_icon("arrow-clockwise", "me-1"), "Refresh"], id="ops-refresh", className="dash-btn-ghost",
                   color="dash"),
    ])
    return shell(path, user, title, subtitle,
                 filters,
                 html.Div(id="ops-notices"),
                 dcc.Store(id="ops-path", data=path),
                 dcc.Interval(id="ops-interval", interval=max(every, 60) * 1000, disabled=not every),
                 dcc.Loading(html.Div(id="ops-content", className="dash-stack"), type="dot",
                             color="var(--accent)", delay_show=400),
                 actions=bar)


RISK_CHOICES = [{"label": "All risk levels", "value": 0}, {"label": "Moderate and above", "value": 1},
                {"label": "High and above", "value": 2}, {"label": "Very High only", "value": 3}]


def assigned_areas(user: dict) -> list[str]:
    """A county/regional officer's area ids (spec §4); empty means no restriction."""
    if user.get("role") != "county_officer":
        return []
    return boundaries.ids_by_name(user.get("assigned_areas") or [])


def area_options(user: dict) -> list[dict]:
    allowed = assigned_areas(user)
    if allowed:
        return boundaries.options(only=allowed)
    return boundaries.options(("sub_county", "ward"))


def effective_filter(user: dict, area: str | None, min_risk) -> dict:
    """The filter actually applied: a county officer never sees beyond the assigned areas."""
    allowed = assigned_areas(user)
    if user.get("role") == "county_officer" and not allowed:
        return {"areas": ["__none__"], "min_risk": int(min_risk or 0), "unassigned": True}
    if allowed:
        areas = [area] if area in allowed else allowed
    else:
        areas = [area] if area else []
    return {"areas": areas, "min_risk": int(min_risk or 0)}


def notices() -> list:
    """Data warnings shown above every page (FR-020, §10.6)."""
    out = []
    rows = registry.status_rows()
    if registry.demo_mode():
        out.append(("info", "easel", "Demo mode: partner systems show synthetic DEMO data, not measurements."))
    for r in rows:
        if r["state"] == "down":
            out.append(("danger", "x-octagon", f"{r['name']} is not responding (last error: "
                                               f"{(r['last_error'] or '')[:90]}). Values from it may be out of date."))
        elif r["freshness"] == base.STALE and r["state"] == "ok":
            out.append(("warn", "hourglass-split", f"{r['name']}: newest data is from {local_time(r['last_observed'])},"
                                                   " older than expected. It is shown as stale."))
    return [html.Div([_icon(ic, "me-2"), text], className=f"dash-alert tone-{tone} mb-2") for tone, ic, text in out]


def filter_header(flt: dict, filtered_page: bool) -> list:
    """What the filter is doing, above the content."""
    if flt.get("unassigned"):
        return [html.Div([_icon("geo", "me-2"), "No areas are assigned to your account yet, so no area data is "
                          "shown. Ask an administrator to assign your sub-counties or wards."],
                         className="dash-alert tone-warn mb-2")]
    if flt["areas"] and filtered_page:
        names = ", ".join(boundaries.label(a) for a in flt["areas"][:3])
        more = f" and {len(flt['areas']) - 3} more" if len(flt["areas"]) > 3 else ""
        return [html.Div([_icon("funnel", "me-2"), "Showing ", html.Strong(names), more,
                          ". Rainfall and the flood chance are city-wide."], className="ops-filter-note")]
    return []


def register_ops_callbacks(app) -> None:
    @app.callback(Output("ops-content", "children"), Output("ops-updated", "children"),
                  Output("ops-notices", "children"),
                  Input("ops-interval", "n_intervals"), Input("ops-refresh", "n_clicks"),
                  Input("ops-area", "value"), Input("ops-risk", "value"),
                  State("ops-path", "data"), State("resolved-theme-store", "data"))
    def ops_content(_n, _clicks, area, min_risk, path, theme):
        user = current_user()
        page = PAGES.get(path or "/ops")
        if page is None or not can(user, "view_ops") or not can(user, page[4]):
            raise PreventUpdate
        flt = effective_filter(user, area, min_risk)
        if ctx.triggered_id in ("ops-area", "ops-risk"):
            get_ops_store().set_preferences(user["user_id"], {"area": area, "min_risk": int(min_risk or 0)})
        return (filter_header(flt, page[6]) + page[5](user, theme or "dark", flt),
                datetime.now(EAT).strftime("%H:%M:%S"), notices())

    @app.callback(Output("ops-area", "value"), Output("ops-risk", "value"),
                  Input("ops-reset", "n_clicks"), prevent_initial_call=True)
    def reset_filters(n):
        if not n:
            raise PreventUpdate
        allowed = assigned_areas(current_user() or {})
        return (allowed[0] if allowed else None), 0

    @app.callback(Output("ops-interval", "interval"), Output("ops-interval", "disabled"),
                  Input("ops-every", "value"), prevent_initial_call=True)
    def set_refresh(every):
        user = current_user()
        if not can(user, "view_ops"):
            raise PreventUpdate
        every = int(every or 0)
        get_ops_store().set_preferences(user["user_id"], {"refresh_s": every})
        return max(every, 60) * 1000, not every
