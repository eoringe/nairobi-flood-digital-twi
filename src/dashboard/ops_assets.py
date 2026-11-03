"""
src.dashboard.ops_assets
========================
Infrastructure, incident and history screens of the operations console
(specification §2.9–§2.11, §6.8–§6.10; FR-010–FR-012; UC-FVD-008, UC-FVD-009).

    /ops/infrastructure  facilities in or near predicted and observed flooding,
                         by type and area; emergency supplies and aggregated
                         beneficiary figures for the roles allowed to see them
    /ops/incidents       community reports and news-reported floods with type,
                         status, date and area filters and their distribution
    /ops/history         rainfall, incidents and model warnings over a chosen
                         period, compared with another period

Infrastructure comes from the Infrastructure Risk Mapping System when it is
connected and from OpenStreetMap until then. Beneficiary data is aggregated
by area; no personal record is requested or shown (spec §2.9, §9.6).
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime, timedelta

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.dashboard.ops_console import (EAT, PAGES, RISK_CLASS, _icon, demo_tag, empty_state, local_time, panel, tile)
from src.geo import boundaries, layers
from src.integrations import base, registry
from src.integrations.store import get_ops_store

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}
LEVEL_KEY = {0: "CLEAR", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}
#: A facility this close to an active community report counts as near observed flooding.
NEAR_REPORT_M = 300
REPORT_KINDS = {"water": "Water on the road", "impassable": "Road impassable", "homes": "Homes flooded"}


def _metres(lat1, lon1, lat2, lon2) -> float:
    kx = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat2 - lat1) * 110574.0, (lon2 - lon1) * kx)


def _style(fig, theme, height=280):
    from src.dashboard.ops_monitoring import _style as s
    return s(fig, theme, height)


# --------------------------------------------------------- infrastructure --
def infrastructure_page(user: dict, theme: str, flt: dict | None = None) -> list:
    flt = flt or {}
    areas = flt.get("areas") or None
    partner = registry.get("infrastructure_system").fetch("assets")
    res = partner if partner.ok else registry.get("infrastructure").fetch("assets", area_ids=areas)
    if not res.ok:
        return [empty_state(res, "Infrastructure")]
    items = res.records
    reports = registry.get("community").fetch("active")
    near = {}
    for a in items:
        for r in reports.records if reports.ok else []:
            if _metres(a["lat"], a["lon"], r["lat"], r["lon"]) <= NEAR_REPORT_M:
                near[a["id"]] = r
                break
    min_level = max(1, int(flt.get("min_risk") or 0))
    exposed = [a for a in items if a.get("level", 0) >= min_level or a["id"] in near]
    by_cat = Counter(a["category"] for a in items)
    ex_cat = Counter(a["category"] for a in exposed)
    source = ("Infrastructure Risk Mapping System" if partner.ok else
              "OpenStreetMap (interim, until the Infrastructure Risk Mapping System connects)")
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Facilities", len(items), "buildings", "info", "in the selected area", res),
        tile("In predicted flood areas", sum(1 for a in items if a.get("level", 0) >= min_level),
             "exclamation-triangle", "warn" if exposed else "ok", "worst hour of the 12-hour outlook", res),
        tile("Near reported flooding", len(near), "megaphone", "danger" if near else "ok",
             f"within {NEAR_REPORT_M} m of an active community report", reports),
        tile("Health facilities exposed", ex_cat["hospital"] + ex_cat["health_centre"], "hospital",
             "danger" if ex_cat["hospital"] + ex_cat["health_centre"] else "ok", "hospitals and health centres", res),
    ])
    cat_rows = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th("Type"), html.Th("Total"), html.Th("Exposed now")])),
        html.Tbody([html.Tr([html.Td([html.Span(style={"display": "inline-block", "width": "9px", "height": "9px",
                                                       "borderRadius": "50%", "marginRight": "7px",
                                                       "background": "rgb({},{},{})".format(*c)}), label]),
                             html.Td(by_cat[k]), html.Td(ex_cat[k], className="ops-above" if ex_cat[k] else None)])
                    for k, (label, c) in layers.CATEGORIES.items()]),
    ]))
    exposed.sort(key=lambda a: (-(a["id"] in near), -a.get("level", 0), a["category"], a["name"]))
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Facility", "Type", "Ward", "Predicted risk", "Observed")])),
        html.Tbody([html.Tr([
            html.Td(a["name"], className="dash-cell-strong"),
            html.Td(layers.CATEGORIES[a["category"]][0][:-1] if a["category"] in layers.CATEGORIES else a["category"]),
            html.Td(boundaries.label(a["ward"]) if a.get("ward") else "—", className="dash-cell-muted"),
            html.Td(html.Span(RISK_CLASS[LEVEL_KEY[a.get("level", 0)]], className=f"ops-risk lvl-{a.get('level', 0)}")),
            html.Td(f"Report {near[a['id']]['kind']} at {near[a['id']]['place']}" if a["id"] in near else "—",
                    className="dash-cell-muted"),
        ]) for a in exposed[:80]]),
    ])) if exposed else html.P("No facilities are in predicted flood areas or near reported flooding.",
                                className="dash-empty")
    out = [kpis,
           html.Div(className="dash-grid cols-5-7", children=[
               panel("Facilities by type", cat_rows, html.P(source, className="dash-cell-muted mt-2"),
                     icon="building"),
               panel("Exposed facilities", table, icon="exclamation-diamond",
                     actions=html.Span("Predicted risk is model output", className="ops-model-label"))])]
    if can(user, "view_supplies"):
        out.append(_supplies_panel())
    if can(user, "view_beneficiaries"):
        out.append(_beneficiaries_panel(areas))
    return out


def _supplies_panel() -> html.Section:
    wh = registry.get("supply").fetch("warehouses")
    dist = registry.get("supply").fetch("distributions")
    if not wh.ok:
        return panel("Emergency supplies", empty_state(wh, "Warehouses, stock and distributions"), icon="box-seam")
    rows = [html.Tr([html.Td(w["name"], className="dash-cell-strong"),
                     html.Td(", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (w.get("stock") or {}).items()))])
            for w in wh.records]
    drows = [html.Tr([html.Td(d.get("destination")), html.Td(d.get("status", "").replace("_", " ")),
                      html.Td(", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (d.get("items") or {}).items()))])
             for d in (dist.records if dist.ok else [])]
    return panel("Emergency supplies", html.Div(className="dash-table-wrap", children=html.Table(className="dash-table",
                 children=[html.Thead(html.Tr([html.Th("Warehouse"), html.Th("Stock")])), html.Tbody(rows)])),
                 html.Div(className="dash-table-wrap mt-3", children=html.Table(className="dash-table", children=[
                     html.Thead(html.Tr([html.Th("Distribution to"), html.Th("Status"), html.Th("Items")])),
                     html.Tbody(drows)])) if drows else None,
                 html.P("Restricted: procurement and financial data are not requested.", className="dash-cell-muted"),
                 icon="box-seam", actions=demo_tag(wh))


def _beneficiaries_panel(areas) -> html.Section:
    r = registry.get("beneficiaries").fetch("aggregate", level="sub_county")
    if not r.ok:
        return panel("Vulnerable communities", empty_state(r, "Aggregated beneficiary figures"), icon="people")
    body = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th("Area"), html.Th("Beneficiaries")])),
        html.Tbody([html.Tr([html.Td(x["area"]), html.Td(x["beneficiaries"])]) for x in r.records])]))
    return panel("Vulnerable communities", body,
                 html.P("Aggregated counts only; no personal records are requested or shown. Restricted view.",
                        className="dash-cell-muted mt-2"), icon="people", actions=demo_tag(r))


# -------------------------------------------------------------- incidents --
def incidents(start: date, end: date, areas: list[str] | None, kinds: list[str] | None = None,
              status: str = "all") -> tuple[list[dict], dict]:
    """Community reports and news-reported floods in a period, as one list, newest first."""
    out, results = [], {}
    comm = registry.get("community").fetch("history", start=start.isoformat(), end=end.isoformat())
    results["community"] = comm
    for r in comm.records if comm.ok else []:
        out.append({"source": "Community report", "type": r["kind"], "type_label": REPORT_KINDS.get(r["kind"], r["kind"]),
                    "severity": {"homes": 3, "impassable": 2, "water": 1}.get(r["kind"], 1),
                    "when": r["created_at"], "place": r["place"], "lat": r["lat"], "lon": r["lon"],
                    "status": r["status"] + (" · verified" if r.get("verified") else ""),
                    "detail": f"{r.get('confirmations', 0)} confirmations" + (f" · {r['note']}" if r.get("note") else "")})
    news = registry.get("news_floods").fetch("events", start=start.isoformat(), end=end.isoformat())
    results["news"] = news
    for e in news.records if news.ok else []:
        out.append({"source": "News report", "type": "news", "type_label": "Flooding in the news", "severity": 1,
                    "when": e["start"], "place": e["name"], "lat": e["lat"], "lon": e["lon"], "status": "historical",
                    "detail": f"{e['area_km2']:.1f} km² area reported · Groundsource"})
    if areas:
        out = [i for i in out if boundaries.point_in(i["lat"], i["lon"], areas)]
    if kinds:
        out = [i for i in out if i["type"] in kinds]
    if status and status != "all":
        out = [i for i in out if i["status"].startswith(status)]
    out.sort(key=lambda i: str(i["when"]), reverse=True)
    return out, results


def incidents_page(user: dict, theme: str, flt: dict | None = None) -> list:
    today = datetime.now(EAT).date()
    controls = html.Div(className="dash-toolbar", children=[
        dcc.DatePickerRange(id="ops-inc-range", start_date=date(2024, 3, 1), end_date=today, max_date_allowed=today,
                            min_date_allowed=date(2000, 1, 1), display_format="D MMM YYYY", className="ops-range"),
        dcc.Dropdown(id="ops-inc-kinds", multi=True, value=[], placeholder="All incident types",
                     className="dash-select ops-years",
                     options=[{"label": v, "value": k} for k, v in REPORT_KINDS.items()] +
                             [{"label": "Flooding in the news", "value": "news"}]),
        dcc.Dropdown(id="ops-inc-status", value="all", clearable=False, searchable=False,
                     className="dash-select dash-filter",
                     options=[{"label": "Any status", "value": "all"}] +
                             [{"label": s.title(), "value": s} for s in ("active", "cleared", "expired", "historical")]),
    ])
    return [panel("Find incidents", controls, html.P("Community reports come from signed-in users; news reports are "
                                                     "Groundsource events (2000-2025), about 60% right in place and "
                                                     "date.", className="dash-hint"), icon="search"),
            html.Div(id="ops-inc-result")]


def incidents_view(start, end, kinds, status, flt: dict, theme: str) -> list:
    s = date.fromisoformat(start[:10]) if start else date(2024, 3, 1)
    e = date.fromisoformat(end[:10]) if end else datetime.now(EAT).date()
    rows, res = incidents(s, e, flt.get("areas") or None, kinds or None, status or "all")
    comm = sum(1 for r in rows if r["source"] == "Community report")
    news = len(rows) - comm
    subs = Counter()
    for r in rows:
        a = boundaries.area_at(r["lat"], r["lon"], "sub_county")
        subs[a["name"] if a else "Outside the mapped areas"] += 1
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Incidents", len(rows), "megaphone", "warn" if rows else "ok", f"{s:%d %b %Y} to {e:%d %b %Y}"),
        tile("Community reports", comm, "people", "info", "from people on the ground", res["community"]),
        tile("News reports", news, "newspaper", "info", "Groundsource, to late 2025", res["news"]),
        tile("Most affected sub-county", subs.most_common(1)[0][0] if subs else "—", "geo-alt", "accent",
             f"{subs.most_common(1)[0][1]} incidents" if subs else "none"),
    ])
    by_month = Counter(str(r["when"])[:7] for r in rows)
    months = sorted(by_month)
    fig_t = go.Figure(go.Bar(x=months, y=[by_month[m] for m in months], marker_color="#a78bfa"))
    fig_t = _style(fig_t, theme, 240)
    top = subs.most_common(12)[::-1]
    fig_s = go.Figure(go.Bar(x=[n for _s, n in top], y=[s_ for s_, _n in top], orientation="h", marker_color="#38bdf8"))
    fig_s = _style(fig_s, theme, 240)
    fig_s.update_layout(hovermode="closest")
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("When", "Type", "Place", "Status", "Detail", "Source")])),
        html.Tbody([html.Tr([html.Td(local_time(r["when"]) if "T" in str(r["when"]) else str(r["when"])),
                             html.Td(html.Span(r["type_label"], className=f"ops-risk lvl-{r['severity']}")),
                             html.Td(r["place"], className="dash-cell-strong"), html.Td(r["status"]),
                             html.Td(r["detail"], className="dash-cell-muted"), html.Td(r["source"])])
                    for r in rows[:150]]),
    ])) if rows else html.P("No incidents match.", className="dash-empty")
    return [kpis,
            html.Div(className="dash-grid cols-7-5", children=[
                panel("Incidents by month", dcc.Graph(figure=fig_t, config=GRAPH_CONFIG, style={"height": "240px"}),
                      icon="bar-chart"),
                panel("By sub-county", dcc.Graph(figure=fig_s, config=GRAPH_CONFIG, style={"height": "240px"}),
                      icon="geo")]),
            panel(f"Incidents ({len(rows)}" + (", newest 150 shown)" if len(rows) > 150 else ")"), table,
                  icon="list-ul", actions=html.Span("Observed, not predicted", className="ops-model-label"))]


# ---------------------------------------------------------------- history --
def history_page(user: dict, theme: str, flt: dict | None = None) -> list:
    return [panel("Compare two periods", html.Div(className="dash-toolbar", children=[
                html.Span("Period A", className="dash-label mb-0"),
                dcc.DatePickerRange(id="ops-hist-a", start_date=date(2024, 3, 1), end_date=date(2024, 5, 31),
                                    min_date_allowed=date(2015, 1, 1), display_format="D MMM YYYY", className="ops-range"),
                html.Span("Period B", className="dash-label mb-0"),
                dcc.DatePickerRange(id="ops-hist-b", start_date=date(2023, 3, 1), end_date=date(2023, 5, 31),
                                    min_date_allowed=date(2015, 1, 1), display_format="D MMM YYYY", className="ops-range"),
            ]), icon="calendar-range"),
            html.Div(id="ops-hist-result")]


def _period_stats(start: date, end: date, areas) -> dict:
    from src.dashboard.ops_monitoring import _chirps_daily, unusual_days
    daily = _chirps_daily()
    days = [d for d in daily if start <= d <= end]
    rain = [daily[d] for d in sorted(days)]
    inc, _r = incidents(start, end, areas)
    runs = get_ops_store().prediction_runs(limit=5000, since=datetime.combine(start, datetime.min.time()).isoformat(),
                                           until=datetime.combine(end, datetime.max.time()).isoformat())
    return {"days": sorted(days), "rain": rain, "total": sum(rain), "wet_days": sum(1 for v in rain if v >= 1),
            "max": max(rain) if rain else 0, "unusual": unusual_days(daily, start, end), "incidents": inc,
            "warning_runs": sum(1 for r in runs if r["city_level"] >= 1), "runs": len(runs)}


def history_view(a0, a1, b0, b1, flt: dict, theme: str) -> list:
    pa = (date.fromisoformat(a0[:10]), date.fromisoformat(a1[:10]))
    pb = (date.fromisoformat(b0[:10]), date.fromisoformat(b1[:10]))
    areas = flt.get("areas") or None
    A, B = _period_stats(*pa, areas), _period_stats(*pb, areas)
    label = lambda p: f"{p[0]:%d %b %Y} – {p[1]:%d %b %Y}"

    def row(name, fa, fb):
        return html.Tr([html.Td(name, className="dash-cell-strong"), html.Td(fa), html.Td(fb)])
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(""), html.Th(f"A: {label(pa)}"), html.Th(f"B: {label(pb)}")])),
        html.Tbody([
            row("Rainfall total (CHIRPS, city average)", f"{A['total']:.0f} mm", f"{B['total']:.0f} mm"),
            row("Days with 1 mm or more", A["wet_days"], B["wet_days"]),
            row("Wettest day", f"{A['max']:.0f} mm", f"{B['max']:.0f} mm"),
            row("Unusually wet days", len(A["unusual"]), len(B["unusual"])),
            row("Incidents (community + news)", len(A["incidents"]), len(B["incidents"])),
            row("Outlooks with a model warning", f"{A['warning_runs']} of {A['runs']}" if A["runs"] else "no runs stored",
                f"{B['warning_runs']} of {B['runs']}" if B["runs"] else "no runs stored"),
        ]),
    ]))
    fig = go.Figure()
    for st, name, colour in ((A, "A", "#38bdf8"), (B, "B", "#f0b93f")):
        import numpy as np
        fig.add_scatter(x=list(range(1, len(st["rain"]) + 1)), y=np.cumsum(st["rain"]) if st["rain"] else [],
                        name=f"{name}: {label(pa if name == 'A' else pb)}", line=dict(color=colour, width=2.2))
    fig = _style(fig, theme, 280)
    fig.update_layout(xaxis=dict(title="day of period"), yaxis=dict(title="cumulative rainfall (mm)"))
    inc = Counter()
    for name, st in (("A", A), ("B", B)):
        for i in st["incidents"]:
            inc[(name, i["type_label"])] += 1
    kinds = sorted({k for _n, k in inc})
    fig_i = go.Figure([go.Bar(x=kinds, y=[inc[(n, k)] for k in kinds], name=n, marker_color=c)
                       for n, c in (("A", "#38bdf8"), ("B", "#f0b93f"))])
    fig_i = _style(fig_i, theme, 280)
    fig_i.update_layout(barmode="group")
    return [panel("Summary", table, html.P("Model warnings exist only for outlooks stored since the operations "
                                           "console was added; the published-alert history arrives with the "
                                           "Automated Alert System.", className="dash-cell-muted mt-2"),
                  icon="table"),
            html.Div(className="dash-grid cols-7-5", children=[
                panel("Cumulative rainfall", dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": "280px"}),
                      icon="cloud-rain"),
                panel("Incidents by type", dcc.Graph(figure=fig_i, config=GRAPH_CONFIG, style={"height": "280px"}),
                      icon="bar-chart")])]


# ------------------------------------------------------------- registration --
_status = PAGES.pop("/ops/status")
PAGES.update({
    "/ops/infrastructure": ("Infrastructure", "building-exclamation", "Infrastructure at risk",
                            "Facilities in or near predicted and observed flooding.", "view_infrastructure",
                            infrastructure_page, True),
    "/ops/incidents": ("Incidents", "megaphone", "Flood incidents",
                       "Community and news-reported flooding by type, status, date and area.", "view_ops",
                       incidents_page, True),
    "/ops/history": ("History", "calendar-range", "Historical analysis",
                     "Rainfall, incidents and model warnings, period against period.", "view_ops", history_page, True),
})
PAGES["/ops/status"] = _status


def register_asset_callbacks(app) -> None:
    from src.dashboard.ops_console import effective_filter

    @app.callback(Output("ops-inc-result", "children"),
                  Input("ops-inc-range", "start_date"), Input("ops-inc-range", "end_date"),
                  Input("ops-inc-kinds", "value"), Input("ops-inc-status", "value"),
                  Input("ops-area", "value"), Input("ops-updated", "children"), State("resolved-theme-store", "data"))
    def show_incidents(start, end, kinds, status, area, _u, theme):
        user = current_user()
        if not can(user, "view_ops"):
            raise PreventUpdate
        return incidents_view(start, end, kinds, status, effective_filter(user, area, 0), theme or "dark")

    @app.callback(Output("ops-hist-result", "children"),
                  Input("ops-hist-a", "start_date"), Input("ops-hist-a", "end_date"),
                  Input("ops-hist-b", "start_date"), Input("ops-hist-b", "end_date"),
                  Input("ops-area", "value"), State("resolved-theme-store", "data"))
    def show_history(a0, a1, b0, b1, area, theme):
        user = current_user()
        if not can(user, "view_ops") or not all((a0, a1, b0, b1)):
            raise PreventUpdate
        return history_view(a0, a1, b0, b1, effective_filter(user, area, 0), theme or "dark")
