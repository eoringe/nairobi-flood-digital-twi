"""
src.dashboard.ops_predictions
=============================
Prediction, comparison and alert screens of the operations console
(specification §2.6–§2.8, §6.6–§6.7; FR-007, FR-009, FR-021; UC-FVD-005,
UC-FVD-006, UC-FVD-010).

    /ops/predictions  the current outlook: risk by area with the hour each
                      place reaches its level, forecast horizons, issue time,
                      the 3-day flood chance and how reliable it has been,
                      model details, and the history of past runs
    /ops/compare      a prediction run beside what was observed in its window:
                      community reports, news-reported floods, the satellite
                      pass and gauges when connected, matched by ward
    /ops/alerts       published alerts from the Automated Alert System, or the
                      outlook's model warnings, labelled as such, until it is
                      connected; severity/status filters and the full record

Predictions are always labelled as model output, observations as observed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.dashboard.ops_console import (EAT, LEVEL_RANK, PAGES, RISK_CLASS, _icon, demo_tag, empty_state, fresh_badge,
                                       local_time, panel, tile)
from src.geo import boundaries
from src.integrations import base, registry
from src.integrations.store import get_ops_store

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}
CALL_TONE = {"likely": "danger", "possible": "warn", "unlikely": "ok"}
#: Observations within this window after a run's issue time count against it.
COMPARE_WINDOW = timedelta(days=3)
COMPARE_BEFORE = timedelta(days=1)


def _model_label() -> html.Span:
    return html.Span("Model output, not observed flooding", className="ops-model-label")


# ------------------------------------------------------------- predictions --
def predictions_page(user: dict, theme: str, flt: dict | None = None) -> list:
    flt = flt or {}
    areas, min_risk = flt.get("areas") or [], int(flt.get("min_risk") or 0)
    pred = registry.get("prediction").fetch("outlook")
    if not pred.ok or not pred.records:
        return [empty_state(pred, "The flood prediction")]
    nc = pred.records[0]
    chance = registry.get("prediction").fetch("chance")
    from src.dashboard.callbacks import predictor
    from src.forecast.flood_chance import _params
    params = _params() or {}
    city = max(nc.get("city_level") or [0])
    c = chance.records[0] if chance.ok and chance.records else None
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Issued", local_time(nc.get("as_of")), "clock", "info",
             "replay: hindsight rainfall" if nc.get("is_replay") else "live forecast", pred, source="12-hour outlook"),
        tile("Worst level, next 12 h", RISK_CLASS[{0: "CLEAR", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}[city]],
             "exclamation-triangle", ["ok", "warn", "warn", "danger"][city], "highest band reached anywhere", pred,
             source="12-hour outlook"),
        tile("Flood chance, next 3 days", f"{round(100 * c['p'])}%" if c else "—", "graph-up-arrow",
             CALL_TONE.get(c["call"], "info") if c else "neutral", f"{c['call_text']} (95% call)" if c else
             "needs 31 days of rainfall", chance, source="Flood chance model"),
        tile("72-hour rainfall, peak", f"{max(nc.get('acc72_mm') or [0]):.0f} mm", "cloud-rain", "info",
             "bias-corrected forecast, CHIRPS scale", pred, source="Open-Meteo"),
    ])
    hours = nc.get("hours") or []
    rows = []
    for w in nc.get("warnings") or []:
        for p in w["places"]:
            if not boundaries.place_in(p, areas) or LEVEL_RANK[w["level"]] < min_risk:
                continue
            ward = boundaries.place_areas().get(p, {}).get("ward")
            rows.append((p, boundaries.label(ward) if ward else "—", w["level"], w.get("level_now", w["level"]),
                         hours[w["hour"]] if w["hour"] < len(hours) else "", w["text"]))
    rows.sort(key=lambda r: (-LEVEL_RANK[r[2]], r[4]))
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Place", "Ward", "Predicted risk", "From", "Outlook")])),
        html.Tbody([html.Tr([html.Td(p, className="dash-cell-strong"), html.Td(ward, className="dash-cell-muted"),
                             html.Td(html.Span(RISK_CLASS[lvl], className=f"ops-risk lvl-{LEVEL_RANK[lvl]}")),
                             html.Td(when), html.Td(text, className="dash-cell-muted")])
                    for p, ward, lvl, _now, when, text in rows[:60]]),
    ])) if rows else html.P("No flooding predicted" + (" in the selected area" if areas else "") +
                            " in the next 12 hours.", className="dash-empty")

    cal = (params.get("evaluation") or {}).get("feed_calibration_cross_fitted") or {}
    call = cal.get("call_95") or {}
    reliability = html.Div([
        html.P(["The 3-day chance is learned from 158 days of news-reported flooding (2015-2025). On years the "
                "calibration did not see, its stated chances matched what happened:"], className="dash-hint"),
        html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
            html.Thead(html.Tr([html.Th("Stated chance"), html.Th("Days"), html.Th("Flood reported")])),
            html.Tbody([html.Tr([html.Td(f"{100 * r['predicted']:.0f}% (band {r['band']})"), html.Td(r["n"]),
                                 html.Td(f"{100 * r['observed']:.0f}%")]) for r in cal.get("reliability", [])]),
        ])),
        html.P([html.Strong("95% call: "), f"{100 * call.get('flood_days_not_called_unlikely', 0):.1f}% of flood days "
                f"were not called 'no flooding', and {100 * call.get('dry_days_not_called_likely', 0):.1f}% of dry "
                "days were not called 'flooding expected'. This is a coverage guarantee, not 95% accuracy; the "
                "event predicted is a flood being reported in the news."], className="dash-hint mt-2"),
    ]) if cal else html.P("Flood-chance evaluation not available.", className="dash-empty")
    m = getattr(predictor, "metrics", {}) or {}
    details = html.Dl(className="dash-facts", children=[
        html.Dt("Extent model"), html.Dd(predictor.model_version),
        html.Dt("Output"), html.Dd("Calibrated probability that each ~70 m cell floods; bands Moderate 25-55%, "
                                   "High 55-82%, Very High above 82%"),
        html.Dt("Test score"), html.Dd(f"F1 {m['f1']:.3f} against constructed labels, 4 unseen storm seasons"
                                       if "f1" in m else "—"),
        html.Dt("Horizons"), html.Dd("12-hour place outlook, hourly; 3-day flood chance"),
        html.Dt("Rainfall input"), html.Dd("Open-Meteo hourly forecast, quantile-mapped to CHIRPS"),
        html.Dt("Flood chance"), html.Dd(params.get("method", "—")),
        html.Dt("Limits"), html.Dd("City-wide rainfall: local storms are missed; places are neighbourhood scale, "
                                   "not streets. See LIMITATIONS.md."),
    ])
    return [kpis,
            panel("Predicted risk by area, next 12 hours", _hour_chart(nc, theme), table, icon="map",
                  actions=_model_label()),
            html.Div(className="dash-grid cols-7-5", children=[
                panel("How reliable is the flood chance?", reliability, icon="patch-check"),
                panel("Model", details, icon="cpu")]),
            panel("Prediction runs", _runs_table(), icon="clock-history",
                  actions=dcc.Link("Compare with observations", href="/ops/compare", className="dash-link"))]


def _hour_chart(nc: dict, theme: str) -> dcc.Graph:
    from src.dashboard.ops_monitoring import _style
    hours = nc.get("hours") or []
    fig = go.Figure()
    fig.add_bar(x=hours, y=nc.get("flooded_pct") or [], name="Area predicted flooded (%)", marker_color="#5b9bff")
    fig.add_scatter(x=hours, y=nc.get("acc72_mm") or [], name="72-hour rainfall (mm)", yaxis="y2", mode="lines+markers",
                    line=dict(color="#e2e8f0"))
    fig = _style(fig, theme, 220)
    fig.update_layout(yaxis=dict(title="% of grid"), yaxis2=dict(title="mm", overlaying="y", side="right",
                                                                  showgrid=False, rangemode="tozero"))
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": "220px"})


def _runs_table(limit: int = 15) -> html.Div:
    runs = get_ops_store().prediction_runs(limit=limit)
    if not runs:
        return html.P("No runs recorded yet. Every live outlook is stored from now on.", className="dash-empty")
    return html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Issued", "Worst level", "Places warned", "72 h rain", "Flood chance",
                                                 "Headline", "Model")])),
        html.Tbody([html.Tr([
            html.Td(local_time(r["issued_at"])),
            html.Td(html.Span(RISK_CLASS[{0: "CLEAR", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}[r["city_level"]]],
                              className=f"ops-risk lvl-{r['city_level']}")),
            html.Td(len(r["place_levels"])),
            html.Td(f"{r['rain_72h_mm']:.0f} mm" if r["rain_72h_mm"] is not None else "—"),
            html.Td(f"{round(100 * r['chance'])}% · {r['chance_call']}" if r["chance"] is not None else "—"),
            html.Td(r["headline"] or "", className="dash-cell-muted"),
            html.Td(r["model_version"].split(".pth")[0], className="dash-mono"),
        ]) for r in runs]),
    ]))


# --------------------------------------------------------------- compare --
def run_options() -> list[dict]:
    from src.forecast.nowcast import SOURCES
    opts = [{"label": f"Replay · {v['label'].split(' - ', 1)[-1]}", "value": f"replay:{k}"}
            for k, v in SOURCES.items() if v["as_of"] is not None]
    opts += [{"label": f"Live run · {local_time(r['issued_at'])} · {r['headline'] or ''}"[:90],
              "value": f"run:{r['run_id']}"} for r in get_ops_store().prediction_runs(limit=40)]
    return opts


def _load_run(key: str) -> dict | None:
    """{issued, levels {place: level}, label, replay} for a stored run or a replay outlook."""
    kind, _, ident = (key or "").partition(":")
    if kind == "run":
        r = get_ops_store().prediction_run(int(ident))
        if r is None:
            return None
        return {"issued": base.as_utc(r["issued_at"]), "levels": r["place_levels"], "replay": False,
                "label": f"Live run issued {local_time(r['issued_at'])}"}
    if kind == "replay":
        from src.dashboard.callbacks import predictor
        from src.forecast.nowcast import get_nowcast
        nc = get_nowcast(ident, predictor)
        if nc.error:
            return None
        levels = {}
        for w in nc.warnings:
            for p in w["places"]:
                if LEVEL_RANK[w["level"]] > LEVEL_RANK.get(levels.get(p, "CLEAR"), 0):
                    levels[p] = w["level"]
        return {"issued": base.as_utc(nc.as_of), "levels": levels, "replay": True, "label": nc.label}
    return None


def observations(issued: datetime, areas: list[str] | None = None) -> list[dict]:
    """Everything observed from a day before the run to three days after: {kind, ward, label, when}."""
    start, end = issued - COMPARE_BEFORE, issued + COMPARE_WINDOW
    out = []
    comm = registry.get("community").fetch("history", start=start.date().isoformat(), end=end.date().isoformat())
    for r in comm.records if comm.ok else []:
        a = boundaries.area_at(r["lat"], r["lon"], "ward")
        out.append({"kind": "Community report", "ward": a["id"] if a else None, "when": r["created_at"],
                    "label": f"{r['place']}: {r['kind']} ({r['status']})"})
    news = registry.get("news_floods").fetch("events", start=start.date().isoformat(), end=end.date().isoformat())
    for e in news.records if news.ok else []:
        if e["area_km2"] > 25:
            continue
        a = boundaries.area_at(e["lat"], e["lon"], "ward")
        out.append({"kind": "News report", "ward": a["id"] if a else None, "when": e["start"], "label": e["name"]})
    iot = registry.get("iot").fetch("stations")
    for s in iot.records if iot.ok else []:
        lvl = (s.get("latest") or {}).get("water_level") or {}
        if s.get("warning_m") and lvl.get("value") is not None and lvl["value"] >= s["warning_m"]:
            a = boundaries.area_at(s["lat"], s["lon"], "ward")
            out.append({"kind": "Gauge above warning" + (" (DEMO)" if iot.demo else ""), "ward": a["id"] if a else None,
                        "when": lvl.get("observed_at"), "label": s["name"]})
    if areas:
        out = [o for o in out if o["ward"] and boundaries.lineage(o["ward"]) and
               ({x["id"] for x in boundaries.lineage(o["ward"])} & set(areas))]
    return out


def compare(key: str, areas: list[str] | None = None) -> dict | None:
    """Predicted wards against observed wards for a run: hits, unconfirmed predictions, missed observations."""
    run = _load_run(key)
    if run is None:
        return None
    pa = boundaries.place_areas()
    predicted: dict[str, str] = {}
    for place, lvl in run["levels"].items():
        ward = pa.get(place, {}).get("ward")
        if ward and boundaries.place_in(place, areas) and LEVEL_RANK[lvl] > LEVEL_RANK.get(predicted.get(ward, "CLEAR"), 0):
            predicted[ward] = lvl
    obs = observations(run["issued"], areas)
    observed: dict[str, list[dict]] = {}
    for o in obs:
        if o["ward"]:
            observed.setdefault(o["ward"], []).append(o)
    wards = set(predicted) | set(observed)
    rows = [{"ward": w, "predicted": predicted.get(w), "observed": observed.get(w, []),
             "result": "Confirmed" if w in predicted and w in observed else
                       ("Not confirmed" if w in predicted else "Missed")} for w in wards]
    rows.sort(key=lambda r: ({"Confirmed": 0, "Missed": 1, "Not confirmed": 2}[r["result"]],
                             -LEVEL_RANK.get(r["predicted"] or "CLEAR", 0)))
    hits = sum(r["result"] == "Confirmed" for r in rows)
    return {"run": run, "rows": rows, "hits": hits, "predicted": len(predicted), "observed": len(observed),
            "observations": len(obs)}


def compare_page(user: dict, theme: str, flt: dict | None = None) -> list:
    opts = run_options()
    return [panel("Choose a prediction run", html.Div(className="dash-toolbar", children=[
                dcc.Dropdown(id="ops-compare-run", options=opts, value=opts[1]["value"] if len(opts) > 1 else None,
                             clearable=False, className="dash-select ops-years")]),
                html.P("Observations from a day before the run to three days after are matched to the run's "
                       "predicted places by ward. Observed: community reports, news-reported floods and, when "
                       "connected, gauges above their warning level.", className="dash-hint"),
                icon="sliders"),
            html.Div(id="ops-compare-result")]


def compare_view(key: str, flt: dict) -> list:
    res = compare(key, flt.get("areas") or None)
    if res is None:
        return [html.P("That run is not available.", className="dash-empty")]
    precision = f"{100 * res['hits'] / res['predicted']:.0f}%" if res["predicted"] else "—"
    recall = f"{100 * res['hits'] / res['observed']:.0f}%" if res["observed"] else "—"
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Wards predicted", res["predicted"], "map", "info", "Moderate or worse, model output"),
        tile("Wards with observed flooding", res["observed"], "eye", "warn", f"{res['observations']} observations"),
        tile("Predictions confirmed", precision, "check2-circle", "ok", f"{res['hits']} of {res['predicted']} wards"),
        tile("Observations predicted", recall, "bullseye", "accent", f"{res['hits']} of {res['observed']} wards"),
    ])
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Ward", "Predicted (model)", "Observed", "Result")])),
        html.Tbody([html.Tr([
            html.Td(boundaries.label(r["ward"]), className="dash-cell-strong"),
            html.Td(html.Span(RISK_CLASS[r["predicted"]], className=f"ops-risk lvl-{LEVEL_RANK[r['predicted']]}")
                    if r["predicted"] else html.Span("—", className="dash-cell-muted")),
            html.Td([html.Div(f"{o['kind']}: {o['label']}", className="dash-cell-muted") for o in r["observed"][:3]]
                    + ([html.Div(f"+{len(r['observed']) - 3} more", className="dash-cell-muted")]
                       if len(r["observed"]) > 3 else []) or "—"),
            html.Td(html.Span(r["result"], className=f"ops-result is-{r['result'].split()[0].lower()}")),
        ]) for r in res["rows"]]),
    ])) if res["rows"] else html.P("Nothing predicted and nothing observed in this window.", className="dash-empty")
    caveat = html.P("A ward with no observation is not evidence that it stayed dry: reports depend on who was there "
                    "and what the news covered. 'Not confirmed' therefore means unverified, not wrong."
                    + (" Replays use rainfall that actually fell, a perfect forecast no live run has."
                       if res["run"]["replay"] else ""), className="dash-hint")
    return [html.Div([html.Strong(res["run"]["label"]), f" · issued {local_time(res['run']['issued'])}"],
                     className="mb-2"), kpis,
            panel("Predicted and observed, by ward", caveat, table, icon="intersect")]


# ----------------------------------------------------------------- alerts --
SEVERITY_ORDER = ["critical", "very high", "high", "moderate", "low"]


def _alerts(flt: dict) -> tuple[list[dict], base.SourceResult, bool]:
    partner = registry.get("alerts").fetch("active")
    published = partner.ok
    res = partner if published else registry.get("outlook_alerts").fetch("active")
    rows = res.records if res.ok else []
    areas = flt.get("areas") or []
    if areas:
        rows = [{**a, "areas": [p for p in a.get("areas") or [] if boundaries.place_in(p, areas)]} for a in rows]
        rows = [a for a in rows if a["areas"]]
    return rows, res, published


def alerts_page(user: dict, theme: str, flt: dict | None = None) -> list:
    flt = flt or {}
    rows, res, published = _alerts(flt)
    banner = None if published else html.Div(
        [_icon("plug", "me-2"), html.Strong("Automated Alert System not connected. "),
         "These are the outlook's model warnings, not published alerts: they have no publication time or "
         "acknowledgements. Alert creation, approval and publication stay with the alert system (spec §2.8)."],
        className="dash-alert tone-info mb-2")
    sev_counts = {s: sum(1 for a in rows if (a.get("severity") or "").lower() == s) for s in SEVERITY_ORDER}
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Active " + ("alerts" if published else "model warnings"), len(rows), "bell", "warn" if rows else "ok",
             "Automated Alert System" if published else "12-hour outlook", res),
        tile("Critical", sev_counts["critical"] + sev_counts["very high"], "exclamation-octagon",
             "danger" if sev_counts["critical"] + sev_counts["very high"] else "ok", "highest severity", res),
        tile("High", sev_counts["high"], "exclamation-triangle", "warn" if sev_counts["high"] else "ok", "", res),
        tile("Places affected", len({p for a in rows for p in a.get("areas") or []}), "geo-alt", "info",
             "named in an alert", res),
    ])
    toolbar = html.Div(className="dash-toolbar", children=[
        dcc.Dropdown(id="ops-alert-severity", value="all", clearable=False, searchable=False,
                     className="dash-select dash-filter",
                     options=[{"label": "All severities", "value": "all"}] +
                             [{"label": s.title(), "value": s} for s in SEVERITY_ORDER]),
        dcc.Dropdown(id="ops-alert-status", value="all", clearable=False, searchable=False,
                     className="dash-select dash-filter",
                     options=[{"label": "All statuses", "value": "all"}] +
                             [{"label": s.replace("_", " ").title(), "value": s}
                              for s in sorted({a.get("status") or "unknown" for a in rows})]),
    ])
    return [x for x in (banner,) if x] + [
        kpis,
        panel("Active " + ("alerts" if published else "model warnings"), toolbar,
              html.Div(id="ops-alert-list", children=alert_cards(rows, user, "all", "all")), icon="bell",
              actions=demo_tag(res)),
        dbc.Modal(id="ops-alert-modal", size="lg", is_open=False, className="dash-modal",
                  children=[dbc.ModalHeader(dbc.ModalTitle(id="ops-alert-modal-title")),
                            dbc.ModalBody(id="ops-alert-modal-body")]),
    ]


def alert_cards(rows: list[dict], user: dict, severity: str, status: str) -> list:
    shown = [a for a in rows if (severity == "all" or (a.get("severity") or "").lower() == severity)
             and (status == "all" or (a.get("status") or "unknown") == status)]
    shown.sort(key=lambda a: SEVERITY_ORDER.index((a.get("severity") or "low").lower())
               if (a.get("severity") or "low").lower() in SEVERITY_ORDER else 9)
    if not shown:
        return [html.P("No active alerts match.", className="dash-empty")]
    out = []
    for a in shown:
        sev = (a.get("severity") or "moderate").lower()
        rank = {"critical": 3, "very high": 3, "high": 2, "moderate": 1}.get(sev, 0)
        ack = (f"{a['acknowledged']} of {a.get('recipients', '?')} acknowledged" if a.get("acknowledged") is not None
               else "Acknowledgements: available when the alert system is connected")
        out.append(html.Div(className=f"ops-alert lvl-{rank}", children=[
            html.Div(className="ops-alert-head", children=[
                html.Span(RISK_CLASS.get(sev.upper(), sev.title()), className=f"ops-risk lvl-{rank}"),
                html.Strong(a.get("title", "Alert")),
                html.Span((a.get("status") or "").replace("_", " "), className="dash-chip"),
            ]),
            html.Div(", ".join(a.get("areas") or [])[:300], className="ops-alert-areas"),
            html.Div(className="ops-alert-meta", children=[
                html.Span(["Created ", local_time(a.get("created_at"))]),
                html.Span(["Published ", local_time(a.get("published_at")) if a.get("published_at") else "not published"]),
                html.Span(["Starts ", local_time(a["starts_at"])]) if a.get("starts_at") else None,
                html.Span(ack)]),
            html.Ul([html.Li(x) for x in a.get("recommended_actions") or []], className="ops-alert-actions"),
            html.Button("Full record", id={"type": "ops-alert-open", "aid": a.get("id", "")}, n_clicks=0,
                        className="dash-btn-ghost btn btn-sm") if can(user, "view_alert_record") else None,
        ]))
    return out


# ------------------------------------------------------------- registration --
_status = PAGES.pop("/ops/status")
PAGES.update({
    "/ops/predictions": ("Predictions", "graph-up-arrow", "Flood predictions",
                         "Predicted risk by area, horizons, uncertainty, model details and past runs.", "view_ops",
                         predictions_page, True),
    "/ops/compare": ("Predicted vs observed", "intersect", "Prediction against observation",
                     "A prediction run beside the flooding observed in its window.", "view_ops", compare_page, True),
    "/ops/alerts": ("Alerts", "bell", "Active alerts",
                    "Alerts with severity, status, affected areas and recommended actions.", "view_ops",
                    alerts_page, True),
})
PAGES["/ops/status"] = _status


def register_prediction_callbacks(app) -> None:
    @app.callback(Output("ops-compare-result", "children"), Input("ops-compare-run", "value"),
                  Input("ops-area", "value"), Input("ops-updated", "children"))
    def show_compare(key, area, _u):
        from src.dashboard.ops_console import effective_filter
        user = current_user()
        if not key or not can(user, "view_ops"):
            raise PreventUpdate
        return compare_view(key, effective_filter(user, area, 0))

    @app.callback(Output("ops-alert-list", "children"),
                  Input("ops-alert-severity", "value"), Input("ops-alert-status", "value"),
                  State("ops-area", "value"), prevent_initial_call=True)
    def filter_alerts(severity, status, area):
        from src.dashboard.ops_console import effective_filter
        user = current_user()
        if not can(user, "view_ops"):
            raise PreventUpdate
        rows, _res, _pub = _alerts(effective_filter(user, area, 0))
        return alert_cards(rows, user, severity or "all", status or "all")

    @app.callback(Output("ops-alert-modal", "is_open"), Output("ops-alert-modal-title", "children"),
                  Output("ops-alert-modal-body", "children"),
                  Input({"type": "ops-alert-open", "aid": ALL}, "n_clicks"), prevent_initial_call=True)
    def open_alert(clicks):
        user = current_user()
        if not ctx.triggered_id or not any(clicks or []) or not can(user, "view_alert_record"):
            raise PreventUpdate
        aid = ctx.triggered_id["aid"]
        partner = registry.get("alerts")
        res = partner.fetch("alert", alert_id=aid) if partner.configured() or partner.demo() else None
        rec = res.records[0] if res is not None and res.ok and res.records else None
        if rec is None:
            rec = next((a for a in registry.get("outlook_alerts").fetch("active").records if a["id"] == aid), None)
        if rec is None:
            return True, "Alert", html.P("This alert is no longer active.", className="dash-empty")
        from src.auth.store import get_store
        get_store().audit("alert_record_viewed", user["user_id"], aid)
        body = html.Dl(className="dash-facts", children=sum(
            ([html.Dt(k.replace("_", " ").capitalize()),
              html.Dd(", ".join(map(str, v)) if isinstance(v, list) else ("—" if v in (None, "") else str(v)))]
             for k, v in rec.items()), []))
        return True, rec.get("title", "Alert"), body
