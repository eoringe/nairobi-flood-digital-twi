"""
src.dashboard.ops_monitoring
============================
Sensor, rainfall and water-level screens of the operations console
(specification §2.3–§2.5, §6.3–§6.5; FR-003–FR-006, FR-015, FR-024;
UC-FVD-002–004).

    /ops/sensors       station list, health, last communication, search;
                       a station's readings over a chosen period, CSV export
    /ops/rainfall      hourly rainfall and 72-hour accumulation from the forecast
                       feed, intensity classes, the daily record since 2015 and
                       unusually wet days; gauges when the IoT platform connects
    /ops/water-levels  river levels against warning and critical thresholds,
                       rising/stable/falling trend, station comparison

Station data comes only from the IoT Sensor Data Platform. Until it is
connected these screens say so; in demo mode they show labelled demo data.
Missing readings are drawn as gaps and counted, never filled in.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.dashboard.ops_console import (EAT, PAGES, _icon, demo_tag, empty_state, fresh_badge, local_time, panel,
                                       tile)
from src.geo import boundaries
from src.integrations import base, registry

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

#: Hourly rainfall intensity (WMO classes). Configurable: replace with the
#: thresholds the partner hydrology team approves (spec §2.4).
HOURLY_INTENSITY = [(0.1, "Light", "#7dd3fc"), (2.5, "Moderate", "#38bdf8"), (10.0, "Heavy", "#f08a3c"),
                    (50.0, "Violent", "#ef4459")]
#: Daily totals (mm/day) used for "unusually wet": above this percentile of the
#: same calendar month in the 2015-2025 record.
UNUSUAL_PERCENTILE = 95
#: Water-level trend: change over the last 3 hours beyond this is rising/falling.
TREND_M = 0.05
HEALTH_LABELS = {"ok": ("Healthy", "ok"), "degraded": ("Degraded", "warn"), "offline": ("Offline", "down"),
                 "unknown": ("Unknown", "warn")}


def _style(fig: go.Figure, theme: str, height: int = 300) -> go.Figure:
    from src.dashboard.admin_dashboard import _style as admin_style
    fig = admin_style(fig, theme, height)
    fig.update_yaxes(tickformat=None)
    return fig


def _stations(flt: dict | None) -> base.SourceResult:
    r = registry.get("iot").fetch("stations")
    areas = (flt or {}).get("areas") or []
    if r.ok and areas:
        r.records = [s for s in r.records if boundaries.point_in(s["lat"], s["lon"], areas)]
    return r


def _health(s: dict) -> html.Span:
    label, state = HEALTH_LABELS.get(s.get("health") or "unknown", HEALTH_LABELS["unknown"])
    return html.Span([html.Span(className=f"dash-dot state-{state} me-2"), label], className="ops-state")


def _latest_text(s: dict) -> list:
    out = []
    for kind, r in (s.get("latest") or {}).items():
        v = r.get("value")
        above = kind == "water_level" and v is not None and s.get("warning_m") and v >= s["warning_m"]
        out.append(html.Div([html.Span(kind.replace("_", " ").capitalize() + ": ", className="dash-cell-muted"),
                             html.Strong("missing" if v is None else f"{v} {r.get('unit') or ''}",
                                         className="ops-above" if above else None),
                             html.Span(" above warning" if above else "", className="ops-above")]))
    return out or [html.Span("No readings", className="dash-cell-muted")]


def _readings(station_id: str, kind: str, start: datetime, end: datetime) -> base.SourceResult:
    return registry.get("iot").fetch("readings", station_id=station_id, kind=kind,
                                     start=start.isoformat(), end=end.isoformat())


def _series_figure(rows: list[dict], theme: str, station: dict | None, kind: str, title: str = "") -> go.Figure:
    """A reading series with gaps where data is missing and threshold lines for water levels."""
    xs = [base.as_utc(r["observed_at"]).astimezone(EAT) for r in rows]
    ys = [r["value"] for r in rows]                                      # None stays None: a gap
    fig = go.Figure()
    if kind == "rainfall":
        fig.add_bar(x=xs, y=ys, name="Rainfall (mm/h)", marker_color="#38bdf8")
    else:
        fig.add_scatter(x=xs, y=ys, mode="lines", name="Water level (m)", connectgaps=False,
                        line=dict(color="#38bdf8", width=2))
        for key, label, colour in (("warning_m", "Warning", "#f0b93f"), ("critical_m", "Critical", "#ef4459")):
            if station and station.get(key):
                fig.add_hline(y=station[key], line=dict(color=colour, dash="dash", width=1.5),
                              annotation_text=f"{label} {station[key]} m", annotation_position="top left")
    missing = sum(v is None for v in ys)
    fig = _style(fig, theme, 300)
    fig.update_layout(hovermode="x unified", showlegend=False,
                      title=dict(text=title + (f" · {missing} missing readings shown as gaps" if missing else ""),
                                 font=dict(size=12), x=0, y=0.98))
    return fig


def trend(rows: list[dict]) -> str:
    """Rising, falling or stable over the last 3 hours of valid readings."""
    valid = [(base.as_utc(r["observed_at"]), r["value"]) for r in rows if r.get("value") is not None]
    if len(valid) < 2:
        return "Not enough data"
    end_t, end_v = valid[-1]
    past = [v for t, v in valid if end_t - t >= timedelta(hours=3)]
    ref = past[-1] if past else valid[0][1]
    change = end_v - ref
    return "Rising" if change > TREND_M else ("Falling" if change < -TREND_M else "Stable")


# ---------------------------------------------------------------- sensors --
def sensors_page(user: dict, theme: str, flt: dict | None = None) -> list:
    r = _stations(flt)
    if not r.ok:
        return [empty_state(r, "Monitoring stations, readings and sensor health"),
                _next_steps_panel()]
    st = r.records
    active = [s for s in st if s.get("status") == "active"]
    by_health = {h: sum(1 for s in st if s.get("health") == h) for h in ("ok", "degraded", "offline")}
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Stations", len(st), "broadcast-pin", "info", f"{len(active)} active · {len(st) - len(active)} inactive", r),
        tile("Healthy", by_health["ok"], "check-circle", "ok", "reporting normally", r),
        tile("Degraded", by_health["degraded"], "exclamation-circle", "warn" if by_health["degraded"] else "ok",
             "readings suspect or intermittent", r),
        tile("Offline", by_health["offline"], "wifi-off", "danger" if by_health["offline"] else "ok",
             "no recent communication", r),
    ])
    toolbar = html.Div(className="dash-toolbar", children=[
        html.Div(className="dash-search", children=[
            _icon("search"), dcc.Input(id="ops-station-search", type="search", debounce=True,
                                       placeholder="Search stations by name or id", className="dash-search-input")]),
        dcc.Dropdown(id="ops-station-type", value="all", clearable=False, searchable=False,
                     className="dash-select dash-filter",
                     options=[{"label": "All sensor types", "value": "all"},
                              {"label": "Rainfall", "value": "rainfall"},
                              {"label": "Water level", "value": "water_level"}]),
    ])
    return [kpis,
            panel("Stations", toolbar, html.Div(id="ops-station-table", children=station_table(st, r)),
                  icon="broadcast-pin", actions=demo_tag(r)),
            panel("Station readings", _range_controls("ops-station"), html.Div(id="ops-station-detail",
                  children=html.P("Choose a station above to see its readings.", className="dash-empty")),
                  icon="graph-up")]


def station_table(stations: list[dict], r: base.SourceResult, query: str | None = None,
                  kind: str | None = None) -> html.Div:
    q = (query or "").strip().lower()
    rows = [s for s in stations if (not q or q in s["name"].lower() or q in s["id"].lower())
            and (not kind or kind == "all" or kind in s.get("types", []))]
    if not rows:
        return html.P("No stations match.", className="dash-empty")
    stale_after = registry.get("iot").stale_after
    live_within = registry.get("iot").live_within
    return html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Station", "Type", "Health", "Last communication", "Latest", "")])),
        html.Tbody([html.Tr([
            html.Td([html.Div(s["name"], className="dash-cell-strong"), html.Div(s["id"], className="dash-mono")]),
            html.Td(", ".join(t.replace("_", " ") for t in s.get("types", []))),
            html.Td(_health(s)),
            html.Td([local_time(s.get("last_communication")), html.Div(fresh_badge(
                base.freshness(s.get("last_communication"), live_within, stale_after), s.get("last_communication")))]),
            html.Td(_latest_text(s)),
            html.Td(dbc.Button("Readings", id={"type": "ops-station-btn", "sid": s["id"]}, size="sm",
                               className="dash-btn-ghost", color="dash")),
        ], className="is-disabled" if s.get("status") == "inactive" else None) for s in rows]),
    ]))


def _range_controls(prefix: str, multi: bool = False) -> html.Div:
    today = datetime.now(EAT).date()
    return html.Div(className="dash-toolbar", children=[
        dcc.Store(id=f"{prefix}-selected"),
        dcc.DatePickerRange(id=f"{prefix}-range", start_date=today - timedelta(days=3), end_date=today,
                            max_date_allowed=today, display_format="D MMM YYYY", className="ops-range"),
        dcc.RadioItems(id=f"{prefix}-kind", options=[{"label": " Water level", "value": "water_level"},
                                                     {"label": " Rainfall", "value": "rainfall"}],
                       value="water_level", inline=True, className="dash-radio ops-kind",
                       inputStyle={"marginRight": "4px", "marginLeft": "12px"}) if not multi else None,
        html.Button([_icon("download", "me-1"), "Export CSV"], id=f"{prefix}-export", n_clicks=0,
                    className="dash-btn-ghost btn") if not multi else None,
        dcc.Download(id=f"{prefix}-download") if not multi else None,
    ])


def _next_steps_panel() -> html.Section:
    return panel("What this screen will show", html.Ul(className="ops-setup", children=[
        html.Li("Every registered rain and river gauge, with health and last communication time."),
        html.Li("Each station's readings over a chosen period, with warning thresholds and gaps for missing data."),
        html.Li("Search by station, filter by sensor type and area, export readings as CSV."),
        html.Li(["Connect by setting ", html.Code("IOT_API_URL"), " and ", html.Code("IOT_API_KEY"),
                 "; TAHMO gauges, once approved, arrive through the same interface."]),
    ]), icon="info-circle")


def _period(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    today = datetime.now(EAT).date()
    s = date.fromisoformat(start[:10]) if start else today - timedelta(days=3)
    e = date.fromisoformat(end[:10]) if end else today
    e_dt = min(datetime.combine(e, datetime.max.time(), EAT), datetime.now(EAT))
    return datetime.combine(s, datetime.min.time(), EAT), e_dt


def station_detail(sid: str, kind: str, start: str | None, end: str | None, theme: str) -> list:
    st = next((s for s in registry.get("iot").fetch("stations").records if s["id"] == sid), None)
    if st is None:
        return [html.P("That station is no longer listed.", className="dash-empty")]
    if kind not in st.get("types", []):
        kind = st["types"][0]
    t0, t1 = _period(start, end)
    r = _readings(sid, kind, t0, t1)
    if not r.ok:
        return [empty_state(r, "Readings")]
    valid = [x["value"] for x in r.records if x["value"] is not None]
    facts = [html.Span(["Readings ", html.Strong(len(r.records))], className="dash-chip"),
             html.Span(["Missing ", html.Strong(len(r.records) - len(valid))], className="dash-chip")]
    if valid:
        unit = "mm" if kind == "rainfall" else "m"
        facts += ([html.Span(["Total ", html.Strong(f"{sum(valid):.1f} mm")], className="dash-chip")] if kind == "rainfall"
                  else [html.Span(["Peak ", html.Strong(f"{max(valid):.2f} {unit}")], className="dash-chip"),
                        html.Span(["Trend ", html.Strong(trend(r.records))], className="dash-chip")])
    return [html.Div([html.Strong(st["name"]), " · ", kind.replace("_", " "), " · ",
                      f"{t0:%d %b} to {t1:%d %b %Y}", demo_tag(r)], className="mb-2"),
            html.Div(facts, className="dash-chips mb-2"),
            dcc.Graph(figure=_series_figure(r.records, theme, st, kind), config=GRAPH_CONFIG, style={"height": "300px"})]


# --------------------------------------------------------------- rainfall --
def _chirps_daily() -> dict[date, float]:
    from src.forecast.flood_chance import _chirps
    return _chirps()


def unusual_days(daily: dict[date, float], start: date, end: date) -> list[tuple[date, float, float]]:
    """(day, mm, threshold) for days above the month's UNUSUAL_PERCENTILE in the full record."""
    import numpy as np
    by_month: dict[int, list[float]] = {}
    for d, v in daily.items():
        if v > 0:
            by_month.setdefault(d.month, []).append(v)
    thr = {m: float(np.percentile(v, UNUSUAL_PERCENTILE)) for m, v in by_month.items() if len(v) >= 20}
    return [(d, v, thr[d.month]) for d, v in sorted(daily.items())
            if start <= d <= end and d.month in thr and v > thr[d.month]]


def rainfall_page(user: dict, theme: str, flt: dict | None = None) -> list:
    feed = registry.get("forecast").fetch("hourly")
    stations = _stations(flt)
    rain_stations = [s for s in stations.records if "rainfall" in s.get("types", [])] if stations.ok else []
    tiles = []
    if feed.ok and feed.records:
        past = [r for r in feed.records if not r["forecast"]]
        nxt = [r for r in feed.records if r["forecast"]]
        last24 = sum(r["rain_mm"] for r in past[-24:])
        tiles = [
            tile("Last 24 hours", f"{last24:.1f} mm", "cloud-rain", "info", "Nairobi, forecast feed", feed,
                 source="Open-Meteo"),
            tile("72-hour accumulation", f"{past[-1]['rain_72h_mm']:.0f} mm" if past else "—", "moisture", "info",
                 "bias-corrected to the CHIRPS scale", feed, source="Open-Meteo"),
            tile("Next 12 hours", f"{sum(r['rain_mm'] for r in nxt):.1f} mm", "cloud-drizzle", "warn"
                 if sum(r['rain_mm'] for r in nxt) >= 10 else "ok", "forecast", feed, source="Open-Meteo"),
            tile("Rain gauges", len(rain_stations) if stations.ok else "Not connected", "broadcast-pin",
                 "info" if stations.ok else "neutral",
                 "reporting stations" if stations.ok else "IoT Sensor Data Platform", stations),
        ]
    controls = html.Div(className="dash-toolbar", children=[
        dcc.Dropdown(id="ops-rain-days", value=7, clearable=False, searchable=False, className="dash-select dash-filter",
                     options=[{"label": f"Last {d} days", "value": d} for d in (2, 7, 14, 30)]),
    ])
    hist = html.Div(className="dash-toolbar", children=[
        dcc.Dropdown(id="ops-rain-season", value="long", clearable=False, searchable=False,
                     className="dash-select dash-filter",
                     options=[{"label": "Long rains (Mar-May)", "value": "long"},
                              {"label": "Short rains (Oct-Dec)", "value": "short"},
                              {"label": "Whole year", "value": "year"}]),
        dcc.Dropdown(id="ops-rain-years", multi=True, value=[2024, 2023], className="dash-select ops-years",
                     options=[{"label": str(y), "value": y} for y in range(2025, 2014, -1)],
                     placeholder="Compare years"),
    ])
    gauge_view = (panel("Rain gauges", station_compare_controls("rain", rain_stations),
                        html.Div(id="ops-rain-gauges"), icon="broadcast-pin", actions=demo_tag(stations))
                  if rain_stations else panel("Rain gauges", empty_state(stations, "Rainfall by station")
                                              if not stations.ok else html.P("No rain gauges in this area.",
                                                                             className="dash-empty"),
                                              icon="broadcast-pin"))
    return [html.Div(tiles, className="dash-kpis cols-4"),
            panel("Hourly rainfall and 72-hour accumulation, Nairobi", controls,
                  dcc.Graph(id="ops-rain-hourly", config=GRAPH_CONFIG, style={"height": "320px"}), html.Div(id="ops-rain-classes"),
                  icon="cloud-rain", actions=html.Span("Observed hours and forecast hours", className="ops-model-label")),
            panel("Seasonal comparison (CHIRPS daily, 2015-2026)", hist,
                  dcc.Graph(id="ops-rain-seasonal", config=GRAPH_CONFIG, style={"height": "320px"}), html.Div(id="ops-rain-unusual"),
                  icon="calendar3"),
            gauge_view]


def hourly_rain_figure(days: int, theme: str) -> tuple[go.Figure, list]:
    feed = registry.get("forecast").fetch("hourly")
    if not feed.ok:
        return go.Figure(), [empty_state(feed, "Hourly rainfall")]
    # The outlook keeps every fetched hour: 31 days back and 2 days ahead.
    from src.dashboard.callbacks import predictor
    from src.forecast.nowcast import _acc72, get_nowcast
    from src.forecast.rain_bias_correction import correct_3day_total
    nc = get_nowcast("live", predictor)
    series, as_of = nc.series, nc.as_of
    times = sorted(t for t in series if as_of - timedelta(days=days) <= t <= as_of + timedelta(hours=48))
    rain = [series[t] for t in times]
    acc = [correct_3day_total(_acc72(series, t)[0]) for t in times]
    colours = [next((c for thr, _l, c in reversed(HOURLY_INTENSITY) if v >= thr), "#64748b") for v in rain]
    fig = go.Figure()
    fig.add_bar(x=times, y=rain, name="Rain per hour (mm)", marker_color=colours, opacity=0.9)
    fig.add_scatter(x=times, y=acc, name="72-hour total (mm, CHIRPS scale)", yaxis="y2", mode="lines",
                    line=dict(color="#e2e8f0", width=1.6))
    fig.add_vline(x=as_of, line=dict(color="#94a3b8", dash="dot"))
    fig.add_annotation(x=as_of, y=1, yref="paper", text="now", showarrow=False, font=dict(size=10))
    fig = _style(fig, theme, 320)
    fig.update_layout(yaxis=dict(title="mm / hour"), yaxis2=dict(title="72 h (mm)", overlaying="y", side="right",
                                                                  showgrid=False, rangemode="tozero"))
    counts = {label: sum(1 for v in rain if lo <= v < (HOURLY_INTENSITY[i + 1][0] if i + 1 < len(HOURLY_INTENSITY) else 1e9))
              for i, (lo, label, _c) in enumerate(HOURLY_INTENSITY)}
    key = html.Div(className="dash-chips mt-2", children=[
        html.Span([html.Span(style={"display": "inline-block", "width": "9px", "height": "9px", "borderRadius": "2px",
                                    "background": c, "marginRight": "5px"}), f"{label} ≥{lo:g} mm/h: {counts[label]} h"],
                  className="dash-chip") for lo, label, c in HOURLY_INTENSITY] + [
        html.Span("WMO intensity classes; replace with the approved thresholds when agreed.", className="dash-cell-muted")])
    return fig, [key]


SEASONS = {"long": (3, 5), "short": (10, 12), "year": (1, 12)}


def seasonal_figure(season: str, years: list[int], theme: str) -> tuple[go.Figure, list]:
    import numpy as np
    daily = _chirps_daily()
    m0, m1 = SEASONS.get(season, SEASONS["long"])
    fig = go.Figure()
    # Long-term mean cumulative curve, by day of season.
    def season_days(y: int) -> list[date]:
        d, out = date(y, m0, 1), []
        while d.month <= m1 and d.year == y:
            out.append(d)
            d += timedelta(days=1)
        return out
    all_years = sorted({d.year for d in daily})
    complete = [y for y in all_years if all(d in daily for d in season_days(y))]
    if complete:
        mat = np.array([np.cumsum([daily[d] for d in season_days(y)]) for y in complete])
        n = mat.shape[1]
        x = list(range(1, n + 1))
        fig.add_scatter(x=x + x[::-1], y=list(np.percentile(mat, 90, axis=0)) + list(np.percentile(mat, 10, axis=0))[::-1],
                        fill="toself", fillcolor="rgba(148,163,184,0.15)", line=dict(width=0), name="10-90% of years",
                        hoverinfo="skip")
        fig.add_scatter(x=x, y=mat.mean(axis=0), name=f"Mean {complete[0]}-{complete[-1]}",
                        line=dict(color="#94a3b8", dash="dash"))
    palette = ["#38bdf8", "#f0b93f", "#ef4459", "#a78bfa", "#34d399"]
    unusual = []
    for i, y in enumerate(sorted(years or [], reverse=True)[:5]):
        days = [d for d in season_days(y) if d in daily]
        if not days:
            continue
        fig.add_scatter(x=list(range(1, len(days) + 1)), y=np.cumsum([daily[d] for d in days]), name=str(y),
                        line=dict(color=palette[i % len(palette)], width=2.2))
        unusual += unusual_days(daily, days[0], days[-1])
    fig = _style(fig, theme, 320)
    fig.update_layout(xaxis=dict(title="day of season"), yaxis=dict(title="cumulative rainfall (mm)"))
    note = [html.Div(className="mt-2", children=[
        html.Strong(f"Unusually wet days in the selected years ({len(unusual)}): "),
        ", ".join(f"{d:%d %b %Y} ({v:.0f} mm)" for d, v, _t in unusual[:12]) or "none",
        " and more" if len(unusual) > 12 else "",
        html.Div(f"A day counts as unusual above the {UNUSUAL_PERCENTILE}th percentile of rainy days in the same "
                 "month, 2015-2025. CHIRPS city-average rainfall; local storms can be heavier.",
                 className="dash-cell-muted")])]
    return fig, note


# ------------------------------------------------------------ water levels --
def station_compare_controls(prefix: str, stations: list[dict]) -> html.Div:
    today = datetime.now(EAT).date()
    return html.Div(className="dash-toolbar", children=[
        dcc.Dropdown(id=f"ops-{prefix}-stations", multi=True, value=[s["id"] for s in stations[:2]],
                     options=[{"label": s["name"], "value": s["id"]} for s in stations],
                     placeholder="Choose stations to compare", className="dash-select ops-years"),
        dcc.DatePickerRange(id=f"ops-{prefix}-range", start_date=today - timedelta(days=3), end_date=today,
                            max_date_allowed=today, display_format="D MMM YYYY", className="ops-range"),
    ])


def water_levels_page(user: dict, theme: str, flt: dict | None = None) -> list:
    r = _stations(flt)
    if not r.ok:
        return [empty_state(r, "River levels, thresholds and trends"), _next_steps_panel()]
    wl = [s for s in r.records if "water_level" in s.get("types", [])]
    if not wl:
        return [html.P("No water-level stations in this area.", className="dash-empty")]
    now = datetime.now(EAT)
    rows, above_w, above_c = [], 0, 0
    for s in wl:
        rd = _readings(s["id"], "water_level", now - timedelta(hours=12), now)
        latest = next((x for x in reversed(rd.records) if x["value"] is not None), None) if rd.ok else None
        v = latest["value"] if latest else None
        old = latest is not None and base.freshness(latest["observed_at"], registry.get("iot").live_within,
                                                    registry.get("iot").stale_after) == base.STALE
        state = ("stale" if old else
                 "critical" if v is not None and s.get("critical_m") and v >= s["critical_m"] else
                 "warning" if v is not None and s.get("warning_m") and v >= s["warning_m"] else
                 "normal" if v is not None else "no data")
        above_w += state in ("warning", "critical")
        above_c += state == "critical"
        rows.append((s, v, latest, state, "Not current" if old else
                     (trend(rd.records) if rd.ok else "Not enough data")))
    kpis = html.Div(className="dash-kpis cols-4", children=[
        tile("Water-level stations", len(wl), "water", "info",
             f"{sum(1 for x in rows if x[1] is not None and x[3] != 'stale')} with a current reading", r),
        tile("Above warning", above_w, "exclamation-triangle", "warn" if above_w else "ok", "at or over the warning level", r),
        tile("Above critical", above_c, "exclamation-octagon", "danger" if above_c else "ok", "at or over critical", r),
        tile("Rising", sum(1 for x in rows if x[4] == "Rising"), "arrow-up-right", "warn", "over the last 3 hours", r),
    ])
    state_tone = {"critical": 3, "warning": 1, "normal": 0, "no data": 0, "stale": 0}
    state_text = {"stale": "Stale reading"}
    table = html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Station", "Level", "Status", "Warning / critical", "Trend (3 h)",
                                                 "Observed", "Health")])),
        html.Tbody([html.Tr([
            html.Td(s["name"], className="dash-cell-strong"),
            html.Td(f"{v:.2f} m" if v is not None else "missing"),
            html.Td(html.Span(state_text.get(state, state.title()),
                              className=f"ops-risk lvl-{state_tone[state]}" + (" is-stale" if state == "stale" else ""))),
            html.Td(f"{s.get('warning_m') or '—'} / {s.get('critical_m') or '—'} m"),
            html.Td([_icon({"Rising": "arrow-up-right", "Falling": "arrow-down-right", "Stable": "arrow-right"}
                           .get(tr, "dash"), "me-1"), tr]),
            html.Td(local_time(latest["observed_at"]) if latest else "—"),
            html.Td(_health(s)),
        ]) for s, v, latest, state, tr in rows]),
    ]))
    return [kpis, panel("River levels", table, icon="water", actions=demo_tag(r)),
            panel("Compare stations", station_compare_controls("wl", wl), dcc.Graph(id="ops-wl-chart", config=GRAPH_CONFIG, style={"height": "320px"}),
                  icon="graph-up")]


def compare_figure(ids: list[str], kind: str, start: str | None, end: str | None, theme: str) -> go.Figure:
    stations = {s["id"]: s for s in registry.get("iot").fetch("stations").records}
    t0, t1 = _period(start, end)
    fig = go.Figure()
    palette = ["#38bdf8", "#f0b93f", "#a78bfa", "#34d399", "#f472b6"]
    for i, sid in enumerate((ids or [])[:5]):
        r = _readings(sid, kind, t0, t1)
        if not r.ok or sid not in stations:
            continue
        xs = [base.as_utc(x["observed_at"]).astimezone(EAT) for x in r.records]
        ys = [x["value"] for x in r.records]
        name = stations[sid]["name"]
        if kind == "rainfall":
            fig.add_bar(x=xs, y=ys, name=name, marker_color=palette[i % 5])
        else:
            fig.add_scatter(x=xs, y=ys, name=name, mode="lines", connectgaps=False,
                            line=dict(color=palette[i % 5], width=2))
            if len(ids) == 1:
                s = stations[sid]
                for key, colour in (("warning_m", "#f0b93f"), ("critical_m", "#ef4459")):
                    if s.get(key):
                        fig.add_hline(y=s[key], line=dict(color=colour, dash="dash"))
    fig = _style(fig, theme, 320)
    fig.update_layout(yaxis=dict(title="mm / hour" if kind == "rainfall" else "metres"))
    return fig


# ------------------------------------------------------------- registration --
_status = PAGES.pop("/ops/status")
PAGES.update({
    "/ops/sensors": ("Sensors", "broadcast-pin", "Sensor monitoring",
                     "Stations, sensor health, last communication and readings.", "view_sensors", sensors_page, True),
    "/ops/rainfall": ("Rainfall", "cloud-rain", "Rainfall analytics",
                      "Accumulated rainfall, intensity, seasonal comparison and unusually wet days.", "view_sensors",
                      rainfall_page, True),
    "/ops/water-levels": ("Water levels", "water", "Water-level analytics",
                          "River levels against warning and critical thresholds, with trends.", "view_sensors",
                          water_levels_page, True),
})
PAGES["/ops/status"] = _status


def register_monitoring_callbacks(app) -> None:
    @app.callback(Output("ops-station-table", "children"),
                  Input("ops-station-search", "value"), Input("ops-station-type", "value"),
                  State("ops-area", "value"), prevent_initial_call=True)
    def filter_stations(query, kind, area):
        from src.dashboard.ops_console import effective_filter
        user = current_user()
        if not can(user, "view_sensors"):
            raise PreventUpdate
        r = _stations(effective_filter(user, area, 0))
        return station_table(r.records, r, query, kind) if r.ok else no_update

    @app.callback(Output("ops-station-selected", "data"),
                  Input({"type": "ops-station-btn", "sid": ALL}, "n_clicks"), prevent_initial_call=True)
    def pick_station(clicks):
        if not ctx.triggered_id or not any(clicks or []):
            raise PreventUpdate
        return ctx.triggered_id["sid"]

    @app.callback(Output("ops-station-detail", "children"),
                  Input("ops-station-selected", "data"), Input("ops-station-range", "start_date"),
                  Input("ops-station-range", "end_date"), Input("ops-station-kind", "value"),
                  Input("ops-updated", "children"), State("resolved-theme-store", "data"))
    def show_station(sid, start, end, kind, _updated, theme):
        if not sid or not can(current_user(), "view_sensors"):
            raise PreventUpdate
        return station_detail(sid, kind or "water_level", start, end, theme or "dark")

    @app.callback(Output("ops-station-download", "data"), Input("ops-station-export", "n_clicks"),
                  State("ops-station-selected", "data"), State("ops-station-range", "start_date"),
                  State("ops-station-range", "end_date"), State("ops-station-kind", "value"),
                  prevent_initial_call=True)
    def export_station(n, sid, start, end, kind):
        user = current_user()
        if not n or not sid or not can(user, "export_readings"):
            raise PreventUpdate
        t0, t1 = _period(start, end)
        r = _readings(sid, kind or "water_level", t0, t1)
        if not r.ok:
            raise PreventUpdate
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["station_id", "kind", "observed_at", "value", "unit", "quality", "source"])
        src = "DEMO DATA - not measurements" if r.demo else "IoT Sensor Data Platform"
        for x in r.records:
            w.writerow([x["station_id"], x["kind"], x["observed_at"], "" if x["value"] is None else x["value"],
                        x.get("unit") or "", x.get("quality") or "", src])
        from src.auth.store import get_store
        get_store().audit("readings_exported", user["user_id"], f"{sid} {kind} {t0:%Y-%m-%d}..{t1:%Y-%m-%d}")
        return dcc.send_string(buf.getvalue(), f"{sid}_{kind}_{t0:%Y%m%d}-{t1:%Y%m%d}.csv")

    @app.callback(Output("ops-rain-hourly", "figure"), Output("ops-rain-classes", "children"),
                  Input("ops-rain-days", "value"), Input("ops-updated", "children"),
                  State("resolved-theme-store", "data"))
    def rain_hourly(days, _u, theme):
        if not can(current_user(), "view_sensors"):
            raise PreventUpdate
        return hourly_rain_figure(int(days or 7), theme or "dark")

    @app.callback(Output("ops-rain-seasonal", "figure"), Output("ops-rain-unusual", "children"),
                  Input("ops-rain-season", "value"), Input("ops-rain-years", "value"),
                  State("resolved-theme-store", "data"))
    def rain_seasonal(season, years, theme):
        if not can(current_user(), "view_sensors"):
            raise PreventUpdate
        return seasonal_figure(season or "long", years or [], theme or "dark")

    @app.callback(Output("ops-rain-gauges", "children"),
                  Input("ops-rain-stations", "value"), Input("ops-rain-range", "start_date"),
                  Input("ops-rain-range", "end_date"), State("resolved-theme-store", "data"))
    def rain_gauges(ids, start, end, theme):
        if not can(current_user(), "view_sensors"):
            raise PreventUpdate
        return dcc.Graph(figure=compare_figure(ids, "rainfall", start, end, theme or "dark"), config=GRAPH_CONFIG,
                         style={"height": "320px"})

    @app.callback(Output("ops-wl-chart", "figure"),
                  Input("ops-wl-stations", "value"), Input("ops-wl-range", "start_date"),
                  Input("ops-wl-range", "end_date"), State("resolved-theme-store", "data"))
    def wl_chart(ids, start, end, theme):
        if not can(current_user(), "view_sensors"):
            raise PreventUpdate
        return compare_figure(ids, "water_level", start, end, theme or "dark")
