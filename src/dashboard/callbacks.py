"""
src.dashboard.callbacks
========================
Nairobi Urban Flood Digital Twin — Interactive Callbacks

FLOW
----
    mode / forecast source ──> nowcast-store ──┐
    rain slider (what-if) ─────────────────────┼──> update_simulation ──> map, KPIs, warnings
    hour slider (live) ────────────────────────┘            │
                                                   scenario-key-store
                                                            │
    start / destination / map pins ──────────────> plan_route ──> route-store ──> map layers

The map document is built once per view (camera target + theme). Everything
else - flood polygons and routes - is patched into the running map through
postMessage, so zoom and pan survive updates.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html, no_update, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from loguru import logger

from src.models.predict import NAIROBI_LOCATIONS
from src.models.predict_v2 import FloodPredictor
from src.dashboard.components.map_3d import (
    ROUTE_LAYER_IDS, summarise_flooded_regions, create_3d_digital_twin_deck,
    get_deck_html_with_embedded_legend,
)
from src.forecast.nowcast import SOURCES, get_nowcast
from src.routing.flood_router import (
    FloodAwareRouter, LEVEL_NAMES, level_of, place_name_at, smooth_for_routing,
)
from src.routing.geocoder import LocationSearch, decode_value
from src.persistence import scenario_store

# Model B on drainage labels: test F1 0.937 on held-out storm seasons.
predictor = FloodPredictor()
scenario_store.init_db()

RISK_COLORS = {
    "CRITICAL": {"bg": "#ef4459", "text": "CRITICAL", "badge": "danger", "dot": "dot-critical"},
    "HIGH": {"bg": "#f08a3c", "text": "HIGH", "badge": "warning", "dot": "dot-high"},
    "MODERATE": {"bg": "#f0b93f", "text": "MODERATE", "badge": "warning", "dot": "dot-moderate"},
    "LOW": {"bg": "#6fe0a8", "text": "LOW", "badge": "success", "dot": "dot-low"},
    "SAFE": {"bg": "#35d495", "text": "SAFE", "badge": "secondary", "dot": "dot-safe"},
}
LEVEL_RANK = {"CLEAR": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

#: Plot colours per theme, matching the tokens in assets/custom.css.
CHART_THEME = {
    "dark": {"bg": "#10161f", "text": "#93a2b3", "grid": "#202b38", "accent": "#35c2d1",
             "past": "#3a4a5c", "marker_edge": "#10161f"},
    "light": {"bg": "#ffffff", "text": "#46576a", "grid": "#e3e9ee", "accent": "#0b8795",
              "past": "#c2ced7", "marker_edge": "#ffffff"},
}
LEVEL_COLORS = {0: "#35d495", 1: "#f0b93f", 2: "#f08a3c", 3: "#ef4459"}

#: Most flooded places to list in the side panel.
MAX_REGION_CARDS = 12
MAX_ALERTS = 8

#: County-average density, used only if the WorldPop grid is missing.
NAIROBI_POP_PER_KM2 = 6300


def _load_population():
    """
    People per grid cell from WorldPop 2025 (src/ingestion/build_population_grid.py).
    Exposure was previously flooded km2 x the county average, which understates
    it along dense river corridors, exactly where flooding concentrates.
    """
    f = Path("data/processed/arrays/population_grid_worldpop2025.npy")
    try:
        grid = np.load(f)
        logger.info(f"Population: WorldPop 2025 grid, {grid.sum():,.0f} people in the model area")
        return grid
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"Population grid unavailable ({exc}); falling back to the county-average "
                       f"density. Run `python -m src.ingestion.build_population_grid`.")
        return None


POP_GRID = _load_population()
CELL_KM2 = 0.067 * 0.079

#: Route colours. Google-Maps-like blue for the recommended route, grey for the
#: fastest-but-flooded alternative; both read on the dark and light basemaps.
ROUTE_SAFE_RGBA = [26, 115, 232, 255]
ROUTE_CASING_RGBA = [255, 255, 255, 235]
ROUTE_FASTEST_RGBA = [120, 132, 145, 210]
PIN_START_RGBA = [16, 146, 95, 255]
PIN_END_RGBA = [207, 38, 64, 255]
#: Start or destination further than this from the road network is refused.
MAX_SNAP_M = 600


# ----------------------------------------------------------------- places --
def _load_place_coords() -> dict[str, tuple[float, float]]:
    try:
        places = json.loads(Path("data/processed/nairobi_places.json").read_text(encoding="utf-8"))["places"]
    except Exception:                                      # noqa: BLE001
        return {}
    out: dict[str, tuple[float, float]] = {}
    for p in places:
        out.setdefault(p["name"], (p["lat"], p["lon"]))
    return out


PLACE_COORDS = _load_place_coords()
location_search = LocationSearch()


def _resolve_point(value: str | None) -> tuple[float, float] | None:
    """
    Trip-planner value -> (lat, lon). Values are 'loc:<lat>,<lon>|<name>' (search
    results), 'place:<name>' (the suburb list) or 'pin:<lat>,<lon>' (map clicks).
    """
    if not value:
        return None
    loc = decode_value(value)
    if loc:
        return loc[0], loc[1]
    kind, _, rest = value.partition(":")
    if kind == "place":
        return PLACE_COORDS.get(rest)
    if kind == "pin":
        try:
            lat, lon = (float(x) for x in rest.split(","))
            return lat, lon
        except ValueError:
            return None
    return None


def _point_label(value: str | None) -> str:
    if not value:
        return ""
    loc = decode_value(value)
    if loc:
        return loc[2]
    kind, _, rest = value.partition(":")
    if kind == "place":
        return rest
    pt = _resolve_point(value)
    return f"pin near {place_name_at(*pt)}" if pt else "pin"


# ----------------------------------------------------- background loading --
_router: FloodAwareRouter | None = None
_router_error: str | None = None
_router_lock = threading.Lock()


def get_router() -> FloodAwareRouter | None:
    """The road graph, or None while it is still loading in the background."""
    return _router


def _load_router() -> None:
    global _router, _router_error
    with _router_lock:
        if _router is not None:
            return
        try:
            _router = FloodAwareRouter()
        except Exception as exc:                            # noqa: BLE001
            _router_error = str(exc)
            logger.error(f"Road network unavailable: {exc}")


def _warm_up() -> None:
    """
    Load the road graph and pre-compute the model runs the interface will ask
    for first. One CPU inference costs ~1.5 s, so doing this after the page is
    open would make the first slider moves and replays feel stuck.
    """
    _load_router()
    from src.dashboard.components.map_3d import _building_rows
    for theme in ("dark", "light"):
        _building_rows(theme)
    for source in SOURCES:
        try:
            get_nowcast(source, predictor)
        except Exception as exc:                            # noqa: BLE001
            logger.warning(f"Warm-up: outlook {source} failed: {exc}")
    for mm in list(range(5, 151, 5)) + list(range(160, 201, 10)):
        predictor.probability_for(float(mm))
    logger.info("Warm-up complete: road network, outlooks and what-if scenarios cached.")


def start_background_warmup() -> None:
    threading.Thread(target=_warm_up, name="twin-warmup", daemon=True).start()


# ------------------------------------------------------------- scenarios --
def _scenario(mode: str, nowcast: dict | None, hour: int | None, rain_slider: float | None):
    """
    The flood field currently selected, as (key, rainfall_mm, probability, date).

    Live: the 72-hour rainfall total at the chosen outlook hour. What-if: the
    slider total. Probability is None when the live forecast is unavailable.
    """
    if mode == "LIVE":
        if not nowcast or nowcast.get("error") or not nowcast.get("acc72_mm"):
            return "LIVE|unavailable", None, None, None
        h = int(min(max(hour or 0, 0), len(nowcast["acc72_mm"]) - 1))
        rain = float(nowcast["acc72_mm"][h])
        d = datetime.fromisoformat(nowcast["as_of"]).date()
        key = f"LIVE|{nowcast['source']}|{nowcast['as_of']}|{h}"
        return key, rain, predictor.probability_for(rain, scenario_date=d), d
    rain = float(rain_slider or 40.0)
    return f"WHATIF|{round(rain)}", rain, predictor.probability_for(rain), date.today()


# -------------------------------------------------------------- renderers --
def _build_region_risk_cards(region_risks: dict, filter_mode: str = "ALL", selected_region: str | None = None) -> list:
    """Build clickable hotspot cards with html.Div wrapper for pattern matching."""
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "SAFE": 4}
    sorted_regions = sorted(region_risks.items(), key=lambda x: severity_order.get(x[1]["risk_level"], 5))[:MAX_REGION_CARDS]

    cards = []
    for reg_name, rr in sorted_regions:
        risk_level = rr["risk_level"]
        if filter_mode == "HIGH_ONLY" and risk_level in ["SAFE", "LOW", "MODERATE"]:
            continue

        risk_info = RISK_COLORS.get(risk_level, RISK_COLORS["SAFE"])
        is_selected = (reg_name == selected_region)
        cards.append(html.Div(
            id={"type": "region-card", "index": reg_name},
            n_clicks=0,
            className="mb-2",
            children=[
                html.Div(
                    className=f"twin-region-card{' is-selected' if is_selected else ''}",
                    style={"borderLeftColor": risk_info["bg"]},
                    children=[
                        dbc.Row(align="center", children=[
                            dbc.Col(width=7, children=[
                                html.Div([html.Span(className=f"status-dot {risk_info['dot']} me-2"), reg_name],
                                         className="twin-region-name"),
                                html.Span(
                                    f"{rr['peak_probability_pct']}% peak probability · {rr['flooded_pct']}% of area",
                                    className="twin-region-meta",
                                ),
                            ]),
                            dbc.Col(width=5, className="text-end", children=[
                                dbc.Badge(risk_info["text"], color=risk_info["badge"]),
                            ]),
                        ]),
                    ],
                )
            ]
        ))

    if not cards:
        msg = ("No flooding on the map for this scenario." if not region_risks
               else "No places at high or critical risk in this scenario.")
        cards.append(html.P(msg, className="twin-empty-state is-good"))
    return cards


def _build_alert_log(region_risks: dict) -> list:
    """What-if alerts: one line per CRITICAL/HIGH place in the scenario."""
    severity_order = {"CRITICAL": 0, "HIGH": 1}
    alerts = sorted(
        (item for item in region_risks.items() if item[1]["risk_level"] in severity_order),
        key=lambda x: severity_order[x[1]["risk_level"]],
    )[:MAX_ALERTS]

    if not alerts:
        return [html.P("No place exceeds moderate risk in this scenario.", className="twin-empty-state is-good")]

    rows = []
    for reg_name, rr in alerts:
        risk_info = RISK_COLORS[rr["risk_level"]]
        rows.append(html.Div(
            className="twin-alert-row",
            children=[
                html.Span([html.Span(className=f"status-dot {risk_info['dot']}"), reg_name],
                          className="twin-alert-label", style={"color": risk_info["bg"]}),
                html.Span(f"{rr['peak_probability_pct']}% · {rr['flooded_pct']}%", className="twin-mono-meta"),
            ],
        ))
    return rows


def _build_timed_warnings(nowcast: dict) -> list:
    """Live warnings, each with when it starts relative to the outlook time."""
    if nowcast.get("error"):
        return [html.P(nowcast["error"], className="twin-empty-state")]
    warnings = nowcast.get("warnings") or []
    if not warnings:
        return [html.P("No flooding expected in the next 12 hours.", className="twin-empty-state is-good")]

    rows = []
    for w in warnings[:MAX_ALERTS]:
        h = w["hour"]
        if w["kind"] in ("ongoing", "rising"):
            when = html.Div(["NOW", html.Small(nowcast["hours"][0])], className="twin-when")
        else:
            when = html.Div([nowcast["hours"][h], html.Small(f"in {h} h")], className="twin-when")
        rows.append(html.Div(
            className=f"twin-warning lvl-{LEVEL_RANK[w['level']]}",
            children=[
                when,
                html.Div([
                    html.Div(w["text"], className="what"),
                    html.Div(w["places_text"], className="where", title=", ".join(w["places"])),
                ]),
            ],
        ))
    if len(warnings) > MAX_ALERTS:
        rows.append(html.P(f"+ {len(warnings) - MAX_ALERTS} later or smaller warnings",
                           className="twin-empty-state"))
    return rows


def _outlook_card(nowcast: dict) -> tuple[list, str]:
    if nowcast.get("error") and not nowcast.get("acc72_mm"):
        return ([html.Span("Outlook", className="eyebrow"),
                 html.Span(nowcast["headline"], className="headline"),
                 html.Div(nowcast["error"], className="detail")],
                "twin-outlook is-error mb-3")
    level = max(nowcast.get("city_level") or [0])
    eyebrow = (f"Replay · as of {nowcast['as_of_label']}" if nowcast["is_replay"]
               else f"Outlook · issued {nowcast['hours'][0]}")
    source = nowcast["provider"]
    if nowcast.get("bias_corrected"):
        source += " · 72-hour totals bias-corrected to the CHIRPS rainfall the model was trained on"
    if nowcast["is_replay"]:
        source += " · replay: the forecast feed's archived hours, so the future is known in hindsight"
    children = [
        html.Span(eyebrow, className="eyebrow"),
        html.Span(nowcast["headline"], className="headline"),
        html.Div(nowcast["detail"], className="detail"),
    ]
    if nowcast.get("notice"):
        children.append(html.Div(nowcast["notice"], className="detail"))
    children.append(html.Span(
        f"{source}. Timing follows the rainfall forecast crossing the model's flooding level; "
        f"it is not a simulation of water flow.", className="source"))
    return children, f"twin-outlook lvl-{level} mb-3"


def _get_rain_context(rainfall_val: float) -> tuple:
    """
    What the model does at this 3-day total. Thresholds are measured from the
    deployed model's response (flooding first appears between 20 and 30 mm,
    high-probability cores by 40 mm, critical by 50 mm), not assumed.
    """
    if rainfall_val < 25:
        return "No flooding predicted at this 3-day total", "ctx-safe"
    elif rainfall_val < 35:
        return "Flooding begins in the most susceptible channels", "ctx-moderate"
    elif rainfall_val < 50:
        return "High-probability flooding along drainage corridors", "ctx-high"
    return "Critical flooding across several corridors", "ctx-critical"


def _build_scenario_history() -> list:
    """Render the most recent persisted scenario runs (src.persistence.scenario_store)."""
    rows = scenario_store.list_recent_scenarios(limit=5)
    if not rows:
        return [html.P("No runs yet this session.", className="twin-empty-state")]

    items = []
    for row in rows:
        ts = row["run_at_utc"].split("T")[1][:5] if "T" in row["run_at_utc"] else row["run_at_utc"]
        dot = "dot-critical" if row["critical_zone_count"] > 0 else "dot-safe"
        items.append(html.Div(
            className="twin-history-row",
            children=[
                html.Span([html.Span(className=f"status-dot {dot} me-2"),
                           f"{ts} UTC · {row['rainfall_mm_day']:.0f} mm"],
                          className="d-flex align-items-center"),
                html.Span(f"{row['max_depth_m']:.1f}% area · {row['critical_zone_count']} critical",
                          className="twin-mono-meta"),
            ],
        ))
    return items


def _chart_layout(fig: go.Figure, theme: str) -> None:
    ct = CHART_THEME.get(theme, CHART_THEME["dark"])
    fig.update_layout(
        paper_bgcolor=ct["bg"], plot_bgcolor=ct["bg"],
        font=dict(color=ct["text"], size=10, family="IBM Plex Mono, Consolas, monospace"),
        margin=dict(l=45, r=45, t=24, b=35),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.15,
    )


def _response_figure(rainfall_val: float, flooded_pct: float, theme: str) -> go.Figure:
    """
    What-if chart: predicted flooded area across rainfall totals, with the
    scenario marked. Only cached model runs are plotted, so drawing the chart
    never waits on inference; the warm-up fills the curve in within a minute or
    two of startup.
    """
    ct = CHART_THEME.get(theme, CHART_THEME["dark"])
    xs = [mm for mm in range(0, 201, 5) if predictor.is_cached(mm)]
    ys = [round(100.0 * float((predictor.probability_for(mm) > 0.5).mean()), 2) for mm in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, name="Flooded area", mode="lines",
        line=dict(color=ct["accent"], width=2.5), fill="tozeroy",
        fillcolor="rgba(53,194,209,0.12)"))
    fig.add_trace(go.Scatter(
        x=[rainfall_val], y=[flooded_pct], name="This scenario", mode="markers",
        marker=dict(size=11, color="#ef4459", line=dict(color=ct["marker_edge"], width=2))))
    _chart_layout(fig, theme)
    fig.update_layout(
        xaxis=dict(title="Rainfall over 3 days (mm)", showgrid=False, zeroline=False),
        yaxis=dict(title="Flooded area (%)", showgrid=True, gridcolor=ct["grid"], zeroline=False),
    )
    return fig


def _outlook_figure(nowcast: dict, hour: int, theme: str) -> go.Figure:
    """
    Live chart: hourly rain for the last 24 h and next 12 h (bars) with the
    72-hour total that drives the model (line). Outlook hours are coloured by
    the worst band anywhere on the map at that hour.
    """
    ct = CHART_THEME.get(theme, CHART_THEME["dark"])
    offs, rain, acc = nowcast["chart_offsets"], nowcast["chart_rain_mm"], nowcast["chart_acc72_mm"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=offs, y=rain, name="Rain per hour",
        marker_color=[ct["past"] if o <= 0 else ct["accent"] for o in offs],
        hovertemplate="%{customdata}<br>%{y:.1f} mm<extra></extra>",
        customdata=nowcast["chart_labels"]))
    fig.add_trace(go.Scatter(
        x=offs, y=acc, name="72-hour total (CHIRPS scale)" if nowcast.get("bias_corrected") else "72-hour total",
        mode="lines", yaxis="y2",
        line=dict(color=ct["text"], width=1.8),
        hovertemplate="%{customdata}<br>72 h: %{y:.0f} mm<extra></extra>",
        customdata=nowcast["chart_labels"]))
    levels = nowcast.get("city_level") or []
    fig.add_trace(go.Scatter(
        x=list(range(len(levels))), y=nowcast["acc72_mm"], yaxis="y2", mode="markers",
        name="Worst flooding", showlegend=False,
        marker=dict(size=7, color=[LEVEL_COLORS[l] for l in levels], line=dict(color=ct["marker_edge"], width=1)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=[f"{nowcast['hours'][i]}: {LEVEL_NAMES[l].lower()}" for i, l in enumerate(levels)]))
    fig.add_vline(x=0, line=dict(color=ct["text"], width=1, dash="dot"))
    if hour:
        fig.add_vline(x=hour, line=dict(color="#ef4459", width=1.5))
    _chart_layout(fig, theme)
    ticks = [o for o in offs if o % 6 == 0]
    fig.update_layout(
        xaxis=dict(tickvals=ticks, ticktext=["now" if o == 0 else nowcast["chart_labels"][offs.index(o)] for o in ticks],
                   showgrid=False, zeroline=False),
        yaxis=dict(title="mm / hour", showgrid=True, gridcolor=ct["grid"], zeroline=False, rangemode="tozero"),
        yaxis2=dict(title="72 h (mm)", overlaying="y", side="right", showgrid=False, zeroline=False,
                    rangemode="tozero"),
    )
    return fig


# ------------------------------------------------------------------ routes --
def _empty_route_layers() -> dict:
    return {lid: [] for lid in ROUTE_LAYER_IDS}


def _route_layers(res: dict, origin: tuple, destination: tuple) -> tuple[dict, list]:
    fast, safe = res["fastest"], res["safe"]
    layers = _empty_route_layers()
    # Widths are metres, clamped to a pixel range by the layer (map_3d), so the
    # line keeps a readable thickness at every zoom.
    same = fast.node_path == safe.node_path
    if not same:
        layers["route-fastest"] = [{"path": fast.coordinates, "color": ROUTE_FASTEST_RGBA, "width": 40}]
    layers["route-casing"] = [{"path": safe.coordinates, "color": ROUTE_CASING_RGBA, "width": 80}]
    layers["route-safe"] = [{"path": safe.coordinates, "color": ROUTE_SAFE_RGBA, "width": 50}]
    layers["route-endpoints"] = [
        {"position": [origin[1], origin[0]], "color": PIN_START_RGBA},
        {"position": [destination[1], destination[0]], "color": PIN_END_RGBA},
    ]
    pts = np.array(safe.coordinates + (fast.coordinates if not same else []))
    fit = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    return layers, fit


def _exposure_text(r) -> str:
    worst = [s for s in r.flooded_stretches if s["level"] == LEVEL_NAMES[r.worst_level]]
    if not worst:
        return "no predicted flooding on the way"
    s = worst[0]
    return f"{s['level'].lower()} flooding on {s['road']} ({s['place']})"


def _time_check(router: FloodAwareRouter, safe, nowcast: dict, hour: int) -> tuple[str, str] | None:
    """
    Check the recommended route against every later outlook hour, since a road
    that is dry now may not be when the driver reaches it. Returns (message,
    css modifier) or None when there is no live outlook.
    """
    if not nowcast or nowcast.get("error") or not nowcast.get("acc72_mm"):
        return None
    d = datetime.fromisoformat(nowcast["as_of"]).date()
    hours = range(hour, len(nowcast["acc72_mm"]))
    grids = [predictor.probability_for(nowcast["acc72_mm"][h], scenario_date=d) for h in hours]
    levels = router.exposure_over_time(safe.edge_ids, grids)
    now_level = levels[0]
    # The earliest worsening is what decides when to travel; a later, worse
    # band is appended rather than reported instead of it.
    worse = [(h, l) for h, l in zip(hours, levels) if l > now_level]
    if not worse:
        if now_level == 0:
            return ("This route stays clear of predicted flooding for the next 12 hours at the current forecast.",
                    "is-good")
        return None
    h, lvl = worse[0]
    p = router.edge_probability(smooth_for_routing(grids[h - hour]), safe.edge_ids)
    e = safe.edge_ids[int(np.argmax(p))]
    road = router.road_names[router.edge_name[e]] or "an unnamed road"
    place = router.place_at(router.node_lat[router.src[e]], router.node_lon[router.src[e]])
    msg = (f"{LEVEL_NAMES[lvl].title()} flooding is forecast to reach this route around "
           f"{nowcast['hours'][h]} (in about {h} h), on {road} near {place}.")
    peak = max(levels)
    if peak > lvl:
        hp = hours[levels.index(peak)]
        msg += f" {LEVEL_NAMES[peak].title()} by {nowcast['hours'][hp]}."
    msg += " Travel before then, or check again."
    return msg, ("is-bad" if peak >= 2 else "is-warn")


def _route_summary(res: dict, origin_value, destination_value, reroute_msg, time_msg) -> html.Div:
    fast, safe = res["fastest"], res["safe"]
    same = fast.node_path == safe.node_path
    extra = safe.duration_min - fast.duration_min
    avoided = sum(fast.exposure_km.values()) - sum(safe.exposure_km.values())

    children = []
    if reroute_msg:
        children.append(html.Div(reroute_msg, className="twin-reroute-flash"))

    if same:
        safe_meta = f"{safe.distance_km:.1f} km · " + (
            "the fastest route already avoids predicted flooding" if safe.worst_level == 0
            else f"no better alternative: {_exposure_text(safe)}")
    else:
        safe_meta = f"{safe.distance_km:.1f} km · " + (
            f"avoids {avoided:.1f} km of flooded road" if avoided > 0.05 else "lower flood exposure")
        if safe.worst_level > 0:
            safe_meta += f"; still passes {_exposure_text(safe)}"
    children.append(html.Div(className="twin-route-option is-safe", children=[
        html.Span(className="twin-route-swatch"),
        html.Div([html.Div("Flood-aware route", className="twin-route-title"),
                  html.Div(safe_meta, className="twin-route-meta")]),
        html.Div([f"{safe.duration_min:.0f} min"] +
                 ([html.Small(f"+{extra:.0f} min")] if not same and extra >= 0.5 else []),
                 className="twin-route-time"),
    ]))
    if not same:
        children.append(html.Div(className="twin-route-option is-fastest", children=[
            html.Span(className="twin-route-swatch"),
            html.Div([html.Div("Fastest route", className="twin-route-title"),
                      html.Div(f"{fast.distance_km:.1f} km · {_exposure_text(fast)}", className="twin-route-meta")]),
            html.Div(f"{fast.duration_min:.0f} min", className="twin-route-time"),
        ]))
    if res.get("note"):
        children.append(html.Div(res["note"], className="twin-route-banner is-bad"))
    if time_msg:
        children.append(html.Div(time_msg[0], className=f"twin-route-banner {time_msg[1]}"))
    if res.get("outside_grid"):
        children.append(html.Div("Part of this route leaves the model area, where no flooding is predicted.",
                                 className="twin-route-banner"))

    return html.Div([
        html.Div(children, className="twin-route-card"),
        html.P(f"{_point_label(origin_value)} → {_point_label(destination_value)}. "
               "A lower-risk suggestion from ~70 m flood predictions, not verified street by street. "
               "Travel times use typical speeds, not live traffic. Roads © OpenStreetMap contributors.",
               className="twin-fine-print"),
    ])


# ============================================================== callbacks ==
def register_callbacks(app) -> None:

    # Theme: an explicit choice is stored; with none, the OS preference decides.
    app.clientside_callback(
        """
        function(n, stored, resolved) {
            if (!n) { return window.dash_clientside.no_update; }
            return (resolved === "light") ? "dark" : "light";
        }
        """,
        Output("theme-store", "data"),
        Input("btn-theme", "n_clicks"),
        State("theme-store", "data"),
        State("resolved-theme-store", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(stored) {
            var t = stored;
            if (t !== "light" && t !== "dark") {
                t = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches)
                    ? "light" : "dark";
            }
            document.documentElement.setAttribute("data-theme", t);
            return [t, t === "light" ? "Dark mode" : "Light mode"];
        }
        """,
        Output("resolved-theme-store", "data"),
        Output("btn-theme", "children"),
        Input("theme-store", "data"),
    )

    # Mode: show the controls that apply.
    app.clientside_callback(
        """
        function(mode) {
            var live = (mode !== "WHATIF");
            return [!live, live];
        }
        """,
        Output("live-controls", "hidden"),
        Output("whatif-controls", "hidden"),
        Input("mode-radio", "value"),
    )

    # Hide loading overlay when map iframe is populated
    app.clientside_callback(
        """
        function(srcDoc) {
            var overlay = document.getElementById('map-loading-overlay');
            if (overlay && srcDoc && srcDoc.length > 100) {
                overlay.style.display = 'none';
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("map-loading-overlay", "style"),
        Input("3d-map-frame", "srcDoc"),
        prevent_initial_call=True,
    )

    # Map fullscreen
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            var elem = document.getElementById('3d-map-frame');
            if (elem) {
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    if (elem.requestFullscreen) { elem.requestFullscreen(); }
                    else if (elem.webkitRequestFullscreen) { elem.webkitRequestFullscreen(); }
                } else {
                    if (document.exitFullscreen) { document.exitFullscreen(); }
                    else if (document.webkitExitFullscreen) { document.webkitExitFullscreen(); }
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("btn-fullscreen-map", "className"),
        Input("btn-fullscreen-map", "n_clicks"),
        prevent_initial_call=True,
    )

    # Push flood geometry into the map without replacing the document.
    app.clientside_callback(
        """
        function(payload) {
            if (!payload) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try {
                    frame.contentWindow.postMessage(
                        {type: "floodUpdate", main: payload.main, halo: payload.halo}, "*");
                } catch (e) { console.warn("flood map live update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-flood", "title"),
        Input("flood-geojson-store", "data"),
        prevent_initial_call=True,
    )

    # Push route layers into the map, and frame the route when it is new.
    app.clientside_callback(
        """
        function(route) {
            if (!route || !route.layers) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try {
                    frame.contentWindow.postMessage(
                        {type: "layerUpdate", layers: route.layers, fit: route.fit_now ? route.fit : null}, "*");
                } catch (e) { console.warn("route live update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-route", "title"),
        Input("route-store", "data"),
        prevent_initial_call=True,
    )

    # Arm "pick on map": the next map click sets the start or destination.
    app.clientside_callback(
        """
        function(nStart, nEnd) {
            var trig = (window.dash_clientside.callback_context.triggered[0] || {}).prop_id || "";
            var target = trig.indexOf("btn-pick-origin") === 0 ? "origin"
                       : (trig.indexOf("btn-pick-destination") === 0 ? "destination" : null);
            if (!target) { return window.dash_clientside.no_update; }
            window.twinPick = target;
            return [
                "Click the map to set the " + (target === "origin" ? "start." : "destination."),
                "twin-pick-btn" + (target === "origin" ? " is-armed" : ""),
                "twin-pick-btn" + (target === "destination" ? " is-armed" : "")
            ];
        }
        """,
        Output("pick-hint", "children"),
        Output("btn-pick-origin", "className"),
        Output("btn-pick-destination", "className"),
        Input("btn-pick-origin", "n_clicks"),
        Input("btn-pick-destination", "n_clicks"),
        prevent_initial_call=True,
    )

    # A map click arrived while picking: add it as a pinned choice and select it.
    @app.callback(
        Output("route-origin", "options"),
        Output("route-destination", "options"),
        Output("route-origin", "value", allow_duplicate=True),
        Output("route-destination", "value", allow_duplicate=True),
        Output("pick-hint", "children", allow_duplicate=True),
        Output("btn-pick-origin", "className", allow_duplicate=True),
        Output("btn-pick-destination", "className", allow_duplicate=True),
        Input("map-pick-store", "data"),
        State("route-origin", "options"),
        prevent_initial_call=True,
    )
    def on_map_pick(pick, options):
        if not pick:
            raise PreventUpdate
        lat, lon = round(float(pick["lat"]), 5), round(float(pick["lon"]), 5)
        value = f"pin:{lat},{lon}"
        option = {"label": f"📍 Pin near {place_name_at(lat, lon)}", "value": value}
        base = [o for o in (options or []) if not str(o["value"]).startswith("pin:")]
        pins = [o for o in (options or []) if str(o["value"]).startswith("pin:")][-3:]
        new_opts = [option] + pins + base
        is_origin = pick.get("target") == "origin"
        return (new_opts, new_opts,
                value if is_origin else no_update,
                value if not is_origin else no_update,
                "", "twin-pick-btn", "twin-pick-btn")

    # Search any location as the user types, in both trip-planner boxes.
    def _register_search(dropdown_id: str) -> None:
        @app.callback(
            Output(dropdown_id, "options", allow_duplicate=True),
            Input(dropdown_id, "search_value"),
            State(dropdown_id, "value"),
            State(dropdown_id, "options"),
            prevent_initial_call=True,
        )
        def _search(query, value, options):
            if not query or len(query.strip()) < 2:
                raise PreventUpdate
            results = location_search.search(query)
            # The current selection and dropped pins must stay in the list, or
            # the box would go blank when its option disappears.
            keep = [o for o in (options or [])
                    if o["value"] == value or str(o["value"]).startswith("pin:")]
            seen = {o["value"] for o in results}
            return results + [o for o in keep if o["value"] not in seen]

    _register_search("route-origin")
    _register_search("route-destination")

    # Region card click -> store selected region name
    @app.callback(
        Output("selected-region-store", "data"),
        Input({"type": "region-card", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_region_card_click(n_clicks_list):
        if not any(n_clicks_list):
            raise PreventUpdate
        triggered_id = ctx.triggered_id
        if triggered_id and isinstance(triggered_id, dict):
            return triggered_id.get("index")
        return no_update

    # Outlook for the chosen forecast source.
    @app.callback(
        Output("nowcast-store", "data"),
        Output("hour-slider", "marks"),
        Output("hour-slider", "value"),
        Output("outlook-card", "children"),
        Output("outlook-card", "className"),
        Output("mode-pill", "children"),
        Output("mode-pill", "className"),
        Input("mode-radio", "value"),
        Input("forecast-source", "value"),
        Input("btn-refresh-forecast", "n_clicks"),
        State("nowcast-store", "data"),
    )
    def load_outlook(mode, source, _refresh, current):
        dot = html.Span(className="twin-live-dot")
        if mode == "WHATIF":
            return (no_update, no_update, no_update, no_update, no_update,
                    [dot, "What-if scenario"], "twin-mode-pill is-whatif me-2")

        source = source or "live"
        refresh = ctx.triggered_id == "btn-refresh-forecast"
        nc = get_nowcast(source, predictor, refresh=refresh).to_client()
        card, card_class = _outlook_card(nc)

        if nc.get("hours"):
            marks = {h: ("Now" if h == 0 else nc["hours"][h]) for h in (0, 3, 6, 9, 12)}
        else:
            marks = {0: "Now", 3: "+3h", 6: "+6h", 9: "+9h", 12: "+12h"}
        # Keep the chosen hour when only switching back into live mode.
        same_source = current and current.get("source") == source and ctx.triggered_id == "mode-radio"
        hour_value = no_update if same_source else 0

        if nc["is_replay"]:
            pill = ([dot, f"Replay · {nc['as_of_label']}"], "twin-mode-pill is-replay me-2")
        elif nc.get("error") and not nc.get("hours"):
            pill = ([dot, "Live · forecast unavailable"], "twin-mode-pill is-whatif me-2")
        elif (nc.get("notice") or "").startswith("Offline"):
            pill = ([dot, f"Offline copy · issued {nc['hours'][0]}"], "twin-mode-pill is-replay me-2")
        else:
            pill = ([dot, f"Live · issued {nc['hours'][0]}"], "twin-mode-pill me-2")
        return (nc, marks, hour_value, card, card_class) + pill

    @app.callback(
        [
            Output("3d-map-frame", "srcDoc"),
            Output("flood-geojson-store", "data"),
            Output("map-built-store", "data"),
            Output("scenario-key-store", "data"),
            Output("metric-rainfall", "children"),
            Output("metric-max-depth", "children"),
            Output("metric-flood-prob", "children"),
            Output("metric-flooded-area", "children"),
            Output("metric-affected-pop", "children"),
            Output("hydrograph-plot", "figure"),
            Output("chart-title", "children"),
            Output("alert-log-body", "children"),
            Output("zone-risk-table", "children"),
            Output("rain-context-label", "children"),
            Output("scenario-history-body", "children"),
        ],
        [
            Input("btn-run", "n_clicks"),
            Input("rain-slider", "value"),
            Input("hour-slider", "value"),
            Input("mode-radio", "value"),
            Input("nowcast-store", "data"),
            Input("display-mode-radio", "value"),
            Input("region-filter-radio", "value"),
            Input("selected-region-store", "data"),
            Input("resolved-theme-store", "data"),
            Input("map-rebuild-request", "data"),
        ],
        [State("map-built-store", "data"), State("route-store", "data")],
    )
    def update_simulation(n_clicks, rainfall_val, hour, mode, nowcast, display_mode, filter_mode,
                          selected_region, theme, _rebuild, map_built, route_data):
        # Wait for the theme to resolve (instant, client side) and, in live
        # mode, for the outlook, rather than rendering a throwaway map first.
        if theme is None or (mode != "WHATIF" and nowcast is None):
            raise PreventUpdate
        mode = mode or "LIVE"
        display_mode = display_mode or "PROBABILITY"
        filter_mode = filter_mode or "ALL"
        hour = int(hour or 0)

        # Camera target: a curated monitoring point, or any named place.
        highlight_coords = None
        lat_c, lon_c, zoom_c = -1.2787, 36.8213, 13.0
        if selected_region and selected_region in NAIROBI_LOCATIONS:
            lat_c, lon_c, zoom_c = NAIROBI_LOCATIONS[selected_region]
            highlight_coords = (lat_c, lon_c)
        elif selected_region and selected_region in PLACE_COORDS:
            lat_c, lon_c = PLACE_COORDS[selected_region]
            zoom_c = 14.5
            highlight_coords = (lat_c, lon_c)

        key, rain, prob, _d = _scenario(mode, nowcast, hour, rainfall_val)
        if prob is None:
            prob = np.zeros((predictor.h, predictor.w), dtype=np.float32)
        flooded = prob > 0.5
        flooded_pct = 100.0 * float(flooded.mean())
        area_km2 = round(float(flooded.sum()) * CELL_KM2, 2)
        if POP_GRID is not None and POP_GRID.shape == flooded.shape:
            pop = float(POP_GRID[flooded].sum())
        else:
            pop = area_km2 * NAIROBI_POP_PER_KM2
        # Rounded: a census projection spread over 70 m cells does not support
        # a figure precise to the person.
        pop = int(round(pop, -2))

        route_layers = (route_data or {}).get("layers") or _empty_route_layers()
        deck = create_3d_digital_twin_deck(
            depth_grid=prob,
            value_is_probability=True,
            center_lat=lat_c,
            center_lon=lon_c,
            zoom=zoom_c,
            pitch=50.0,
            bearing=-15.0,
            display_mode=display_mode,
            highlight_region=selected_region,
            highlight_coords=highlight_coords,
            theme=theme,
            route_layers=route_layers,
        )
        # The document carries the basemap and building extrusions, which change
        # only with the camera target or theme. Other updates ship geometry alone.
        view_key = f"{selected_region}|{display_mode}|{theme}"
        needs_rebuild = (map_built != view_key) or ctx.triggered_id == "map-rebuild-request"
        map_html = get_deck_html_with_embedded_legend(deck) if needs_rebuild else no_update

        flood_payload = {
            "main": {"type": "FeatureCollection", "features": getattr(deck, "_flood_features", [])},
            "halo": {"type": "FeatureCollection", "features": getattr(deck, "_halo_features", [])},
        }

        # The panel lists the places the map drew water over.
        region_risks = summarise_flooded_regions(getattr(deck, "_flood_features", []))
        region_cards = _build_region_risk_cards(region_risks, filter_mode, selected_region)
        rain_label, rain_class = _get_rain_context(float(rainfall_val or 40.0))

        live_ok = mode == "LIVE" and nowcast and nowcast.get("acc72_mm") and not nowcast.get("error")
        if live_ok:
            fig = _outlook_figure(nowcast, hour, theme)
            chart_title = "Rainfall · Last 24 h and Next 12 h"
            alerts = _build_timed_warnings(nowcast)
        elif mode == "LIVE":
            fig = go.Figure()
            _chart_layout(fig, theme)
            chart_title = "Rainfall · Forecast Unavailable"
            alerts = _build_timed_warnings(nowcast or {"error": "Forecast not loaded."})
        else:
            fig = _response_figure(float(rain), flooded_pct, theme)
            chart_title = "Rainfall Response · What-if"
            alerts = _build_alert_log(region_risks)

        if ctx.triggered_id == "btn-run":
            scenario_store.save_scenario_run(
                rainfall_mm_day=float(rain or 0.0), time_hour=float(hour if mode == "LIVE" else 0.0),
                display_mode=display_mode, max_depth_m=flooded_pct, flooded_area_km2=area_km2,
                est_affected_pop=pop, region_risks=region_risks,
            )

        at_risk = sum(1 for v in region_risks.values() if v["risk_level"] in ("CRITICAL", "HIGH"))
        worst_name, worst = max(region_risks.items(), key=lambda kv: kv[1]["flooded_pct"],
                                default=("--", {"flooded_pct": 0.0}))
        # Some OSM names are lower-case with a comma-separated alias
        # ("kona mbaya,black spot"); the tile shows the first name, capitalised.
        short = worst_name.split(' &')[0].split(' (')[0].split(',')[0][:18]
        short = short.title() if short.islower() else short
        worst_label = "--" if worst["flooded_pct"] <= 0 else f"{short} {worst['flooded_pct']:.0f}%"

        return (
            map_html,
            flood_payload,
            view_key,
            key,
            "--" if rain is None else f"{rain:.0f} mm",
            f"{at_risk} of {len(region_risks)}",
            worst_label,
            f"{area_km2:.2f} km²",
            f"{pop:,}",
            fig,
            chart_title,
            alerts,
            region_cards,
            html.Div(rain_label, className=f"twin-context-label {rain_class}"),
            _build_scenario_history(),
        )

    # Plan, and re-plan, the trip.
    @app.callback(
        Output("route-store", "data"),
        Output("route-summary", "children"),
        Output("route-origin", "value", allow_duplicate=True),
        Output("route-destination", "value", allow_duplicate=True),
        Input("route-origin", "value"),
        Input("route-destination", "value"),
        Input("btn-route-clear", "n_clicks"),
        Input("scenario-key-store", "data"),
        State("route-store", "data"),
        State("mode-radio", "value"),
        State("nowcast-store", "data"),
        State("hour-slider", "value"),
        State("rain-slider", "value"),
        prevent_initial_call=True,
    )
    def plan_route(origin_value, destination_value, _clear, scenario_key, prev,
                   mode, nowcast, hour, rain_slider):
        trig = ctx.triggered_id
        cleared = {"layers": _empty_route_layers(), "active": False}
        hint = html.P("Choose a start and destination to see a route that avoids predicted flooding.",
                      className="twin-empty-state")

        if trig == "btn-route-clear":
            return cleared, hint, None, None
        if not origin_value or not destination_value:
            if prev and prev.get("active"):
                return cleared, hint, no_update, no_update
            return no_update, hint, no_update, no_update
        # A forecast change only matters to a route that is on screen.
        if trig == "scenario-key-store" and not (prev and prev.get("active")):
            raise PreventUpdate

        router = get_router()
        if router is None:
            msg = (f"Road network failed to load: {_router_error}" if _router_error
                   else "The road network is still loading (about 15 seconds after the server starts). "
                        "Choose the route again in a moment.")
            return no_update, html.P(msg, className="twin-empty-state"), no_update, no_update

        origin, destination = _resolve_point(origin_value), _resolve_point(destination_value)
        if origin is None or destination is None:
            return no_update, html.P("That location could not be found.", className="twin-empty-state"), no_update, no_update

        key, _rain, prob, _d = _scenario(mode or "LIVE", nowcast, hour, rain_slider)
        res = router.route(origin, destination, prob)
        # A point far from any mapped road would otherwise snap silently to a
        # road that may be kilometres away.
        far = [(lbl, m) for lbl, m in zip((_point_label(origin_value), _point_label(destination_value)),
                                           res["snap_m"]) if m > MAX_SNAP_M]
        if far:
            lbl, m = far[0]
            return cleared, html.P(f"{lbl} is {m / 1000:.1f} km from the nearest mapped road, so no route "
                                   f"can be planned to it. Choose a point closer to a road.",
                                   className="twin-empty-state"), no_update, no_update
        if res["safe"] is None:
            return cleared, html.P(res.get("note") or "No route found.", className="twin-empty-state"), no_update, no_update

        # Rerouting: the same trip, a different flood field, and a different
        # recommended path. Explain what changed on the old path.
        reroute_msg = None
        same_trip = prev and prev.get("od") == [origin_value, destination_value]
        if trig == "scenario-key-store" and same_trip and prev.get("safe_nodes") != res["safe"].node_path:
            old_edges = router._edges_of(prev["safe_nodes"])
            p_old = router.edge_probability(smooth_for_routing(prob) if prob is not None
                                            else np.zeros((predictor.h, predictor.w), np.float32), old_edges)
            worst = int(np.argmax(p_old)) if len(p_old) else 0
            old_level = int(level_of(p_old[worst])) if len(p_old) else 0
            if old_level > 0:
                e = old_edges[worst]
                road = router.road_names[router.edge_name[e]] or "an unnamed road"
                place = router.place_at(router.node_lat[router.src[e]], router.node_lon[router.src[e]])
                reroute_msg = f"Rerouted: {road} near {place} is now at {LEVEL_NAMES[old_level].lower()} flood risk."
            else:
                reroute_msg = "Route updated: flooding has eased and a quicker route is open."

        time_msg = _time_check(router, res["safe"], nowcast, int(hour or 0)) if mode != "WHATIF" else None
        layers, fit = _route_layers(res, origin, destination)
        store = {
            "active": True,
            "od": [origin_value, destination_value],
            "scenario": key,
            "layers": layers,
            "fit": fit,
            # Frame the camera on a new trip, not on every re-plan of the same one.
            "fit_now": not same_trip,
            "safe_nodes": res["safe"].node_path,
        }
        logger.info(f"Route {_point_label(origin_value)} -> {_point_label(destination_value)} [{key}]: "
                    f"fastest {res['fastest'].duration_min:.0f} min ({LEVEL_NAMES[res['fastest'].worst_level]}), "
                    f"flood-aware {res['safe'].duration_min:.0f} min ({LEVEL_NAMES[res['safe'].worst_level]}) "
                    f"in {res.get('latency_sec')}s")
        return store, _route_summary(res, origin_value, destination_value, reroute_msg, time_msg), no_update, no_update
