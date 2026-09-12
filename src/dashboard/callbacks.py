"""
src.dashboard.callbacks
========================
Nairobi Urban Flood Digital Twin — Interactive Callbacks

PURPOSE
-------
1. Uses get_deck_html_with_embedded_legend for 100% visible legend in Fullscreen mode.
2. North-Up exact raster transform so flood polygons align 100% precisely over Globe Roundabout & Nairobi River.
3. Target 3d-map-frame directly for native Fullscreen map canvas expansion.
4. Live Weather Sync Callback (Open-Meteo API).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html, no_update, dcc, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from loguru import logger

from src.models.predict import NAIROBI_LOCATIONS
from src.models.predict_v2 import FloodPredictor
from src.dashboard.components.map_3d import summarise_flooded_regions, create_3d_digital_twin_deck, get_deck_html_with_embedded_legend
from src.ingestion.live_weather import fetch_live_nairobi_weather
from src.persistence import scenario_store

# Model B on drainage labels: test F1 0.937 on held-out storm seasons.
# Replaces the ConvLSTM, which was trained on the dataset whose rainfall
# join left only 6 of 703 samples flood-positive (see RESULTS.md 4.1).
predictor = FloodPredictor()
scenario_store.init_db()

RISK_COLORS = {
    "CRITICAL": {"bg": "#ef4459", "text": "CRITICAL", "badge": "danger", "dot": "dot-critical"},
    "HIGH": {"bg": "#f08a3c", "text": "HIGH", "badge": "warning", "dot": "dot-high"},
    "MODERATE": {"bg": "#f0b93f", "text": "MODERATE", "badge": "warning", "dot": "dot-moderate"},
    "LOW": {"bg": "#6fe0a8", "text": "LOW", "badge": "success", "dot": "dot-low"},
    "SAFE": {"bg": "#35d495", "text": "SAFE", "badge": "secondary", "dot": "dot-safe"},
}


#: Most flooded places to list in the side panel. The map may cover a hundred
#: places in a heavy scenario; a scrolling list that long is not readable, and
#: the ones below the cut are by construction the least affected.
MAX_REGION_CARDS = 12
MAX_ALERTS = 8


def _build_region_risk_cards(region_risks: dict, filter_mode: str = "ALL", selected_region: str | None = None) -> list:
    """Build clickable hotspot cards with html.Div wrapper for pattern matching."""
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "SAFE": 4}
    sorted_regions = sorted(region_risks.items(), key=lambda x: severity_order.get(x[1]["risk_level"], 5))[:MAX_REGION_CARDS]

    cards = []
    for reg_name, rr in sorted_regions:
        risk_level = rr["risk_level"]

        if filter_mode == "HIGH_ONLY" and risk_level in ["SAFE", "LOW"]:
            continue

        risk_info = RISK_COLORS.get(risk_level, RISK_COLORS["SAFE"])
        is_selected = (reg_name == selected_region)

        card = html.Div(
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
                                html.Div(
                                    [
                                        html.Span(className=f"status-dot {risk_info['dot']} me-2"),
                                        reg_name,
                                    ],
                                    className="twin-region-name",
                                ),
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
        )
        cards.append(card)

    if not cards:
        cards.append(html.P("No locations match the high-risk filter under this rainfall intensity.", className="twin-empty-state is-good"))

    return cards


def _build_alert_log(region_risks: dict) -> list:
    """Render one alert line per CRITICAL/HIGH region for the current scenario."""
    severity_order = {"CRITICAL": 0, "HIGH": 1}
    alerts = sorted(
        (item for item in region_risks.items() if item[1]["risk_level"] in severity_order),
        key=lambda x: severity_order[x[1]["risk_level"]],
    )[:MAX_ALERTS]

    if not alerts:
        return [html.P("No active alerts — no zone exceeds HIGH risk under this scenario.", className="twin-empty-state is-good")]

    rows = []
    for reg_name, rr in alerts:
        risk_info = RISK_COLORS[rr["risk_level"]]
        rows.append(
            html.Div(
                className="twin-alert-row",
                children=[
                    html.Span(
                        [html.Span(className=f"status-dot {risk_info['dot']}"), reg_name],
                        className="twin-alert-label",
                        style={"color": risk_info["bg"]},
                    ),
                    html.Span(f"{rr['peak_probability_pct']}% · {rr['flooded_pct']}%", className="twin-mono-meta"),
                ],
            )
        )
    return rows


#: Nairobi County population density, used to translate flooded area into an
#: order-of-magnitude exposure figure. ~4.4M people over ~700 km2.
NAIROBI_POP_PER_KM2 = 6300


def _region_risks_from_probability(prob_grid: np.ndarray, threshold: float) -> dict:
    """
    Per-region summary from a probability field.

    Reports the flooded share of each region and its peak probability. It does
    NOT report a depth: the model predicts extent, and no depth is estimated
    anywhere in the pipeline.
    """
    from src.models.predict import NAIROBI_LOCATIONS
    from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

    lats = np.linspace(LAT_NORTH, LAT_SOUTH, GRID_H)
    lons = np.linspace(LON_WEST, LON_EAST, GRID_W)
    out = {}
    for name, (lat_t, lon_t, _zoom) in NAIROBI_LOCATIONS.items():
        r_c = int(np.abs(lats - lat_t).argmin())
        c_c = int(np.abs(lons - lon_t).argmin())
        r1, r2 = max(0, r_c - 4), min(GRID_H, r_c + 5)
        c1, c2 = max(0, c_c - 4), min(GRID_W, c_c + 5)
        patch = prob_grid[r1:r2, c1:c2]
        flooded_pct = round(100.0 * float((patch > threshold).mean()), 1)
        peak = round(100.0 * float(patch.max()), 1)
        # Key and vocabulary must match RISK_COLORS and the severity_order map
        # in _build_region_risk_cards: CRITICAL, HIGH, MODERATE, LOW, SAFE.
        if flooded_pct >= 40:
            level = "CRITICAL"
        elif flooded_pct >= 15:
            level = "HIGH"
        elif flooded_pct >= 5:
            level = "MODERATE"
        elif flooded_pct > 0:
            level = "LOW"
        else:
            level = "SAFE"

        out[name] = {
            "peak_probability_pct": peak,
            "flooded_pct": flooded_pct,
            "risk_level": level,
        }
    return out

_RESPONSE_CURVE: tuple[list[float], list[float]] | None = None


def _rainfall_response_curve() -> tuple[list[float], list[float]]:
    """
    Predicted flooded extent across the rainfall range, computed once.

    A genuine characteristic of the model rather than a fabricated time series:
    it answers "how much of the city floods at each rainfall depth", which is
    what the scenario slider is exploring. Cached because it costs one inference
    per point and does not change between requests.
    """
    global _RESPONSE_CURVE
    if _RESPONSE_CURVE is None:
        xs = list(range(0, 201, 10))
        ys = []
        for mm in xs:
            try:
                ys.append(round(100.0 * predictor.predict(rainfall_mm=float(mm))["flooded_fraction"], 2))
            except Exception:                              # noqa: BLE001
                ys.append(0.0)
        _RESPONSE_CURVE = (xs, ys)
    return _RESPONSE_CURVE


def _build_flood_probability_label(prob_grid: np.ndarray, threshold: float = 0.5) -> str:
    """
    Headline KPI: mean predicted probability across the cells reported flooded.

    The model emits a probability directly, so no conversion is applied. An
    earlier version divided by a 2.2 m depth reference - the scaling the old
    depth-regression model needed - which under-reported this figure by a factor
    of 2.2 and pinned it near 44% regardless of the scenario.

    Averaged over flooded cells rather than the whole grid, since the 99% of
    cells that are dry would drag any grid-wide mean to nearly zero and say
    nothing about the flooding being displayed.
    """
    flooded = prob_grid[prob_grid >= threshold]
    if flooded.size == 0:
        return "0%"
    return f"{100.0 * float(flooded.mean()):.0f}%"


def _build_scenario_history() -> list:
    """Render the most recent persisted scenario runs (src.persistence.scenario_store)."""
    rows = scenario_store.list_recent_scenarios(limit=5)
    if not rows:
        return [html.P("No runs yet this session.", className="twin-empty-state")]

    items = []
    for row in rows:
        ts = row["run_at_utc"].split("T")[1][:5] if "T" in row["run_at_utc"] else row["run_at_utc"]
        dot = "dot-critical" if row["critical_zone_count"] > 0 else "dot-safe"
        items.append(
            html.Div(
                className="twin-history-row",
                children=[
                    html.Span(
                        [html.Span(className=f"status-dot {dot} me-2"), f"{ts} UTC · {row['rainfall_mm_day']:.0f}mm/day"],
                        className="d-flex align-items-center",
                    ),
                    html.Span(f"{row['max_depth_m']:.1f}% · {row['critical_zone_count']} critical", className="twin-mono-meta"),
                ],
            )
        )
    return items


def _get_rain_context(rainfall_val: float) -> tuple:
    """Return context label and color for a given rainfall value."""
    if rainfall_val < 15:
        return "SAFE — no flooding expected (< 15 mm/day)", "ctx-safe"
    elif rainfall_val < 40:
        return "MODERATE — surface pooling begins in low basins", "ctx-moderate"
    elif rainfall_val < 75:
        return "HEAVY — significant flooding along drainage channels", "ctx-high"
    elif rainfall_val < 110:
        return "VERY HEAVY — widespread urban inundation expected", "ctx-critical"
    else:
        return "EXTREME — severe flash flooding, high risk to life", "ctx-critical"


def register_callbacks(app) -> None:

    # 0. Hide loading overlay when map iframe is populated
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

    # 0b. DIRECT MAP IFRAME FULLSCREEN CLIENTSIDE CALLBACK
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            var elem = document.getElementById('3d-map-frame');
            if (elem) {
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    if (elem.requestFullscreen) { elem.requestFullscreen(); }
                    else if (elem.webkitRequestFullscreen) { elem.webkitRequestFullscreen(); }
                    else if (elem.msRequestFullscreen) { elem.msRequestFullscreen(); }
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

    # Return-period presets were removed. They mapped labels to fixed slider
    # positions (10-YR -> 40mm ... 100-YR -> 135mm) that no analysis supported.
    # Fitting a Gumbel distribution to the 12 annual maxima of 3-day rainfall in
    # the CHIRPS record puts the 2-year level at 99mm and the 10-year at 133mm,
    # so the "100-YR" button was roughly a 10-year storm and "10-YR" was below a
    # 2-year one. Separately, return periods beyond about 10 years cannot be
    # estimated from 12 years of record at all. The slider states a rainfall
    # depth, which is a fact; a return-period label is a statistical claim.

    # Push flood geometry into the map iframe without replacing the document.
    # A full srcDoc swap reloads deck.gl, flickers, and resets the viewer's zoom
    # and pan; postMessage patches the two flood layers in place. If the iframe
    # has not signalled readiness the message is simply dropped - the server
    # still sends a full document on the next camera change.
    app.clientside_callback(
        """
        function(payload) {
            if (!payload) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (!frame || !frame.contentWindow) { return window.dash_clientside.no_update; }
            try {
                frame.contentWindow.postMessage(
                    {type: "floodUpdate", main: payload.main, halo: payload.halo}, "*");
            } catch (e) {
                console.warn("flood map live update failed:", e);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("flood-geojson-store", "data", allow_duplicate=True),
        Input("flood-geojson-store", "data"),
        prevent_initial_call=True,
    )

    # 2. Live Weather Sync Callback (Open-Meteo API)
    @app.callback(
        [
            Output("rain-slider", "value", allow_duplicate=True),
            Output("live-weather-banner", "children"),
        ],
        Input("btn-sync-live-weather", "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_live_weather(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        weather = fetch_live_nairobi_weather()
        today_rain = weather["today_rainfall_mm"]
        peak_rain = weather["peak_7day_rainfall_mm"]
        source = weather["source"]

        slider_val = max(10, min(150, int(today_rain if today_rain > 5 else peak_rain)))

        banner_children = [
            html.Span("Live weather synced", className="headline"),
            html.Div(f"Today {today_rain} mm/day · 7-day peak {peak_rain} mm/day", className="detail"),
            html.Span(f"Source: {source}", className="source"),
        ]

        return slider_val, banner_children

    # 3. Handle region card click -> store selected region name
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
            selected_reg = triggered_id.get("index")
            logger.info(f"Location card clicked: {selected_reg}")
            return selected_reg
        return no_update

    # 4. Simulation play/pause
    # The storm-progression slider and its play button were removed. They scaled
    # the rainfall input by sin(pi * hour/24) ** 0.8, a bell curve peaking at
    # noon that bottoms out at BOTH ends of the slider. At 55 mm that left
    # 16.5 mm at hour 24, below the 30 mm that defines a flood, so the map
    # emptied at one end of the slider and filled in the middle for no
    # hydrological reason. The model has no sub-daily resolution at all - CHIRPS
    # is daily - so intra-day progression is not something it can represent.

    @app.callback(
        [
            Output("3d-map-frame", "srcDoc"),
            Output("flood-geojson-store", "data"),
            Output("map-built-store", "data"),
            Output("metric-rainfall", "children"),
            Output("metric-max-depth", "children"),
            Output("metric-flood-prob", "children"),
            Output("metric-flooded-area", "children"),
            Output("metric-affected-pop", "children"),
            Output("hydrograph-plot", "figure"),
            Output("alert-log-body", "children"),
            Output("zone-risk-table", "children"),
            Output("rain-context-label", "children"),
            Output("scenario-history-body", "children"),
        ],
        [
            Input("btn-run", "n_clicks"),
            Input("rain-slider", "value"),
            Input("display-mode-radio", "value"),
            Input("region-filter-radio", "value"),
            Input("selected-region-store", "data"),
        ],
        [State("map-built-store", "data")],
    )
    def update_simulation(n_clicks, rainfall_val, display_mode, filter_mode,
                          selected_region, map_built):
        rainfall_val = float(rainfall_val or 10.0)
        display_mode = display_mode or "PROBABILITY"
        filter_mode = filter_mode or "ALL"

        # Determine center focus based on selected location card
        highlight_coords = None
        if selected_region and selected_region in NAIROBI_LOCATIONS:
            lat_c, lon_c, zoom_c = NAIROBI_LOCATIONS[selected_region]
            highlight_coords = (lat_c, lon_c)
        else:
            lat_c, lon_c, zoom_c = -1.2787, 36.8213, 13.0

        # Persist a scenario-history row only for an explicit "Run Simulation"
        # click — not on every slider-drag frame or auto-play tick, both of
        # which also fire this callback dozens of times per interaction.
        should_persist = ctx.triggered_id == "btn-run"

        # Time progression
        # The slider value is the scenario, used as given.
        effective_rain = float(rainfall_val)

        # U-Net inference. The output is a per-cell flood PROBABILITY in [0, 1],
        # not a depth in metres - the model predicts extent, and satellites
        # cannot measure depth (RESULTS.md 4.9).
        res = predictor.predict(rainfall_mm=effective_rain)

        prob_grid = res["probability_grid"]
        flooded_pct = 100.0 * res["flooded_fraction"]
        area_km2 = res["flooded_area_km2"]
        pop = int(area_km2 * NAIROBI_POP_PER_KM2)
        # region_risks is derived below, from the polygons actually drawn.

        # Generate Pydeck 3D map with embedded legend
        deck = create_3d_digital_twin_deck(
            depth_grid=prob_grid,
            value_is_probability=True,
            center_lat=lat_c,
            center_lon=lon_c,
            zoom=zoom_c,
            pitch=50.0,
            bearing=-15.0,
            display_mode=display_mode,
            highlight_region=selected_region,
            highlight_coords=highlight_coords,
        )
        # The document carries the basemap and building extrusions, which never
        # change. It is rebuilt only on first render or when the selected region
        # moves the camera; every other update ships geometry alone, which is
        # ~0.7 MB against ~4.6 MB and leaves the viewer's zoom and pan intact.
        view_key = f"{selected_region}|{display_mode}"
        needs_rebuild = (map_built != view_key)
        map_html = get_deck_html_with_embedded_legend(deck) if needs_rebuild else no_update

        flood_payload = {
            "main": {"type": "FeatureCollection",
                     "features": getattr(deck, "_flood_features", [])},
            "halo": {"type": "FeatureCollection",
                     "features": getattr(deck, "_halo_features", [])},
        }

        # The panel lists the places the map drew water over, ranked by severity
        # then flooded area, rather than a fixed shortlist sampled independently.
        region_risks = summarise_flooded_regions(getattr(deck, "_flood_features", []))

        # Build region risk cards
        region_cards = _build_region_risk_cards(
            region_risks=region_risks,
            filter_mode=filter_mode,
            selected_region=selected_region
        )

        # Rainfall context label
        rain_label, rain_class = _get_rain_context(rainfall_val)

        # Hydrograph chart
        # Response curve: how predicted flooded extent varies with rainfall,
        # with a marker at the current scenario. This replaces a 24-hour
        # hydrograph whose bars and curve were both invented - the model has no
        # sub-daily resolution, so it could not have produced them. The curve is
        # computed once at import and is a genuine property of the model.
        curve_x, curve_y = _rainfall_response_curve()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_x, y=curve_y, name="Flooded area", mode="lines",
            line=dict(color="#35c2d1", width=2.5),
            fill="tozeroy", fillcolor="rgba(53,194,209,0.12)"))
        fig.add_trace(go.Scatter(
            x=[rainfall_val], y=[flooded_pct], name="This scenario",
            mode="markers", marker=dict(size=11, color="#ef4459",
                                        line=dict(color="#10161f", width=2))))
        fig.update_layout(
            paper_bgcolor="#10161f", plot_bgcolor="#10161f",
            font=dict(color="#93a2b3", size=10, family="IBM Plex Mono, Consolas, monospace"),
            margin=dict(l=45, r=20, t=20, b=35),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=9)),
            xaxis=dict(title="Rainfall over 3 days (mm)", showgrid=False, zeroline=False),
            yaxis=dict(title="Flooded area (%)", showgrid=True, gridcolor="#202b38",
                       zeroline=False, title_font=dict(color="#35c2d1")),
        )

        if should_persist:
            scenario_store.save_scenario_run(
                rainfall_mm_day=rainfall_val, time_hour=0.0, display_mode=display_mode,
                max_depth_m=flooded_pct, flooded_area_km2=area_km2, est_affected_pop=pop,
                region_risks=region_risks,
            )

        # Each readout reports a distinct quantity. Flooded extent appears once,
        # in km2; the percentage tile previously beside it was the same number in
        # other units, and the tile before that reported a probability that sat
        # at ~98% for every scenario and so carried no information.
        at_risk = sum(1 for v in region_risks.values()
                      if v["risk_level"] in ("CRITICAL", "HIGH"))
        flooded_places = len(region_risks)
        worst_name, worst = max(region_risks.items(),
                                key=lambda kv: kv[1]["flooded_pct"],
                                default=("--", {"flooded_pct": 0.0}))
        worst_label = ("--" if worst["flooded_pct"] <= 0
                       else f"{worst_name.split(' &')[0].split(' (')[0][:18]} {worst['flooded_pct']:.0f}%")

        return (
            map_html,
            flood_payload,
            view_key,
            f"{rainfall_val:.0f} mm",
            f"{at_risk} of {flooded_places}",
            worst_label,
            f"{area_km2:.2f} km²",
            f"{pop:,}",
            fig,
            _build_alert_log(region_risks),
            region_cards,
            html.Span(rain_label, className=f"twin-context-label {rain_class}"),
            _build_scenario_history(),
        )
