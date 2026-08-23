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

from src.models.predict import FloodSurrogatePredictor, NAIROBI_LOCATIONS
from src.dashboard.components.map_3d import create_3d_digital_twin_deck, get_deck_html_with_embedded_legend
from src.ingestion.live_weather import fetch_live_nairobi_weather
from src.persistence import scenario_store

predictor = FloodSurrogatePredictor()
scenario_store.init_db()

RISK_COLORS = {
    "CRITICAL": {"bg": "#ef4459", "text": "CRITICAL", "badge": "danger", "dot": "dot-critical"},
    "HIGH": {"bg": "#f08a3c", "text": "HIGH", "badge": "warning", "dot": "dot-high"},
    "MODERATE": {"bg": "#f0b93f", "text": "MODERATE", "badge": "warning", "dot": "dot-moderate"},
    "LOW": {"bg": "#6fe0a8", "text": "LOW", "badge": "success", "dot": "dot-low"},
    "SAFE": {"bg": "#35d495", "text": "SAFE", "badge": "secondary", "dot": "dot-safe"},
}


def _build_region_risk_cards(region_risks: dict, filter_mode: str = "ALL", selected_region: str | None = None) -> list:
    """Build clickable hotspot cards with html.Div wrapper for pattern matching."""
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "SAFE": 4}
    sorted_regions = sorted(region_risks.items(), key=lambda x: severity_order.get(x[1]["risk_level"], 5))

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
                                    f"{rr['max_depth']}m max · {rr['flooded_pct']}% area flooded",
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
    )

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
                    html.Span(f"{rr['max_depth']}m · {rr['flooded_pct']}%", className="twin-mono-meta"),
                ],
            )
        )
    return rows


def _build_flood_probability_label(depth_grid: np.ndarray) -> str:
    """
    Headline flood-probability KPI, using the same depth->probability scaling
    src.dashboard.components.map_3d applies in PROBABILITY display mode
    (depth / 2.2m reference clipped to 99%), area-weighted over pixels
    already at or above the moderate-depth threshold rather than just the
    single peak pixel.
    """
    flooded = depth_grid[depth_grid >= 0.2]
    if flooded.size == 0:
        return "0%"
    mean_prob = float(np.clip((flooded.mean() / 2.2) * 100.0, 0.0, 99.0))
    return f"{mean_prob:.0f}%"


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
                    html.Span(f"{row['max_depth_m']:.2f}m · {row['critical_zone_count']} critical", className="twin-mono-meta"),
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

    # 1. Preset return period buttons
    @app.callback(
        Output("rain-slider", "value"),
        [
            Input("btn-10yr", "n_clicks"),
            Input("btn-25yr", "n_clicks"),
            Input("btn-50yr", "n_clicks"),
            Input("btn-100yr", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def update_slider_preset(*args):
        triggered = ctx.triggered_id
        presets = {"btn-10yr": 40, "btn-25yr": 65, "btn-50yr": 95, "btn-100yr": 135}
        return presets.get(triggered, 65)

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
    @app.callback(
        [
            Output("simulation-interval", "disabled"),
            Output("btn-play-sim", "children"),
            Output("btn-play-sim", "color"),
        ],
        Input("btn-play-sim", "n_clicks"),
        State("simulation-interval", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, is_disabled):
        if is_disabled:
            return False, "⏸ Pause Simulation", "danger"
        return True, "▶ Play Live Storm Simulation", "outline-success"

    # 5. Interval step
    @app.callback(
        Output("time-slider", "value"),
        Input("simulation-interval", "n_intervals"),
        State("time-slider", "value"),
        prevent_initial_call=True,
    )
    def step_time(n_intervals, current_hr):
        next_hr = int(current_hr or 1) + 1
        return 1 if next_hr > 24 else next_hr

    # 6. Main forecast callback — FIRES ON PAGE LOAD & REGION SELECTION & FILTERS
    @app.callback(
        [
            Output("3d-map-frame", "srcDoc"),
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
            Input("time-slider", "value"),
            Input("display-mode-radio", "value"),
            Input("region-filter-radio", "value"),
            Input("selected-region-store", "data"),
        ],
    )
    def update_simulation(n_clicks, rainfall_val, time_hour, display_mode, filter_mode, selected_region):
        rainfall_val = float(rainfall_val or 10.0)
        time_hour = float(time_hour or 12.0)
        display_mode = display_mode or "DEPTH"
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
        time_factor = float(np.sin(np.pi * (time_hour / 24.0))) ** 0.8
        effective_rain = max(5.0, rainfall_val * (0.3 + 0.7 * time_factor))

        # Run 100% dynamic AI prediction
        res = predictor.predict_scenario(rainfall_mm_day=effective_rain)

        depth_grid = res["depth_grid"]
        max_depth = res["max_depth_m"]
        area_km2 = res["flooded_area_km2"]
        pop = res["est_affected_pop"]
        region_risks = res.get("region_risks", {})

        # Generate Pydeck 3D map with embedded legend
        deck = create_3d_digital_twin_deck(
            depth_grid=depth_grid,
            center_lat=lat_c,
            center_lon=lon_c,
            zoom=zoom_c,
            pitch=50.0,
            bearing=-15.0,
            display_mode=display_mode,
            highlight_region=selected_region,
            highlight_coords=highlight_coords,
        )
        map_html = get_deck_html_with_embedded_legend(deck)

        # Build region risk cards
        region_cards = _build_region_risk_cards(
            region_risks=region_risks,
            filter_mode=filter_mode,
            selected_region=selected_region
        )

        # Rainfall context label
        rain_label, rain_class = _get_rain_context(rainfall_val)

        # Hydrograph chart
        hours = ["04:00", "08:00", "12:00", "16:00", "20:00", "24:00", "+4h"]
        rain_hist = [round(effective_rain * (0.15 + 0.12 * i), 1) for i in range(7)]
        depths = [round(max_depth * (i / 6.0) ** 1.3, 2) for i in range(7)]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=rain_hist, name="Rainfall (mm)", marker_color="#35c2d1", opacity=0.85))
        fig.add_trace(go.Scatter(x=hours, y=depths, name="Water Depth (m)", mode="lines+markers",
                                 line=dict(color="#ef4459", width=3),
                                 marker=dict(size=5, color="#ef4459"), yaxis="y2"))
        fig.update_layout(
            paper_bgcolor="#10161f", plot_bgcolor="#10161f",
            font=dict(color="#93a2b3", size=10, family="IBM Plex Mono, Consolas, monospace"),
            margin=dict(l=35, r=35, t=20, b=25),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
            yaxis=dict(title="Rainfall (mm)", showgrid=True, gridcolor="#202b38", zeroline=False, title_font=dict(color="#35c2d1")),
            yaxis2=dict(title="Depth (m)", overlaying="y", side="right", showgrid=False, zeroline=False, title_font=dict(color="#ef4459")),
            xaxis=dict(showgrid=False),
        )

        if should_persist:
            scenario_store.save_scenario_run(
                rainfall_mm_day=rainfall_val, time_hour=time_hour, display_mode=display_mode,
                max_depth_m=max_depth, flooded_area_km2=area_km2, est_affected_pop=pop,
                region_risks=region_risks,
            )

        return (
            map_html,
            f"{rainfall_val:.0f} mm",
            f"{max_depth:.2f} m",
            _build_flood_probability_label(depth_grid),
            f"{area_km2:.2f} km²",
            f"{pop:,}",
            fig,
            _build_alert_log(region_risks),
            region_cards,
            html.Span(rain_label, className=f"twin-context-label {rain_class}"),
            _build_scenario_history(),
        )
