"""
src.dashboard.layouts
======================
Nairobi Urban Flood Digital Twin — Dashboard Layout

Visual system lives in src/dashboard/assets/custom.css (loaded
automatically by Dash). This module only builds structure: a command
topbar, a fixed control rail, the map viewport, and a right-hand
telemetry column of uniform panels.
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc


def _panel(*, header_label: str, dot_class: str, body_id: str | None = None,
           body_children=None, header_extra=None, body_style: dict | None = None,
           className: str = "") -> html.Div:
    """A .twin-panel: eyebrow header (status dot + label) over a body."""
    header_row = [
        html.P(
            [html.Span(className=f"status-dot {dot_class}"), header_label],
            className="twin-eyebrow",
        )
    ]
    if header_extra is not None:
        header_row.append(header_extra)

    body_kwargs = {"className": "twin-panel-body"}
    if body_style:
        body_kwargs["style"] = body_style
    if body_id:
        body_kwargs["id"] = body_id
    if body_children is not None:
        body_kwargs["children"] = body_children

    return html.Div(
        className=f"twin-panel {className}".strip(),
        children=[
            html.Div(className="twin-panel-header", children=header_row),
            html.Div(**body_kwargs),
        ],
    )


def build_dashboard_layout() -> html.Div:
    return html.Div(
        className="twin-app",
        children=[
            dbc.Container(fluid=True, style={"padding": "16px 20px"}, children=[

                # ═══════════════════════════════════════════════════════
                # TOPBAR
                # ═══════════════════════════════════════════════════════
                dbc.Row(
                    align="center",
                    className="twin-topbar mb-3 pb-3",
                    children=[
                        dbc.Col(
                            width=7,
                            children=html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                children=[
                                    html.Div(className="twin-brand-mark"),
                                    html.Div([
                                        html.H2("Nairobi Flood Digital Twin", className="twin-title"),
                                        html.Span(
                                            "Predictive WebGL Terrain Model · Real-Time Risk Forecast",
                                            className="twin-subtitle",
                                        ),
                                    ]),
                                ],
                            ),
                        ),
                        dbc.Col(
                            width=5,
                            className="text-end",
                            children=[
                                dbc.Button(
                                    "↻ Sync Live Weather",
                                    id="btn-sync-live-weather",
                                    color="success",
                                    size="sm",
                                    className="me-2 twin-glyph-btn",
                                ),
                                html.Span(
                                    [html.Span("⚡", style={"marginRight": "5px"}), "Latency < 0.1s"],
                                    className="twin-latency-pill me-2",
                                ),
                                html.Span(
                                    [html.Span(className="twin-live-dot"), "Live"],
                                    className="twin-live-pill",
                                ),
                            ],
                        ),
                    ],
                ),

                # ═══════════════════════════════════════════════════════
                # MAIN GRID
                # ═══════════════════════════════════════════════════════
                dbc.Row(
                    className="g-3",
                    children=[

                        # ---------------------------------------------
                        # LEFT: CONTROL RAIL
                        # ---------------------------------------------
                        dbc.Col(
                            width=3,
                            children=[
                                html.Details(
                                    className="twin-help-box mb-3",
                                    children=[
                                        html.Summary("How to use this console"),
                                        html.Ol([
                                            html.Li([html.Strong("Live forecast — "), "sync real Nairobi weather to drive the model from today's conditions."]),
                                            html.Li([html.Strong("Click a location — "), "any card in the risk list zooms the map to that exact spot."]),
                                            html.Li([html.Strong("Simulate a storm — "), "drag the rainfall slider or pick a return-period preset."]),
                                        ]),
                                    ],
                                ),

                                _panel(
                                    header_label="Forecast & Simulation Controls",
                                    dot_class="dot-accent",
                                    className="mb-3",
                                    body_children=[
                                        html.Div(
                                            id="live-weather-banner",
                                            className="twin-live-banner mb-3",
                                            children=[
                                                html.Span("Live weather available", className="headline"),
                                                html.Div("Sync above to drive the forecast from today's real conditions.", className="detail"),
                                            ],
                                        ),

                                        html.Label("24-Hour Rainfall Intensity", className="twin-field-label"),
                                        dcc.Slider(
                                            id="rain-slider",
                                            min=5,
                                            max=150,
                                            step=5,
                                            value=10,
                                            marks={
                                                5: "5mm",
                                                20: "20mm",
                                                50: "50mm",
                                                80: "80mm",
                                                120: "120mm",
                                                150: "150mm",
                                            },
                                            tooltip={"placement": "top", "always_visible": True},
                                            className="mb-2",
                                        ),

                                        html.Div(
                                            id="rain-context-label",
                                            className="twin-context-label ctx-safe mb-3",
                                            children="SAFE — no flooding expected (< 15 mm/day)",
                                        ),

                                        html.Label("Storm Progression", className="twin-field-label"),
                                        dcc.Slider(
                                            id="time-slider",
                                            min=1,
                                            max=24,
                                            step=1,
                                            value=12,
                                            marks={1: "1h", 6: "6h", 12: "12h", 18: "18h", 24: "24h"},
                                            tooltip={"placement": "top", "always_visible": True},
                                            className="mb-2",
                                        ),

                                        dbc.Button(
                                            "▶ Play Live Storm Simulation",
                                            id="btn-play-sim",
                                            color="outline-success",
                                            size="sm",
                                            className="w-100 mb-3 twin-glyph-btn",
                                        ),

                                        dcc.Interval(
                                            id="simulation-interval",
                                            interval=1200,
                                            n_intervals=0,
                                            disabled=True,
                                        ),

                                        html.Label("Preset Return Periods", className="twin-field-label"),
                                        dbc.ButtonGroup(
                                            className="w-100 mb-3",
                                            children=[
                                                dbc.Button("10-YR", id="btn-10yr", color="outline-info", size="sm"),
                                                dbc.Button("25-YR", id="btn-25yr", color="outline-info", size="sm"),
                                                dbc.Button("50-YR", id="btn-50yr", color="outline-warning", size="sm"),
                                                dbc.Button("100-YR", id="btn-100yr", color="outline-danger", size="sm"),
                                            ],
                                        ),

                                        html.Label("Map Overlay Mode", className="twin-field-label"),
                                        dbc.RadioItems(
                                            id="display-mode-radio",
                                            className="twin-toggle-group mb-3",
                                            options=[
                                                {"label": "Water Depth", "value": "DEPTH"},
                                                {"label": "Flood Probability", "value": "PROBABILITY"},
                                            ],
                                            value="DEPTH",
                                            inline=True,
                                        ),

                                        dcc.Dropdown(
                                            id="basin-dropdown",
                                            options=[{"label": "Nairobi County", "value": "ALL"}],
                                            value="ALL",
                                            style={"display": "none"},
                                        ),

                                        dbc.Button(
                                            "Run Forecast Prediction",
                                            id="btn-run",
                                            color="primary",
                                            className="w-100 fw-bold",
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ---------------------------------------------
                        # CENTER: MAP VIEWPORT
                        # ---------------------------------------------
                        dbc.Col(
                            id="map-column",
                            width=5,
                            children=[
                                html.Div(
                                    className="twin-map-bezel mb-3",
                                    style={"height": "650px", "position": "relative"},
                                    children=[
                                        html.Div(
                                            className="twin-panel-header",
                                            children=[
                                                html.P(
                                                    [html.Span(className="status-dot dot-accent"), "3D WebGL Digital Twin Viewport"],
                                                    className="twin-eyebrow",
                                                ),
                                                dbc.Button(
                                                    "⤢ Fullscreen",
                                                    id="btn-fullscreen-map",
                                                    color="outline-info",
                                                    size="sm",
                                                    className="py-0 twin-glyph-btn",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={"position": "relative", "height": "calc(100% - 42px)"},
                                            children=[
                                                html.Div(
                                                    id="map-loading-overlay",
                                                    className="twin-map-loading",
                                                    children=[
                                                        dbc.Spinner(color="info", size="lg"),
                                                        html.P("Generating AI flood prediction"),
                                                    ],
                                                ),
                                                html.Iframe(
                                                    id="3d-map-frame",
                                                    style={"width": "100%", "height": "100%", "border": "none"},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ---------------------------------------------
                        # RIGHT: TELEMETRY COLUMN
                        # ---------------------------------------------
                        dbc.Col(
                            width=4,
                            children=[

                                # READOUT STRIP
                                html.Div(
                                    className="twin-readout-strip mb-3",
                                    children=[
                                        html.Div(className="twin-readout-cell rc-rain", children=[
                                            html.Span("Rainfall", className="twin-readout-label"),
                                            html.Span("10 mm", id="metric-rainfall", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-depth", children=[
                                            html.Span("Max Depth", className="twin-readout-label"),
                                            html.Span("0.00 m", id="metric-max-depth", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-prob", children=[
                                            html.Span("Flood Prob", className="twin-readout-label"),
                                            html.Span("--", id="metric-flood-prob", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-area", children=[
                                            html.Span("Flood Area", className="twin-readout-label"),
                                            html.Span("0.00 km²", id="metric-flooded-area", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-pop", children=[
                                            html.Span("Pop At Risk", className="twin-readout-label"),
                                            html.Span("0", id="metric-affected-pop", className="twin-readout-value"),
                                        ]),
                                    ],
                                ),

                                _panel(
                                    header_label="Recent Scenario Runs",
                                    dot_class="dot-accent",
                                    className="mb-3 twin-scroll",
                                    body_id="scenario-history-body",
                                    body_style={"maxHeight": "110px", "overflowY": "auto"},
                                    body_children=[html.P("No runs yet this session.", className="twin-empty-state")],
                                ),

                                _panel(
                                    header_label="Nairobi Locations — Click To Zoom",
                                    dot_class="dot-moderate",
                                    className="mb-3",
                                    header_extra=dbc.RadioItems(
                                        id="region-filter-radio",
                                        className="twin-toggle-group",
                                        options=[
                                            {"label": "All", "value": "ALL"},
                                            {"label": "High Risk", "value": "HIGH_ONLY"},
                                        ],
                                        value="ALL",
                                        inline=True,
                                    ),
                                    body_id="zone-risk-table",
                                    body_style={"maxHeight": "320px", "overflowY": "auto"},
                                    body_children=[html.P("Loading region risk data…", className="twin-empty-state")],
                                ),

                                _panel(
                                    header_label="24-Hour Storm Hydrograph",
                                    dot_class="dot-accent",
                                    className="mb-3",
                                    body_children=[
                                        dcc.Graph(
                                            id="hydrograph-plot",
                                            config={"displayModeBar": False},
                                            style={"height": "170px"},
                                        ),
                                    ],
                                ),

                                _panel(
                                    header_label="Active Alerts",
                                    dot_class="dot-critical",
                                    body_id="alert-log-body",
                                    body_style={"maxHeight": "160px", "overflowY": "auto"},
                                    body_children=[html.P("Loading alerts…", className="twin-empty-state")],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Store(id="selected-region-store", data=None),
            ]),
        ],
    )
