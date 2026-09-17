"""
src.dashboard.layouts
======================
Nairobi Urban Flood Digital Twin — Dashboard Layout

Visual system lives in src/dashboard/assets/custom.css (loaded
automatically by Dash). This module only builds structure: a command
topbar, a control rail (flood outlook + trip planner), the map viewport,
and a right-hand telemetry column of uniform panels.
"""

from __future__ import annotations

import json
from pathlib import Path

from dash import dcc, html
import dash_bootstrap_components as dbc

from src.forecast.nowcast import SOURCES

PLACES_FILE = Path("data/processed/nairobi_places.json")


def place_options() -> list[dict]:
    """
    Trip-planner choices: every named place in the gazetteer, once each.

    The gazetteer can hold the same name twice (a place mapped as both a node
    and an area); the first occurrence is kept so a name maps to one point.
    """
    try:
        places = json.loads(PLACES_FILE.read_text(encoding="utf-8"))["places"]
    except Exception:                                      # noqa: BLE001
        return []
    seen, opts = set(), []
    for p in places:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        opts.append({"label": p["name"], "value": f"place:{p['name']}"})
    return sorted(opts, key=lambda o: o["label"].lower())


def _panel(*, header_label, dot_class: str, body_id: str | None = None,
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
    trip_places = place_options()

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
                            xs=12, lg=7,
                            children=html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                children=[
                                    html.Div(className="twin-brand-mark"),
                                    html.Div([
                                        html.H2("Nairobi Flood Digital Twin", className="twin-title"),
                                        html.Span(
                                            "Flood outlook · Flood-aware routing · WebGL terrain model",
                                            className="twin-subtitle",
                                        ),
                                    ]),
                                ],
                            ),
                        ),
                        dbc.Col(
                            xs=12, lg=5,
                            className="text-end",
                            children=[
                                html.Span(
                                    id="mode-pill",
                                    className="twin-mode-pill me-2",
                                    children=[html.Span(className="twin-live-dot"), "Live"],
                                ),
                                dbc.Button(
                                    "Light mode",
                                    id="btn-theme",
                                    size="sm",
                                    className="twin-theme-btn",
                                    title="Switch between light and dark themes",
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
                            xs=12, lg=3,
                            children=[
                                html.Details(
                                    className="twin-help-box mb-3",
                                    children=[
                                        html.Summary("How to use this console"),
                                        html.Ol([
                                            html.Li([html.Strong("Live — "), "the outlook uses Nairobi's hourly rainfall forecast and warns when flooding is expected to start. Drag the hour slider to see the map later on."]),
                                            html.Li([html.Strong("Replay — "), "pick the April 2024 storm under Forecast source to see warnings on a real event."]),
                                            html.Li([html.Strong("Plan a trip — "), "search any place, road or landmark in Nairobi for the start and destination, or pick them on the map. The blue route avoids predicted flooding and updates as the forecast changes."]),
                                            html.Li([html.Strong("What-if — "), "set any 3-day rainfall total to explore a scenario."]),
                                        ]),
                                    ],
                                ),

                                _panel(
                                    header_label="Flood Outlook",
                                    dot_class="dot-accent",
                                    className="mb-3",
                                    body_children=[
                                        dbc.RadioItems(
                                            id="mode-radio",
                                            className="twin-segmented mb-3",
                                            options=[
                                                {"label": "Live forecast", "value": "LIVE"},
                                                {"label": "What-if", "value": "WHATIF"},
                                            ],
                                            value="LIVE",
                                            inline=True,
                                        ),

                                        html.Div(id="live-controls", children=[
                                            html.Label("Forecast source", className="twin-field-label"),
                                            dcc.Dropdown(
                                                id="forecast-source",
                                                options=[{"label": v["label"], "value": k} for k, v in SOURCES.items()],
                                                value="live",
                                                clearable=False,
                                                searchable=False,
                                                className="mb-3",
                                            ),
                                            dcc.Loading(
                                                type="dot",
                                                color="var(--accent)",
                                                children=html.Div(
                                                    id="outlook-card",
                                                    className="twin-outlook mb-3",
                                                    children=[
                                                        html.Span("Outlook", className="eyebrow"),
                                                        html.Span("Loading the rainfall forecast…", className="headline"),
                                                    ],
                                                ),
                                            ),
                                            html.Label("Show the map at", className="twin-field-label"),
                                            dcc.Slider(
                                                id="hour-slider",
                                                min=0, max=12, step=1, value=0,
                                                marks={0: "Now", 3: "+3h", 6: "+6h", 9: "+9h", 12: "+12h"},
                                                tooltip={"placement": "top", "always_visible": False},
                                                allow_direct_input=False,
                                                className="mb-2",
                                            ),
                                            dbc.Button(
                                                "↻ Refresh forecast",
                                                id="btn-refresh-forecast",
                                                color="outline-info",
                                                size="sm",
                                                className="w-100 mb-3 twin-glyph-btn",
                                            ),
                                        ]),

                                        html.Div(id="whatif-controls", hidden=True, children=[
                                            html.Label("Rainfall over 3 days", className="twin-field-label"),
                                            dcc.Slider(
                                                id="rain-slider",
                                                min=5,
                                                max=150,
                                                step=5,
                                                value=40,
                                                marks={5: "5", 30: "30", 60: "60", 90: "90", 120: "120", 150: "150 mm"},
                                                # The number box beside the slider shows the
                                                # value; an always-on tooltip covered the label.
                                                tooltip={"placement": "top", "always_visible": False},
                                                className="mb-2",
                                            ),
                                            html.Div(id="rain-context-label", className="mb-3"),
                                        ]),

                                        html.Label("Map overlay", className="twin-field-label"),
                                        dbc.RadioItems(
                                            id="display-mode-radio",
                                            className="twin-toggle-group mb-3",
                                            options=[
                                                # No depth option: the model predicts flood
                                                # EXTENT, and no depth is estimated anywhere
                                                # in the pipeline.
                                                {"label": "Flood Probability", "value": "PROBABILITY"},
                                                {"label": "Flooded Area", "value": "EXTENT"},
                                            ],
                                            value="PROBABILITY",
                                            inline=True,
                                        ),

                                        dcc.Dropdown(
                                            id="basin-dropdown",
                                            options=[{"label": "Nairobi County", "value": "ALL"}],
                                            value="ALL",
                                            style={"display": "none"},
                                        ),

                                        dbc.Button(
                                            "Save Scenario Run",
                                            id="btn-run",
                                            color="primary",
                                            className="w-100 fw-bold",
                                        ),
                                    ],
                                ),

                                _panel(
                                    header_label="Plan a Trip",
                                    dot_class="dot-accent",
                                    className="mb-3",
                                    body_children=[
                                        html.Div(className="twin-trip-row", children=[
                                            html.Span(className="twin-trip-dot is-start"),
                                            dcc.Dropdown(
                                                id="route-origin",
                                                options=trip_places,
                                                placeholder="Search a start: place, road, landmark",
                                                clearable=True,
                                            ),
                                            dbc.Button("Pick", id="btn-pick-origin", size="sm",
                                                       className="twin-pick-btn",
                                                       title="Click, then click the map to set the start"),
                                        ]),
                                        html.Div(className="twin-trip-row", children=[
                                            html.Span(className="twin-trip-dot is-end"),
                                            dcc.Dropdown(
                                                id="route-destination",
                                                options=trip_places,
                                                placeholder="Search a destination",
                                                clearable=True,
                                            ),
                                            dbc.Button("Pick", id="btn-pick-destination", size="sm",
                                                       className="twin-pick-btn",
                                                       title="Click, then click the map to set the destination"),
                                        ]),
                                        html.Div(id="pick-hint", className="twin-pick-hint"),
                                        dcc.Loading(
                                            type="dot",
                                            color="var(--accent)",
                                            children=html.Div(id="route-summary", children=html.P(
                                                "Choose a start and destination to see a route that avoids predicted flooding.",
                                                className="twin-empty-state")),
                                        ),
                                        dbc.Button(
                                            "Clear route",
                                            id="btn-route-clear",
                                            color="outline-info",
                                            size="sm",
                                            className="w-100 mt-2 twin-glyph-btn",
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
                            xs=12, lg=5,
                            children=[
                                html.Div(
                                    className="twin-map-bezel mb-3",
                                    style={"height": "720px", "position": "relative"},
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
                            xs=12, lg=4,
                            children=[

                                # READOUT STRIP
                                html.Div(
                                    className="twin-readout-strip mb-3",
                                    children=[
                                        html.Div(className="twin-readout-cell rc-rain", children=[
                                            html.Span("Rain · 3 days", className="twin-readout-label"),
                                            html.Span("--", id="metric-rainfall", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-depth", children=[
                                            html.Span("Zones At Risk", className="twin-readout-label"),
                                            html.Span("--", id="metric-max-depth", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-prob", children=[
                                            html.Span("Worst Zone", className="twin-readout-label"),
                                            html.Span("--", id="metric-flood-prob", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-area", children=[
                                            html.Span("Flood Area", className="twin-readout-label"),
                                            html.Span("--", id="metric-flooded-area", className="twin-readout-value"),
                                        ]),
                                        html.Div(className="twin-readout-cell rc-pop", children=[
                                            html.Span("Pop At Risk", className="twin-readout-label"),
                                            html.Span("--", id="metric-affected-pop", className="twin-readout-value"),
                                        ]),
                                    ],
                                ),

                                _panel(
                                    header_label="Flood Warnings",
                                    dot_class="dot-critical",
                                    className="mb-3",
                                    body_id="alert-log-body",
                                    body_style={"maxHeight": "250px", "overflowY": "auto"},
                                    body_children=[html.P("Loading warnings…", className="twin-empty-state")],
                                ),

                                _panel(
                                    header_label=html.Span("Rainfall · Last 24 h and Next 12 h", id="chart-title"),
                                    dot_class="dot-accent",
                                    className="mb-3",
                                    body_children=[
                                        dcc.Graph(
                                            id="hydrograph-plot",
                                            config={"displayModeBar": False},
                                            style={"height": "190px"},
                                        ),
                                    ],
                                ),

                                _panel(
                                    header_label="Flooded Places On The Map",
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
                                    body_style={"maxHeight": "280px", "overflowY": "auto"},
                                    body_children=[html.P("Loading…", className="twin-empty-state")],
                                ),

                                _panel(
                                    header_label="Recent Scenario Runs",
                                    dot_class="dot-accent",
                                    className="mb-3 twin-scroll",
                                    body_id="scenario-history-body",
                                    body_style={"maxHeight": "110px", "overflowY": "auto"},
                                    body_children=[html.P("No runs yet this session.", className="twin-empty-state")],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Store(id="selected-region-store", data=None),
                # Flood geometry for the live-update path. Only this crosses the
                # wire on a scenario change; the ~4 MB basemap and building
                # document stays in the iframe untouched.
                dcc.Store(id="flood-geojson-store", data=None),
                # Tracks which view the map document was built for, so it is
                # rebuilt only when the camera or theme must change.
                dcc.Store(id="map-built-store", data=None),
                # Written by assets/twin_bridge.js when an in-place update fails.
                dcc.Store(id="map-rebuild-request", data=None),
                # The 12-hour outlook for the chosen forecast source.
                dcc.Store(id="nowcast-store", data=None),
                # Identifies the flood field currently on the map; the router
                # listens to it so an active route is re-planned when it changes.
                dcc.Store(id="scenario-key-store", data=None),
                dcc.Store(id="route-store", data=None),
                # Written by assets/twin_bridge.js when a map click drops a pin.
                dcc.Store(id="map-pick-store", data=None),
                # The viewer's explicit theme choice persists across visits;
                # resolved-theme folds in the OS preference when there is none.
                dcc.Store(id="theme-store", storage_type="local", data=None),
                dcc.Store(id="resolved-theme-store", data=None),
                html.Div(id="bridge-sink-flood", hidden=True),
                html.Div(id="bridge-sink-route", hidden=True),
            ]),
        ],
    )
