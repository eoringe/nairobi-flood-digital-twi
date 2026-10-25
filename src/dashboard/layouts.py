"""
src.dashboard.layouts
======================
Nairobi Urban Flood Digital Twin — Dashboard Layout

Visual system lives in src/dashboard/assets/custom.css (loaded
automatically by Dash). This module only builds structure: a slim top bar,
then a full-height workspace of three columns: the control rail (flood
outlook and trip planner), the map edge to edge with its readouts floating
over it, and the insights rail (warnings, role tools, rainfall, places).
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


def _icon(name: str, cls: str = "") -> html.I:
    return html.I(className=f"bi bi-{name} {cls}".strip())


def _panel(*, header_label, dot_class: str, body_id: str | None = None,
           body_children=None, header_extra=None, body_style: dict | None = None,
           className: str = "", icon: str | None = None) -> html.Div:
    """A .twin-panel: a header (icon, or a status dot, then the title) over a body."""
    mark = (html.Span(_icon(icon), className=f"twin-panel-icon {dot_class}") if icon
            else html.Span(className=f"status-dot {dot_class}"))
    header_row = [html.P([mark, header_label], className="twin-eyebrow")]
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


def _readout(cls: str, icon: str, label: str, value_id: str) -> html.Div:
    """One tile of the readout strip floating over the map."""
    return html.Div(className=f"twin-readout-cell {cls}", children=[
        html.Span(_icon(icon), className="twin-readout-icon"),
        html.Div(className="twin-readout-text", children=[
            html.Span(label, className="twin-readout-label"),
            html.Span("--", id=value_id, className="twin-readout-value"),
        ]),
    ])


def _topbar() -> html.Header:
    return html.Header(className="twin-topbar", children=[
        html.Div(className="twin-brand", children=[
            html.Div(className="twin-brand-mark"),
            html.Div([
                html.H1("Nairobi Flood Twin", className="twin-title"),
                html.Span("Flood outlook and flood-aware routing", className="twin-subtitle"),
            ]),
        ]),
        html.Div(className="twin-topbar-actions", children=[
            html.Span(
                id="mode-pill",
                className="twin-mode-pill",
                children=[html.Span(className="twin-live-dot"), "Live"],
            ),
            # Shows a sun in dark mode and a moon in light mode (custom.css);
            # the theme callback keeps the tooltip in step.
            html.Button(
                [_icon("sun", "twin-theme-sun"), _icon("moon-stars", "twin-theme-moon")],
                id="btn-theme",
                type="button",
                className="twin-theme-btn",
                title="Switch between light and dark themes",
            ),
            # Sign-in link, or who is signed in and their links
            # (filled per user by auth_pages.route_page).
            html.Div(id="account-area", className="twin-account-area"),
        ]),
    ])


def _outlook_panel() -> html.Div:
    return _panel(
        header_label="Flood outlook",
        dot_class="tone-accent",
        icon="cloud-rain-heavy",
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
                    [_icon("arrow-clockwise", "me-2"), "Refresh forecast"],
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
                    {"label": "Flood probability", "value": "PROBABILITY"},
                    {"label": "Flooded area", "value": "EXTENT"},
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
                [_icon("bookmark-plus", "me-2"), "Save scenario run"],
                id="btn-run",
                color="primary",
                className="w-100 fw-bold",
            ),
        ],
    )


def _trip_panel(trip_places: list[dict]) -> html.Div:
    return _panel(
        header_label="Plan a trip",
        dot_class="tone-info",
        icon="sign-turn-right",
        body_children=[
            html.Div(className="twin-trip-stops", children=[
                html.Div(className="twin-trip-row", children=[
                    html.Span(className="twin-trip-dot is-start"),
                    dcc.Dropdown(
                        id="route-origin",
                        options=trip_places,
                        placeholder="Start: place, road or landmark",
                        clearable=True,
                    ),
                    dbc.Button(_icon("crosshair"), id="btn-pick-origin", size="sm",
                               className="twin-pick-btn",
                               title="Click, then click the map to set the start"),
                ]),
                html.Div(className="twin-trip-row", children=[
                    html.Span(className="twin-trip-dot is-end"),
                    dcc.Dropdown(
                        id="route-destination",
                        options=trip_places,
                        placeholder="Destination",
                        clearable=True,
                    ),
                    dbc.Button(_icon("crosshair"), id="btn-pick-destination", size="sm",
                               className="twin-pick-btn",
                               title="Click, then click the map to set the destination"),
                ]),
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
                [_icon("x-lg", "me-2"), "Clear route"],
                id="btn-route-clear",
                color="outline-info",
                size="sm",
                className="w-100 mt-2 twin-glyph-btn",
            ),
        ],
    )


def _help_box() -> html.Details:
    return html.Details(
        className="twin-help-box",
        children=[
            html.Summary([_icon("lightbulb"), html.Span("How to use the map")]),
            html.Ol([
                html.Li([html.Strong("Live: "), "the outlook uses Nairobi's hourly rainfall forecast and warns when flooding is expected to start. Drag the hour slider to see the map later on."]),
                html.Li([html.Strong("Replay: "), "pick the April 2024 storm under Forecast source to see warnings on a real event."]),
                html.Li([html.Strong("Plan a trip: "), "search any place, road or landmark in Nairobi for the start and destination, or pick them on the map. The blue route avoids predicted flooding and updates as the forecast changes."]),
                html.Li([html.Strong("What-if: "), "planners set any 3-day rainfall total to explore a scenario (sign in as a planner)."]),
            ]),
        ],
    )


def _map_stage() -> html.Div:
    """The map, edge to edge, with the readouts and the full-screen button over it."""
    return html.Div(className="twin-map-bezel", children=[
        html.Div(className="twin-map-overlay", children=[
            html.Div(className="twin-readout-strip", children=[
                _readout("rc-rain", "cloud-drizzle", "Rain, 3 days", "metric-rainfall"),
                _readout("rc-depth", "exclamation-triangle", "Zones at risk", "metric-max-depth"),
                _readout("rc-prob", "geo-alt", "Worst zone", "metric-flood-prob"),
                _readout("rc-area", "water", "Flood area", "metric-flooded-area"),
                _readout("rc-pop", "people", "People at risk", "metric-affected-pop"),
            ]),
            html.Button(
                _icon("arrows-fullscreen"),
                id="btn-fullscreen-map",
                type="button",
                className="twin-map-btn",
                title="Show the map full screen",
            ),
        ]),
        html.Div(
            id="map-loading-overlay",
            className="twin-map-loading",
            children=[
                html.Div(className="twin-signing-spinner"),
                html.P("Preparing the flood map…"),
            ],
        ),
        html.Iframe(id="3d-map-frame", className="twin-map-frame"),
    ])


def build_dashboard_layout() -> html.Div:
    trip_places = place_options()

    return html.Div(
        className="twin-app twin-shell",
        children=[
            _topbar(),
            html.Div(className="twin-workspace", children=[

                # LEFT: controls
                html.Aside(className="twin-rail twin-rail-left", children=[
                    _outlook_panel(),
                    _trip_panel(trip_places),
                    _help_box(),
                ]),

                # CENTRE: the map
                html.Main(id="map-column", className="twin-map-column", children=_map_stage()),

                # RIGHT: insights
                html.Aside(className="twin-rail twin-rail-right", children=[
                    _panel(
                        header_label="Flood warnings",
                        dot_class="tone-danger",
                        icon="bell",
                        body_id="alert-log-body",
                        body_style={"maxHeight": "260px", "overflowY": "auto"},
                        body_children=[html.P("Loading warnings…", className="twin-empty-state")],
                    ),

                    # Alerts for the signed-in user's areas, and the
                    # planner / responder tools (auth_pages.role_tools).
                    html.Div(id="role-tools-panel", className="twin-rail-group"),

                    _panel(
                        header_label=html.Span("Rainfall: last 24 h and next 12 h", id="chart-title"),
                        dot_class="tone-accent",
                        icon="bar-chart-line",
                        body_children=[
                            dcc.Graph(
                                id="hydrograph-plot",
                                config={"displayModeBar": False},
                                style={"height": "200px"},
                            ),
                        ],
                    ),

                    _panel(
                        header_label="Flooded places",
                        dot_class="tone-warn",
                        icon="pin-map",
                        header_extra=dbc.RadioItems(
                            id="region-filter-radio",
                            className="twin-toggle-group is-compact",
                            options=[
                                {"label": "All", "value": "ALL"},
                                {"label": "High risk", "value": "HIGH_ONLY"},
                            ],
                            value="ALL",
                            inline=True,
                        ),
                        body_id="zone-risk-table",
                        body_style={"maxHeight": "300px", "overflowY": "auto"},
                        body_children=[html.P("Loading…", className="twin-empty-state")],
                    ),

                    html.Div(id="history-wrap", hidden=True, children=_panel(
                        header_label="Recent scenario runs",
                        dot_class="tone-accent",
                        icon="clock-history",
                        className="twin-scroll",
                        body_id="scenario-history-body",
                        body_style={"maxHeight": "130px", "overflowY": "auto"},
                        body_children=[html.P("No runs yet this session.", className="twin-empty-state")],
                    )),
                ]),
            ]),

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
            # What is on the map now, for comparison, export and reports.
            dcc.Store(id="scenario-summary-store", data=None),
            # The planner's held scenario, kept for this browser tab.
            dcc.Store(id="held-scenario-store", storage_type="session", data=None),
            dcc.Download(id="download-file"),
            html.Div(id="bridge-sink-flood", hidden=True),
            html.Div(id="bridge-sink-route", hidden=True),
        ],
    )
