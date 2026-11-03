"""
src.dashboard.map_layers_ui
===========================
The "Map layers" checklist and the area filter on the live map (specification
§2.2, §2.12, FR-013, FR-014, UC-FVD-007, UC-FVD-008).

    boundaries      sub-county and ward outlines (geoBoundaries)
    rivers          rivers, streams and water bodies (OpenStreetMap)
    infrastructure  facilities with the predicted risk at their location, for
                    the outlook on screen (live or replay)
    stations        monitoring stations (staff only; IoT Sensor Data Platform)
    alerts          wards with published alerts, or with the outlook's model
                    warnings while the Automated Alert System is not connected

Choosing an area zooms the map to it and outlines it. Layer rows go to the map
document by postMessage, like the other overlays, so the camera is kept.
"""

from __future__ import annotations

from collections import Counter

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.geo import boundaries, layers

BASE_OPTIONS = [
    {"label": "Sub-county and ward boundaries", "value": "boundaries"},
    {"label": "Rivers and water bodies", "value": "rivers"},
    {"label": "Schools, hospitals, bridges and public buildings", "value": "infrastructure"},
    {"label": "Alert areas", "value": "alerts"},
]
STATION_OPTION = {"label": "Monitoring stations", "value": "stations"}

#: Every layer on the live map, for the layer catalogue (spec §8 MapLayers, FR-014).
LAYER_DEFINITIONS = [
    {"layer_key": "flood-main", "name": "Predicted flooding", "category": "Prediction", "default_on": 1,
     "description": "Probability that each ~70 m cell floods in the outlook shown (model output, not observed).",
     "source": "Predictive Flood Modeling System (U-Net)"},
    {"layer_key": "flood-reports", "name": "Community flood reports", "category": "Observation", "default_on": 1,
     "description": "Flooding reported by signed-in users in the last few hours.", "source": "Digital twin reports"},
    {"layer_key": "satellite", "name": "Satellite flood map", "category": "Observation", "default_on": 0,
     "description": "Water seen by Sentinel-1 radar, about every six days; blind between buildings.",
     "source": "Copernicus Global Flood Monitoring"},
    {"layer_key": "past-floods", "name": "Past floods in the news", "category": "History", "default_on": 0,
     "description": "Places the news reported flooded, 2000-2025; a record, not a forecast.",
     "source": "Groundsource (Google Research), CC BY 4.0"},
    {"layer_key": "admin-boundaries", "name": "Sub-county and ward boundaries", "category": "Reference",
     "default_on": 0, "description": "Administrative areas used for filtering.",
     "source": "geoBoundaries (IEBC, OCHA; CC BY)"},
    {"layer_key": "rivers", "name": "Rivers and water bodies", "category": "Reference", "default_on": 0,
     "description": "Named rivers, streams, dams and ponds.", "source": "OpenStreetMap (ODbL)"},
    {"layer_key": "infrastructure", "name": "Infrastructure", "category": "Exposure", "default_on": 0,
     "description": "Schools, health facilities, bridges, public buildings and emergency services, ringed by "
                    "predicted risk.", "source": "OpenStreetMap until the Infrastructure Risk Mapping System connects"},
    {"layer_key": "stations", "name": "Monitoring stations", "category": "Observation", "default_on": 0,
     "description": "Rain and river gauges by health; staff only.", "source": "IoT Sensor Data Platform",
     "min_role": "staff"},
    {"layer_key": "alert-areas", "name": "Alert areas", "category": "Alerts", "default_on": 0,
     "description": "Wards with a published alert, or a model warning while the alert system is not connected.",
     "source": "Automated Alert System / model warnings"},
]


def _swatch(rgb: list[int], round_: bool = False) -> html.Span:
    return html.Span(style={"display": "inline-block", "width": "10px", "height": "10px", "marginRight": "6px",
                            "borderRadius": "50%" if round_ else "2px", "background": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"})


def _alerts_from_nowcast(nowcast: dict | None) -> list[dict]:
    return [{"severity": w["level"].lower(), "areas": w["places"], "title": w["text"]}
            for w in (nowcast or {}).get("warnings") or []]


def _probability_for(nowcast: dict | None):
    """The flood probability at the outlook's wettest hour, matching the outlook on screen."""
    if not nowcast or not nowcast.get("acc72_mm"):
        return None
    from datetime import datetime
    from src.dashboard.callbacks import predictor
    peak = max(nowcast["acc72_mm"])
    day = datetime.fromisoformat(nowcast["as_of"]).date() if nowcast.get("as_of") else None
    return predictor.probability_for(float(round(peak)), scenario_date=day)


def build(selected: list[str], area: str | None, nowcast: dict | None, user: dict | None,
          categories: list[str] | None = None) -> tuple[dict, list]:
    """Rows for every operational layer (empty when off) and the note beside the checklist."""
    from src.dashboard.components.map_3d import OPS_LAYER_IDS
    from src.integrations import registry
    selected = set(selected or [])
    rows = {lid: [] for lid in OPS_LAYER_IDS}
    note = []
    area_ids = [area] if area else []
    if "boundaries" in selected or area:
        rows["admin-boundaries"] = layers.boundary_rows(highlight=area_ids)
        rows["label-subcounties"] = layers.subcounty_label_rows(highlight=area_ids)
        if "boundaries" in selected:
            note.append(html.Div([html.Strong("Boundaries: "), "19 sub-counties (thick, labelled) and 71 wards "
                                  "(thin), geoBoundaries 2020 (IEBC; CC BY). Hover a ward for its name."], className="mb-1"))
    if "rivers" in selected:
        rows["rivers"], rows["water-bodies"] = layers.river_rows(), layers.water_rows()
        rows["label-rivers"] = layers.river_label_rows()
        note.append(html.Div([_swatch([56, 189, 248]), html.Strong("Rivers and water: "),
                              f"{len(rows['rivers'])} river and stream lines, {len(rows['water-bodies'])} water "
                              "bodies (OpenStreetMap)."], className="mb-1"))
    if "infrastructure" in selected:
        items = layers.exposure(_probability_for(nowcast))
        cats = categories or list(layers.CATEGORIES)
        rows["infrastructure"] = layers.infrastructure_rows(items, categories=cats, area_ids=area_ids or None)
        rows["label-facilities"] = layers.facility_label_rows(items, categories=cats, area_ids=area_ids or None)
        at_risk = Counter(a["category"] for a in items if a.get("level", 0) >= 1 and a["category"] in cats
                          and (not area_ids or {a.get("ward"), a.get("sub_county")} & set(area_ids)))
        key = [html.Span([_swatch(layers.CATEGORIES[k][1], True), layers.CATEGORIES[k][0]], className="me-2",
                         style={"whiteSpace": "nowrap"}) for k in cats]
        risk = (", ".join(f"{n} {layers.CATEGORIES[k][0].lower()}" for k, n in at_risk.most_common())
                if at_risk else "none")
        note.append(html.Div([html.Strong("Infrastructure: "), f"{len(rows['infrastructure'])} facilities shown. ",
                              html.Div(key, className="mt-1"),
                              html.Div(["A coloured ring marks predicted flooding at the facility in this outlook "
                                        "(amber moderate, orange high, red very high). In a predicted flood area "
                                        "now: ", html.Strong(risk), "."], className="mt-1"),
                              html.Div("OpenStreetMap, used until the Infrastructure Risk Mapping System is "
                                       "connected. Informal settlements are mapped less completely.",
                                       className="twin-fine-print")], className="mb-1"))
    if "stations" in selected and can(user, "view_sensors"):
        r = registry.get("iot").fetch("stations")
        if r.ok:
            st = [s for s in r.records if boundaries.point_in(s["lat"], s["lon"], area_ids or None)]
            rows["stations"] = layers.station_rows(st, demo=r.demo)
            note.append(html.Div([html.Strong("Stations: "), f"{len(st)} shown, coloured by health (green ok, "
                                  "amber degraded, grey offline); a red ring is a water level above warning.",
                                  html.Span(" DEMO DATA", className="ops-demo-tag ms-1") if r.demo else None],
                                 className="mb-1"))
        else:
            note.append(html.Div([html.Strong("Stations: "), r.message], className="mb-1"))
    if "alerts" in selected:
        partner = registry.get("alerts").fetch("active")
        if partner.ok:
            alerts, src = partner.records, "Automated Alert System" + (" (DEMO DATA)" if partner.demo else "")
        else:
            alerts, src = _alerts_from_nowcast(nowcast), "Model warnings: the Automated Alert System is not connected"
        rows["alert-areas"] = layers.alert_area_rows(alerts, src)
        wards = len({r_["title"] for r_ in rows["alert-areas"]})
        note.append(html.Div([html.Strong("Alert areas: "),
                              f"{wards} ward{'s' if wards != 1 else ''} with an alert. " if wards else "No alerts. ",
                              html.Span(src, className="twin-fine-print")], className="mb-1"))
    if area:
        a = boundaries.get(area)
        if a:
            note.insert(0, html.Div([html.Strong("Showing "), boundaries.label(area),
                                     ". Clear the box to see the whole city."], className="mb-1"))
    return rows, note


def register_map_layer_callbacks(app) -> None:
    try:
        from src.integrations.store import get_ops_store
        get_ops_store().upsert_layers(LAYER_DEFINITIONS)
    except Exception as exc:                                    # noqa: BLE001  catalogue only; never block start-up
        from loguru import logger
        logger.warning(f"Could not record the map layer catalogue: {exc}")

    @app.callback(Output("ops-layers", "options"), Output("map-area-filter", "options"),
                  Output("ops-layers", "value"),
                  Input("account-area", "children"))
    def layer_options(_account):
        """Options for this viewer, and the layers a signed-in user had on last time (FR-014)."""
        user = current_user()
        opts = BASE_OPTIONS + ([STATION_OPTION] if can(user, "view_sensors") else [])
        value = no_update
        if user:
            from src.integrations.store import get_ops_store
            saved = get_ops_store().preferences(user["user_id"]).get("map_layers")
            if saved is not None:
                allowed = {o["value"] for o in opts}
                value = [v for v in saved if v in allowed]
        return opts, boundaries.options(("sub_county", "ward")), value

    @app.callback(Output("bridge-sink-ops", "role"), Input("ops-layers", "value"), prevent_initial_call=True)
    def save_layers(value):
        user = current_user()
        if user:
            from src.integrations.store import get_ops_store
            get_ops_store().set_preferences(user["user_id"], {"map_layers": value or []})
        return no_update

    @app.callback(Output("ops-layers-store", "data"), Output("ops-layers-note", "children"),
                  Input("ops-layers", "value"), Input("map-area-filter", "value"), Input("nowcast-store", "data"),
                  Input("infra-categories", "value"), State("ops-layers-store", "data"))
    def ops_layers(selected, area, nowcast, categories, current):
        if not selected and not area and not (current or {}).get("any"):
            raise PreventUpdate
        rows, note = build(selected or [], area, nowcast, current_user(), categories)
        fit = boundaries.bounds([area]) if area and area != (current or {}).get("area") else None
        return {"rows": rows, "any": any(rows.values()), "area": area, "fit": fit}, note

    @app.callback(Output("infra-categories", "style"), Input("ops-layers", "value"))
    def show_categories(selected):
        return None if "infrastructure" in (selected or []) else {"display": "none"}

    app.clientside_callback(
        """
        function(data) {
            if (!data) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try {
                    var msg = {type: "layerUpdate", layers: data.rows || {}};
                    if (data.fit) { msg.fit = data.fit; }
                    frame.contentWindow.postMessage(msg, "*");
                } catch (e) { console.warn("map layers update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-ops", "title"),
        Input("ops-layers-store", "data"),
        prevent_initial_call=True,
    )
