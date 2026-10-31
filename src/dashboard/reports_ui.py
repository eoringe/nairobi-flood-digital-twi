"""
src.dashboard.reports_ui
========================
Community flood reports and the satellite layer on the map page.

* Anyone sees the reports: pins on the map and the "Reported flooding" list,
  refreshed every REFRESH_SEC for every viewer.
* Signed-in users report flooding (their location, a point picked on the map,
  or a searched place), and confirm "still flooded" or "cleared" on others'
  reports. Responders and administrators verify or remove reports.
* The flood-aware route treats reported flooding like predicted flooding
  (src.routing.flood_router.report_probability).
* The satellite switch shows the newest Copernicus Sentinel-1 flood map
  (src.reports.satellite) and says plainly how old it is and how little of
  the city the radar can see.
* The past-floods switch shows where flooding was reported in the news
  (src.reports.history, Google's Groundsource) and, during a storm replay,
  picks out the places reported flooded that week.

Every action re-checks the permission on the server and is written to the
audit log; who made a report is never shown to other users.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.auth.store import get_store
from src.models.predict import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST
from src.reports import history, satellite
from src.reports.store import KINDS, NOTE_MAX, ReportError, get_reports

REFRESH_SEC = 30
#: Pin colour per kind (also the halo on the map).
KIND_COLOURS = {"water": [56, 152, 236], "impassable": [240, 138, 60], "homes": [239, 68, 89]}
KIND_ICONS = {"water": "droplet-half", "impassable": "sign-stop", "homes": "house-exclamation"}


# ---------------------------------------------------------------- helpers --
def _ago(iso: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    mins = max(0, int((now - datetime.fromisoformat(iso)).total_seconds() // 60))
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    return f"{mins // 60} h {mins % 60:02d} min ago" if mins % 60 else f"{mins // 60} h ago"


def _confirmed_text(r: dict) -> str:
    n = r["confirmations"]
    return "" if n == 0 else (" · confirmed by 1 person" if n == 1 else f" · confirmed by {n} people")


def map_rows(reports: list[dict]) -> list[dict]:
    """Report pins for the map, with the tooltip text (no reporter details)."""
    now = datetime.now(timezone.utc)
    rows = []
    for r in reports:
        rows.append({
            "position": [r["lon"], r["lat"]],
            "pin": KIND_COLOURS[r["kind"]] + [255],
            "color": KIND_COLOURS[r["kind"]] + [70],
            "line": KIND_COLOURS[r["kind"]] + [200],
            "title": KINDS[r["kind"]],
            "detail": f"Reported in {r['place']} {_ago(r['updated_at'], now)}{_confirmed_text(r)}"
                      + (" · verified by a responder" if r["verified_by"] else ""),
            "note": r["note"] or "",
        })
    return rows


def reports_payload() -> dict:
    """What the browser keeps: the map rows and a version that changes with the set."""
    active = get_reports().active()
    version = "|".join(f"{r['report_id']}:{r['kind']}:{r['confirmations']}:{r['verified_by'] or 0}" for r in active)
    return {"version": version, "rows": map_rows(active),
            "points": [{"lat": r["lat"], "lon": r["lon"], "kind": r["kind"]} for r in active]}


def _inside_map(lat: float, lon: float) -> bool:
    return LAT_SOUTH <= lat <= LAT_NORTH and LON_WEST <= lon <= LON_EAST


def report_list(user: dict | None) -> list:
    """The "Reported flooding" panel for this user (vote buttons when signed in)."""
    active = get_reports().active()
    if not active:
        return [html.P("No flooding reported by people on the ground right now.", className="twin-empty-state is-good")]
    now = datetime.now(timezone.utc)
    signed_in = can(user, "report_flood")
    moderator = can(user, "moderate_reports")
    items = []
    for r in active[:12]:
        mine = signed_in and r["user_id"] == user["user_id"]
        vote = get_reports().vote_of(r["report_id"], user["user_id"]) if signed_in else None
        buttons = []
        if signed_in:
            if not mine:
                buttons.append(dbc.Button("Still flooded", id={"type": "report-act", "id": r["report_id"], "act": "still"},
                                          size="sm", color="outline-info", disabled=vote == "still",
                                          className="twin-report-btn"))
            buttons.append(dbc.Button("Cleared", id={"type": "report-act", "id": r["report_id"], "act": "cleared"},
                                      size="sm", color="outline-info", disabled=vote == "cleared",
                                      className="twin-report-btn"))
        if moderator:
            if not r["verified_by"]:
                buttons.append(dbc.Button("Verify", id={"type": "report-act", "id": r["report_id"], "act": "verify"},
                                          size="sm", color="outline-success", className="twin-report-btn"))
            buttons.append(dbc.Button("Remove", id={"type": "report-act", "id": r["report_id"], "act": "remove"},
                                      size="sm", color="outline-danger", className="twin-report-btn"))
        items.append(html.Div(className=f"twin-report kind-{r['kind']}", children=[
            html.Span(html.I(className=f"bi bi-{KIND_ICONS[r['kind']]}"), className="twin-report-icon"),
            html.Div(className="twin-report-body", children=[
                html.Div([KINDS[r["kind"]],
                          html.Span([html.I(className="bi bi-patch-check-fill me-1"), "Verified"],
                                    className="twin-report-verified") if r["verified_by"] else None],
                         className="twin-report-title"),
                html.Div(f"{r['place']} · {_ago(r['updated_at'], now)}{_confirmed_text(r)}"
                         + (" · your report" if mine else ""), className="twin-report-meta"),
                html.Div(f"“{r['note']}”", className="twin-report-note") if r["note"] else None,
                html.Div(buttons, className="twin-report-actions") if buttons else None,
            ]),
        ]))
    return items


def _report_form() -> list:
    """The report dialog's form. Always rendered, so its callbacks find their inputs."""
    return [
        html.Div(id="report-signin", className="twin-report-signin", children=[
            html.P("Sign in to report flooding. It takes one click with Google, and your name is never shown "
                   "on a report.", className="twin-empty-state"),
            dcc.Link("Sign in", href="/login", className="twin-auth-btn"),
        ]),
        html.Div(id="report-form", children=[
            html.Label("Where is the flooding?", className="twin-field-label"),
            html.Div(className="d-flex gap-2 mb-2", children=[
                dbc.Button([html.I(className="bi bi-geo-alt-fill me-2"), "Use my location"], id="btn-report-myloc",
                           size="sm", color="outline-info", className="flex-fill"),
                dbc.Button([html.I(className="bi bi-crosshair me-2"), "Pick on the map"], id="btn-report-pick",
                           size="sm", color="outline-info", className="flex-fill"),
            ]),
            dcc.Dropdown(id="report-place", placeholder="Or search a place, road or landmark", clearable=True,
                         options=[], className="mb-2"),
            html.Div(id="report-point-label", className="twin-report-point"),
            html.Label("What do you see?", className="twin-field-label"),
            dbc.RadioItems(id="report-kind", className="twin-report-kinds",
                           options=[{"label": v, "value": k} for k, v in KINDS.items()], value="water"),
            html.Label("Anything to add? (optional)", className="twin-field-label"),
            dcc.Textarea(id="report-note", maxLength=NOTE_MAX, className="form-control twin-input twin-report-text",
                         placeholder="For example: knee-deep near the stage, matatus turning back"),
            html.Div(id="report-status", className="mt-2"),
        ]),
    ]


def report_modal() -> dbc.Modal:
    return dbc.Modal(id="report-modal", is_open=False, centered=True, className="dash-modal", children=[
        dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-megaphone me-2"), "Report flooding"])),
        dbc.ModalBody(_report_form()),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-report-cancel", color="outline-info", size="sm"),
            dbc.Button([html.I(className="bi bi-send me-2"), "Send report"], id="btn-report-submit",
                       color="primary", size="sm"),
        ]),
    ])


def _audit(action: str, user: dict | None, detail: str) -> None:
    get_store().audit(action, user["user_id"] if user else None, detail)


# -------------------------------------------------------------- callbacks --
def register_report_callbacks(app) -> None:
    # Reports for every viewer: on load, every REFRESH_SEC, and after an action.
    @app.callback(Output("reports-store", "data"), Input("reports-poll", "n_intervals"))
    def poll(_n):
        return reports_payload()

    @app.callback(Output("reports-list", "children"),
                  Input("reports-store", "data"), Input("role-tools-panel", "children"))
    def show_list(_payload, _role_panel):
        return report_list(current_user())

    # Pins onto the map without rebuilding it.
    app.clientside_callback(
        """
        function(payload) {
            if (!payload) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try {
                    frame.contentWindow.postMessage({type: "layerUpdate", layers: {
                        "flood-reports": payload.rows, "flood-reports-halo": payload.rows}}, "*");
                } catch (e) { console.warn("report pins update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-reports", "title"),
        Input("reports-store", "data"),
    )

    # Open and close the report dialog; signed-out viewers see a sign-in prompt.
    @app.callback(
        Output("report-modal", "is_open"),
        Output("report-signin", "hidden"),
        Output("report-form", "hidden"),
        Output("btn-report-submit", "disabled"),
        Input("btn-report-open", "n_clicks"),
        Input("btn-report-cancel", "n_clicks"),
        Input("report-reopen", "data"),
        prevent_initial_call=True,
    )
    def toggle(_open, _cancel, reopen):
        if ctx.triggered_id == "btn-report-cancel":
            return False, no_update, no_update, no_update
        if ctx.triggered_id == "report-reopen" and not reopen:
            raise PreventUpdate
        if ctx.triggered_id == "btn-report-open" and not _open:
            raise PreventUpdate
        allowed = can(current_user(), "report_flood")
        return True, allowed, not allowed, not allowed

    # "Use my location": the browser asks permission, then fills the point.
    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            var set = window.dash_clientside.set_props;
            if (!navigator.geolocation) {
                set("report-status", {children: "This browser cannot share its location. Pick the place on the map instead."});
                return window.dash_clientside.no_update;
            }
            set("report-status", {children: "Finding your location\\u2026"});
            navigator.geolocation.getCurrentPosition(function (pos) {
                set("report-point", {data: {lat: pos.coords.latitude, lon: pos.coords.longitude,
                                           label: "Your location", t: Date.now()}});
                set("report-status", {children: ""});
            }, function () {
                set("report-status", {children: "Your location could not be read. Allow location access, or pick the place on the map."});
            }, {enableHighAccuracy: true, timeout: 12000, maximumAge: 60000});
            return window.dash_clientside.no_update;
        }
        """,
        Output("report-point", "data", allow_duplicate=True),
        Input("btn-report-myloc", "n_clicks"),
        prevent_initial_call=True,
    )

    # "Pick on the map": close the dialog; the next map click comes back as a point.
    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            window.twinPick = "report";
            window.dash_clientside.set_props("pick-hint", {children: "Click the map where the flooding is."});
            return false;
        }
        """,
        Output("report-modal", "is_open", allow_duplicate=True),
        Input("btn-report-pick", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("report-point", "data", allow_duplicate=True),
        Output("report-reopen", "data"),
        Output("pick-hint", "children", allow_duplicate=True),
        Input("report-pick-store", "data"),
        prevent_initial_call=True,
    )
    def picked(pick):
        if not pick:
            raise PreventUpdate
        return {"lat": pick["lat"], "lon": pick["lon"], "label": "Point picked on the map"}, pick.get("t"), ""

    # Search a place for the report, from the same index as the trip planner.
    @app.callback(
        Output("report-place", "options"),
        Input("report-place", "search_value"),
        State("report-place", "value"),
        State("report-place", "options"),
        prevent_initial_call=True,
    )
    def search(query, value, options):
        from src.dashboard.callbacks import location_search
        if not query or len(query.strip()) < 2:
            raise PreventUpdate
        keep = [o for o in (options or []) if o["value"] == value]
        results = [o for o in location_search.search(query) if not o.get("disabled")]
        return keep + [o for o in results if o["value"] != value]

    @app.callback(Output("report-point", "data", allow_duplicate=True),
                  Input("report-place", "value"), prevent_initial_call=True)
    def searched(value):
        from src.dashboard.callbacks import _point_label, _resolve_point
        point = _resolve_point(value)
        if point is None:
            raise PreventUpdate
        return {"lat": point[0], "lon": point[1], "label": _point_label(value)}

    @app.callback(Output("report-point-label", "children"), Input("report-point", "data"))
    def show_point(point):
        from src.routing.flood_router import place_name_at
        if not point:
            return html.Span("No place chosen yet.", className="twin-empty-state")
        if not _inside_map(point["lat"], point["lon"]):
            return html.Span(f"{point['label']} is outside the map area (Nairobi).", className="twin-auth-error")
        return html.Span([html.I(className="bi bi-geo-alt-fill me-1"), html.Strong(point["label"]),
                          f" · {place_name_at(point['lat'], point['lon'])}"])

    @app.callback(
        Output("report-status", "children", allow_duplicate=True),
        Output("report-modal", "is_open", allow_duplicate=True),
        Output("reports-store", "data", allow_duplicate=True),
        Output("report-note", "value"),
        Input("btn-report-submit", "n_clicks"),
        State("report-point", "data"),
        State("report-kind", "value"),
        State("report-note", "value"),
        prevent_initial_call=True,
    )
    def submit(n, point, kind, note):
        from src.routing.flood_router import place_name_at
        if not n:
            raise PreventUpdate
        user = current_user()
        if not can(user, "report_flood"):
            return html.Span("Please sign in to report flooding.", className="twin-auth-error"), no_update, no_update, no_update
        if not point:
            return html.Span("Choose where the flooding is first.", className="twin-auth-error"), no_update, no_update, no_update
        lat, lon = float(point["lat"]), float(point["lon"])
        if not _inside_map(lat, lon):
            return html.Span("That place is outside the map area.", className="twin-auth-error"), no_update, no_update, no_update
        place = place_name_at(lat, lon)
        try:
            report, merged = get_reports().add(user["user_id"], lat, lon, kind, place, note)
        except ReportError as exc:
            return html.Span(str(exc), className="twin-auth-error"), no_update, no_update, no_update
        _audit("report_confirmed" if merged else "report_created", user,
               f"#{report['report_id']} {kind} in {place}")
        return "", False, reports_payload(), ""

    # Still flooded / cleared, and the moderators' verify / remove.
    @app.callback(
        Output("reports-store", "data", allow_duplicate=True),
        Input({"type": "report-act", "id": ALL, "act": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def act(clicks):
        trig = ctx.triggered_id
        if not trig or not any(clicks or []):
            raise PreventUpdate
        user = current_user()
        rid, action = int(trig["id"]), trig["act"]
        store = get_reports()
        try:
            if action in ("still", "cleared") and can(user, "report_flood"):
                r = store.vote(rid, user["user_id"], action)
                _audit("report_voted", user, f"#{rid} {action}" + (" (closed)" if r["status"] != "active" else ""))
            elif action == "verify" and can(user, "moderate_reports"):
                store.verify(rid, user["user_id"])
                _audit("report_verified", user, f"#{rid}")
            elif action == "remove" and can(user, "moderate_reports"):
                store.remove(rid, user["user_id"])
                _audit("report_removed", user, f"#{rid}")
            else:
                raise PreventUpdate
        except ReportError:
            pass
        return reports_payload()

    # Satellite flood map: the newest pass on or before the time on the map.
    @app.callback(
        Output("satellite-store", "data"),
        Output("satellite-note", "children"),
        Input("satellite-toggle", "value"),
        Input("nowcast-store", "data"),
        State("satellite-store", "data"),
    )
    def satellite_layer(on, nowcast, current):
        if not on:
            return ({"visible": False}, "") if (current or {}).get("visible") else (no_update, "")
        anchor = date.today()
        if nowcast and nowcast.get("is_replay") and nowcast.get("as_of"):
            anchor = datetime.fromisoformat(nowcast["as_of"]).date()
        if current and current.get("visible") and current.get("anchor") == anchor.isoformat():
            raise PreventUpdate
        obs = satellite.observed(anchor)
        if obs is None:
            return ({"visible": False},
                    html.Span("The satellite flood map could not be reached, or no Sentinel-1 pass was found in "
                              "the last two weeks.", className="twin-auth-error"))
        day = date.fromisoformat(obs["date"])
        age = (anchor - day).days
        when = "the same day" if age == 0 else (f"{age} day before" if age == 1 else f"{age} days before")
        note = [
            html.Div([html.Strong(f"Sentinel-1 pass {day.day} {day.strftime('%b %Y')}"),
                      f" ({when} the time on the map): ",
                      f"{obs['flooded_pct']}% of the area seen flooded."]),
            html.Div(f"Radar cannot see water between buildings, so only {obs['visible_pct']}% of the area could be "
                     "checked (the rest is shaded grey). Passes are days apart: this is not a live view.",
                     className="twin-fine-print"),
        ]
        return ({"visible": True, "anchor": anchor.isoformat(), "date": obs["date"],
                 "image": f"/data/satellite-flood.png?anchor={anchor.isoformat()}"}, note)

    app.clientside_callback(
        """
        function(sat) {
            if (!sat) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try { frame.contentWindow.postMessage({type: "satellite", visible: !!sat.visible, image: sat.image || null}, "*"); }
                catch (e) { console.warn("satellite layer update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-satellite", "title"),
        Input("satellite-store", "data"),
        prevent_initial_call=True,
    )

    # Past floods in the news, picking out the replayed storm's week.
    @app.callback(
        Output("history-store", "data"),
        Output("history-note", "children"),
        Input("history-toggle", "value"),
        Input("nowcast-store", "data"),
        State("history-store", "data"),
    )
    def history_layer(on, nowcast, current):
        if not on:
            return ({"visible": False, "rows": []}, "") if (current or {}).get("visible") else (no_update, "")
        around = None
        if nowcast and nowcast.get("is_replay") and nowcast.get("as_of"):
            around = datetime.fromisoformat(nowcast["as_of"]).date()
        key = around.isoformat() if around else None
        if current and current.get("visible") and current.get("around") == key:
            raise PreventUpdate
        info = history.summary(around)
        if not info["events"]:
            return ({"visible": False, "rows": []},
                    html.Span("The past-floods file is missing (run src.ingestion.fetch_groundsource).",
                              className="twin-auth-error"))
        note = [html.Div([html.Strong(f"{info['events']} news reports"),
                          f" of flooding in {info['places']} places, {info['first'].year}–{info['last'].year}. "
                          f"The {info['shown']} neighbourhood-sized areas (up to {history.MAP_MAX_KM2:g} km²) are "
                          "drawn; hover one for how often and when."])]
        if around:
            note.append(html.Div(
                [html.Strong(f"{info['near_places']} place{'s' if info['near_places'] != 1 else ''}"),
                 " reported flooded in the news around this storm (yellow)."]
                if info["near_places"] else "No flooding was reported in the news around this storm."))
        note.append(html.Div("From news articles, read by Google's Gemini (Groundsource, CC BY 4.0). About 60% "
                             "are right in both place and date, and some areas get far more coverage than "
                             "others: no reports does not mean no flooding.", className="twin-fine-print"))
        return {"visible": True, "around": key, "rows": history.map_rows(around)}, note

    app.clientside_callback(
        """
        function(hist) {
            if (!hist) { return window.dash_clientside.no_update; }
            var frame = document.getElementById("3d-map-frame");
            if (frame && frame.contentWindow) {
                try {
                    frame.contentWindow.postMessage({type: "layerUpdate", layers: {
                        "past-floods": hist.visible ? hist.rows : []}}, "*");
                } catch (e) { console.warn("past floods update failed:", e); }
            }
            return "";
        }
        """,
        Output("bridge-sink-history", "title"),
        Input("history-store", "data"),
        prevent_initial_call=True,
    )
