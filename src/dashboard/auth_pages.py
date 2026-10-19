"""
src.dashboard.auth_pages
========================
Role-specific pages and tools: sign-in, the citizen's alert areas, the planner's
and responder's tools beside the map, and the administrator's console.

    /           the map. Public; signed-in users also get their role's tools panel
    /login      sign-in and registration: email + password, or Google (Firebase)
    /2fa        administrators: authenticator code, or first-time app setup
    /2fa-codes  shown once after setup: the recovery codes
    /account, /admin...  the signed-in dashboard (src/dashboard/admin_dashboard.py)

Each page and each tool is rendered on the server for the current user, and
every action re-checks the permission when it runs (src.auth.access).
"""

from __future__ import annotations

import html as html_escape
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import session

from src.auth import google, totp
from src.auth.access import ROLE_LABELS, can, csrf_token, current_user, mfa_subject
from src.auth.store import AuthError, get_store

EAT = timezone(timedelta(hours=3))


# ----------------------------------------------------------------- shared --
def _card(title: str, *children, className: str = "mb-3") -> html.Div:
    return html.Div(className=f"twin-panel {className}", children=[
        html.Div(className="twin-panel-header", children=[
            html.P([html.Span(className="status-dot dot-accent"), title], className="twin-eyebrow")]),
        html.Div(className="twin-panel-body", children=list(children)),
    ])


def _page(title: str, *children) -> html.Div:
    """Frame for the non-map pages: brand, a way back to the map, the content."""
    return dbc.Container(fluid=True, style={"padding": "16px 20px", "maxWidth": "1100px"}, children=[
        html.Div(className="twin-topbar mb-3 pb-3 d-flex align-items-center justify-content-between", children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                html.Div(className="twin-brand-mark"),
                html.Div([html.H2(title, className="twin-title"),
                          html.Span("Nairobi Flood Digital Twin", className="twin-subtitle")]),
            ]),
            dcc.Link("← Back to the map", href="/", className="twin-auth-link"),
        ]),
        *children,
    ])


def _narrow(*children) -> dbc.Row:
    return dbc.Row(justify="center", children=dbc.Col(md=7, lg=5, children=list(children)))


def _denied(message: str) -> html.Div:
    return _page("Not available", _card("Access", html.P(message, className="twin-empty-state"),
                                        dcc.Link("Sign in", href="/login", className="twin-auth-link")))


def _error_from(search: str | None) -> str:
    return (parse_qs((search or "").lstrip("?")).get("error") or [""])[0]


def account_area(user: dict | None) -> list:
    """Top-bar account controls."""
    if not user:
        return [dcc.Link("Sign in", href="/login", className="twin-auth-btn")]
    shown_role = ROLE_LABELS[user.get("real_role") or user["role"]]
    links = [html.Span([html.Strong(user.get("display_name") or user["email"] or user["username"]),
                        f" · {shown_role}"], className="twin-auth-who"),
             dcc.Link("Account", href="/account", className="twin-auth-link")]
    if user.get("mfa_pending"):
        links.append(dcc.Link("Verify 2FA to use admin tools", href="/2fa", className="twin-auth-btn"))
    if can(user, "manage_users"):
        links.append(dcc.Link("Admin", href="/admin", className="twin-auth-link"))
    links.append(html.A("Sign out", href="/auth/logout", className="twin-auth-link"))
    return links


def _field(label: str, name: str, kind: str = "text", auto: str | None = None) -> html.Div:
    fid = f"f-{name}-{kind}"
    return html.Div(className="mb-2", children=[
        html.Label(label, className="twin-field-label", htmlFor=fid),
        dcc.Input(id=fid, name=name, type=kind, required=True, autoComplete=auto,
                  className="form-control twin-input"),
    ])


def _firebase_options() -> list:
    """
    'Continue with Google' and 'Email me a sign-in link', when Firebase is
    configured. Both run in assets/firebase_login.js and work for every account:
    the account is matched by its verified email.
    """
    if not google.firebase_config():
        return []
    return [
        html.Div(className="twin-auth-divider", children=html.Span("or")),
        html.Button([html.Span("G", className="twin-google-g"), "Continue with Google"],
                    id="btn-google", type="button", className="btn twin-google-btn w-100"),
        html.Div(id="google-error", className="twin-auth-error mt-2"),
        html.Div(className="twin-auth-divider", children=html.Span("or, without a password")),
        html.Label("Email me a sign-in link", className="twin-field-label", htmlFor="f-link-email"),
        html.Div(className="d-flex gap-2", children=[
            dcc.Input(id="f-link-email", type="email", placeholder="you@example.com", autoComplete="email",
                      className="form-control twin-input"),
            html.Button("Send link", id="btn-email-link", type="button",
                        className="btn btn-outline-info text-nowrap"),
        ]),
        html.Div(id="link-status", className="mt-2"),
    ]


# ------------------------------------------------------------------ login --
def render_login(search: str | None) -> html.Div:
    """Sign in or register: email and password, or Google, for every account."""
    q = parse_qs((search or "").lstrip("?"))
    error = (q.get("error") or [""])[0]
    tab = (q.get("tab") or ["login"])[0]
    token = csrf_token()

    login_form = html.Form(action="/auth/login", method="POST", children=[
        dcc.Input(type="hidden", name="csrf", value=token),
        _field("Email", "email", "email", auto="email"),
        _field("Password", "password", "password", auto="current-password"),
        html.Button("Sign in", type="submit", className="btn btn-primary w-100 fw-bold mt-2"),
    ])
    register_form = html.Form(action="/auth/register", method="POST", children=[
        dcc.Input(type="hidden", name="csrf", value=token),
        _field("Email", "email", "email", auto="email"),
        _field("Password (at least 8 characters)", "password", "password", auto="new-password"),
        _field("Confirm password", "confirm", "password", auto="new-password"),
        html.Button("Create citizen account", type="submit", className="btn btn-primary w-100 fw-bold mt-2"),
        html.P("Planner, responder and administrator accounts are created by an administrator.",
               className="twin-empty-state mt-2"),
    ])

    return _page("Sign in", _narrow(
        _card("Account",
              html.Div(error, className="twin-auth-error mb-2") if error else None,
              dbc.Tabs(active_tab=tab if tab in ("login", "register") else "login", className="mb-3", children=[
                  dbc.Tab(login_form, tab_id="login", label="Sign in"),
                  dbc.Tab(register_form, tab_id="register", label="Register"),
              ]),
              *_firebase_options()),
        html.P("The flood map, outlook and routing are open to everyone without an account. "
               "Sign in with your password, Google or an emailed link to subscribe to alerts for "
               "your area or to use staff tools. Administrators also confirm a code from their "
               "authenticator app.",
               className="twin-empty-state"),
    ))


# ---------------------------------------------------- authenticator (2FA) --
def render_2fa(search: str | None) -> html.Div:
    """The second step for administrators: enter a code, or set up the app the first time."""
    user = mfa_subject()
    if user is None:
        return _page("Two-step sign-in", _narrow(_card(
            "Sign-in expired",
            html.P("Your sign-in has expired or is already complete. Sign in again to continue.",
                   className="twin-empty-state"),
            dcc.Link("Sign in", href="/login", className="twin-auth-btn"))))
    error = _error_from(search)
    form_children = [
        dcc.Input(type="hidden", name="csrf", value=csrf_token()),
        html.Div(className="mb-2", children=[
            html.Label("6-digit code", className="twin-field-label", htmlFor="f-code"),
            dcc.Input(id="f-code", name="code", type="text", required=True, autoComplete="one-time-code",
                      inputMode="numeric", maxLength=12, className="form-control twin-input twin-code-input"),
        ]),
        html.Button("Verify", type="submit", className="btn btn-primary w-100 fw-bold mt-2"),
    ]

    if user["totp_enabled"]:
        body = [
            html.P(["Signed in as ", html.Strong(user["email"]), ". Open your authenticator app and enter the "
                    "current code for ", html.Strong(totp.ISSUER), "."], className="twin-empty-state"),
            html.Div(error, className="twin-auth-error mb-2") if error else None,
            html.Form(action="/auth/2fa", method="POST", children=form_children),
            html.P("Lost your phone? Enter one of your recovery codes instead (for example k7m2-q9xd).",
                   className="twin-empty-state mt-3"),
        ]
        return _page("Two-step sign-in", _narrow(_card("Authenticator code", *body),
                                                 html.A("Cancel and sign out", href="/auth/logout",
                                                        className="twin-auth-link")))

    # First time: a new secret for this sign-in, confirmed before it is saved.
    secret = session.get("totp_setup")
    if not secret:
        secret = session["totp_setup"] = totp.new_secret()
    uri = totp.provisioning_uri(secret, user["email"])
    grouped = " ".join(secret[i:i + 4] for i in range(0, len(secret), 4))
    body = [
        html.P("Administrators must use an authenticator app. Set it up once:", className="twin-empty-state"),
        html.Ol(className="twin-steps", children=[
            html.Li("Install Google Authenticator, Microsoft Authenticator or Authy on your phone."),
            html.Li("In the app, add an account and scan this QR code."),
            html.Li("Enter the 6-digit code the app shows."),
        ]),
        html.Div(className="twin-qr", children=html.Img(src=totp.qr_svg_data_uri(uri), alt="Authenticator QR code",
                                                        width=210, height=210)),
        html.P(["Cannot scan? Enter this key in the app: ", html.Code(grouped, className="twin-secret")],
               className="twin-empty-state"),
        html.Div(error, className="twin-auth-error mb-2") if error else None,
        html.Form(action="/auth/2fa", method="POST", children=form_children),
    ]
    return _page("Set up two-step sign-in", _narrow(_card("Authenticator app", *body),
                                                    html.A("Cancel and sign out", href="/auth/logout",
                                                           className="twin-auth-link")))


def render_recovery_codes() -> html.Div:
    """The recovery codes, shown once straight after setup."""
    codes = session.pop("recovery_codes_once", None)
    if not codes:
        return _page("Recovery codes", _narrow(_card(
            "Recovery codes", html.P("Recovery codes are shown only once, right after setup.",
                                     className="twin-empty-state"),
            dcc.Link("Go to the admin console", href="/admin", className="twin-auth-btn"))))
    return _page("Recovery codes", _narrow(_card(
        "Save these recovery codes",
        html.P("Two-step sign-in is on. If you lose your phone, each of these codes signs you in once. "
               "Store them somewhere safe; they will not be shown again.", className="twin-empty-state"),
        html.Div(className="twin-recovery-grid", children=[html.Code(c) for c in codes]),
        html.Div(className="mt-3", children=dcc.Link("I have saved them: go to the admin console", href="/admin",
                                                    className="twin-auth-btn")),
    )))


# ---------------------------------------------------------------- account --
def _place_names() -> list[str]:
    from src.dashboard.layouts import place_options
    return [o["label"] for o in place_options()]


def _alerts_for(user: dict, nowcast: dict | None) -> list:
    """Warnings in the live outlook that touch this user's alert areas."""
    subs = set(get_store().subscriptions(user["user_id"]))
    if not subs:
        return [html.P(["No alert areas chosen. ",
                        dcc.Link("Choose your areas", href="/account", className="twin-auth-link"), "."],
                       className="twin-empty-state")]
    if not nowcast or nowcast.get("error") or nowcast.get("is_replay"):
        return [html.P("The live outlook is not loaded; alerts appear here when it is.",
                       className="twin-empty-state")]
    rows = []
    for w in nowcast.get("warnings") or []:
        hit = sorted(subs.intersection(w.get("places") or []))
        if not hit:
            continue
        when = "now" if w["kind"] in ("ongoing", "rising") else f"in about {w['hour']} h"
        rows.append(html.Div(className=f"twin-warning lvl-{['CLEAR', 'MODERATE', 'HIGH', 'CRITICAL'].index(w['level'])}",
                             children=[html.Div(when.upper(), className="twin-when"),
                                       html.Div([html.Div(w["text"], className="what"),
                                                 html.Div(", ".join(hit), className="where")])]))
    if not rows:
        return [html.P(f"No flooding expected in your {len(subs)} alert area(s) in the next 12 hours.",
                       className="twin-empty-state is-good")]
    return rows


# ------------------------------------------------------------- role tools --
def role_tools(user: dict | None) -> list:
    """The panel beside the map, built for this user's role."""
    if not user:
        return [_card("Accounts",
                      html.P("Sign in to get flood alerts for your area. Planners and responders sign in "
                             "for scenario, export and report tools.", className="twin-empty-state"),
                      dcc.Link("Sign in or register", href="/login", className="twin-auth-btn"))]

    panels = [_card("My Flood Alerts", html.Div(id="my-alerts-body", children=html.P(
        "Waiting for the live outlook…", className="twin-empty-state")),
        dcc.Link("Manage alert areas", href="/account", className="twin-auth-link"))]

    if can(user, "compare") or can(user, "export_map"):
        # Every tool is always in the panel and hidden when the role lacks it, so
        # the callbacks that share these buttons always find their inputs. The
        # callbacks still check the permission; hiding is not the control.
        def shown(perm):
            return None if can(user, perm) else {"display": "none"}

        tools = [
            html.Div(style=shown("compare"), children=[
                html.P("Hold the scenario on the map, change the rainfall, and compare.",
                       className="twin-empty-state"),
                html.Div(className="d-flex gap-2 mb-2", children=[
                    dbc.Button("Hold this scenario", id="btn-hold", size="sm", color="outline-info"),
                    dbc.Button("Clear", id="btn-hold-clear", size="sm", color="outline-secondary"),
                ]),
                html.Div(id="compare-body", className="mb-2"),
            ]),
            dbc.Button("⬇ Export flood map (GeoJSON)", id="btn-export", size="sm", style=shown("export_map"),
                       color="outline-info", className="w-100 mb-2 twin-glyph-btn"),
            dbc.Button("⬇ Emergency report", id="btn-report", size="sm", style=shown("emergency_report"),
                       color="primary", className="w-100 mb-2"),
            html.Div(id="tools-status"),
        ]
        title = {"planner": "Planning Tools", "responder": "Response Tools", "admin": "Staff Tools"}[user["role"]]
        panels.insert(0, _card(title, *tools))
    return panels


def _fmt_summary_rows(held: dict, cur: dict) -> html.Table:
    def delta(a, b, unit="", digits=0):
        d = b - a
        sign = "+" if d > 0 else ""
        return f"{sign}{d:,.{digits}f}{unit}"

    def at_risk(s):
        return sum(1 for r in s["regions"] if r["level"] in ("CRITICAL", "HIGH"))

    rows = [
        ("Rainfall (3 days)", f"{held['rain']:.0f} mm", f"{cur['rain']:.0f} mm", delta(held["rain"], cur["rain"], " mm")),
        ("Flooded area", f"{held['area_km2']:.2f} km²", f"{cur['area_km2']:.2f} km²",
         delta(held["area_km2"], cur["area_km2"], " km²", 2)),
        ("People at risk", f"{held['pop']:,}", f"{cur['pop']:,}", delta(held["pop"], cur["pop"])),
        ("Places high or critical", str(at_risk(held)), str(at_risk(cur)), delta(at_risk(held), at_risk(cur))),
    ]
    new = sorted({r["name"] for r in cur["regions"]} - {r["name"] for r in held["regions"]})
    table = html.Table(className="table table-sm twin-table mb-1", children=[
        html.Thead(html.Tr([html.Th(""), html.Th("Held"), html.Th("Current"), html.Th("Change")])),
        html.Tbody([html.Tr([html.Td(a), html.Td(b), html.Td(c), html.Td(d)]) for a, b, c, d in rows]),
    ])
    note = html.Small(f"Newly flooded in the current scenario: {', '.join(new[:8])}"
                      + (f" and {len(new) - 8} more" if len(new) > 8 else "") if new else
                      "No newly flooded places in the current scenario.", className="twin-mono-meta")
    return html.Div([html.Small(f"Held: {held['label']}  ·  Current: {cur['label']}",
                                className="twin-mono-meta d-block mb-1"), table, note])


def emergency_report_html(summary: dict, user: dict, nowcast: dict | None) -> str:
    """A self-contained, printable HTML report (FR-11)."""
    e = html_escape.escape
    now = datetime.now(EAT).strftime("%d %b %Y, %H:%M EAT")
    regions = sorted(summary["regions"], key=lambda r: (["CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE"]
                                                        .index(r["level"]) if r["level"] in
                                                        ("CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE") else 9,
                                                        -r["flooded_pct"]))
    region_rows = "".join(
        f"<tr><td>{e(r['name'])}</td><td class='lvl {e(r['level']).lower()}'>{e(r['level'])}</td>"
        f"<td>{r['peak_pct']}%</td><td>{r['flooded_pct']}%</td></tr>" for r in regions
    ) or "<tr><td colspan='4'>No place is predicted to flood in this scenario.</td></tr>"
    warnings = ""
    if summary["mode"] == "LIVE" and nowcast and nowcast.get("warnings"):
        items = "".join(f"<li><b>{e(w['level'].title())}</b>, "
                        f"{'now' if w['kind'] in ('ongoing', 'rising') else 'in about ' + str(w['hour']) + ' h'}: "
                        f"{e(w['places_text'])}</li>" for w in nowcast["warnings"][:15])
        warnings = f"<h2>Timed warnings (12-hour outlook)</h2><ul>{items}</ul>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Emergency flood report - {e(now)}</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;color:#111;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.45}}
 h1{{font-size:22px;margin-bottom:2px}} h2{{font-size:16px;margin-top:22px;border-bottom:1px solid #ccc;padding-bottom:3px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ccc;padding:4px 6px;text-align:left}}
 th{{background:#f1f3f5}} .meta{{color:#555;font-size:13px}} .lvl{{font-weight:bold}}
 .critical{{color:#b3001b}} .high{{color:#c25400}} .moderate{{color:#8a6100}}
 .kpi{{display:flex;gap:10px;flex-wrap:wrap}} .kpi div{{border:1px solid #ccc;border-radius:6px;padding:8px 12px;min-width:150px}}
 .kpi b{{display:block;font-size:18px}} .note{{font-size:12px;color:#444;background:#f7f7f7;padding:8px;border-left:3px solid #888}}
 @media print{{body{{margin:0}}}}
</style></head><body>
<h1>Emergency Flood Report: Nairobi</h1>
<div class="meta">Generated {e(now)} by {e(user.get('email') or user['username'])} ({e(ROLE_LABELS[user['role']])}) ·
Model {e(summary.get('model') or 'U-Net Model B')} · Nairobi Urban Flood Digital Twin</div>
<h2>Scenario</h2>
<p>{e(summary['label'])}. Three-day rainfall total used by the model: <b>{summary['rain']:.0f} mm</b>.</p>
<div class="kpi">
 <div>Predicted flooded area<b>{summary['area_km2']:.2f} km²</b>{summary['flooded_pct']:.1f}% of the mapped area</div>
 <div>People living in predicted flood cells<b>{summary['pop']:,}</b>WorldPop 2025, upper bound</div>
 <div>Places at high or critical risk<b>{sum(1 for r in regions if r['level'] in ('CRITICAL', 'HIGH'))}</b>of {len(regions)} flooded places</div>
</div>
{warnings}
<h2>Affected settlements</h2>
<table><tr><th>Place</th><th>Risk level</th><th>Peak flood probability</th><th>Share of place flooded</th></tr>
{region_rows}</table>
<h2>How to read this report</h2>
<p class="note">Values are probabilities of flooding from a U-Net model, validated at neighbourhood scale (about 70 m cells),
not street by street, and not water depths. People at risk counts residents of predicted flood cells, which is broader than
people affected. Live timing follows the rainfall forecast crossing the model's flooding level; it is not a simulation of
water flow. Verify on the ground before deploying resources.</p>
</body></html>"""


# -------------------------------------------------------------- callbacks --
def register_auth_callbacks(app) -> None:
    # The page for this address, and the role-specific parts of the map page.
    @app.callback(
        Output("page-twin", "hidden"),
        Output("page-other", "hidden"),
        Output("page-other", "children"),
        Output("account-area", "children"),
        Output("role-tools-panel", "children"),
        Output("mode-radio", "options"),
        Output("btn-run", "style"),
        Output("history-wrap", "hidden"),
        Input("url", "pathname"),
        Input("url", "search"),
        State("resolved-theme-store", "data"),
    )
    def route_page(pathname, search, theme):
        from src.dashboard.admin_dashboard import render_account, render_admin
        user = current_user()
        path = (pathname or "/").rstrip("/") or "/"
        modes = [{"label": "Live forecast", "value": "LIVE"}]
        if can(user, "run_whatif"):
            modes.append({"label": "What-if", "value": "WHATIF"})
        run_style = None if can(user, "save_scenario") else {"display": "none"}
        twin_parts = (account_area(user), role_tools(user), modes, run_style, not can(user, "save_scenario"))
        if path == "/login":
            if user:
                return (False, True, None) + twin_parts
            return (True, False, render_login(search)) + twin_parts
        if path == "/2fa":
            return (True, False, render_2fa(search)) + twin_parts
        if path == "/2fa-codes":
            return (True, False, render_recovery_codes()) + twin_parts
        if path == "/account":
            return (True, False, render_account(user, theme or "dark")) + twin_parts
        if path == "/admin" or path.startswith("/admin/"):
            return (True, False, render_admin(user, path, theme or "dark")) + twin_parts
        return (False, True, None) + twin_parts

    # Google sign-in: the popup and token exchange run in assets/firebase_login.js.

    @app.callback(Output("my-alerts-body", "children"),
                  Input("nowcast-store", "data"), Input("role-tools-panel", "children"))
    def my_alerts(nowcast, _panel):
        user = current_user()
        if not can(user, "subscribe_alerts"):
            raise PreventUpdate
        return _alerts_for(user, nowcast)

    @app.callback(Output("subs-status", "children"), Output("account-alerts", "children"),
                  Input("btn-save-subs", "n_clicks"), State("sub-places", "value"),
                  prevent_initial_call=True)
    def save_subs(_n, places):
        if not _n:
            raise PreventUpdate
        user = current_user()
        if not can(user, "subscribe_alerts"):
            return html.Span("Please sign in again.", className="twin-auth-error"), no_update
        valid = set(_place_names())
        saved = get_store().set_subscriptions(user["user_id"], [p for p in places or [] if p in valid])
        get_store().audit("subscriptions_updated", user["user_id"], f"{len(saved)} areas")
        from src.forecast.nowcast import cached_nowcast
        nc = cached_nowcast("live")
        return (html.Span(f"Saved {len(saved)} alert area(s).", className="twin-auth-ok"),
                _alerts_for(user, nc.to_client() if nc else None))

    @app.callback(Output("pw-status", "children"),
                  Input("btn-change-pw", "n_clicks"),
                  State("pw-current", "value"), State("pw-new", "value"), State("pw-confirm", "value"),
                  prevent_initial_call=True)
    def change_password(_n, current, new, confirm):
        if not _n:
            raise PreventUpdate
        user = current_user()
        store = get_store()
        if not user or not user["has_password"]:
            return html.Span("Please sign in again.", className="twin-auth-error")
        if not store.verify_password(user["email"], current or ""):
            store.audit("password_change_failed", user["user_id"])
            return html.Span("Your current password is not correct.", className="twin-auth-error")
        if (new or "") != (confirm or ""):
            return html.Span("The new passwords do not match.", className="twin-auth-error")
        try:
            store.set_password(user["user_id"], new or "")
        except AuthError as exc:
            return html.Span(str(exc), className="twin-auth-error")
        store.audit("password_changed", user["user_id"])
        return html.Span("Password changed.", className="twin-auth-ok")

    # ---- planner / responder tools ----
    @app.callback(Output("held-scenario-store", "data"),
                  Input("btn-hold", "n_clicks"), Input("btn-hold-clear", "n_clicks"),
                  State("scenario-summary-store", "data"), prevent_initial_call=True)
    def hold(_h, _c, summary):
        if not (_h or _c):
            raise PreventUpdate
        if not can(current_user(), "compare"):
            raise PreventUpdate
        if ctx.triggered_id == "btn-hold-clear":
            return None
        return summary

    @app.callback(Output("compare-body", "children"),
                  Input("held-scenario-store", "data"), Input("scenario-summary-store", "data"),
                  Input("role-tools-panel", "children"))
    def compare(held, current, _panel):
        if not can(current_user(), "compare"):
            raise PreventUpdate
        if not held:
            return html.Small("Nothing held yet.", className="twin-mono-meta")
        if not current:
            raise PreventUpdate
        return _fmt_summary_rows(held, current)

    @app.callback(Output("download-file", "data"), Output("tools-status", "children"),
                  Input("btn-export", "n_clicks"), Input("btn-report", "n_clicks"),
                  State("flood-geojson-store", "data"), State("scenario-summary-store", "data"),
                  State("nowcast-store", "data"), prevent_initial_call=True)
    def download(_e, _r, flood, summary, nowcast):
        # Dash fires these once when the panel is inserted; act only on a click.
        if not (_e or _r):
            raise PreventUpdate
        user = current_user()
        store = get_store()
        if not summary:
            return no_update, html.Small("The map is still loading.", className="twin-mono-meta")
        stamp = datetime.now(EAT).strftime("%Y%m%d_%H%M")
        if ctx.triggered_id == "btn-export":
            if not can(user, "export_map"):
                return no_update, html.Span("Not permitted for your role.", className="twin-auth-error")
            fc = {
                "type": "FeatureCollection",
                "metadata": {"scenario": summary["label"], "rainfall_3day_mm": summary["rain"],
                             "flooded_area_km2": summary["area_km2"], "people_at_risk": summary["pop"],
                             "model": summary.get("model"), "crs": "EPSG:4326 (WGS 84)",
                             "values": "flood probability bands, not water depth",
                             "exported_by": user.get("email") or user["username"], "exported_at": stamp},
                # Display colours stay behind; GIS users get the level, place and probability.
                "features": [
                    {"type": "Feature", "geometry": f.get("geometry"),
                     "properties": {k: v for k, v in (f.get("properties") or {}).items()
                                    if k in ("level", "name", "region", "value", "value_label")}}
                    for f in ((flood or {}).get("main") or {}).get("features") or []
                ],
            }
            store.audit("export_geojson", user["user_id"], summary["label"])
            return (dict(content=json.dumps(fc), filename=f"flood_extent_{stamp}.geojson",
                         type="application/geo+json"),
                    html.Small(f"Exported {len(fc['features'])} flood areas.", className="twin-mono-meta"))
        if not can(user, "emergency_report"):
            return no_update, html.Span("Not permitted for your role.", className="twin-auth-error")
        store.audit("emergency_report", user["user_id"], summary["label"])
        return (dict(content=emergency_report_html(summary, user, nowcast),
                     filename=f"emergency_report_{stamp}.html", type="text/html"),
                html.Small("Report downloaded. Open it and print to PDF if needed.", className="twin-mono-meta"))
