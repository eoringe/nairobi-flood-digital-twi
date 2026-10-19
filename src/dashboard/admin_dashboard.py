"""
src.dashboard.admin_dashboard
=============================
The signed-in workspace: a sidebar dashboard for the account page and the
administrator's console.

    /account          any signed-in user: profile, alert areas, sign-in settings
    /admin            administrators: overview (KPIs, sign-in trend, activity, health)
    /admin/users      users and roles: search, filter, create, change role, disable
    /admin/activity   the audit log, filtered by kind or text
    /admin/system     model, forecast feed and service status

Pages are rendered on the server for the current user; every action checks the
permission again when it runs (src.auth.access).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from src.auth.access import ROLE_LABELS, can, current_user
from src.auth.store import ROLES, AuthError, get_store

ROLE_TONES = {"citizen": "neutral", "planner": "accent", "responder": "warn", "admin": "danger"}

SIGN_IN = ("login",)
FAILURES = ("login_failed", "2fa_failed", "login_locked", "2fa_locked", "password_change_failed")
ACTION_KINDS = {
    "signin": ("Sign-ins", {"login", "login_first_step", "logout", "register"}),
    "failure": ("Failures", set(FAILURES)),
    "account": ("Account changes", {"user_created", "user_updated", "admin_bootstrapped", "2fa_enrolled",
                                    "2fa_reset", "password_changed", "password_reset", "subscriptions_updated"}),
    "data": ("Data and reports", {"scenario_saved", "export_geojson", "emergency_report", "forecast_refreshed"}),
}
ACTION_LABELS = {
    "login": "Signed in", "login_first_step": "Password or Google step passed", "logout": "Signed out",
    "register": "Registered", "login_failed": "Sign-in failed", "2fa_failed": "Wrong authenticator code",
    "login_locked": "Sign-in locked", "2fa_locked": "Authenticator locked",
    "password_change_failed": "Password change failed", "user_created": "User created",
    "user_updated": "User updated", "admin_bootstrapped": "First admin created", "2fa_enrolled": "2FA enrolled",
    "2fa_reset": "2FA reset", "password_changed": "Password changed", "password_reset": "Password reset",
    "subscriptions_updated": "Alert areas updated", "scenario_saved": "Scenario saved",
    "export_geojson": "Flood map exported", "emergency_report": "Emergency report", "forecast_refreshed":
        "Forecast refreshed",
}


def _kind_of(action: str) -> str:
    for kind, (_label, actions) in ACTION_KINDS.items():
        if action in actions:
            return kind
    return "other"


def _utc(days: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


def _icon(name: str, cls: str = "") -> html.I:
    return html.I(className=f"bi bi-{name} {cls}".strip())


def _initials(user: dict) -> str:
    name = (user.get("display_name") or user.get("email") or "?").split("@")[0]
    parts = [p for p in name.replace(".", " ").replace("_", " ").split() if p]
    return ("".join(p[0] for p in parts[:2]) or name[:2]).upper()


def _avatar(user: dict, size: str = "") -> html.Span:
    return html.Span(_initials(user), className=f"dash-avatar tone-{ROLE_TONES.get(user['role'], 'neutral')} {size}")


def _role_badge(role: str) -> html.Span:
    return html.Span(ROLE_LABELS[role], className=f"dash-badge tone-{ROLE_TONES[role]}")


def _panel(title: str, *children, icon: str | None = None, actions=None, className: str = "") -> html.Div:
    head = [html.H3([_icon(icon, "me-2") if icon else None, title], className="dash-panel-title")]
    if actions is not None:
        head.append(html.Div(actions, className="dash-panel-actions"))
    return html.Section(className=f"dash-panel {className}".strip(), children=[
        html.Header(head, className="dash-panel-head"),
        html.Div(list(children), className="dash-panel-body"),
    ])


def _kpi(label: str, value, icon: str, tone: str = "accent", sub=None) -> html.Div:
    return html.Div(className="dash-kpi", children=[
        html.Div(_icon(icon), className=f"dash-kpi-icon tone-{tone}"),
        html.Div(className="dash-kpi-text", children=[
            html.Span(label, className="dash-kpi-label"),
            html.Span(value, className="dash-kpi-value"),
            html.Span(sub, className="dash-kpi-sub") if sub is not None else None,
        ]),
    ])


# ------------------------------------------------------------------ shell --
def shell(path: str, user: dict, title: str, subtitle: str, *content, actions=None) -> html.Div:
    """Sidebar + header frame shared by the account page and the admin console."""
    def link(label, href, icon):
        active = path == href
        return dcc.Link([_icon(icon), html.Span(label)], href=href,
                        className="dash-nav-link" + (" is-active" if active else ""))

    nav = [html.P("Workspace", className="dash-nav-section"),
           link("Live map", "/", "map"),
           link("My account", "/account", "person-circle")]
    if can(user, "manage_users"):
        nav += [html.P("Administration", className="dash-nav-section"),
                link("Overview", "/admin", "speedometer2"),
                link("Users", "/admin/users", "people"),
                link("Activity", "/admin/activity", "clock-history"),
                link("System", "/admin/system", "cpu")]
    elif user.get("mfa_pending"):
        nav += [html.P("Administration", className="dash-nav-section"),
                dcc.Link([_icon("shield-lock"), html.Span("Verify 2FA")], href="/2fa", className="dash-nav-link")]

    shown_role = user.get("real_role") or user["role"]
    sidebar = html.Aside(className="dash-sidebar", children=[
        html.Div(className="dash-brand", children=[
            html.Div(className="twin-brand-mark"),
            html.Div([html.Span("Flood Twin", className="dash-brand-name"),
                      html.Span("Nairobi", className="dash-brand-sub")]),
        ]),
        html.Nav(nav, className="dash-nav"),
        html.Div(className="dash-sidebar-foot", children=[
            html.Div(className="dash-user", children=[
                _avatar({**user, "role": shown_role}),
                html.Div([html.Span(user.get("display_name") or user["email"], className="dash-user-name"),
                          html.Span(ROLE_LABELS[shown_role], className="dash-user-role")]),
            ]),
            html.A([_icon("box-arrow-right"), " Sign out"], href="/auth/logout", className="dash-signout"),
        ]),
    ])
    header = html.Header(className="dash-header", children=[
        html.Div([html.H1(title, className="dash-title"), html.P(subtitle, className="dash-subtitle")]),
        html.Div(actions, className="dash-header-actions") if actions is not None else None,
    ])
    return html.Div(className="dash-shell", children=[
        sidebar, html.Main(className="dash-main", children=[header, *content])])


# ------------------------------------------------------------------ users --
def _methods(u: dict) -> list:
    chips = [html.Span("Google", className="dash-chip"), html.Span("Email link", className="dash-chip")]
    if u["role"] == "admin":
        chips.append(html.Span([_icon("shield-check", "me-1"), "2FA"], className="dash-chip tone-ok")
                     if u["totp_enabled"] else html.Span("2FA pending", className="dash-chip tone-warn"))
    return chips


def users_table(query: str | None = None, role: str | None = None) -> html.Div:
    q = (query or "").strip().lower()
    users = [u for u in get_store().list_users()
             if (not role or role == "all" or u["role"] == role)
             and (not q or q in (u["email"] or "").lower() or q in (u.get("display_name") or "").lower())]
    if not users:
        return html.P("No users match.", className="dash-empty")
    rows = []
    for u in users:
        rows.append(html.Tr(className="" if u["active"] else "is-disabled", children=[
            html.Td(html.Div(className="dash-user-cell", children=[
                _avatar(u, "is-sm"),
                html.Div([html.Span(u.get("display_name") or u["email"], className="dash-cell-strong"),
                          html.Span(u["email"], className="dash-cell-muted") if u.get("display_name") else None]),
            ])),
            html.Td(html.Div(_methods(u), className="dash-chips")),
            html.Td(dcc.Dropdown(id={"type": "role-dd", "uid": u["user_id"]},
                                 options=[{"label": ROLE_LABELS[r], "value": r} for r in ROLES],
                                 value=u["role"], clearable=False, className="dash-select")),
            html.Td(dbc.Switch(id={"type": "active-sw", "uid": u["user_id"]}, value=u["active"],
                               label="Active" if u["active"] else "Disabled", className="dash-switch")),
            html.Td(html.Span((u.get("last_login_at") or "Never").replace("T", " ")[:16], className="dash-cell-muted")),
            html.Td(dbc.Button("Save", id={"type": "save-user", "uid": u["user_id"]}, size="sm",
                               color="dash", className="dash-btn-ghost")),
        ]))
    return html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("User", "Sign-in", "Role", "Status", "Last sign-in (UTC)", "")])),
        html.Tbody(rows),
    ]))


def _new_user_modal() -> dbc.Modal:
    return dbc.Modal(id="new-user-modal", is_open=False, centered=True, className="dash-modal", children=[
        dbc.ModalHeader(dbc.ModalTitle("New user")),
        dbc.ModalBody([
            html.Label("Email", className="dash-label", htmlFor="new-email"),
            dbc.Input(id="new-email", type="email", placeholder="name@example.com", className="dash-input mb-3"),
            html.Label("Role", className="dash-label"),
            dcc.Dropdown(id="new-role", options=[{"label": ROLE_LABELS[r], "value": r} for r in ROLES],
                         value="planner", clearable=False, className="dash-select mb-3"),
            html.P("They sign in with Google or an emailed link for this address; no password is set. "
                   "Administrators also set up an authenticator app at their first sign-in.",
                   className="dash-hint"),
            html.Div(id="create-user-status", className="mt-2"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-cancel-new-user", color="dash", className="dash-btn-ghost"),
            dbc.Button("Create user", id="btn-create-user", className="dash-btn-primary"),
        ]),
    ])


# --------------------------------------------------------------- activity --
def audit_table(query: str | None = None, kind: str | None = None, limit: int = 60) -> html.Div:
    q = (query or "").strip().lower()
    matches = [a for a in get_store().recent_audit(1000)
               if (not kind or kind == "all" or _kind_of(a["action"]) == kind)
               and (not q or q in (a.get("email") or "").lower() or q in a["action"].lower()
                    or q in (ACTION_LABELS.get(a["action"], "")).lower()
                    or q in (a.get("detail") or "").lower())]
    entries = matches[:limit]
    if not entries:
        return html.P("No activity matches.", className="dash-empty")
    rows = [html.Tr([
        html.Td(html.Span(a["occurred_at"].replace("T", " ")[:19], className="dash-cell-muted dash-mono")),
        html.Td(a.get("email") or html.Span("system / anonymous", className="dash-cell-muted")),
        html.Td(html.Span(ACTION_LABELS.get(a["action"], a["action"]),
                          className=f"dash-badge kind-{_kind_of(a['action'])}")),
        html.Td(html.Span(a.get("detail") or "", className="dash-cell-muted")),
    ]) for a in entries]
    more = (html.P(f"Showing the latest {len(entries)} of {len(matches)} events. Search or filter to narrow them.",
                   className="dash-hint mt-3 mb-0") if len(matches) > len(entries) else None)
    return html.Div([html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Time (UTC)", "User", "Event", "Detail")])),
        html.Tbody(rows),
    ])), more])


def _activity_feed(limit: int = 8) -> list:
    entries = get_store().recent_audit(limit)
    if not entries:
        return [html.P("No activity yet.", className="dash-empty")]
    items = []
    for a in entries:
        kind = _kind_of(a["action"])
        icon = {"signin": "box-arrow-in-right", "failure": "exclamation-triangle", "account": "person-gear",
                "data": "file-earmark-arrow-down"}.get(kind, "dot")
        items.append(html.Li(className="dash-feed-item", children=[
            html.Span(_icon(icon), className=f"dash-feed-icon kind-{kind}"),
            html.Div([html.Span(ACTION_LABELS.get(a["action"], a["action"]), className="dash-cell-strong"),
                      html.Span(f" · {a.get('email') or 'anonymous'}", className="dash-cell-muted"),
                      html.Div(a["occurred_at"].replace("T", " ")[:16] + " UTC", className="dash-feed-time")]),
        ]))
    return [html.Ul(items, className="dash-feed")]


# ----------------------------------------------------------------- system --
def health() -> list[tuple[str, str, str]]:
    """(component, state 'ok' | 'warn' | 'down', detail) for the system panel."""
    from src.dashboard.callbacks import predictor, warmup_status
    from src.forecast.nowcast import cached_nowcast
    status = warmup_status()
    nc = cached_nowcast("live")
    if nc is None:
        forecast = ("warn", "Not loaded yet")
    elif nc.error and not nc.hours:
        forecast = ("down", nc.error)
    elif (nc.notice or "").startswith("Offline"):
        forecast = ("warn", f"Offline copy, issued {nc.hours[0].strftime('%H:%M')}")
    else:
        forecast = ("ok", f"{nc.provider}, issued {nc.hours[0].strftime('%H:%M')}")
    try:
        get_store().count_active_admins()
        accounts = ("ok", "Reachable")
    except Exception as exc:                                # noqa: BLE001
        accounts = ("down", str(exc))
    return [
        ("Flood model", "ok", predictor.model_version),
        ("Warm-up", "ok" if status.get("warmup_complete") else "warn",
         "Complete" if status.get("warmup_complete") else "In progress"),
        ("Road network", "ok" if status.get("road_network_loaded") else "warn",
         "Loaded" if status.get("road_network_loaded") else "Loading"),
        ("Live forecast (Open-Meteo)", *forecast),
        ("Account database", *accounts),
    ]


def _health_list(items) -> html.Ul:
    return html.Ul(className="dash-health", children=[
        html.Li([html.Span(className=f"dash-dot state-{state}"),
                 html.Span(name, className="dash-cell-strong"),
                 html.Span(detail if len(detail) <= 70 else detail[:67].rstrip() + "…",
                           title=detail, className="dash-health-detail")]) for name, state, detail in items])


# ----------------------------------------------------------------- charts --
def _chart_colours(theme: str) -> dict:
    from src.dashboard.callbacks import CHART_THEME
    return CHART_THEME.get(theme or "dark", CHART_THEME["dark"])


def _style(fig: go.Figure, theme: str, height: int) -> go.Figure:
    c = _chart_colours(theme)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=c["text"], size=11,
                                                              family="IBM Plex Sans, Segoe UI, sans-serif"),
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
                      bargap=0.35, hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=c["grid"], tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=c["grid"], zeroline=False, rangemode="tozero", tickformat="d")
    return fig


def signin_chart(theme: str, days: int = 14) -> go.Figure:
    store = get_store()
    since = (datetime.now(timezone.utc) - timedelta(days=days - 1)).date()
    ok = store.actions_by_day(SIGN_IN, since.isoformat())
    bad = store.actions_by_day(FAILURES, since.isoformat())
    dates = [since + timedelta(days=i) for i in range(days)]
    keys = [d.isoformat() for d in dates]
    labels = [d.strftime("%d %b") for d in dates]
    c = _chart_colours(theme)
    fig = go.Figure([
        go.Bar(x=labels, y=[ok.get(k, 0) for k in keys], name="Successful", marker_color=c["accent"]),
        go.Bar(x=labels, y=[bad.get(k, 0) for k in keys], name="Failed or locked", marker_color="#ef4459"),
    ])
    fig.update_layout(barmode="stack")
    return _style(fig, theme, 240)


def roles_chart(theme: str) -> go.Figure:
    counts = get_store().role_counts()
    colours = {"citizen": "#7b8794", "planner": _chart_colours(theme)["accent"], "responder": "#f0a35e",
               "admin": "#ef4459"}
    order = list(reversed(ROLES))
    fig = go.Figure(go.Bar(x=[counts[r] for r in order], y=[ROLE_LABELS[r] for r in order], orientation="h",
                           marker_color=[colours[r] for r in order], text=[counts[r] for r in order],
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x}<extra></extra>"))
    fig = _style(fig, theme, 240)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, ticksuffix="   ", automargin=True)
    return fig


GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


# ------------------------------------------------------------------ pages --
def _overview(theme: str) -> list:
    store = get_store()
    users = store.list_users()
    counts = store.role_counts()
    admins = [u for u in users if u["role"] == "admin" and u["active"]]
    new_week = sum(1 for u in users if (u.get("created_at") or "") >= _utc(7))
    signins = store.count_actions(SIGN_IN, _utc(1))
    failures = store.count_actions(FAILURES, _utc(1))
    states = [s for _n, s, _d in health()]
    status = ("Healthy", "ok") if all(s == "ok" for s in states) else (
        ("Degraded", "warn") if "down" not in states else ("Attention", "danger"))
    kpis = html.Div(className="dash-kpis cols-3", children=[
        _kpi("Active users", sum(counts.values()), "people", "accent", f"+{new_week} this week"),
        _kpi("Staff accounts", counts["planner"] + counts["responder"], "briefcase", "info",
             f"{counts['planner']} planners · {counts['responder']} responders"),
        _kpi("Administrators", len(admins), "shield-lock", "danger",
             f"{sum(1 for a in admins if a['totp_enabled'])} of {len(admins)} with 2FA"),
        _kpi("Sign-ins (24 h)", signins, "box-arrow-in-right", "ok", f"{failures} failed or locked"),
        _kpi("Alert subscribers", store.subscriber_count(), "bell", "warn", "following at least one area"),
        _kpi("System", status[0], "activity", status[1], "see System for detail"),
    ])
    return [
        kpis,
        html.Div(className="dash-grid cols-8-4", children=[
            _panel("Sign-ins, last 14 days", dcc.Graph(figure=signin_chart(theme), config=GRAPH_CONFIG),
                   icon="bar-chart"),
            _panel("Accounts by role", dcc.Graph(figure=roles_chart(theme), config=GRAPH_CONFIG), icon="pie-chart"),
        ]),
        html.Div(className="dash-grid cols-7-5", children=[
            _panel("Recent activity", *_activity_feed(), icon="clock-history",
                   actions=dcc.Link("View all", href="/admin/activity", className="dash-link")),
            _panel("System health", _health_list(health()), icon="heart-pulse",
                   actions=dcc.Link("Details", href="/admin/system", className="dash-link")),
        ]),
    ]


def _users_page() -> list:
    toolbar = html.Div(className="dash-toolbar", children=[
        html.Div(className="dash-search", children=[
            _icon("search"),
            dcc.Input(id="users-search", type="search", placeholder="Search by name or email", debounce=True,
                      className="dash-search-input"),
        ]),
        dcc.Dropdown(id="users-role-filter", value="all", clearable=False, className="dash-select dash-filter",
                     options=[{"label": "All roles", "value": "all"}] +
                             [{"label": ROLE_LABELS[r], "value": r} for r in ROLES]),
    ])
    return [
        _panel("Users and roles", toolbar, html.Div(id="users-status", className="mb-2"),
               html.Div(id="users-table", children=users_table()), icon="people"),
        _new_user_modal(),
    ]


def _activity_page() -> list:
    toolbar = html.Div(className="dash-toolbar", children=[
        html.Div(className="dash-search", children=[
            _icon("search"),
            dcc.Input(id="audit-search", type="search", placeholder="Search user, event or detail", debounce=True,
                      className="dash-search-input"),
        ]),
        dcc.Dropdown(id="audit-kind", value="all", clearable=False, className="dash-select dash-filter",
                     options=[{"label": "All events", "value": "all"}] +
                             [{"label": label, "value": k} for k, (label, _a) in ACTION_KINDS.items()]),
        dbc.Button([_icon("arrow-clockwise", "me-1"), "Refresh"], id="btn-audit-refresh", color="dash", className="dash-btn-ghost"),
    ])
    return [_panel("Audit log", toolbar, html.Div(id="audit-table", children=audit_table()),
                   icon="journal-text")]


def _system_page() -> list:
    from src.dashboard.callbacks import predictor
    m = getattr(predictor, "metrics", None) or {}
    cards = html.Div(className="dash-kpis cols-4", children=[
        _kpi("Held-out test F1", f"{m['f1']:.3f}" if "f1" in m else "—", "bullseye", "accent", "4 unseen storm seasons"),
        _kpi("IoU", f"{m['iou']:.3f}" if "iou" in m else "—", "intersect", "info", "flooded-area overlap"),
        _kpi("Precision", f"{m['precision']:.3f}" if "precision" in m else "—", "crosshair", "ok",
             "share of warnings that were right"),
        _kpi("Recall", f"{m['recall']:.3f}" if "recall" in m else "—", "broadcast", "warn",
             "share of flooding caught"),
    ])
    return [
        cards,
        html.Div(className="dash-grid cols-7-5", children=[
            _panel("Service status", html.Div(id="admin-status", children=_health_list(health())), icon="heart-pulse",
                   actions=dbc.Button([_icon("arrow-clockwise", "me-1"), "Refresh forecast"],
                                      id="btn-admin-refresh", color="dash", className="dash-btn-ghost")),
            _panel("About the model", html.P(
                ["The deployed U-Net (Model B, drainage labels) maps a 3-day rainfall total to the calibrated "
                 "probability that each ~70 m cell floods. Scores are against held-out storm seasons; see ",
                 html.Code("RESULTS.md"), " for baselines, calibration and the live-chain backtest."],
                className="dash-hint"), html.Div(id="admin-refresh-status"), icon="info-circle"),
        ]),
    ]


ADMIN_SECTIONS = {
    "/admin": ("Overview", "Who is using the twin, how sign-ins are going, and whether the services are up.",
               _overview),
    "/admin/users": ("Users", "Create accounts, assign roles and disable access.", None),
    "/admin/activity": ("Activity", "Every sign-in, change, export and report, newest first.", None),
    "/admin/system": ("System", "Model, forecast feed and service status.", None),
}


def render_admin(user: dict | None, path: str = "/admin", theme: str = "dark") -> html.Div:
    from src.dashboard.auth_pages import _card, _denied, _narrow, _page
    if not can(user, "manage_users"):
        if user and user.get("mfa_pending"):
            return _page("Administration", _narrow(_card(
                "Two-step sign-in required",
                html.P("Confirm a code from your authenticator app to use the administration console.",
                       className="twin-empty-state"),
                dcc.Link("Continue", href="/2fa", className="twin-auth-btn"))))
        return _denied("The administration console is only available to administrators.")
    path = path if path in ADMIN_SECTIONS else "/admin"
    title, subtitle, _ = ADMIN_SECTIONS[path]
    if path == "/admin/users":
        content, actions = _users_page(), dbc.Button([_icon("person-plus", "me-1"), "New user"],
                                                     id="btn-new-user", className="dash-btn-primary")
    elif path == "/admin/activity":
        content, actions = _activity_page(), None
    elif path == "/admin/system":
        content, actions = _system_page(), None
    else:
        content, actions = _overview(theme), dcc.Link([_icon("person-plus", "me-1"), "Manage users"],
                                                       href="/admin/users", className="dash-btn-primary")
    return shell(path, user, title, subtitle, *content, actions=actions)


def render_account(user: dict | None, theme: str = "dark") -> html.Div:
    from src.dashboard.auth_pages import _alerts_for, _denied, alert_area_options
    if not user:
        return _denied("Sign in to manage your alert areas.")
    from src.forecast.nowcast import cached_nowcast
    nc = cached_nowcast("live")
    store = get_store()
    subs = store.subscriptions(user["user_id"])
    sub_options, sub_values = alert_area_options(user)
    role = user.get("real_role") or user["role"]
    methods = ["Google", "Email link"]
    profile = _panel("Profile", html.Div(className="dash-profile", children=[
        _avatar({**user, "role": role}, "is-lg"),
        html.Div([html.Span(user.get("display_name") or user["email"], className="dash-profile-name"),
                  html.Span(user["email"], className="dash-cell-muted"),
                  html.Div(_role_badge(role), className="mt-2")]),
    ]), html.Dl(className="dash-facts", children=[
        html.Dt("Sign in with"), html.Dd(html.Div([html.Span(m, className="dash-chip") for m in methods],
                                                  className="dash-chips")),
        html.Dt("Two-step sign-in"), html.Dd(
            "On (authenticator app)" if user["totp_enabled"] else
            ("Required, set up at next sign-in" if role == "admin" else "Not required for this role")),
        html.Dt("Member since"), html.Dd((user.get("created_at") or "")[:10]),
        html.Dt("Last sign-in"), html.Dd((user.get("last_login_at") or "—").replace("T", " ")[:16] + " UTC"),
    ]), icon="person-badge")
    alerts = _panel("Alert areas", html.P("Search any neighbourhood or place you care about, such as your home "
                                          "area, school or workplace. Warnings are issued per neighbourhood, so a "
                                          "place is followed through the neighbourhood it is in.",
                                          className="dash-hint"),
                    dcc.Dropdown(id="sub-places", options=sub_options, value=sub_values, multi=True,
                                 placeholder="Search a place, e.g. Strathmore University or Mathare",
                                 className="dash-select mb-3"),
                    dbc.Button([_icon("check2", "me-1"), "Save alert areas"], id="btn-save-subs",
                               className="dash-btn-primary"),
                    html.Div(id="subs-status", className="mt-2"), icon="bell")
    current = _panel("Current alerts for your areas", html.Div(id="account-alerts", children=_alerts_for(
        user, nc.to_client() if nc else None)), icon="exclamation-diamond")
    left = [profile]
    kpis = html.Div(className="dash-kpis cols-3", children=[
        _kpi("Role", ROLE_LABELS[role], "person-badge", ROLE_TONES[role] if ROLE_TONES[role] != "neutral" else "info"),
        _kpi("Alert areas", len(subs), "geo-alt", "warn", "places you follow"),
        _kpi("Live outlook", "Loaded" if nc and not nc.error else "Unavailable", "cloud-rain",
             "ok" if nc and not nc.error else "danger", "Open-Meteo hourly forecast"),
    ])
    return shell("/account", user, "My account", "Your profile, alert areas and sign-in settings.",
                 kpis, html.Div(className="dash-grid cols-5-7", children=[
                     html.Div(left, className="dash-stack"), html.Div([alerts, current], className="dash-stack")]))


# -------------------------------------------------------------- callbacks --
def register_admin_callbacks(app) -> None:
    @app.callback(Output("new-user-modal", "is_open", allow_duplicate=True),
                  Input("btn-new-user", "n_clicks"), Input("btn-cancel-new-user", "n_clicks"),
                  prevent_initial_call=True)
    def toggle_new_user(open_clicks, cancel_clicks):
        if not (open_clicks or cancel_clicks):
            raise PreventUpdate
        return ctx.triggered_id == "btn-new-user"

    @app.callback(Output("users-table", "children", allow_duplicate=True),
                  Input("users-search", "value"), Input("users-role-filter", "value"),
                  prevent_initial_call=True)
    def filter_users(query, role):
        if not can(current_user(), "manage_users"):
            raise PreventUpdate
        return users_table(query, role)

    @app.callback(Output("create-user-status", "children"),
                  Output("users-status", "children", allow_duplicate=True),
                  Output("users-table", "children", allow_duplicate=True),
                  Output("new-user-modal", "is_open", allow_duplicate=True),
                  Input("btn-create-user", "n_clicks"),
                  State("new-email", "value"), State("new-role", "value"),
                  State("users-search", "value"), State("users-role-filter", "value"),
                  prevent_initial_call=True)
    def create_user(n, email, role, query, role_filter):
        if not n:
            raise PreventUpdate
        me = current_user()
        if not can(me, "manage_users"):
            return html.Span("Not permitted.", className="twin-auth-error"), no_update, no_update, no_update
        store = get_store()
        try:
            u = store.create_user(email or "", role or "citizen")
        except AuthError as exc:
            return html.Span(str(exc), className="twin-auth-error"), no_update, no_update, no_update
        store.audit("user_created", me["user_id"], f"{u['email']} as {u['role']}")
        how = "They sign in with Google or an emailed link."
        if u["role"] == "admin":
            how += " They will set up an authenticator app at their first sign-in."
        return (None, html.Div([_icon("check-circle", "me-2"), f"Created {u['email']} ({ROLE_LABELS[u['role']]}). {how}"],
                               className="dash-alert tone-ok"), users_table(query, role_filter), False)

    @app.callback(Output("users-status", "children", allow_duplicate=True),
                  Output("users-table", "children", allow_duplicate=True),
                  Input({"type": "save-user", "uid": ALL}, "n_clicks"),
                  State({"type": "role-dd", "uid": ALL}, "value"), State({"type": "role-dd", "uid": ALL}, "id"),
                  State({"type": "active-sw", "uid": ALL}, "value"),
                  State("users-search", "value"), State("users-role-filter", "value"),
                  prevent_initial_call=True)
    def save_user(clicks, roles, ids, actives, query, role_filter):
        me = current_user()
        if not can(me, "manage_users"):
            return html.Div("Not permitted.", className="dash-alert tone-danger"), no_update
        if not ctx.triggered_id or not any(clicks):
            raise PreventUpdate
        uid = ctx.triggered_id["uid"]
        i = [d["uid"] for d in ids].index(uid)
        store = get_store()
        before = store.get_user(uid)
        try:
            after = store.update_user(uid, role=roles[i], active=bool(actives[i]))
        except AuthError as exc:
            return (html.Div([_icon("exclamation-triangle", "me-2"), str(exc)], className="dash-alert tone-danger"),
                    users_table(query, role_filter))
        store.audit("user_updated", me["user_id"],
                    f"{after['email']}: role {before['role']}->{after['role']}, "
                    f"active {before['active']}->{after['active']}")
        return (html.Div([_icon("check-circle", "me-2"), f"Saved {after['email']}."], className="dash-alert tone-ok"),
                users_table(query, role_filter))

    @app.callback(Output("audit-table", "children"),
                  Input("btn-audit-refresh", "n_clicks"), Input("audit-search", "value"), Input("audit-kind", "value"),
                  prevent_initial_call=True)
    def filter_audit(_n, query, kind):
        if not can(current_user(), "view_audit"):
            raise PreventUpdate
        return audit_table(query, kind)

    @app.callback(Output("admin-refresh-status", "children"), Output("admin-status", "children"),
                  Input("btn-admin-refresh", "n_clicks"), prevent_initial_call=True)
    def admin_refresh(n):
        if not n:
            raise PreventUpdate
        me = current_user()
        if not can(me, "refresh_data"):
            return html.Div("Not permitted.", className="dash-alert tone-danger"), no_update
        from src.dashboard.callbacks import predictor
        from src.forecast.nowcast import get_nowcast
        nc = get_nowcast("live", predictor, refresh=True)
        get_store().audit("forecast_refreshed", me["user_id"], nc.error or "ok")
        msg = (html.Div([_icon("exclamation-triangle", "me-2"), f"Forecast refresh failed: {nc.error}"],
                        className="dash-alert tone-danger") if nc.error else
               html.Div([_icon("check-circle", "me-2"), "Live forecast refreshed."], className="dash-alert tone-ok"))
        return msg, _health_list(health())
