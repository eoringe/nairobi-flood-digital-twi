"""
src.dashboard.ops_reports
=========================
The reports screen (specification §2.13, §6.11, FR-018, UC-FVD-011): choose a
report, its area (from the filter bar) and period, preview it, export it as
PDF or CSV. Every export is recorded (generated_reports) and audited.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from src.auth.access import can, current_user
from src.dashboard.ops_console import EAT, PAGES, _icon, local_time, panel
from src.integrations.store import get_ops_store
from src.reporting import builder

PREVIEW_ROWS = 25


def reports_page(user: dict, theme: str, flt: dict | None = None) -> list:
    store = get_ops_store()
    store.upsert_report_definitions(builder.DEFINITIONS)
    defs = store.report_definitions()
    today = datetime.now(EAT).date()
    controls = html.Div(className="dash-toolbar", children=[
        dcc.Dropdown(id="ops-report-key", value="flood_summary", clearable=False, searchable=False,
                     className="dash-select ops-years",
                     options=[{"label": d["name"], "value": d["report_key"]} for d in defs]),
        dcc.DatePickerRange(id="ops-report-range", start_date=today - timedelta(days=7), end_date=today,
                            max_date_allowed=today, min_date_allowed=date(2000, 1, 1), display_format="D MMM YYYY",
                            className="ops-range"),
        html.Button([_icon("filetype-pdf", "me-1"), "Export PDF"], id="ops-report-pdf", n_clicks=0,
                    className="dash-btn-primary btn"),
        html.Button([_icon("filetype-csv", "me-1"), "Export CSV"], id="ops-report-csv", n_clicks=0,
                    className="dash-btn-ghost btn"),
        dcc.Download(id="ops-report-download"),
    ])
    return [panel("Report", controls, html.Div(id="ops-report-desc", className="dash-hint"), icon="file-earmark-text"),
            dcc.Loading(html.Div(id="ops-report-preview"), type="dot", color="var(--accent)"),
            panel("Recently generated", html.Div(id="ops-report-recent", children=recent_table()), icon="clock-history")]


def recent_table() -> html.Div:
    rows = get_ops_store().generated_reports(limit=15)
    if not rows:
        return html.P("No reports generated yet.", className="dash-empty")
    names = {d["report_key"]: d["name"] for d in builder.DEFINITIONS}
    return html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("When", "Report", "Format", "Filters", "Rows")])),
        html.Tbody([html.Tr([html.Td(local_time(r["created_at"])), html.Td(names.get(r["report_key"], r["report_key"])),
                             html.Td(r["format"].upper()),
                             html.Td(" · ".join(f"{k}: {v}" for k, v in r["filters"].items()), className="dash-cell-muted"),
                             html.Td(r["rows"])]) for r in rows]),
    ]))


def preview(rep: builder.Report) -> list:
    out = [html.Div([html.Strong(rep.title), f" · generated {rep.generated_at:%d %b %Y %H:%M} · ",
                     " · ".join(f"{k}: {v}" for k, v in rep.filters.items()),
                     html.Span(" DEMO DATA", className="ops-demo-tag ms-2") if rep.demo else None], className="mb-2")]
    for s in rep.sections:
        body = []
        if s.source:
            body.append(html.Div(f"Source: {s.source}", className="dash-cell-muted mb-1"))
        if s.note:
            body.append(html.P(s.note, className="dash-hint"))
        if s.columns and s.rows:
            body.append(html.Div(className="dash-table-wrap", children=html.Table(className="dash-table", children=[
                html.Thead(html.Tr([html.Th(c) for c in s.columns])),
                html.Tbody([html.Tr([html.Td(str(v)) for v in row]) for row in s.rows[:PREVIEW_ROWS]])])))
            if len(s.rows) > PREVIEW_ROWS:
                body.append(html.P(f"Preview shows {PREVIEW_ROWS} of {len(s.rows)} rows; exports include all.",
                                   className="dash-cell-muted"))
        out.append(panel(s.heading, *body, icon="table"))
    return out


_status = PAGES.pop("/ops/status")
PAGES["/ops/reports"] = ("Reports", "file-earmark-text", "Reports",
                         "Monitoring summaries, risk, station, rainfall, alert and incident reports, as PDF or CSV.",
                         "generate_reports", reports_page, True)
PAGES["/ops/status"] = _status


def register_report_page_callbacks(app) -> None:
    from src.dashboard.ops_console import effective_filter

    @app.callback(Output("ops-report-preview", "children"), Output("ops-report-desc", "children"),
                  Input("ops-report-key", "value"), Input("ops-report-range", "start_date"),
                  Input("ops-report-range", "end_date"), Input("ops-area", "value"))
    def show_preview(key, start, end, area):
        user = current_user()
        if not key or not can(user, "generate_reports"):
            raise PreventUpdate
        flt = effective_filter(user, area, 0)
        rep = builder.build(key, flt["areas"], start, end)
        desc = next((d["description"] for d in builder.DEFINITIONS if d["report_key"] == key), "")
        return preview(rep), desc

    @app.callback(Output("ops-report-download", "data"), Output("ops-report-recent", "children"),
                  Input("ops-report-pdf", "n_clicks"), Input("ops-report-csv", "n_clicks"),
                  State("ops-report-key", "value"), State("ops-report-range", "start_date"),
                  State("ops-report-range", "end_date"), State("ops-area", "value"), prevent_initial_call=True)
    def export(n_pdf, n_csv, key, start, end, area):
        from dash import ctx
        user = current_user()
        if not ctx.triggered_id or not (n_pdf or n_csv) or not can(user, "generate_reports"):
            raise PreventUpdate
        fmt = "pdf" if ctx.triggered_id == "ops-report-pdf" else "csv"
        flt = effective_filter(user, area, 0)
        rep = builder.build(key, flt["areas"], start, end)
        get_ops_store().record_report(key, user["user_id"], fmt, rep.filters, rep.row_count)
        from src.auth.store import get_store
        get_store().audit("report_generated", user["user_id"], f"{key} {fmt} {rep.filters}")
        name = f"{key}_{rep.generated_at:%Y%m%d_%H%M}.{fmt}"
        data = (dcc.send_bytes(builder.to_pdf(rep), name) if fmt == "pdf"
                else dcc.send_string(builder.to_csv(rep), name))
        return data, recent_table()
