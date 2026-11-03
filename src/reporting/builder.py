"""
src.reporting.builder
=====================
Monitoring reports (specification §2.13, FR-018, UC-FVD-011): built once as
plain tables, then previewed on screen, exported as CSV or PDF, or returned
by the API.

    flood_summary       current outlook, flood chance, areas at risk, alerts,
                        incidents and data-source state
    risk_summary        predicted risk by ward and exposed facilities
    station_report      every station with health, last contact and readings
    rainfall            daily rainfall and unusually wet days for a period
    alerts_incidents    active alerts and incidents in a period

Every report states its filters, when it was generated, and the source and
freshness of each section; sections from a partner system that is not
connected say so instead of being empty, and demo data is marked.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from src.geo import boundaries
from src.integrations import base, registry

EAT = timezone(timedelta(hours=3))
LEVEL_KEY = {0: "CLEAR", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}
RISK_CLASS = {"CLEAR": "Low", "MODERATE": "Moderate", "HIGH": "High", "CRITICAL": "Very High"}

DEFINITIONS = [
    {"report_key": "flood_summary", "name": "Flood monitoring summary", "formats": "pdf,csv",
     "permission": "generate_reports",
     "description": "Current outlook, flood chance, areas at risk, alerts, incidents and data-source status."},
    {"report_key": "risk_summary", "name": "Geographical risk summary", "formats": "pdf,csv",
     "permission": "generate_reports", "description": "Predicted risk by ward and the facilities exposed."},
    {"report_key": "station_report", "name": "Station report", "formats": "pdf,csv",
     "permission": "generate_reports",
     "description": "Every monitoring station with health, last communication and latest readings."},
    {"report_key": "rainfall", "name": "Rainfall report", "formats": "pdf,csv", "permission": "generate_reports",
     "description": "Daily rainfall over a period with unusually wet days."},
    {"report_key": "alerts_incidents", "name": "Alert and incident summary", "formats": "pdf,csv",
     "permission": "generate_reports", "description": "Active alerts and the incidents reported in a period."},
]


@dataclass
class Section:
    heading: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    note: str = ""
    source: str = ""


@dataclass
class Report:
    key: str
    title: str
    generated_at: datetime
    filters: dict
    sections: list[Section]
    demo: bool = False

    @property
    def row_count(self) -> int:
        return sum(len(s.rows) for s in self.sections)


def _when(t) -> str:
    """'01 Nov 2026 12:39' in Nairobi time, or the input unchanged if it is not a timestamp."""
    try:
        u = base.as_utc(t)
    except (TypeError, ValueError):
        return str(t)
    return u.astimezone(EAT).strftime("%d %b %Y %H:%M") if u else "—"


def _src_note(r: base.SourceResult) -> str:
    s = registry.get(r.source)
    if r.state == base.NOT_CONNECTED:
        return f"{s.system} is not connected; no data."
    if r.state == base.ERROR:
        return f"{s.name} unavailable: {r.message}"
    when = r.observed_at.astimezone(EAT).strftime("%d %b %Y %H:%M") if r.observed_at else "unknown time"
    return f"{s.name} · {r.freshness} · observed {when}" + (" · DEMO DATA, not measurements" if r.demo else "")


def _area_label(areas: list[str] | None) -> str:
    return ", ".join(boundaries.label(a) for a in areas) if areas else "All of Nairobi"


def _filters(areas, start, end) -> dict:
    out = {"area": _area_label(areas)}
    if start:
        out["from"] = str(start)
    if end:
        out["to"] = str(end)
    return out


# ------------------------------------------------------------------ build --
def flood_summary(areas=None, start=None, end=None) -> Report:
    pred = registry.get("prediction").fetch("outlook")
    chance = registry.get("prediction").fetch("chance")
    nc = pred.records[0] if pred.ok and pred.records else {}
    secs = []
    c = chance.records[0] if chance.ok and chance.records else None
    secs.append(Section("Outlook", ["Item", "Value"], [
        ["Issued", nc.get("as_of_label") or nc.get("as_of") or "—"],
        ["Headline", nc.get("headline", "—")],
        ["Worst level, next 12 h", RISK_CLASS[LEVEL_KEY[max(nc.get("city_level") or [0])]]],
        ["72-hour rainfall, peak (mm)", f"{max(nc.get('acc72_mm') or [0]):.0f}"],
        ["Flood chance, next 3 days", f"{round(100 * c['p'])}% · {c['call_text']} (95% call)" if c else "—"],
    ], note="Model output, not observed flooding.", source=_src_note(pred)))
    secs.append(risk_section(nc, areas, pred))
    secs.append(alerts_section(areas))
    inc = incidents_section(areas, start or (datetime.now(EAT).date() - timedelta(days=7)), end)
    secs.append(inc)
    rows = [[r["name"], r["system"], {"ok": "Connected", "down": "Down", "not_connected": "Not connected",
                                      "unknown": "Not checked"}[r["state"]], r["freshness"],
             _when(r["last_success"]) if r["last_success"] else "—"]
            for r in registry.status_rows()]
    secs.append(Section("Data sources", ["Source", "System", "State", "Data", "Last success"], rows))
    return Report("flood_summary", "Flood monitoring summary", datetime.now(EAT), _filters(areas, start, end), secs,
                  demo=registry.demo_mode())


def risk_section(nc: dict, areas, pred: base.SourceResult) -> Section:
    rows, seen = [], set()
    pa = boundaries.place_areas()
    rank = {"CLEAR": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    for w in sorted(nc.get("warnings") or [], key=lambda w: -rank[w["level"]]):
        for p in w["places"]:
            if p in seen or not boundaries.place_in(p, areas):
                continue
            seen.add(p)
            ward = pa.get(p, {}).get("ward")
            rows.append([p, boundaries.label(ward) if ward else "—", RISK_CLASS[w["level"]], w["text"]])
    return Section("Predicted risk by area, next 12 hours", ["Place", "Ward", "Risk", "Outlook"], rows,
                   note="No flooding predicted." if not rows else "Model output, not observed flooding.",
                   source=_src_note(pred))


def alerts_section(areas) -> Section:
    partner = registry.get("alerts").fetch("active")
    res = partner if partner.ok else registry.get("outlook_alerts").fetch("active")
    rows = []
    for a in res.records if res.ok else []:
        places = [p for p in a.get("areas") or [] if boundaries.place_in(p, areas)]
        if places:
            rows.append([a.get("title"), (a.get("severity") or "").title(), (a.get("status") or "").replace("_", " "),
                         ", ".join(places)[:200], _when(a["published_at"]) if a.get("published_at") else "not published"])
    note = "" if partner.ok else "Automated Alert System not connected: these are model warnings, not published alerts."
    return Section("Active alerts", ["Alert", "Severity", "Status", "Areas", "Published"], rows,
                   note=note or ("No active alerts." if not rows else ""), source=_src_note(res))


def incidents_section(areas, start, end) -> Section:
    from src.dashboard.ops_assets import incidents
    s = start if isinstance(start, date) else date.fromisoformat(str(start)[:10])
    e = end if isinstance(end, date) else (date.fromisoformat(str(end)[:10]) if end else datetime.now(EAT).date())
    rows, res = incidents(s, e, areas)
    return Section(f"Incidents {s:%d %b %Y} to {e:%d %b %Y}", ["When", "Type", "Place", "Status", "Source"],
                   [[_when(r["when"]) if "T" in str(r["when"]) else str(r["when"]), r["type_label"], r["place"],
                     r["status"], r["source"]]
                    for r in rows], note="" if rows else "No incidents in this period.",
                   source=" | ".join(_src_note(x) for x in res.values()))


def risk_summary(areas=None, start=None, end=None) -> Report:
    pred = registry.get("prediction").fetch("outlook")
    nc = pred.records[0] if pred.ok and pred.records else {}
    infra = registry.get("infrastructure_system").fetch("assets")
    if not infra.ok:
        infra = registry.get("infrastructure").fetch("assets", area_ids=areas, min_level=1)
    rows = [[a["name"], a["category"].replace("_", " "), boundaries.label(a["ward"]) if a.get("ward") else "—",
             RISK_CLASS[LEVEL_KEY[a.get("level", 0)]]] for a in (infra.records if infra.ok else [])]
    return Report("risk_summary", "Geographical risk summary", datetime.now(EAT), _filters(areas, None, None), [
        risk_section(nc, areas, pred),
        Section("Facilities in predicted flood areas", ["Facility", "Type", "Ward", "Predicted risk"], rows,
                note="" if rows else "No facilities in predicted flood areas.", source=_src_note(infra)),
    ], demo=infra.demo)


def station_report(areas=None, start=None, end=None) -> Report:
    r = registry.get("iot").fetch("stations")
    rows = []
    for s in r.records if r.ok else []:
        if not boundaries.point_in(s["lat"], s["lon"], areas):
            continue
        latest = "; ".join(f"{k.replace('_', ' ')} {('missing' if v.get('value') is None else v['value'])} "
                           f"{v.get('unit') or ''}".strip() for k, v in (s.get("latest") or {}).items())
        rows.append([s["id"], s["name"], ", ".join(s.get("types", [])), s.get("health"), s.get("status"),
                     _when(s.get("last_communication")) if s.get("last_communication") else "—", latest or "—",
                     f"{s.get('warning_m') or '—'} / {s.get('critical_m') or '—'}"])
    return Report("station_report", "Station report", datetime.now(EAT), _filters(areas, None, None), [
        Section("Monitoring stations", ["Id", "Station", "Types", "Health", "Status", "Last communication", "Latest",
                                        "Warning / critical (m)"], rows,
                note="" if rows else ("No stations." if r.ok else "IoT Sensor Data Platform not connected."),
                source=_src_note(r))], demo=r.demo)


def rainfall(areas=None, start=None, end=None) -> Report:
    from src.dashboard.ops_monitoring import _chirps_daily, unusual_days
    daily = _chirps_daily()
    e = date.fromisoformat(str(end)[:10]) if end else max(daily)
    s = date.fromisoformat(str(start)[:10]) if start else e - timedelta(days=30)
    days = [d for d in sorted(daily) if s <= d <= e]
    unusual = {d for d, _v, _t in unusual_days(daily, s, e)}
    rows = [[d.isoformat(), f"{daily[d]:.1f}", "yes" if d in unusual else ""] for d in days]
    total = sum(daily[d] for d in days)
    return Report("rainfall", "Rainfall report", datetime.now(EAT), _filters(None, s, e), [
        Section("Summary", ["Item", "Value"], [["Days", len(days)], ["Total (mm)", f"{total:.0f}"],
                                               ["Days with 1 mm or more", sum(1 for d in days if daily[d] >= 1)],
                                               ["Unusually wet days", len(unusual)]]),
        Section("Daily rainfall", ["Date", "Rainfall (mm)", "Unusually wet"], rows,
                note="CHIRPS city-average daily rainfall; local storms can be heavier. The record ends "
                     f"{max(daily):%d %b %Y}." + (" Station rainfall is added when the IoT platform connects."),
                source="CHIRPS (Climate Hazards Center), via the project archive"),
    ])


def alerts_incidents(areas=None, start=None, end=None) -> Report:
    s = start or (datetime.now(EAT).date() - timedelta(days=30))
    return Report("alerts_incidents", "Alert and incident summary", datetime.now(EAT), _filters(areas, s, end),
                  [alerts_section(areas), incidents_section(areas, s, end)], demo=registry.demo_mode())


BUILDERS = {"flood_summary": flood_summary, "risk_summary": risk_summary, "station_report": station_report,
            "rainfall": rainfall, "alerts_incidents": alerts_incidents}


def build(key: str, areas=None, start=None, end=None) -> Report:
    return BUILDERS[key](areas=areas or None, start=start, end=end)


# ----------------------------------------------------------------- export --
def to_csv(rep: Report) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([rep.title])
    w.writerow([f"Generated {rep.generated_at:%d %b %Y %H:%M} Nairobi time"] +
               [f"{k}: {v}" for k, v in rep.filters.items()] + (["DEMO DATA included"] if rep.demo else []))
    for s in rep.sections:
        w.writerow([])
        w.writerow([s.heading])
        if s.source:
            w.writerow([f"Source: {s.source}"])
        if s.note:
            w.writerow([s.note])
        if s.columns:
            w.writerow(s.columns)
        w.writerows(s.rows)
    return buf.getvalue()


def to_pdf(rep: Report) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title=rep.title,
                            author="Nairobi Flood Digital Twin")
    st = getSampleStyleSheet()
    small = st["BodyText"].clone("small", fontSize=8, leading=10)
    muted = small.clone("muted", textColor=colors.HexColor("#555555"))
    cell = small.clone("cell", fontSize=7.5, leading=9)
    story = [Paragraph(rep.title, st["Title"]),
             Paragraph(f"Generated {rep.generated_at:%d %b %Y %H:%M} (Nairobi time) · " +
                       " · ".join(f"{k}: {v}" for k, v in rep.filters.items()), muted)]
    if rep.demo:
        story.append(Paragraph("<b>Contains DEMO DATA from partner placeholders: not measurements.</b>",
                               small.clone("demo", textColor=colors.HexColor("#a16207"))))
    for s in rep.sections:
        story += [Spacer(1, 6 * mm), Paragraph(s.heading, st["Heading2"])]
        if s.source:
            story.append(Paragraph(f"Source: {s.source}", muted))
        if s.note:
            story.append(Paragraph(s.note, small))
        if s.columns and s.rows:
            data = [[Paragraph(f"<b>{c}</b>", cell) for c in s.columns]] + \
                   [[Paragraph(str(v), cell) for v in row] for row in s.rows[:500]]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
            ]))
            story += [Spacer(1, 2 * mm), t]
            if len(s.rows) > 500:
                story.append(Paragraph(f"First 500 of {len(s.rows)} rows; the CSV export has all of them.", muted))
    story += [Spacer(1, 8 * mm), Paragraph("Predictions are model output, not observed flooding. Nairobi Flood "
                                           "Digital Twin · Integrated Flood Monitoring System.", muted)]
    doc.build(story)
    return buf.getvalue()
