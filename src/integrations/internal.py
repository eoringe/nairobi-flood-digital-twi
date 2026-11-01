"""
src.integrations.internal
=========================
The digital twin's own components seen through the same adapter interface as
the partner systems, so screens and the API treat every source alike.

    forecast      Open-Meteo hourly rainfall forecast (live outlook input)
    prediction    the Predictive Flood Modeling component: U-Net extent,
                  12-hour place outlook, 3-day flood chance
    outlook_alerts  warnings raised by the outlook. They stand in for the
                  Automated Alert System until it is connected, and are labelled
                  as model warnings, not published alerts.
    community     community flood reports (observations)
    news_floods   Groundsource news-reported floods, 2000-2025 (observations)
    satellite     Copernicus GFM Sentinel-1 flood maps (observations)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.integrations.base import DataSource, utc_now

EAT = timezone(timedelta(hours=3))


def _live_nowcast():
    from src.dashboard.callbacks import predictor
    from src.forecast.nowcast import get_nowcast
    return get_nowcast("live", predictor)


class ForecastFeed(DataSource):
    key = "forecast"
    name = "Rainfall forecast"
    system = "Open-Meteo forecast API"
    kind = "external"
    description = "Hourly rainfall for Nairobi: the past 31 days and the next 2 days."
    provides = ("hourly",)
    probe = "hourly"
    live_within = timedelta(hours=2)
    stale_after = timedelta(hours=6)

    def _fetch(self, what: str, **params):
        nc = _live_nowcast()
        if nc.error and not nc.hours:
            raise RuntimeError(nc.error)
        rows = [{"time": t.isoformat(), "rain_mm": v, "rain_72h_mm": a, "forecast": t > nc.as_of}
                for t, v, a in zip(nc.chart_times, nc.chart_rain_mm, nc.chart_acc72_mm)]
        return rows, nc.as_of


class PredictionModel(DataSource):
    key = "prediction"
    name = "Flood prediction"
    system = "Predictive Flood Modeling System"
    description = ("U-Net flood extent and 12-hour place outlook, with the 3-day flood chance learned "
                   "from reported floods.")
    provides = ("outlook", "chance", "model")
    probe = "chance"
    live_within = timedelta(hours=2)
    stale_after = timedelta(hours=6)

    def _fetch(self, what: str, **params):
        if what == "model":
            from src.dashboard.callbacks import predictor
            from src.forecast.flood_chance import _params
            p = _params() or {}
            return [{"model_version": predictor.model_version, "output": "flood extent probability per ~70 m cell",
                     "chance_model": p.get("method"), "chance_confidence": p.get("confidence")}], None
        nc = _live_nowcast()
        if nc.error and not nc.hours:
            raise RuntimeError(nc.error)
        if what == "chance":
            return ([{**nc.chance, "issued_at": nc.as_of.isoformat(), "horizon_days": 3}] if nc.chance else []), nc.as_of
        return [nc.to_client()], nc.as_of


class OutlookAlerts(DataSource):
    key = "outlook_alerts"
    name = "Model warnings"
    system = "Digital twin outlook (interim alert source)"
    description = ("Warnings raised by the 12-hour outlook. Shown in place of published alerts until the "
                   "Automated Alert System is connected.")
    provides = ("active",)
    probe = "active"
    live_within = timedelta(hours=2)
    stale_after = timedelta(hours=6)

    def _fetch(self, what: str, **params):
        nc = _live_nowcast()
        if nc.error and not nc.hours:
            raise RuntimeError(nc.error)
        rows = []
        for i, w in enumerate(nc.warnings):
            start = nc.hours[w["hour"]]
            rows.append({
                "id": f"outlook-{nc.as_of:%Y%m%d%H}-{i + 1}",
                "title": w["text"], "severity": w["level"].lower(), "status": "model_warning",
                "areas": w["places"], "created_at": nc.as_of.isoformat(), "published_at": None,
                "starts_at": start.isoformat(),
                "description": f"{w['text']} in {w['places_text']}.",
                "recommended_actions": RECOMMENDED_ACTIONS.get(w["level"], []),
                "acknowledged": None, "source_label": "Model warning (not a published alert)",
            })
        return rows, nc.as_of


#: Standard advice by warning level, shown with model warnings.
RECOMMENDED_ACTIONS = {
    "MODERATE": ["Avoid walking or driving through flowing water", "Clear drains around your home"],
    "HIGH": ["Move valuables and documents above floor level", "Avoid river crossings and underpasses",
             "Plan a route to higher ground"],
    "CRITICAL": ["Move to higher ground now if you live beside a river", "Do not cross flooded roads or bridges",
                 "Follow instructions from county emergency services"],
}


class CommunityReports(DataSource):
    key = "community"
    name = "Community reports"
    system = "Digital twin community reporting"
    description = "Flooding reported by signed-in users: water on the road, impassable roads, flooded homes."
    provides = ("active", "history")
    probe = "active"
    live_within = timedelta(hours=1)
    stale_after = timedelta(hours=6)

    def _fetch(self, what: str, **params):
        from src.reports.store import get_reports
        store = get_reports()
        if what == "active":
            # Checked just now: no active reports is current information, not missing data.
            return store.active(), utc_now()
        rows = store.history(params.get("start"), params.get("end"))
        return rows, max((r["updated_at"] for r in rows), default=None)


class NewsFloods(DataSource):
    key = "news_floods"
    name = "News-reported floods"
    system = "Groundsource (Google Research), CC BY 4.0"
    kind = "external"
    description = "Floods reported in the news inside the model area, 2000-2025, about 60% right in place and date."
    provides = ("events",)
    probe = "events"
    archive = True

    def _fetch(self, what: str, **params):
        from src.reports import history
        start, end = params.get("start"), params.get("end")
        rows = []
        for e in history.load_events():
            if start and e["start"] < date.fromisoformat(str(start)[:10]):
                continue
            if end and e["start"] > date.fromisoformat(str(end)[:10]):
                continue
            rows.append({"id": e["uuid"], "start": e["start"].isoformat(), "end": e["end"].isoformat(),
                         "area_km2": e["area_km2"], "name": history.area_name(e["geom"]),
                         "lat": e["geom"].representative_point().y, "lon": e["geom"].representative_point().x})
        newest = max((datetime.combine(date.fromisoformat(r["end"]), datetime.min.time(), EAT) for r in rows),
                     default=None)
        return rows, newest


class SatelliteFloodMap(DataSource):
    key = "satellite"
    name = "Satellite flood map"
    system = "Copernicus Global Flood Monitoring (Sentinel-1)"
    kind = "external"
    description = "Observed flood extent from radar, about every six days; cannot see water between buildings."
    provides = ("latest",)
    probe = "latest"
    live_within = timedelta(days=7)
    stale_after = timedelta(days=14)

    def _fetch(self, what: str, **params):
        from src.reports import satellite
        anchor = params.get("anchor") or date.today()
        day = satellite.latest_pass(anchor)
        if day is None:
            return [], None
        return [{"pass_date": day.isoformat()}], datetime.combine(day, datetime.min.time(), EAT)
