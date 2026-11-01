"""
src.integrations.partners
=========================
Adapters for the Integrated Flood Monitoring System components that belong to
other teams (specification §9.1, §9.3–§9.6). None has published its API yet.

Each adapter is written against the contract in docs/capstone/TASKLIST.md
("Integration contracts"). Until its `*_API_URL` is set, the adapter reports
*not connected* and the screens say which system is missing. If a partner's
real API differs from the contract, only `_fetch` / `_normalise_*` here change.

DEMO MODE
---------
With INTEGRATION_DEMO=1 and no URL set, an adapter serves fixed demo records so
the screens can be shown and acceptance-tested. Every demo record is named
"DEMO ..." and every result carries demo=True, which the interface turns into a
"Demo data" watermark. Demo values are synthetic: they are not measurements and
must never be presented as such.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from src.integrations.base import HttpSource, utc_now

EAT = timezone(timedelta(hours=3))


def _hour_floor(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _parse_time(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=EAT)


# ------------------------------------------------------------- IoT sensors --
#: Demo stations: real river crossings and open spaces, fictional instruments.
DEMO_STATIONS = [
    {"id": "DEMO-WL-01", "name": "DEMO Nairobi River, Museum Hill", "types": ["water_level"],
     "lat": -1.2745, "lon": 36.8155, "warning_m": 2.5, "critical_m": 3.5, "health": "ok"},
    {"id": "DEMO-WL-02", "name": "DEMO Mathare River, Juja Road", "types": ["water_level", "rainfall"],
     "lat": -1.2615, "lon": 36.8560, "warning_m": 2.0, "critical_m": 3.0, "health": "ok"},
    {"id": "DEMO-WL-03", "name": "DEMO Ngong River, Nairobi Dam", "types": ["water_level"],
     "lat": -1.3135, "lon": 36.7860, "warning_m": 2.2, "critical_m": 3.2, "health": "degraded"},
    {"id": "DEMO-WL-04", "name": "DEMO Nairobi River, Kariobangi", "types": ["water_level"],
     "lat": -1.2560, "lon": 36.8790, "warning_m": 2.8, "critical_m": 3.8, "health": "offline"},
    {"id": "DEMO-RG-01", "name": "DEMO Rain gauge, CBD", "types": ["rainfall"],
     "lat": -1.2864, "lon": 36.8172, "health": "ok"},
    {"id": "DEMO-RG-02", "name": "DEMO Rain gauge, Eastleigh", "types": ["rainfall"],
     "lat": -1.2730, "lon": 36.8480, "health": "ok"},
    {"id": "DEMO-RG-03", "name": "DEMO Rain gauge, Kibera", "types": ["rainfall"],
     "lat": -1.3125, "lon": 36.7880, "health": "ok"},
]


def _demo_value(station: dict, kind: str, t: datetime) -> float | None:
    """Deterministic synthetic series; one gap per day so missing data can be seen."""
    h = int(t.timestamp() // 3600)
    seed = sum(ord(c) for c in station["id"])
    if (h + seed) % 24 == 5:
        return None
    if kind == "rainfall":
        pulse = max(0.0, math.sin((h + seed) / 9.0)) ** 6
        return round(12.0 * pulse * (1 + (seed % 3) / 3), 1)
    base = station["warning_m"] * 0.45
    return round(base + 0.9 * max(0.0, math.sin((h + seed) / 14.0)) ** 3 + 0.05 * math.sin(h / 2.0), 2)


class IoTSensorPlatform(HttpSource):
    key = "iot"
    name = "IoT sensors"
    system = "IoT Sensor Data Platform"
    description = "Monitoring stations, rainfall and water-level readings, sensor health."
    provides = ("stations", "readings", "health")
    probe = "stations"
    env_url, env_key = "IOT_API_URL", "IOT_API_KEY"
    live_within = timedelta(minutes=30)
    stale_after = timedelta(hours=3)

    def _fetch(self, what: str, **params):
        if self.demo():
            return self._demo(what, **params)
        if what == "stations":
            raw = self._get("/stations")
            rows = [self._normalise_station(s) for s in raw]
        elif what == "health":
            raw = self._get("/stations/health")
            rows = [self._normalise_station(s) for s in raw]
        else:
            raw = self._get(f"/stations/{params['station_id']}/readings",
                            {"from": params.get("start"), "to": params.get("end"), "type": params.get("kind")})
            rows = [self._normalise_reading(r) for r in raw]
        newest = max((_parse_time(r.get("observed_at") or r.get("last_communication")) for r in rows),
                     default=None, key=lambda t: t or datetime.min.replace(tzinfo=timezone.utc))
        return rows, newest

    @staticmethod
    def _normalise_station(s: dict) -> dict:
        return {
            "id": str(s["id"]), "name": s.get("name", s["id"]), "types": s.get("types") or [s.get("type")],
            "lat": float(s["lat"]), "lon": float(s["lon"]),
            "status": s.get("status", "active"), "health": s.get("health", "unknown"),
            "last_communication": s.get("last_communication"),
            "warning_m": s.get("warning_m"), "critical_m": s.get("critical_m"),
            "latest": s.get("latest", {}),
        }

    @staticmethod
    def _normalise_reading(r: dict) -> dict:
        v = r.get("value")
        return {"station_id": str(r["station_id"]), "kind": r["kind"], "observed_at": r["observed_at"],
                "value": None if v is None else float(v), "unit": r.get("unit"),
                "quality": r.get("quality", "valid" if v is not None else "missing")}

    def _demo(self, what: str, **params):
        now = _hour_floor(utc_now())
        if what in ("stations", "health"):
            rows = []
            for s in DEMO_STATIONS:
                last = utc_now() - (timedelta(hours=9) if s["health"] == "offline" else timedelta(minutes=10))
                latest = {k: {"value": _demo_value(s, k, last), "observed_at": last.isoformat(),
                              "unit": "mm/h" if k == "rainfall" else "m"} for k in s["types"]}
                rows.append({**{k: v for k, v in s.items() if k != "health"}, "health": s["health"],
                             "status": "inactive" if s["health"] == "offline" else "active",
                             "last_communication": last.isoformat(), "latest": latest})
            # The newest contact, not the hour it falls in: healthy demo stations report every 10 minutes.
            return rows, utc_now() - timedelta(minutes=10)
        station = next((s for s in DEMO_STATIONS if s["id"] == params.get("station_id")), None)
        if station is None:
            raise KeyError(f"unknown station {params.get('station_id')}")
        kind = params.get("kind") or station["types"][0]
        end = min(_parse_time(params.get("end")) or now, now)
        start = _parse_time(params.get("start")) or end - timedelta(days=3)
        if station["health"] == "offline":
            end = min(end, now - timedelta(hours=9))
        rows, t = [], _hour_floor(start)
        while t <= end:
            v = _demo_value(station, kind, t)
            rows.append({"station_id": station["id"], "kind": kind, "observed_at": t.isoformat(), "value": v,
                         "unit": "mm/h" if kind == "rainfall" else "m", "quality": "valid" if v is not None else "missing"})
            t += timedelta(hours=1)
        return rows, end


# ------------------------------------------------------------ alert system --
class AutomatedAlertSystem(HttpSource):
    key = "alerts"
    name = "Alert system"
    system = "Automated Alert System"
    description = "Published flood alerts, their status and acknowledgements."
    provides = ("active", "alert")
    probe = "active"
    env_url, env_key = "ALERTS_API_URL", "ALERTS_API_KEY"
    live_within = timedelta(minutes=15)
    stale_after = timedelta(hours=2)

    def _fetch(self, what: str, **params):
        if self.demo():
            now = utc_now()
            rows = [
                {"id": "DEMO-ALERT-1", "title": "DEMO Flood warning: Mathare River corridor", "severity": "high",
                 "status": "published", "areas": ["Mathare", "Mabatini", "Kiamaiko"],
                 "created_at": (now - timedelta(hours=2)).isoformat(), "published_at": (now - timedelta(hours=1, minutes=40)).isoformat(),
                 "description": "Demo alert for interface testing.",
                 "recommended_actions": ["Move valuables above floor level", "Avoid crossing the river"],
                 "acknowledged": 14, "recipients": 120},
                {"id": "DEMO-ALERT-2", "title": "DEMO Flood watch: Ngong River", "severity": "moderate",
                 "status": "published", "areas": ["Kibera", "Lindi"],
                 "created_at": (now - timedelta(hours=5)).isoformat(), "published_at": (now - timedelta(hours=4, minutes=50)).isoformat(),
                 "description": "Demo alert for interface testing.", "recommended_actions": ["Monitor updates"],
                 "acknowledged": 3, "recipients": 80},
            ]
            if what == "alert":
                rows = [r for r in rows if r["id"] == params.get("alert_id")]
            return rows, now
        if what == "alert":
            rows = [self._get(f"/alerts/{params['alert_id']}")]
        else:
            rows = self._get("/alerts", {"status": "active"})
        newest = max((_parse_time(r.get("published_at") or r.get("created_at")) for r in rows), default=None,
                     key=lambda t: t or datetime.min.replace(tzinfo=timezone.utc))
        return rows, newest


# --------------------------------------------------- infrastructure system --
class InfrastructureRiskSystem(HttpSource):
    key = "infrastructure_system"
    name = "Infrastructure risk"
    system = "Infrastructure Risk Mapping System"
    description = "Infrastructure assets with their exposure and risk classification."
    provides = ("assets",)
    probe = "assets"
    env_url, env_key = "INFRA_API_URL", "INFRA_API_KEY"
    live_within = timedelta(days=1)
    stale_after = timedelta(days=30)

    def demo(self) -> bool:
        # The interim OpenStreetMap layer (src.integrations.infrastructure) already
        # gives real facilities, so no demo assets are invented here.
        return False

    def _fetch(self, what: str, **params):
        rows = self._get("/assets", {"bbox": params.get("bbox"), "category": params.get("category")})
        return rows, utc_now()


# -------------------------------------------------------------- supply chain --
class SupplyChainSystem(HttpSource):
    key = "supply"
    name = "Emergency supplies"
    system = "Supply Chain Management System"
    description = "Warehouses, emergency stock and distribution progress (no financial data)."
    provides = ("warehouses", "distributions")
    probe = "warehouses"
    env_url, env_key = "SUPPLY_API_URL", "SUPPLY_API_KEY"
    live_within = timedelta(hours=6)
    stale_after = timedelta(days=2)

    def _fetch(self, what: str, **params):
        if self.demo():
            now = utc_now()
            if what == "warehouses":
                return [
                    {"id": "DEMO-WH-1", "name": "DEMO Warehouse, Industrial Area", "lat": -1.3060, "lon": 36.8500,
                     "stock": {"tarpaulins": 400, "water_kits": 900, "food_packs": 1200}},
                    {"id": "DEMO-WH-2", "name": "DEMO Warehouse, Kasarani", "lat": -1.2350, "lon": 36.8900,
                     "stock": {"tarpaulins": 150, "water_kits": 300, "food_packs": 500}},
                ], now
            return [{"id": "DEMO-DIST-1", "destination": "Mathare", "status": "in_transit",
                     "items": {"food_packs": 200}, "updated_at": now.isoformat()}], now
        path = "/warehouses" if what == "warehouses" else "/distributions"
        return self._get(path, params or None), utc_now()


# ------------------------------------------------------------- beneficiaries --
class BeneficiarySystem(HttpSource):
    key = "beneficiaries"
    name = "Beneficiaries"
    system = "Student Beneficiary Management System"
    description = "Aggregated beneficiary counts by area. No personal records are requested or shown."
    provides = ("aggregate",)
    probe = "aggregate"
    env_url, env_key = "BENEFICIARY_API_URL", "BENEFICIARY_API_KEY"
    live_within = timedelta(days=1)
    stale_after = timedelta(days=30)

    def _fetch(self, what: str, **params):
        if self.demo():
            return [{"area": a, "level": "sub_county", "beneficiaries": n} for a, n in
                    (("DEMO Mathare", 420), ("DEMO Kibra", 380), ("DEMO Embakasi Central", 150))], utc_now()
        rows = self._get("/beneficiaries/aggregate", {"area_level": params.get("level", "sub_county")})
        # Defence in depth: keep only aggregate fields even if the partner sends more.
        return [{"area": r.get("area"), "level": r.get("level"), "beneficiaries": r.get("beneficiaries")}
                for r in rows], utc_now()
