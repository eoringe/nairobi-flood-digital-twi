"""
src.geo.layers
==============
Rows for the operations map layers (specification §2.2, §6.2) and the interim
infrastructure source (§2.9):

    admin-boundaries   sub-county and ward outlines, filtered areas highlighted
    rivers             named rivers and streams (OpenStreetMap)
    water-bodies       lakes, dams and ponds (OpenStreetMap)
    infrastructure     schools, health facilities, bridges, public buildings and
                       emergency services, coloured by category and ringed by the
                       predicted risk at their location
    stations           monitoring stations from the IoT platform, by health
    alert-areas        wards containing places under an alert or model warning

Every row carries `title`, `detail` and `source` for the shared map tooltip.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.geo import boundaries
from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

RIVERS_FILE = Path("data/processed/nairobi_rivers.json")
INFRA_FILE = Path("data/processed/nairobi_infrastructure.json")
OSM = "OpenStreetMap contributors (ODbL)"

CATEGORIES = {
    "school": ("Schools", [96, 165, 250]),
    "hospital": ("Hospitals", [239, 68, 89]),
    "health_centre": ("Health centres", [244, 114, 182]),
    "bridge": ("Bridges", [250, 204, 21]),
    "emergency_service": ("Police and fire stations", [251, 146, 60]),
    "public_building": ("Public buildings", [167, 139, 250]),
}
LEVEL_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
RISK_RING = {0: [255, 255, 255, 140], 1: [240, 185, 63, 255], 2: [240, 138, 60, 255], 3: [239, 68, 89, 255]}
SEVERITY_FILL = {"moderate": [240, 185, 63], "high": [240, 138, 60], "critical": [239, 68, 89],
                 "very high": [239, 68, 89], "low": [52, 211, 153]}


def _cell(lat: float, lon: float) -> int | None:
    """Flat grid index of the cell containing a point, or None outside the grid."""
    r = round((LAT_NORTH - lat) / (LAT_NORTH - LAT_SOUTH) * (GRID_H - 1))
    c = round((lon - LON_WEST) / (LON_EAST - LON_WEST) * (GRID_W - 1))
    return r * GRID_W + c if 0 <= r < GRID_H and 0 <= c < GRID_W else None


# ------------------------------------------------------------ boundaries --
def _rings(geom) -> list[list[list[float]]]:
    polys = getattr(geom, "geoms", [geom])
    return [[[round(x, 5), round(y, 5)] for x, y in p.exterior.coords] for p in polys]


def boundary_rows(highlight: list[str] | None = None) -> list[dict]:
    highlight = set(highlight or [])
    rows = []
    src = boundaries.attribution()
    # Widths in metres; the layer keeps them between 1 and 6 pixels.
    for lvl, colour, width in (("ward", [203, 213, 225, 90], 10), ("sub_county", [226, 232, 240, 190], 30)):
        for a in boundaries.load()[lvl]:
            on = a["id"] in highlight
            for ring in _rings(a["geom"]):
                rows.append({"path": ring, "color": [129, 140, 248, 255] if on else colour,
                             "width": 60 if on else width, "title": a["name"],
                             "detail": boundaries.label(a["id"]).replace(a["name"] + " ", "").capitalize(),
                             "source": src})
    return rows


# ---------------------------------------------------------- rivers, water --
@lru_cache(maxsize=1)
def _rivers() -> dict:
    try:
        return json.loads(RIVERS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"rivers": [], "water_bodies": []}


def river_name(name: str) -> str:
    """English display name: OSM often carries the Swahili 'Mto Nairobi' (mto = river)."""
    return f"{name[4:]} River" if name.startswith("Mto ") else name


def river_rows() -> list[dict]:
    rows = []
    for r in _rivers()["rivers"]:
        named = bool(r["name"])
        rows.append({"path": r["path"], "color": [56, 189, 248, 230 if r["kind"] == "river" else 150],
                     "width": 28 if r["kind"] == "river" else (14 if named else 7),          # metres
                     "title": river_name(r["name"]) or f"Unnamed {r['kind']}", "detail": r["kind"].title(),
                     "source": OSM})
    return rows


def water_rows() -> list[dict]:
    return [{"polygon": w["polygon"], "fill": [56, 189, 248, 80], "edge": [56, 189, 248, 200],
             "title": w["name"] or "Water body", "detail": w["kind"].replace("_", " ").title(), "source": OSM}
            for w in _rivers()["water_bodies"]]


# -------------------------------------------------------- infrastructure --
@lru_cache(maxsize=1)
def assets() -> tuple[dict, ...]:
    """OSM facilities with their grid cell, ward and sub-county."""
    try:
        raw = json.loads(INFRA_FILE.read_text(encoding="utf-8"))["assets"]
    except FileNotFoundError:
        return ()
    wards, subs = boundaries.cell_index("ward"), boundaries.cell_index("sub_county")
    wl, sl = boundaries.load()["ward"], boundaries.load()["sub_county"]
    out = []
    for a in raw:
        cell = _cell(a["lat"], a["lon"])
        out.append({**a, "cell": cell,
                    "ward": wl[wards[cell]]["id"] if cell is not None and wards[cell] >= 0 else None,
                    "sub_county": sl[subs[cell]]["id"] if cell is not None and subs[cell] >= 0 else None})
    return tuple(out)


def exposure(probability: np.ndarray | None) -> list[dict]:
    """Every asset with the predicted risk level at its location (0 when no prediction)."""
    if probability is None:
        return [{**a, "level": 0} for a in assets()]
    from scipy.ndimage import gaussian_filter
    from src.routing.flood_router import MAP_SMOOTHING_SIGMA, level_of
    lv = level_of(gaussian_filter(probability.astype(np.float32), sigma=MAP_SMOOTHING_SIGMA)).ravel()
    return [{**a, "level": int(lv[a["cell"]]) if a["cell"] is not None else 0} for a in assets()]


def infrastructure_rows(items: list[dict], categories: list[str] | None = None,
                        area_ids: list[str] | None = None) -> list[dict]:
    rows = []
    for a in items:
        if categories and a["category"] not in categories:
            continue
        if area_ids and not ({a.get("ward"), a.get("sub_county")} & set(area_ids)):
            continue
        lvl = a.get("level", 0)
        where = boundaries.label(a["ward"]) if a.get("ward") else "outside the mapped wards"
        rows.append({"position": [a["lon"], a["lat"]], "color": CATEGORIES[a["category"]][1] + [235],
                     "line": RISK_RING[lvl], "radius": 7 if lvl >= 2 else 5,
                     "title": a["name"],
                     "detail": f"{CATEGORIES[a['category']][0][:-1]} · {where} · predicted risk {LEVEL_NAMES[lvl]}",
                     "source": f"{OSM} (interim, until the Infrastructure Risk Mapping System connects)"})
    return rows


# --------------------------------------------------------------- stations --
HEALTH_COLOUR = {"ok": [52, 211, 153], "degraded": [240, 185, 63], "offline": [148, 163, 184],
                 "unknown": [148, 163, 184]}


def station_rows(stations: list[dict], demo: bool = False) -> list[dict]:
    rows = []
    for s in stations:
        latest = s.get("latest") or {}
        parts = []
        above = False
        for kind, r in latest.items():
            v = r.get("value")
            unit = r.get("unit") or ""
            parts.append(f"{kind.replace('_', ' ')} {'missing' if v is None else f'{v} {unit}'}")
            if kind == "water_level" and v is not None and s.get("warning_m") and v >= s["warning_m"]:
                above = True
        rows.append({"position": [s["lon"], s["lat"]], "color": HEALTH_COLOUR.get(s.get("health"), HEALTH_COLOUR["unknown"]) + [255],
                     "line": [239, 68, 89, 255] if above else [255, 255, 255, 255],
                     "title": s["name"], "detail": f"{s.get('health', 'unknown').title()} · " + (", ".join(parts) or "no readings")
                     + (" · ABOVE WARNING" if above else ""),
                     "source": ("DEMO DATA, not measurements" if demo else "IoT Sensor Data Platform")
                     + f" · last contact {s.get('last_communication') or 'unknown'}"})
    return rows


# ------------------------------------------------------------ alert areas --
def alert_area_rows(alerts: list[dict], source: str) -> list[dict]:
    """Wards that contain places named in alerts, filled by the worst severity."""
    order = ["low", "moderate", "high", "very high", "critical"]
    worst: dict[str, tuple[str, list[str]]] = {}
    pa = boundaries.place_areas()
    for a in alerts:
        sev = (a.get("severity") or "moderate").lower()
        for place in a.get("areas") or []:
            ward = pa.get(place, {}).get("ward")
            if not ward:
                continue
            cur = worst.get(ward)
            if cur is None or order.index(sev) > order.index(cur[0]):
                worst[ward] = (sev, (cur[1] if cur else []) + [a.get("title", "")])
            else:
                cur[1].append(a.get("title", ""))
    rows = []
    for ward_id, (sev, titles) in worst.items():
        w = boundaries.get(ward_id)
        for ring in _rings(w["geom"]):
            rows.append({"polygon": ring, "fill": SEVERITY_FILL.get(sev, [240, 185, 63]) + [60],
                         "edge": SEVERITY_FILL.get(sev, [240, 185, 63]) + [230],
                         "title": f"{w['name']} ward: {sev.title()} alert",
                         "detail": "; ".join(dict.fromkeys(t for t in titles if t))[:240],
                         "source": source})
    return rows


# ----------------------------------------------------------------- labels --
def subcounty_label_rows(highlight: list[str] | None = None) -> list[dict]:
    """One label per sub-county, at a point guaranteed inside it."""
    highlight = set(highlight or [])
    rows = []
    for a in boundaries.load()["sub_county"]:
        p = a["geom"].representative_point()
        rows.append({"position": [round(p.x, 5), round(p.y, 5)], "text": a["name"].upper(),
                     "color": [165, 180, 252, 255] if a["id"] in highlight else [226, 232, 240, 235]})
    return rows


def _segment_angle(path: list[list[float]], i: int) -> float:
    """Screen angle (degrees, counter-clockwise) of the path at vertex i, kept upright."""
    import math
    a, b = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
    ang = math.degrees(math.atan2(b[1] - a[1], (b[0] - a[0]) * math.cos(math.radians(a[1]))))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return round(ang, 1)


def _length(path: list[list[float]]) -> float:
    return sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(path, path[1:]))


#: Spacing of repeated river names along a river (degrees, about 2.5 km).
RIVER_LABEL_SPACING = 0.0225


def river_label_rows(max_per_name: int = 8) -> list[dict]:
    """Named rivers and streams, labelled every ~2.5 km along their course."""
    by_name: dict[str, list[list]] = {}
    for r in _rivers()["rivers"]:
        if r["name"] and len(r["path"]) >= 3:
            by_name.setdefault(r["name"], []).append(r["path"])
    rows = []
    for name, paths in by_name.items():
        placed = []
        for path in sorted(paths, key=_length, reverse=True):
            if _length(path) < 0.004:            # under ~450 m: too short to carry a name
                continue
            run, next_at = 0.0, min(RIVER_LABEL_SPACING / 2, _length(path) / 2)
            for k in range(1, len(path)):
                run += ((path[k][0] - path[k - 1][0]) ** 2 + (path[k][1] - path[k - 1][1]) ** 2) ** 0.5
                if run < next_at:
                    continue
                next_at += RIVER_LABEL_SPACING
                pos = path[k]
                # Paths of one river overlap; skip a spot already named nearby.
                if any(abs(pos[0] - q[0]) + abs(pos[1] - q[1]) < RIVER_LABEL_SPACING * 0.6 for q in placed):
                    continue
                placed.append(pos)
                rows.append({"position": pos, "text": river_name(name), "angle": _segment_angle(path, k),
                             "color": [125, 211, 252, 255]})
                if len(placed) >= max_per_name:
                    break
            if len(placed) >= max_per_name:
                break
    return rows


#: Which facility keeps its label when several fall in one ~130 m square.
LABEL_PRIORITY = ["hospital", "health_centre", "emergency_service", "public_building", "bridge", "school"]
LABEL_CELL_DEG = 0.0012


def facility_label_rows(items: list[dict], categories: list[str] | None = None,
                        area_ids: list[str] | None = None) -> list[dict]:
    """Names beside named facilities; generic 'Bridge on ... road' names are left unlabelled."""
    rows = []
    for a in items:
        if categories and a["category"] not in categories:
            continue
        if area_ids and not ({a.get("ward"), a.get("sub_county")} & set(area_ids)):
            continue
        if a["category"] == "bridge" and a["name"].startswith("Bridge on "):
            continue
        rows.append({"position": [a["lon"], a["lat"]], "text": a["name"] if len(a["name"]) <= 30 else a["name"][:29] + "…",
                     "category": a["category"],
                     "color": CATEGORIES[a["category"]][1] + [255] if a["category"] in CATEGORIES else [226, 232, 240, 255]})
    # Thin to one label per ~130 m square so names do not pile up in dense areas
    # (the deck.gl bundle has no collision filter); every point still has a tooltip.
    rank = {c: i for i, c in enumerate(LABEL_PRIORITY)}
    rows.sort(key=lambda r: (rank.get(r["category"], 99), len(r["text"])))
    kept, taken = [], set()
    for r in rows:
        cell = (round(r["position"][1] / LABEL_CELL_DEG), round(r["position"][0] / LABEL_CELL_DEG))
        if cell in taken:
            continue
        taken.add(cell)
        kept.append(r)
    return kept
