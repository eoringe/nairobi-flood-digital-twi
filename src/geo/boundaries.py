"""
src.geo.boundaries
==================
Counties, sub-counties and wards (src.ingestion.fetch_boundaries), and the
lookups that geographical filtering needs (specification §2.12, FR-013,
UC-FVD-007): which area a point, a grid cell or a named place falls in, and
which places an area contains.

Area ids are geoBoundaries shape ids. Filters pass ids; screens show names.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely import wkt as shapely_wkt

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

FILE = Path("data/processed/nairobi_boundaries.json")
LEVELS = ("county", "sub_county", "ward")
LEVEL_LABELS = {"county": "County", "sub_county": "Sub-county", "ward": "Ward"}


@lru_cache(maxsize=1)
def load() -> dict[str, list[dict]]:
    """{level: [{id, name, level, parent, geom}]}; empty levels if the file is missing."""
    try:
        raw = json.loads(FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {lvl: [] for lvl in LEVELS}
    out = {}
    for lvl in LEVELS:
        rows = []
        for r in raw.get(lvl, []):
            g = shapely_wkt.loads(r["wkt"])
            shapely.prepare(g)
            rows.append({"id": r["id"], "name": r["name"], "level": lvl, "parent": r.get("parent"), "geom": g})
        out[lvl] = rows
    return out


@lru_cache(maxsize=1)
def by_id() -> dict[str, dict]:
    return {r["id"]: r for rows in load().values() for r in rows}


def get(area_id: str | None) -> dict | None:
    return by_id().get(area_id) if area_id else None


def attribution() -> str:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))["source"]
    except Exception:                                           # noqa: BLE001
        return "geoBoundaries"


def area_at(lat: float, lon: float, level: str = "ward") -> dict | None:
    p = shapely.Point(lon, lat)
    return next((r for r in load()[level] if r["geom"].contains(p)), None)


def lineage(area_id: str) -> list[dict]:
    """The area and its parents, smallest first: ward, sub-county, county."""
    out, a = [], get(area_id)
    while a is not None:
        out.append(a)
        a = get(a["parent"])
    return out


def label(area_id: str) -> str:
    """'Mabatini ward, Mathare' style label."""
    chain = lineage(area_id)
    if not chain:
        return area_id
    head = f"{chain[0]['name']} {LEVEL_LABELS[chain[0]['level']].lower()}"
    return head + (f", {chain[1]['name']}" if len(chain) > 1 else "")


@lru_cache(maxsize=3)
def cell_index(level: str) -> np.ndarray:
    """Flat (GRID_H*GRID_W) index into load()[level] for each grid cell centre; -1 outside every area."""
    lats = np.linspace(LAT_NORTH, LAT_SOUTH, GRID_H)
    lons = np.linspace(LON_WEST, LON_EAST, GRID_W)
    gy, gx = np.meshgrid(lats, lons, indexing="ij")
    out = np.full(GRID_H * GRID_W, -1, dtype=np.int32)
    for i, r in enumerate(load()[level]):
        inside = shapely.contains_xy(r["geom"], gx.ravel(), gy.ravel())
        out[inside & (out < 0)] = i
    return out


def cell_mask(area_id: str) -> np.ndarray:
    """Boolean (GRID_H, GRID_W) mask of the cells in an area."""
    a = get(area_id)
    if a is None:
        return np.zeros((GRID_H, GRID_W), dtype=bool)
    idx = load()[a["level"]].index(a)
    return (cell_index(a["level"]) == idx).reshape(GRID_H, GRID_W)


@lru_cache(maxsize=1)
def place_areas() -> dict[str, dict[str, str]]:
    """Gazetteer place name -> {level: area id} for the area its point lies in."""
    from src.routing.flood_router import PLACES_FILE
    places = json.loads(Path(PLACES_FILE).read_text(encoding="utf-8"))["places"]
    out: dict[str, dict[str, str]] = {}
    for p in places:
        found = {}
        for lvl in LEVELS:
            a = area_at(p["lat"], p["lon"], lvl)
            if a:
                found[lvl] = a["id"]
        out.setdefault(p["name"], found)
    return out


def place_in(place: str, area_ids: list[str] | set[str] | None) -> bool:
    """True if a named place lies in any of the areas; no filter means everything matches."""
    if not area_ids:
        return True
    return bool(set(place_areas().get(place, {}).values()) & set(area_ids))


def point_in(lat: float, lon: float, area_ids: list[str] | set[str] | None) -> bool:
    if not area_ids:
        return True
    p = shapely.Point(lon, lat)
    return any((a := get(i)) is not None and a["geom"].contains(p) for i in area_ids)


def options(levels: tuple[str, ...] = LEVELS, only: list[str] | None = None) -> list[dict]:
    """Dropdown options, grouped by level: 'Sub-county · Mathare'."""
    out = []
    for lvl in levels:
        for r in load()[lvl]:
            if only and r["id"] not in only:
                continue
            parent = get(r["parent"])
            suffix = f" ({parent['name']})" if parent and lvl == "ward" else ""
            out.append({"label": f"{LEVEL_LABELS[lvl]} · {r['name']}{suffix}", "value": r["id"]})
    return out


def bounds(area_ids: list[str]) -> list[float] | None:
    """[west, south, east, north] around the areas, for zooming the map to a filter."""
    geoms = [a["geom"] for i in area_ids if (a := get(i))]
    if not geoms:
        return None
    return list(shapely.union_all(geoms).bounds)


def ids_by_name(names: list[str]) -> list[str]:
    """Area ids for names, e.g. a county officer's assigned areas stored by name or id."""
    out = []
    for n in names:
        if get(n):
            out.append(n)
            continue
        out += [r["id"] for rows in load().values() for r in rows if r["name"].lower() == n.lower()]
    return out
