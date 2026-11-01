"""
src.reports.history
===================
Past Nairobi floods reported in the news, from Google's Groundsource dataset
(src.ingestion.fetch_groundsource).

Each event is a polygon for the place a news article reported flooded and the
dates it was reported. The same place is usually reported many times (one
polygon per neighbourhood, one event per storm), so the map shows one area per
place with how often, and when, flooding was reported there.

Two cautions travel with every use, and the interface states both:
  * Google's review found 60% of events right in both place and timing.
  * News covers some areas far more than others. A place with no reports may
    simply not have been written about; it is not evidence that it stays dry.

Also used by src.validation.validate_groundsource to compare the model with
floods that were actually reported, and by the outlook to put the most
flood-prone places first (`flood_prone_rank`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

EVENTS_FILE = Path("data/processed/nairobi_groundsource.json")
#: Named places (src.ingestion.fetch_place_names), used to name each area.
PLACES_FILE = Path("data/processed/nairobi_places.json")
#: Place kinds tried first when naming an area: the names people use.
NAME_ORDER = ("suburb", "neighbourhood", "quarter", "village", "locality", "hamlet")
ATTRIBUTION = "Groundsource (Google Research), news reports 2000–2026, CC BY 4.0"
#: Around a replayed storm, reports this many days before and after count as
#: "reported that week".
NEAR_BEFORE = timedelta(days=2)
NEAR_AFTER = timedelta(days=4)
#: Events counted towards "flooded here before": neighbourhood-sized polygons
#: only, as in the spatial validation, smoothed over ~500 m (sigma in cells).
PRONE_MAX_KM2 = 25.0
PRONE_SIGMA = 4.0
#: The map shows neighbourhood-sized areas only. Larger polygons ("Eastlands",
#: a whole sub-county) cover much of the city and bury the rest.
MAP_MAX_KM2 = 5.0

#: Map colours: violet, distinct from the risk hues (amber to red) and the water.
EDGE = [192, 132, 252]
FILL = [168, 85, 247]
EDGE_NEAR = [250, 204, 21]
FILL_NEAR = [250, 204, 21]


@lru_cache(maxsize=1)
def load_events() -> tuple[dict, ...]:
    """Every event with its polygon (shapely) and dates. Empty if the file is missing."""
    if not EVENTS_FILE.exists():
        return ()
    import shapely
    raw = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))["events"]
    return tuple({
        "uuid": e["uuid"],
        "start": date.fromisoformat(e["start"]),
        "end": date.fromisoformat(e["end"]),
        "area_km2": e["area_km2"],
        "wkt": e["wkt"],
        "geom": shapely.from_wkt(e["wkt"]),
    } for e in raw)


_CELLS: tuple[np.ndarray, np.ndarray] | None = None


def cell_mask(geom) -> np.ndarray:
    """(GRID_H, GRID_W) bool: grid cells whose centre lies inside `geom`.

    A polygon smaller than a cell still marks the cell it falls in, so no event
    drops out of a comparison for being tiny.
    """
    global _CELLS
    import shapely
    if _CELLS is None:
        lats = np.linspace(LAT_NORTH, LAT_SOUTH, GRID_H)
        lons = np.linspace(LON_WEST, LON_EAST, GRID_W)
        gy, gx = np.meshgrid(lats, lons, indexing="ij")
        _CELLS = (gx, gy)
    gx, gy = _CELLS
    mask = shapely.contains_xy(geom, gx, gy)
    if not mask.any():
        p = geom.representative_point()
        r = int(round((LAT_NORTH - p.y) / (LAT_NORTH - LAT_SOUTH) * (GRID_H - 1)))
        c = int(round((p.x - LON_WEST) / (LON_EAST - LON_WEST) * (GRID_W - 1)))
        if 0 <= r < GRID_H and 0 <= c < GRID_W:
            mask[r, c] = True
    return mask


def _fmt(d: date) -> str:
    return f"{d.day} {d.strftime('%b %Y')}"


@lru_cache(maxsize=1)
def _named_points() -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    if not PLACES_FILE.exists():
        return np.empty(0), np.empty(0), [], []
    pts = json.loads(PLACES_FILE.read_text(encoding="utf-8"))["places"]
    rank = {k: i for i, k in enumerate(NAME_ORDER)}
    return (np.array([q["lon"] for q in pts]), np.array([q["lat"] for q in pts]),
            [q["name"] for q in pts], [rank.get(q.get("place_type"), len(NAME_ORDER)) for q in pts])


def area_name(geom) -> str:
    """
    A name people recognise for an area: the named places inside it (suburbs
    first, at most three), or "near <place>" for an area too small to contain
    one. Empty if no place names are available.
    """
    import shapely
    xs, ys, names, ranks = _named_points()
    if not names:
        return ""
    inside = np.nonzero(shapely.contains_xy(geom, xs, ys))[0]
    if len(inside):
        picked = sorted(inside, key=lambda i: (ranks[i], names[i]))
        label = ", ".join(names[i] for i in picked[:3])
        return label + (f" and {len(picked) - 3} more" if len(picked) > 3 else "")
    c = geom.representative_point()
    kx = np.cos(np.radians(c.y))
    i = int(np.argmin(((xs - c.x) * kx) ** 2 + (ys - c.y) ** 2))
    return f"Near {names[i]}"


@lru_cache(maxsize=1)
def places() -> tuple[dict, ...]:
    """One entry per reported place: its polygon and the days flooding was reported there."""
    by_place: dict[str, dict] = {}
    days: dict[str, set] = defaultdict(set)
    for e in load_events():
        p = by_place.setdefault(e["wkt"], {"geom": e["geom"], "area_km2": e["area_km2"]})
        days[e["wkt"]].add(e["start"])
    out = []
    for wkt, p in by_place.items():
        p["days"] = tuple(sorted(days[wkt]))
        p["name"] = area_name(p["geom"])
        out.append(p)
    # Largest first, so smaller places draw on top and stay hoverable.
    out.sort(key=lambda p: -p["area_km2"])
    return tuple(out)


def map_rows(around: date | None = None) -> list[dict]:
    """
    Polygon rows for the map's "past-floods" layer. With `around` (a replayed
    storm), places reported flooded that week are picked out.
    """
    rows = []
    for p in places():
        if p["area_km2"] > MAP_MAX_KM2:
            continue
        n = len(p["days"])
        near = [d for d in p["days"] if around and around - NEAR_BEFORE <= d <= around + NEAR_AFTER]
        polys = list(getattr(p["geom"], "geoms", [p["geom"]]))
        count = f"Flooding reported in the news on {n} day{'s' if n != 1 else ''}"
        first, last = p["days"][0], p["days"][-1]
        detail = (f"{_fmt(first)}" if n == 1 else f"{first.year}–{last.year} · most recently {_fmt(last)}")
        alpha = int(min(80, 14 + 8 * n))
        for poly in polys:
            rows.append({
                "polygon": [[round(x, 5), round(y, 5)] for x, y in poly.exterior.coords],
                "fill": (FILL_NEAR + [70]) if near else (FILL + [alpha]),
                "edge": (EDGE_NEAR + [240]) if near else (EDGE + [170]),
                "width": 2 if near else 1,
                "name": p["name"] or "Past flood area",
                "title": count,
                "detail": detail,
                "near": f"In the news that week: {', '.join(_fmt(d) for d in near)}" if near else "",
                "source": ATTRIBUTION,
            })
    if around:
        # Places reported that week draw last, on top of the rest.
        rows.sort(key=lambda r: bool(r["near"]))
    return rows


def summary(around: date | None = None) -> dict:
    """Counts for the note beside the switch."""
    ev = load_events()
    if not ev:
        return {"events": 0, "places": 0}
    shown = [p for p in places() if p["area_km2"] <= MAP_MAX_KM2]
    out = {"events": len(ev), "places": len(places()), "shown": len(shown),
           "first": min(e["start"] for e in ev), "last": max(e["start"] for e in ev)}
    if around:
        near = [e for e in ev if around - NEAR_BEFORE <= e["start"] <= around + NEAR_AFTER]
        out["near_events"] = len(near)
        out["near_places"] = len({e["wkt"] for e in near if e["area_km2"] <= MAP_MAX_KM2})
    return out


@lru_cache(maxsize=4)
def report_density(until: date | None = None) -> np.ndarray:
    """How often each cell lies in (or near) a reported flood area, smoothed."""
    from scipy.ndimage import gaussian_filter
    count = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    masks: dict[str, np.ndarray] = {}
    for e in load_events():
        if e["area_km2"] > PRONE_MAX_KM2 or (until is not None and e["start"] > until):
            continue
        if e["wkt"] not in masks:
            masks[e["wkt"]] = cell_mask(e["geom"])
        count += masks[e["wkt"]]
    return gaussian_filter(count, PRONE_SIGMA)


def flood_prone_rank(susceptibility: np.ndarray, until: date | None = None) -> np.ndarray:
    """
    Drainage susceptibility combined with where floods were reported before:
    the mean of the two city-wide ranks, in 0..1. On 2022-2025 reports, with
    reports up to 2021 as history, it ranks reported areas above the rest with
    AUC 0.78, against 0.73 for susceptibility alone and 0.72 for history alone
    (src.validation.validate_flood_prone). With no events it is susceptibility.
    """
    from scipy.stats import rankdata
    n = susceptibility.size
    r_susc = (rankdata(susceptibility.ravel()) - 1) / max(n - 1, 1)
    density = report_density(until)
    if not density.any():
        return r_susc.reshape(susceptibility.shape)
    r_hist = (rankdata(density.ravel()) - 1) / max(n - 1, 1)
    return ((r_susc + r_hist) / 2).reshape(susceptibility.shape)
