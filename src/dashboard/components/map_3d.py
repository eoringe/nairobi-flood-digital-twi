"""
src.dashboard.components.map_3d
================================
Nairobi Urban Flood Digital Twin — Pydeck 3D WebGL Canvas Component

PURPOSE
-------
1. Hardware-accelerated 3D WebGL rendering using Pydeck (Deck.gl core).
2. North-Up exact raster transform so flood polygons align 100% precisely over streets and rivers.
3. PRECISE STREET-ONLY FLOOD OVERLAYS: Building footprints are subtracted (cut out) using Shapely so flood water renders ONLY on streets, roundabouts, and river channels — NEVER on top of buildings.
4. Floating legend pill embedded directly inside Pydeck HTML canvas document.
5. THE CITY IN 3D: every OpenStreetMap building, streamed as vector tiles from
   OpenFreeMap and raised to its real height (towers such as Times Tower and
   the GTC reach 130-150 m), lit by the sun in the light theme and as a night
   skyline in the dark theme. The local footprints are the offline fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio.features
import rasterio.transform
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
import pydeck as pdk
from loguru import logger

from src.models.predict import (
    LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST, GRID_H, GRID_W, NAIROBI_LOCATIONS,
)

BUILDINGS_JSON = Path("data/processed/nairobi_buildings_3d.json")

_POINT_FALLBACK_BOX_DEG = 0.00018  # ~20m box, used only when a feature has no real footprint polygon


def _footprint_polygon(feature: dict) -> Polygon:
    """
    Real footprint polygon when available (src.preprocessing.building_processor
    now parses the source CSV's WKT geometry column); falls back to a small
    synthetic box around the centroid for older Point-geometry feature sets.
    """
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return Polygon(geom["coordinates"][0])
    lon, lat = geom["coordinates"]
    d = _POINT_FALLBACK_BOX_DEG
    return Polygon([[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]])


# Pre-compute merged Shapely building footprint geometry at module import time.
# Only structures above ~120m2 are subtracted from the water fill — most of
# the 25k footprints are single-room informal-settlement structures (~30m2),
# and cutting a precise hole for every single one produced a dense uniform
# "checkerboard" of tiny rectangles that read as a GIS layer rather than
# real floodwater. Real overland flow does not stay crisply out of every hut
# either. A small buffer+simplify pass afterwards rounds the corners of the
# structures that ARE kept, so their edges against the water fill aren't
# perfectly rectangular.
BUILDINGS_UNION = None
_BUILDING_AREA_MIN_M2 = 120.0
_BUILDING_EDGE_SOFTEN_DEG = 0.00003  # ~3m round-corner buffer
if BUILDINGS_JSON.exists():
    try:
        with open(BUILDINGS_JSON, "r") as f:
            _geo_data = json.load(f)

        _b_polys = [
            _footprint_polygon(_feat)
            for _feat in _geo_data.get("features", [])[:8000]
            if _feat.get("properties", {}).get("area_m2", 999.0) >= _BUILDING_AREA_MIN_M2
        ]

        BUILDINGS_UNION = unary_union(_b_polys)
        if _BUILDING_EDGE_SOFTEN_DEG > 0 and not BUILDINGS_UNION.is_empty:
            BUILDINGS_UNION = BUILDINGS_UNION.buffer(_BUILDING_EDGE_SOFTEN_DEG).buffer(-_BUILDING_EDGE_SOFTEN_DEG)
        if not BUILDINGS_UNION.is_valid:
            BUILDINGS_UNION = make_valid(BUILDINGS_UNION)
        logger.info(f"Pre-computed 3D building footprint union ({len(_b_polys)} structures >= {_BUILDING_AREA_MIN_M2:.0f}m2) for street-only flood overlay.")
    except Exception as _e:
        logger.warning(f"Could not pre-compute building footprints: {_e}")


#: Named places across Nairobi, from OpenStreetMap via
#: src/ingestion/fetch_place_names.py. Several hundred suburbs, neighbourhoods,
#: quarters and villages, so a flooded patch anywhere on the prediction grid can
#: be named rather than only those near a handful of curated points.
_PLACES_FILE = Path("data/processed/nairobi_places.json")

#: With places this dense the nearest one is almost always the right one, so the
#: bare name is used out to ~1.7 km and a "Near X" form to ~3.3 km. Beyond that
#: the polygon really is between named places and no claim is made.
_REGION_EXACT_DEG = 0.015
_REGION_NEAR_DEG = 0.03


def _load_places() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the gazetteer as parallel arrays for vectorised nearest-neighbour.

    Falls back to the ten curated monitoring points if the cache is absent, so
    a fresh clone still labels regions - just more coarsely - rather than
    failing to render.
    """
    try:
        data = json.loads(_PLACES_FILE.read_text(encoding="utf-8"))
        pl = data["places"]
        logger.info(
            f"Region labelling: {len(pl)} OpenStreetMap places "
            f"(fetched {data.get('fetched', 'unknown')})"
        )
        return (np.array([p["lat"] for p in pl], dtype=np.float64),
                np.array([p["lon"] for p in pl], dtype=np.float64),
                [p["name"] for p in pl])
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(
            f"Region labelling: {_PLACES_FILE} unavailable ({exc}); falling back "
            f"to the {len(NAIROBI_LOCATIONS)} curated monitoring points. "
            f"Run `python -m src.ingestion.fetch_place_names` for full coverage."
        )
        names = list(NAIROBI_LOCATIONS.keys())
        return (np.array([NAIROBI_LOCATIONS[n][0] for n in names]),
                np.array([NAIROBI_LOCATIONS[n][1] for n in names]),
                names)


def _iter_polys(geom):
    """Yield the Polygon parts of any geometry, ignoring lines and points."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type == "Polygon"]
    return []


#: Minimum polygon area to render, in square degrees. Defined at module
#: scope because the place-splitter needs it too.
MIN_AREA_DEG2 = 4.0e-6  # ~25m x 25m — drops stray single-pixel speckle

_PLACE_LATS, _PLACE_LONS, _PLACE_NAMES = _load_places()


def _build_place_partition():
    """
    Partition Nairobi into one cell per named place (a Voronoi diagram).

    Flood contours follow river corridors and can run many kilometres. Naming
    such a polygon from a single interior point labels its entire length after
    one place, so hovering the Mathare end of a ribbon that reaches Kariobangi
    returned "Kariobangi South". Splitting each polygon against this partition
    gives every piece the name of the place it actually sits in.

    Returns (STRtree over the cells, cell list, name per cell) or None if the
    geometry libraries are unavailable, in which case labelling falls back to
    one name per polygon.
    """
    try:
        from shapely.ops import voronoi_diagram
        from shapely.geometry import MultiPoint, Point
        from shapely import STRtree

        pts = MultiPoint([(lon, lat) for lon, lat in zip(_PLACE_LONS, _PLACE_LATS)])
        cells = list(voronoi_diagram(pts, envelope=pts.buffer(0.05)).geoms)

        # voronoi_diagram does not preserve input order, so each cell is matched
        # to the place it was generated from by containment.
        names = []
        for cell in cells:
            hit = "Unmonitored area"
            for lat, lon, nm in zip(_PLACE_LATS, _PLACE_LONS, _PLACE_NAMES):
                if cell.contains(Point(lon, lat)):
                    hit = nm
                    break
            names.append(hit)

        logger.info(f"Region labelling: place partition built ({len(cells)} cells)")
        return STRtree(cells), cells, names
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(
            f"Region labelling: place partition unavailable ({exc}); each flood "
            f"polygon will carry a single name along its whole length."
        )
        return None


_PLACE_PARTITION = _build_place_partition()


def _split_by_place(poly):
    """
    Split a flood polygon into (piece, place name) parts.

    Pieces below a minimum area are dropped rather than rendered as slivers at
    cell boundaries. Falls back to the whole polygon under its nearest place if
    no partition is available.
    """
    if _PLACE_PARTITION is None:
        c = poly.representative_point()
        return [(poly, _nearest_region(c.y, c.x))]

    tree, cells, names = _PLACE_PARTITION
    out = []
    # Splitting produces slivers where a polygon clips the corner of a cell.
    # They carry no information, are invisible at any usable zoom, and each one
    # costs a geometry serialised into the map document, so the floor for a
    # split piece is higher than for a whole polygon.
    min_piece = MIN_AREA_DEG2 * 8
    # predicate="intersects" makes the index return only cells that genuinely
    # touch the polygon. Without it the query returns every cell whose bounding
    # box overlaps, which for a long river-corridor ribbon is most of the county,
    # and each one costs a full intersection that returns empty.
    try:
        candidates = tree.query(poly, predicate="intersects")
    except TypeError:                                      # older shapely
        candidates = tree.query(poly)
    for idx in candidates:
        piece = poly.intersection(cells[int(idx)])
        if piece.is_empty or piece.area < min_piece:
            continue
        for part in _iter_polys(piece):
            if part.area >= min_piece:
                out.append((part, names[int(idx)]))
    if not out:
        c = poly.representative_point()
        return [(poly, _nearest_region(c.y, c.x))]
    return out


def _nearest_region(lat: float, lon: float) -> str:
    """
    Name the place a flood polygon sits in or beside.

    Returns the bare name when the polygon is essentially on the place, a
    "Near X" form when it is in the vicinity, and declines to guess beyond that.
    Naming the nearest place unconditionally would attribute water to somewhere
    kilometres away whenever it fell in a genuinely unnamed gap, which reads as
    authoritative while being wrong.
    """
    if not len(_PLACE_NAMES):
        return "Unmonitored area"
    d2 = (_PLACE_LATS - lat) ** 2 + (_PLACE_LONS - lon) ** 2
    i = int(np.argmin(d2))
    d = float(d2[i]) ** 0.5
    if d <= _REGION_EXACT_DEG:
        return _PLACE_NAMES[i]
    if d <= _REGION_NEAR_DEG:
        return f"Near {_PLACE_NAMES[i]}"
    return "Unmonitored area"
    d = best_d2 ** 0.5
    if d <= _REGION_EXACT_DEG:
        return best
    if d <= _REGION_NEAR_DEG:
        return f"Near {best}"
    return "Unmonitored area"


def _band_value_in(poly, grid: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                   lo: float, hi: float) -> float:
    """
    Representative value for a polygon, within its own band.

    The risk bands are nested: the polygon for the moderate band covers every
    cell at or above the moderate threshold, including the severe core drawn on
    top of it. So the maximum inside a moderate polygon is legitimately a severe
    value, and reporting it contradicts the polygon's own label.

    This takes the cells inside the polygon whose values fall within [lo, hi] -
    the annulus the band actually describes - and reports their median. Falls
    back to the midpoint of the band if containment testing is unavailable.
    """
    minx, miny, maxx, maxy = poly.bounds
    r0, r1 = np.searchsorted(-lats, -maxy), np.searchsorted(-lats, -miny)
    c0, c1 = np.searchsorted(lons, minx), np.searchsorted(lons, maxx)
    r0, r1 = max(0, r0 - 1), min(len(lats), r1 + 1)
    c0, c1 = max(0, c0 - 1), min(len(lons), c1 + 1)
    if r1 <= r0 or c1 <= c0:
        return (lo + hi) / 2.0

    window = grid[r0:r1, c0:c1]
    # Point-in-polygon over a mesh is the expensive part, and splitting by place
    # produces many small pieces. Below a few dozen cells the window is
    # essentially the piece already, so the containment test buys accuracy that
    # the band filter below would impose anyway.
    if window.size <= 48:
        vals = window.ravel()
    else:
        try:
            from shapely import contains_xy
            gy, gx = np.meshgrid(lats[r0:r1], lons[c0:c1], indexing="ij")
            inside = contains_xy(poly, gx, gy)
            vals = window[inside] if inside.any() else window.ravel()
        except Exception:                                  # noqa: BLE001
            vals = window.ravel()

    in_band = vals[(vals >= lo) & (vals <= hi)]
    if in_band.size:
        return float(np.median(in_band))
    return float(np.clip(np.median(vals), lo, hi))


#: Square kilometres per square degree at Nairobi's latitude, for converting
#: polygon areas. 1 deg latitude ~ 110.57 km; 1 deg longitude ~ 111.32*cos(1.29) km.
_KM2_PER_DEG2 = 110.57 * 111.32 * 0.99975


def summarise_flooded_regions(features: list[dict]) -> dict:
    """
    Per-place summary built from the polygons actually drawn on the map.

    The side panel previously listed ten hard-coded locations and sampled a
    fixed patch around each, so it could report a place as safe while the map
    drew water across it, and could not mention anywhere outside its shortlist.
    Aggregating the rendered pieces instead means the panel and the map always
    describe the same flooding.

    `flooded_pct` is the share of that place's Voronoi cell covered by water,
    which is what "how much of this area is flooded" means when places are
    points rather than boundaries.
    """
    cell_area: dict[str, float] = {}
    if _PLACE_PARTITION is not None:
        _tree, cells, names = _PLACE_PARTITION
        for cell, nm in zip(cells, names):
            cell_area[nm] = cell_area.get(nm, 0.0) + cell.area

    agg: dict[str, dict] = {}
    for f in features:
        pr = f["properties"]
        region = pr.get("region", "Unmonitored area")
        try:
            area = shape(f["geometry"]).area
        except Exception:                                  # noqa: BLE001
            continue
        rec = agg.setdefault(region, {"area_deg2": 0.0, "peak": 0.0, "level": 0})
        rec["area_deg2"] += area
        rec["peak"] = max(rec["peak"], float(pr.get("value", 0.0)))
        rec["level"] = max(rec["level"], int(pr.get("level", 0)))

    out: dict[str, dict] = {}
    for region, rec in agg.items():
        denom = cell_area.get(region, 0.0)
        pct = 100.0 * rec["area_deg2"] / denom if denom > 0 else 0.0
        pct = min(pct, 100.0)
        # Severity follows the band the polygons reached, so a place shaded red
        # on the map cannot be listed as MODERATE in the panel beside it.
        level = {3: "CRITICAL", 2: "HIGH", 1: "MODERATE"}.get(rec["level"], "LOW")
        out[region] = {
            "risk_level": level,
            "peak_probability_pct": round(100.0 * rec["peak"], 1),
            "flooded_pct": round(pct, 1),
            "flooded_area_km2": round(rec["area_deg2"] * _KM2_PER_DEG2, 3),
        }

    # Every flooded place is returned, ranked by severity then area. Capping
    # here would make the count of at-risk zones equal the cap, so display
    # limits belong in the components that render, not in the summary.
    order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    return dict(sorted(out.items(),
                       key=lambda kv: (order.get(kv[1]["risk_level"], 4),
                                       -kv[1]["flooded_area_km2"])))


#: Per-theme map styling. The light theme uses CARTO Voyager rather than the
#: near-white Positron: routing needs roads to be legible, and Voyager draws
#: them with class-graded colour. Flood fills are more opaque on the light
#: basemap because amber at the dark theme's alpha disappears against white.
MAP_THEMES = {
    "dark": {
        "basemap": pdk.map_styles.CARTO_DARK,
        # (water fill, edge). The fill is the water's colour, more opaque the
        # likelier the flooding; the edge carries the risk band's colour.
        "flood": {
            1: ([22, 92, 168, 175],  [240, 185, 63, 170]),
            2: ([22, 92, 168, 205],  [240, 138, 60, 180]),
            3: ([22, 92, 168, 230],  [239, 68, 89, 190]),
        },
        "halo": {1: [22, 92, 168, 45], 2: [22, 92, 168, 55], 3: [22, 92, 168, 65]},
        "buildings": ([91, 155, 255, 205], [74, 98, 140, 180], [88, 98, 118, 150]),
        "tooltip": {"backgroundColor": "#171d29", "color": "#eef1f7", "border": "1px solid rgba(255,255,255,0.08)",
                    "boxShadow": "0 12px 32px rgba(0,0,0,0.42)"},
        "legend": {"bg": "rgba(17, 22, 31, 0.80)", "border": "rgba(255, 255, 255, 0.08)", "text": "#eef1f7",
                   "muted": "#8a93a6", "shadow": "0 12px 32px rgba(0, 0, 0, 0.42)", "bldg": "#58627a",
                   "water": "#165ca8"},
        "pin_stroke": [17, 22, 31, 255],
    },
    "light": {
        "basemap": pdk.map_styles.CARTO_ROAD,
        "flood": {
            1: ([30, 120, 210, 170], [214, 150, 10, 170]),
            2: ([30, 120, 210, 200], [214, 100, 20, 180]),
            3: ([30, 120, 210, 228], [200, 30, 55, 190]),
        },
        "halo": {1: [30, 120, 210, 40], 2: [30, 120, 210, 50], 3: [30, 120, 210, 60]},
        "buildings": ([37, 99, 235, 170], [140, 158, 186, 170], [178, 188, 202, 150]),
        "tooltip": {"backgroundColor": "#ffffff", "color": "#0f1626", "border": "1px solid #e5e9f0",
                    "boxShadow": "0 8px 24px rgba(16,24,40,0.12)"},
        "legend": {"bg": "rgba(255, 255, 255, 0.86)", "border": "#e5e9f0", "text": "#0f1626",
                   "muted": "#7b8497", "shadow": "0 8px 24px rgba(16, 24, 40, 0.12)", "bldg": "#b0bac6",
                   "water": "#1e78d2"},
        "pin_stroke": [255, 255, 255, 255],
    },
}


def generate_flood_contour_geojson(
    depth_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    display_mode: str = "PROBABILITY",
    value_is_probability: bool = False,
    theme: str = "dark",
) -> tuple[dict, dict]:
    """
    Extract precise, street-only flood polygons with building footprints subtracted.
    Water renders strictly on streets, roundabouts, alleys, and river channels.

    Returns (main_geojson, halo_geojson). `main` is the readable, semi-
    transparent water fill; `halo` is a wider, much fainter duplicate of the
    same geometry meant to render underneath it, so the edge fades into the
    basemap instead of stopping at a hard vector outline (deck.gl has no
    native edge-feather, so this fakes one).
    """
    h, w = depth_grid.shape
    empty = {"type": "FeatureCollection", "features": []}
    # The bail-out threshold depends on what the field means: 0.2 m of water,
    # or a 5% flood probability.
    if np.max(depth_grid) < (0.05 if value_is_probability else 0.2):
        return empty, empty

    # Wider smoothing than the raw 30m grid so contours read as organic
    # water edges rather than a blocky raster-to-vector staircase.
    smooth_grid = gaussian_filter(depth_grid.astype(np.float64), sigma=2.4)

    risk_mask = np.zeros((h, w), dtype=np.int32)

    # Water is rendered as water — a blue/teal depth ramp echoing the app's
    # own accent color (src/dashboard/assets/custom.css --accent #35c2d1),
    # not the amber/orange/red risk-severity ramp used for badges and alerts
    # elsewhere in the UI. Alpha stays low enough that streets and terrain
    # under the CARTO_DARK basemap remain visible through the fill.
    # The model outputs a per-cell probability. There is no depth mode: depth is
    # not estimated anywhere in the pipeline, and rendering probability through a
    # metre-scaled ramp invented a quantity the system cannot produce.
    #
    # PROBABILITY : graded probability bands
    # EXTENT      : binary - flooded or not at the operating threshold
    if display_mode == "EXTENT":
        prob_grid = np.clip(smooth_grid * 100.0, 0.0, 100.0)
        risk_mask[prob_grid >= 50.0] = 3
        names = {1: "", 2: "", 3: "Flooded (>50% probability)"}
        bands = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.50, 1.00)}
    else:
        prob_grid = np.clip(smooth_grid * 100.0, 0.0, 100.0)
        risk_mask[prob_grid >= 25.0] = 1
        risk_mask[prob_grid >= 55.0] = 2
        risk_mask[prob_grid >= 82.0] = 3
        names = {1: "MODERATE (25-55%)", 2: "HIGH (55-82%)", 3: "CRITICAL (>82%)"}
        bands = {1: (0.25, 0.55), 2: (0.55, 0.82), 3: (0.82, 1.00)}

    # Drawn as water standing in the streets (the shader in CITY_JS adds ripples
    # and reflections). Likelier flooding is more opaque water, never "deeper"
    # water, since no depth is estimated. Each polygon's edge carries its risk
    # band's colour, the palette of RISK_COLORS in src/dashboard/callbacks.py, so
    # a region card reading CRITICAL in red matches a red-edged patch of water.
    palette = MAP_THEMES.get(theme, MAP_THEMES["dark"])
    styles = {
        lvl: {"fillColor": fill, "lineColor": line, "name": names[lvl]}   # amber / orange / red
        for lvl, (fill, line) in palette["flood"].items()
    }
    halo_styles = palette["halo"]

    transform = rasterio.transform.from_bounds(
        LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, w, h
    )
    BUFFER_R = 0.0011       # ~120m open/close radius — real rounded water edges,
                            # not a street-precise vector trace (see module docstring)
    SIMPLIFY_TOL = 0.00022  # Reduce vertex count along the now-rounded curve
    HALO_BUFFER = 0.0014    # Extra dilation for the faded edge-blend halo

    features = []
    halo_features = []

    for level in sorted(styles.keys()):
        level_mask = (risk_mask == level).astype(np.int32)

        raw_polys = []
        for geom, val in rasterio.features.shapes(level_mask, transform=transform):
            if int(val) == 1:
                try:
                    poly = shape(geom)
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    if poly.area >= MIN_AREA_DEG2:
                        raw_polys.append(poly)
                except Exception:
                    continue

        if not raw_polys:
            continue

        try:
            merged = unary_union(raw_polys)
            smoothed = merged.buffer(BUFFER_R).buffer(-BUFFER_R * 0.55)
            smoothed = smoothed.simplify(SIMPLIFY_TOL, preserve_topology=True)

            if not smoothed.is_valid:
                smoothed = make_valid(smoothed)

            # SUBTRACT BUILDING FOOTPRINTS so water flows strictly on streets
            if BUILDINGS_UNION is not None and not BUILDINGS_UNION.is_empty:
                smoothed = smoothed.difference(BUILDINGS_UNION)
                if not smoothed.is_valid:
                    smoothed = make_valid(smoothed)

        except Exception as e:
            logger.warning(f"Shapely smoothing/difference failed for level {level}: {e}")
            smoothed = unary_union(raw_polys)

        if smoothed.is_empty:
            continue

        try:
            halo_geom = smoothed.buffer(HALO_BUFFER)
            if BUILDINGS_UNION is not None and not BUILDINGS_UNION.is_empty:
                halo_geom = halo_geom.difference(BUILDINGS_UNION)
            if not halo_geom.is_valid:
                halo_geom = make_valid(halo_geom)
        except Exception:
            halo_geom = None


        style = styles[level]
        for poly in _iter_polys(smoothed):
            if poly.area < MIN_AREA_DEG2:
                continue

            # One label per polygon is wrong for these shapes: contours follow
            # river corridors and can run kilometres across several places. Each
            # polygon is split against the place partition so every piece is
            # named after the place it actually covers.
            for piece, region in _split_by_place(poly):
                lo, hi = bands[level]
                val = _band_value_in(piece, smooth_grid, lats, lons, lo, hi)
                if value_is_probability:
                    value_label = f"{100.0 * val:.0f}% probability"
                else:
                    value_label = f"{val:.2f} m"

                features.append({
                    "type": "Feature",
                    "geometry": mapping(piece),
                    "properties": {
                        "level": level,
                        "fillColor": style["fillColor"],
                        "lineColor": style["lineColor"],
                        "name": style["name"],
                        "region": region,
                        "value_label": value_label,
                        "value": float(val),
                    },
                })

        for poly in _iter_polys(halo_geom):
            if poly.area < MIN_AREA_DEG2:
                continue
            halo_features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"level": level, "fillColor": halo_styles[level]},
            })

    return (
        {"type": "FeatureCollection", "features": features},
        {"type": "FeatureCollection", "features": halo_features},
    )

_BUILDING_ROWS: dict[str, list[dict]] = {}


def _building_rows(theme: str) -> list[dict]:
    """
    Extrusion rows for the 3D buildings, built once per theme.

    The footprints never change, but they were re-read from disk and re-coloured
    on every map update, which was over half of the server time for a flood
    update (~1 s of ~2 s).
    """
    if theme in _BUILDING_ROWS:
        return _BUILDING_ROWS[theme]
    tall, mid, low = MAP_THEMES.get(theme, MAP_THEMES["dark"])["buildings"]
    rows: list[dict] = []
    if BUILDINGS_JSON.exists():
        try:
            with open(BUILDINGS_JSON, "r") as f:
                geo_data = json.load(f)
            # The 4,000 tallest and largest, which carry the skyline; the file's
            # first 4,000 were small structures scattered over the county.
            feats = sorted(geo_data.get("features", []),
                           key=lambda f: (f["properties"].get("height", 0), f["properties"].get("area_m2", 0)),
                           reverse=True)
            for feat in feats[:4000]:
                h_val = feat["properties"].get("height", 14.0)
                rows.append({
                    "polygon": list(_footprint_polygon(feat).exterior.coords),
                    "height": h_val,
                    "color": tall if h_val > 25 else (mid if h_val > 12 else low),
                })
        except Exception as e:                                  # noqa: BLE001
            logger.warning(f"Error loading building footprints: {e}")
    _BUILDING_ROWS[theme] = rows
    return rows


#: Vector tiles with every OpenStreetMap building and its height
#: (OpenMapTiles schema: `render_height` from the `height` tag, or the number of
#: floors). Free, no key; the TileJSON names the current tile set.
CITY_TILES = "https://tiles.openfreemap.org/planet"

#: Named buildings for the hover labels (src.ingestion.fetch_building_names),
#: served by the app at this path (src/dashboard/app.py).
BUILDING_NAMES_URL = "/data/building-names.json"
#: The offline fallback's building footprints, fetched only when the tiles fail.
FALLBACK_BUILDINGS_URL = "/data/buildings-fallback.json"

#: How the city is drawn and lit in each theme: one building colour, the
#: surface material, the lights, and the windows. Day is a warm afternoon sun
#: with glass windows; night is a dim blue sky with lit windows, some warm and
#: some cool, so the towers read as an occupied skyline.
#:
#: Windows (drawn by a shader on every wall, see CITY_JS): one per `bay` metres
#: along the wall and per `floor` metres up it, on buildings at least `min_height`
#: tall; at night `lit` is the share of windows with the lights on.
CITY_STYLE = {
    "dark": {
        "color": [50, 58, 78],
        "material": {"ambient": 0.5, "diffuse": 0.65, "shininess": 64, "specularColor": [120, 150, 210]},
        "ambient": {"color": [140, 160, 215], "intensity": 1.15},
        "lights": [{"color": [255, 214, 170], "intensity": 0.85, "direction": [-1, -3, -2.5]},
                   {"color": [90, 130, 255], "intensity": 0.55, "direction": [2, 1.5, -1]}],
        "sun": False,
        # Night: a deep blue sky, so the water reads as water, with the warm
        # glints of street and window light.
        "water": {"horizon": [60, 105, 165], "zenith": [16, 38, 86], "glint": [255, 208, 150],
                  "glint_strength": 1.1, "light_height": 0.22, "ripple": 2.2},
        "windows": {"night": True, "floor": 3.4, "bay": 3.2, "min_height": 7.0, "lit": 0.42,
                    "glass": [16, 21, 32], "warm": [255, 200, 120], "cool": [205, 222, 255]},
    },
    "light": {
        "color": [234, 231, 225],
        "material": {"ambient": 0.6, "diffuse": 0.55, "shininess": 40, "specularColor": [60, 64, 72]},
        "ambient": {"color": [255, 255, 255], "intensity": 1.3},
        "lights": [{"color": [200, 215, 255], "intensity": 0.45, "direction": [2, 2, -1]}],
        # A mid-afternoon sun over Nairobi (15:30 EAT), so walls facing it are
        # lit and the others fall into shade.
        "sun": {"timestamp": 1761913800000, "color": [255, 246, 228], "intensity": 1.15},
        # Day: a blue sky (the horizon kept well short of white, or the water
        # turns milky), and sun glints.
        "water": {"horizon": [140, 190, 235], "zenith": [60, 130, 210], "glint": [255, 250, 236],
                  "glint_strength": 1.3, "light_height": 0.35, "ripple": 2.2},
        "windows": {"night": False, "floor": 3.4, "bay": 3.2, "min_height": 7.0, "lit": 0.0,
                    "glass": [118, 138, 164], "warm": [150, 172, 200], "cool": [96, 114, 140]},
    },
}

#: A transparent 1x1 image: the satellite layer's picture while it is hidden.
BLANK_PNG = ("data:image/png;base64,"
             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

#: Layer ids the live-update bridge can patch. Every one is created in every
#: document, empty if need be: a layer that does not exist cannot be updated by
#: id, so a route requested after the map loaded would have nowhere to go.
ROUTE_LAYER_IDS = ("route-fastest", "route-casing", "route-safe", "route-endpoints")


#: Operational map layers (src.geo.layers), patched by the "Map layers" checklist.
OPS_LAYER_IDS = ("admin-boundaries", "rivers", "water-bodies", "infrastructure", "stations", "alert-areas",
                 "label-subcounties", "label-rivers", "label-facilities")


def create_3d_digital_twin_deck(
    depth_grid: np.ndarray | None = None,
    value_is_probability: bool = False,
    center_lat: float = -1.2787,
    center_lon: float = 36.8213,
    zoom: float = 13.0,
    pitch: float = 45.0,
    bearing: float = -15.0,
    display_mode: str = "PROBABILITY",
    highlight_region: str | None = None,
    highlight_coords: tuple[float, float] | None = None,
    theme: str = "dark",
    route_layers: dict | None = None,
    report_rows: list[dict] | None = None,
    satellite: dict | None = None,
    history_rows: list[dict] | None = None,
    ops_rows: dict | None = None,
) -> pdk.Deck:
    """
    Construct Pydeck 3D Viewport with street-accurate flood overlay and crisp 3D buildings.

    `route_layers` maps each id in ROUTE_LAYER_IDS to its data rows, so a
    rebuilt document (theme or camera change) keeps an active route on screen.
    `report_rows` are the community flood report pins
    (src.dashboard.reports_ui.map_rows) and `satellite` the satellite layer's
    state ({"visible", "image"}), kept across a rebuild in the same way, as are
    `history_rows`, the past-floods areas (src.reports.history.map_rows), and
    `ops_rows`, the operational layers by id (OPS_LAYER_IDS, src.geo.layers).
    """
    ops_rows = ops_rows or {}
    palette = MAP_THEMES.get(theme, MAP_THEMES["dark"])
    route_layers = route_layers or {}
    layers = []

    if depth_grid is None:
        depth_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    h, w = depth_grid.shape
    lats = np.linspace(LAT_NORTH, LAT_SOUTH, h)
    lons = np.linspace(LON_WEST, LON_EAST, w)

    # 1. Street-accurate Vector GeoJSON Flood Polygon Layer (renders ground water on streets)
    flood_geojson, halo_geojson = generate_flood_contour_geojson(
        depth_grid=depth_grid,
        value_is_probability=value_is_probability,
        lats=lats,
        lons=lons,
        display_mode=display_mode,
        theme=theme,
    )

    # 1a. Faded halo underneath - feathers the water edge into the basemap
    # instead of stopping at a hard outline (see generate_flood_contour_geojson).
    # Created unconditionally, like every patchable layer.
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        halo_geojson,
        id="flood-halo",
        opacity=1.0,
        stroked=False,
        filled=True,
        extruded=False,
        get_fill_color="properties.fillColor",
        pickable=False,
        material=False,
    ))

    # 1b. The readable flood fill itself - translucent so streets and terrain
    # stay visible underneath.
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        flood_geojson,
        id="flood-main",
        opacity=1.0,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="properties.fillColor",
        get_line_color="properties.lineColor",
        get_line_width=1.5,
        line_width_min_pixels=1.5,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 60],
        material=False,
    ))

    no_depth = {"depthCompare": "always", "depthWriteEnabled": False}

    # 1c. Past floods reported in the news (src.reports.history), drawn over the
    # water but under the buildings. Patched in place by the "Past floods" switch.
    layers.append(pdk.Layer(
        "PolygonLayer", history_rows or [], id="past-floods",
        get_polygon="polygon", get_fill_color="fill", get_line_color="edge",
        get_line_width="width", line_width_units="'pixels'", line_width_min_pixels=1,
        stroked=True, filled=True, extruded=False,
        pickable=True, auto_highlight=True, highlight_color=[255, 255, 255, 50],
        parameters=no_depth,
    ))

    # 1d. Operational layers (src.geo.layers), patched in place by the "Map
    # layers" checklist: area outlines and alert areas on the ground, rivers and
    # water over the flood fill, facilities and stations as pins (drawn after the
    # buildings, below).
    layers.append(pdk.Layer(
        "PolygonLayer", ops_rows.get("alert-areas", []), id="alert-areas",
        get_polygon="polygon", get_fill_color="fill", get_line_color="edge", line_width_min_pixels=2,
        stroked=True, filled=True, extruded=False, pickable=True, auto_highlight=True,
        highlight_color=[255, 255, 255, 40], parameters=no_depth,
    ))
    layers.append(pdk.Layer(
        "PolygonLayer", ops_rows.get("water-bodies", []), id="water-bodies",
        get_polygon="polygon", get_fill_color="fill", get_line_color="edge", line_width_min_pixels=1,
        stroked=True, filled=True, extruded=False, pickable=True, parameters=no_depth,
    ))
    layers.append(pdk.Layer(
        "PathLayer", ops_rows.get("rivers", []), id="rivers",
        get_path="path", get_color="color", get_width="width", width_units="'meters'",
        width_min_pixels=1, width_max_pixels=6, cap_rounded=True, joint_rounded=True, pickable=True,
        auto_highlight=True, highlight_color=[255, 255, 255, 120], parameters=no_depth,
    ))
    layers.append(pdk.Layer(
        "PathLayer", ops_rows.get("admin-boundaries", []), id="admin-boundaries",
        get_path="path", get_color="color", get_width="width", width_units="'meters'",
        width_min_pixels=1, width_max_pixels=6, pickable=True, auto_highlight=True,
        highlight_color=[129, 140, 248, 200],
        parameters=no_depth,
    ))

    # 2. LOD2 Solid 3D Volumetric Building Footprint Extrusions
    # The offline fallback's footprints (~1 MB) are not embedded: the layer starts
    # empty and CITY_JS fetches FALLBACK_BUILDINGS_URL only if the tiles fail.
    if BUILDINGS_JSON.exists():
        layers.append(pdk.Layer(
            "PolygonLayer",
            [],
            id="buildings",
            get_polygon="polygon",
            get_elevation="height",
            get_fill_color="color",
            # Wireframe edges default to black, which is invisible on the dark
            # basemap but turns every building into a black speck on the light one.
            get_line_color="color",
            extruded=True,
            wireframe=True,
            pickable=False,
            opacity=0.85,
            # The offline fallback: hidden while the full city streams from
            # CITY_TILES, shown if those tiles cannot be reached.
            visible=False,
        ))

    # 3. Routes. Drawn last with depth testing off, so a route is never hidden
    # behind an extruded building when the camera is pitched.
    #
    # Widths are in metres, clamped to a pixel range, NOT in pixels. With
    # width_units="pixels", deck.gl 9.2 drops the whole path once the camera
    # zooms out past ~12 (it switches projection mode there): the route
    # vanished on zoom-out while the pins beside it stayed. Measured: a pixel
    # path visible at zoom 12.5 is gone at 11.9; the same path in metres with
    # a pixel minimum stays drawn at every zoom.
    # pydeck turns plain strings into JavaScript expressions ("meters" becomes an
    # undefined variable), so literal string props are written quoted: "'meters'".
    path_common = dict(get_path="path", get_color="color", get_width="width",
                       width_units="'meters'", cap_rounded=True, joint_rounded=True,
                       pickable=False, parameters=no_depth)
    layers.append(pdk.Layer("PathLayer", route_layers.get("route-fastest", []),
                            id="route-fastest", width_min_pixels=4, width_max_pixels=7, **path_common))
    layers.append(pdk.Layer("PathLayer", route_layers.get("route-casing", []),
                            id="route-casing", width_min_pixels=9, width_max_pixels=13, **path_common))
    layers.append(pdk.Layer("PathLayer", route_layers.get("route-safe", []),
                            id="route-safe", width_min_pixels=5, width_max_pixels=8, **path_common))
    layers.append(pdk.Layer(
        "ScatterplotLayer", route_layers.get("route-endpoints", []),
        id="route-endpoints", get_position="position", get_fill_color="color",
        get_line_color=palette["pin_stroke"], stroked=True, line_width_min_pixels=3,
        get_radius=9, radius_units="'pixels'", pickable=False, parameters=no_depth,
    ))

    # 4. Community flood reports: a soft ring on the ground and a pin above it,
    # always on top. Patched in place every refresh (layer ids in REPORT_LAYER_IDS).
    report_rows = report_rows or []
    layers.append(pdk.Layer(
        "ScatterplotLayer", report_rows, id="flood-reports-halo",
        get_position="position", get_fill_color="color", get_line_color="line",
        get_radius=45, radius_units="'meters'", radius_min_pixels=10,
        stroked=True, line_width_min_pixels=2, pickable=False, parameters=no_depth,
    ))
    layers.append(pdk.Layer(
        "ScatterplotLayer", report_rows, id="flood-reports",
        get_position="position", get_fill_color="pin", get_line_color=[255, 255, 255, 255],
        get_radius=9, radius_units="'pixels'", stroked=True, line_width_min_pixels=3,
        pickable=True, auto_highlight=True, parameters=no_depth,
    ))

    for lid, radius in (("infrastructure", "radius"), ("stations", 9)):
        layers.append(pdk.Layer(
            "ScatterplotLayer", ops_rows.get(lid, []), id=lid,
            get_position="position", get_fill_color="color", get_line_color="line",
            get_radius=radius, radius_units="'pixels'", stroked=True, line_width_min_pixels=2,
            pickable=True, auto_highlight=True, parameters=no_depth,
        ))

    # Labels (src.geo.layers *_label_rows), drawn last so text sits on top. Sizes in
    # metres with a pixel cap, so sub-county names stay readable at city scale.
    # Facility names are switched on only from FACILITY_LABEL_ZOOM (updater script),
    # where they are large enough to read. Colours follow the theme: pale text with
    # a dark halo on the dark map, slate text with a white halo on the light one.
    light = theme == "light"
    halo = [255, 255, 255, 235] if light else [10, 14, 24, 235]
    label_common = dict(get_position="position", get_text="text",
                        font_family="'Inter, Segoe UI, sans-serif'", character_set="'auto'",
                        font_settings={"sdf": True, "fontSize": 64, "buffer": 8},
                        outline_width=3, outline_color=halo, billboard=True,
                        pickable=False, parameters=no_depth)
    layers.append(pdk.Layer("TextLayer", ops_rows.get("label-subcounties", []), id="label-subcounties",
                            get_size=320, size_units="'meters'", size_min_pixels=11, size_max_pixels=17,
                            get_color=[51, 65, 85, 235] if light else "color", font_weight=700, **label_common))
    layers.append(pdk.Layer("TextLayer", ops_rows.get("label-rivers", []), id="label-rivers",
                            get_size=260, size_units="'meters'", size_min_pixels=10, size_max_pixels=14,
                            get_color=[3, 105, 161, 255] if light else "color", get_angle="angle",
                            font_weight=600, **label_common))
    layers.append(pdk.Layer("TextLayer", ops_rows.get("label-facilities", []), id="label-facilities",
                            get_size=30, size_units="'meters'", size_min_pixels=9, size_max_pixels=13,
                            get_color=[30, 41, 59, 255] if light else "color", visible=False,
                            get_text_anchor="'start'", get_alignment_baseline="'center'",
                            get_pixel_offset=[10, 0], font_weight=500, **label_common))

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )

    # minZoom/maxZoom are Controller options, not top-level View props - deck.gl
    # silently ignores them if passed as `pdk.View(min_zoom=...)` instead of
    # inside `controller`, which is why the camera could still be pulled back
    # past Nairobi County into a regional view. This keeps it to Nairobi.
    nairobi_view = pdk.View(
        type="MapView",
        controller={"minZoom": 10.3, "maxZoom": 20},
    )

    tooltip_style = {
        **palette["tooltip"],
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "fontSize": "12.5px",
        "borderRadius": "10px",
        "padding": "8px 12px",
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[nairobi_view],
        map_provider="carto",
        map_style=palette["basemap"],
        tooltip={
            "html": ("<div style='font-weight:600;font-size:13px;margin-bottom:3px'>{region}</div>"
                     "<div style='opacity:0.85'>{name}</div>"
                     "<div style='opacity:0.7;font-size:11.5px;margin-top:2px'>{value_label}</div>"),
            "style": tooltip_style,
        },
    )
    deck._tooltip_style = tooltip_style
    # The side panel summarises the polygons that were actually drawn, so they
    # travel with the deck rather than being recomputed and risking divergence
    # between what the map shows and what the panel reports.
    deck._flood_features = flood_geojson.get("features", [])
    deck._halo_features = halo_geojson.get("features", [])
    deck._theme = theme
    # The satellite layer is created in the browser (CITY_JS): pydeck would read
    # its image address as an expression.
    deck._satellite = satellite or {}
    deck._history_rows = history_rows or []
    return deck


def get_deck_html_with_embedded_legend(deck: pdk.Deck) -> str:
    """
    Generate Pydeck HTML string with an embedded floating Legend Pill.
    """
    base_html = deck.to_html(as_string=True)
    lg = MAP_THEMES.get(getattr(deck, "_theme", "dark"), MAP_THEMES["dark"])["legend"]
    history_rows = getattr(deck, "_history_rows", [])
    history_near = any(r.get("near") for r in history_rows)

    # Matches the token system in src/dashboard/assets/custom.css. This HTML is
    # rendered inside the map's own iframe and cannot reach that stylesheet, so
    # the theme's colours are written in directly.
    legend_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        #pydeck-embedded-legend .sw {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
        #pydeck-embedded-legend .wt, #past-floods-legend .wt {{ width:14px; height:10px; border-radius:3px; display:inline-block;
            background:{lg['water']}; border:2px solid; box-sizing:border-box; }}
        #pydeck-embedded-legend .it, #past-floods-legend .it {{ display:flex; align-items:center; gap:6px; }}
        #pydeck-embedded-legend .mu, #past-floods-legend .mu {{ color:{lg['muted']}; }}
        @media (max-width: 640px) {{ #pydeck-embedded-legend .wide {{ display:none; }} }}
    </style>
    <div id="pydeck-embedded-legend" style="
        position: fixed;
        bottom: 20px;
        left: calc(50% + (var(--inset-left, 0px) - var(--inset-right, 0px)) / 2);
        transform: translateX(-50%);
        transition: left 0.25s ease;
        background-color: {lg['bg']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {lg['border']};
        border-radius: 999px;
        padding: 9px 18px;
        z-index: 99999;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: {lg['text']};
        font-size: 12px;
        font-weight: 500;
        box-shadow: {lg['shadow']};
        pointer-events: none;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span class="mu" style="font-weight: 600;">Likely flooding</span>
        <span class="it"><span class="wt" style="border-color:#f0b93f; opacity:0.7;"></span>Moderate <span class="mu">25&ndash;55%</span></span>
        <span class="it"><span class="wt" style="border-color:#f08a3c; opacity:0.85;"></span>High <span class="mu">55&ndash;82%</span></span>
        <span class="it"><span class="wt" style="border-color:#ef4459;"></span>Critical <span class="mu">&gt;82%</span></span>
        <span class="mu wide" style="font-size:11px;">Probability of flooding, not depth</span>
        <span class="it wide mu"><span class="sw" style="background:{lg['bldg']}; border-radius:2px;"></span>3D buildings</span>
    </div>
    <div id="past-floods-legend" style="
        position: fixed; bottom: 64px; left: calc(var(--inset-left, 0px) + 16px); transition: left 0.25s ease;
        background-color: {lg['bg']}; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid {lg['border']}; border-radius: 999px; padding: 7px 14px; z-index: 99999;
        font-family: 'Inter', 'Segoe UI', sans-serif; color: {lg['text']}; font-size: 12px; font-weight: 500;
        box-shadow: {lg['shadow']}; pointer-events: none; white-space: nowrap;
        display: {'flex' if history_rows else 'none'}; align-items: center; gap: 12px;">
        <span class="mu" style="font-weight: 600;">Past floods in the news</span>
        <span class="it"><span class="wt" style="background:rgba(168,85,247,0.45); border-color:#c084fc;"></span>Flooded before <span class="mu">darker = more often</span></span>
        <span class="it" id="past-floods-legend-near" style="display:{'flex' if history_near else 'none'};"><span class="wt" style="background:rgba(250,204,21,0.35); border-color:#facc15;"></span>Reported this storm's week</span>
    </div>
    """

    # Live-update bridge. Without this the whole ~4.6 MB document is rebuilt and
    # the iframe reloaded on every change, which flickers and discards the
    # viewer's zoom and pan. The parent page posts only the data for the layers
    # that changed, and this patches them in place.
    #
    # pydeck's template creates `deckInstance` in a script placed after </body>,
    # which runs after this one, so the instance is looked up lazily.
    updater_js = """
<script>
(function () {
  function findDeck() {
    try { if (typeof deckInstance !== "undefined" && deckInstance) return deckInstance; } catch (e) {}
    return null;
  }

  function post(msg) { try { window.parent.postMessage(msg, "*"); } catch (e) {} }

  function patchLayers(d, updates) {
    var layers = d.props.layers.map(function (layer) {
      return Object.prototype.hasOwnProperty.call(updates, layer.id)
        ? layer.clone({ data: updates[layer.id] }) : layer;
    });
    // setProps leaves viewState untouched, so zoom and pan survive the update.
    d.setProps({ layers: layers });
  }

  function fitTo(d, b) {
    // b = [west, south, east, north]. Web Mercator is effectively linear this
    // close to the equator, so zoom follows directly from the span.
    var w = window.innerWidth || 800, h = window.innerHeight || 600;
    var lonSpan = Math.max(b[2] - b[0], 0.004), latSpan = Math.max(b[3] - b[1], 0.004);
    var zoom = Math.min(Math.log2(w * 360 / (512 * lonSpan)), Math.log2(h * 360 / (512 * latSpan))) - 0.5;
    d.setProps({ initialViewState: {
      longitude: (b[0] + b[2]) / 2, latitude: (b[1] + b[3]) / 2,
      zoom: Math.max(10.3, Math.min(zoom, 16.5)), pitch: 30, bearing: 0,
      transitionDuration: 900
    }});
  }

  window.addEventListener("message", function (ev) {
    var msg = ev.data;
    // The floating panels cover the map's edges; keep the legends in the clear part.
    if (msg && msg.type === "insets") {
      var r = document.documentElement.style;
      r.setProperty("--inset-left", (msg.left || 0) + "px");
      r.setProperty("--inset-right", (msg.right || 0) + "px");
      return;
    }
    if (!msg || (msg.type !== "floodUpdate" && msg.type !== "layerUpdate")) return;
    var d = findDeck();
    if (!d || !d.props || !d.props.layers) {
      post({ type: "layerUpdateAck", ok: false, error: "no deck" });
      return;
    }
    try {
      var updates = msg.layers || {};
      if (msg.type === "floodUpdate") {
        if (msg.main) updates["flood-main"] = msg.main;
        if (msg.halo) updates["flood-halo"] = msg.halo;
      }
      patchLayers(d, updates);
      if (updates["past-floods"]) {
        var rows = updates["past-floods"], lg = document.getElementById("past-floods-legend");
        if (lg) lg.style.display = rows.length ? "flex" : "none";
        var nr = document.getElementById("past-floods-legend-near");
        if (nr) nr.style.display = rows.some(function (r) { return r.near; }) ? "flex" : "none";
      }
      if (msg.fit) fitTo(d, msg.fit);
      post({ type: "layerUpdateAck", ok: true });
    } catch (e) {
      // Reported so the parent can force a full document rebuild rather than
      // silently leaving stale geometry on screen.
      post({ type: "layerUpdateAck", ok: false, error: String(e) });
    }
  });

  // Facility names only from zoom 15, where they are readable; below that they
  // would be specks. Checked a few times a second, re-cloned only on a change.
  var FACILITY_LABEL_ZOOM = 15;
  setInterval(function () {
    var d = findDeck();
    if (!d || !d.props || !d.props.layers) return;
    var vp = d.getViewports && d.getViewports()[0];
    if (!vp) return;
    var want = vp.zoom >= FACILITY_LABEL_ZOOM;
    var l = d.props.layers.find(function (x) { return x.id === "label-facilities"; });
    if (l && l.props.visible !== want) {
      d.setProps({ layers: d.props.layers.map(function (x) {
        return x.id === "label-facilities" ? x.clone({ visible: want }) : x; }) });
    }
  }, 300);

  // Map clicks go to the parent page, which uses them to drop a start or
  // destination pin when the user has asked to pick a point on the map.
  var tries = 0;
  (function attach() {
    var d = findDeck();
    if (!d) { if (tries++ < 100) setTimeout(attach, 100); return; }
    d.setProps({ onClick: function (info) {
      if (info && info.coordinate) post({ type: "mapClick", lon: info.coordinate[0], lat: info.coordinate[1] });
    }});
    post({ type: "floodMapReady" });
  })();
})();
</script>
"""

    city_js = CITY_JS.replace("__CITY__", json.dumps({
        "tiles": CITY_TILES, "names": BUILDING_NAMES_URL,
        "fallback": FALLBACK_BUILDINGS_URL + "?theme=" + getattr(deck, "_theme", "dark"),
        "tooltip": getattr(deck, "_tooltip_style", {}),
        "satellite": {**getattr(deck, "_satellite", {}), "blank": BLANK_PNG,
                      "bounds": [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH]},
        **CITY_STYLE.get(getattr(deck, "_theme", "dark"), CITY_STYLE["dark"])}))
    extra = legend_html + updater_js + city_js
    if "</body>" in base_html:
        return base_html.replace("</body>", f"{extra}\n</body>")
    return base_html + extra


#: Draws the city in the map document. The buildings come from CITY_TILES at
#: their real heights and the scene is lit for the theme (CITY_STYLE). If the
#: tiles cannot be reached, the local footprints (layer "buildings") are shown.
CITY_JS = r"""
<script>
(function () {
  var CITY = __CITY__;
  function post(msg) { try { window.parent.postMessage(msg, "*"); } catch (e) {} }

  function findDeck() {
    try { if (typeof deckInstance !== "undefined" && deckInstance) return deckInstance; } catch (e) {}
    return null;
  }

  function glslVec3(c) {
    return "vec3(" + c.map(function (v) { return (v / 255).toFixed(4); }).join(", ") + ")";
  }
  function glslFloat(v) { return Number(v).toFixed(4); }

  // Windows, drawn on the walls in the shader rather than modelled: a grid of
  // one window per bay along each wall and per floor up it, measured in metres.
  // The wall shader knows where along the wall edge and how far up it a point
  // is, and the building's height. By day the windows are glass; at night a
  // share of them, picked at random per window, are lit warm or cool.
  // Far away, where a window would be smaller than a pixel, the grid fades to
  // its average colour instead of shimmering.
  function windowsExtension(D) {
    var w = CITY.windows;
    if (!w || !D.LayerExtension) return null;
    var fs = [
      "if (vTwinWall > 0.5 && vTwinHeight >= " + glslFloat(w.min_height) + ") {",
      "  vec2 cell = vec2(vTwinWallPos.x / " + glslFloat(w.bay) + ", vTwinWallPos.y / " + glslFloat(w.floor) + ");",
      "  vec2 f = fract(cell);",
      "  vec2 aa = max(fwidth(cell), vec2(1e-4));",
      "  float inX = smoothstep(0.2 - aa.x, 0.2 + aa.x, f.x) * (1.0 - smoothstep(0.8 - aa.x, 0.8 + aa.x, f.x));",
      "  float inY = smoothstep(0.3 - aa.y, 0.3 + aa.y, f.y) * (1.0 - smoothstep(0.82 - aa.y, 0.82 + aa.y, f.y));",
      // no windows on the ground floor or in the parapet at the top
      "  float floors = vTwinHeight / " + glslFloat(w.floor) + ";",
      "  float band = step(1.0, cell.y) * step(cell.y, floors - 0.5);",
      "  float win = inX * inY * band;",
      "  vec2 id = floor(cell);",
      "  float r = fract(sin(dot(id, vec2(12.9898, 78.233)) + vTwinHeight * 1.618) * 43758.5453);",
      "  float r2 = fract(r * 17.31);",
      "  vec3 glass = " + glslVec3(w.glass) + ";",
      w.night
        ? "  float lit = step(1.0 - " + glslFloat(w.lit) + ", r);\n" +
          "  vec3 light = mix(" + glslVec3(w.warm) + ", " + glslVec3(w.cool) + ", step(0.7, r2)) * (0.7 + 0.45 * r2);\n" +
          "  vec3 detail = mix(glass, light, lit);\n" +
          "  vec3 average = mix(glass, " + glslVec3(w.warm) + " * 0.85, " + glslFloat(w.lit) + ");"
        : "  vec3 detail = mix(" + glslVec3(w.cool) + ", " + glslVec3(w.warm) + ", r2) * 0.9 + glass * 0.1;\n" +
          "  vec3 average = mix(" + glslVec3(w.cool) + ", " + glslVec3(w.warm) + ", 0.5);",
      "  float sharp = clamp(1.6 - 2.0 * max(aa.x, aa.y), 0.0, 1.0);",
      "  vec3 winColor = mix(average, detail, sharp);",
      "  float coverage = mix(0.6 * 0.52 * band, win, sharp);",
      "  color.rgb = mix(color.rgb, winColor, coverage);",
      "}"
    ].join("\n");
    class WindowsExtension extends D.LayerExtension {
      getShaders() { return windowShaders; }
    }
    WindowsExtension.extensionName = "TwinWindowsExtension";
    var windowShaders = { inject: {
        "vs:#decl": "out vec2 vTwinWallPos;\nout float vTwinHeight;\nout float vTwinWall;\n",
        // Only the wall shader defines IS_SIDE_VERTEX; roofs get no windows.
        "vs:#main-end": [
          "#ifdef IS_SIDE_VERTEX",
          "vec3 twinA = project_position(pos, pos64Low);",
          "vec3 twinB = project_position(nextPos, nextPos64Low);",
          "float twinLen = length((twinB - twinA).xy) / project.commonUnitsPerMeter.x;",
          "vTwinWallPos = vec2(positions.x * twinLen, positions.y * elevations);",
          "vTwinHeight = elevations;",
          "vTwinWall = 1.0;",
          "#else",
          "vTwinWallPos = vec2(0.0);",
          "vTwinHeight = 0.0;",
          "vTwinWall = 0.0;",
          "#endif"
        ].join("\n"),
        "fs:#decl": "in vec2 vTwinWallPos;\nin float vTwinHeight;\nin float vTwinWall;\n",
        "fs:DECKGL_FILTER_COLOR": fs
      }};
    return new WindowsExtension();
  }

  function lighting(D) {
    var lights = { ambient: new D.AmbientLight(CITY.ambient) };
    CITY.lights.forEach(function (l, i) { lights["light" + i] = new D.DirectionalLight(l); });
    if (CITY.sun) { lights.sun = new D._SunLight(CITY.sun); }
    return new D.LightingEffect(lights);
  }

  // ---- water ---------------------------------------------------------------
  // The flood polygons drawn as standing water. The surface is a few octaves of
  // noise a couple of metres across, drifting slowly downstream (east, the way
  // Nairobi's rivers run) and churning as it goes; its slope tilts the surface
  // normal. The normal then decides what the water shows: the sky mirrored by
  // a reflection that strengthens at low viewing angles (Schlick's Fresnel),
  // the murky body of the water underneath, and sharp glints where a ripple
  // catches the light. Where ripples would be smaller than a pixel they settle
  // to a calm surface instead of shimmering. The fill's alpha is kept, so
  // likelier flooding stays more opaque. This is the look of water only: the
  // model predicts where flooding is likely, not depth or flow.
  function waterExtension(D) {
    var w = CITY.water;
    if (!w || !D.LayerExtension) return null;
    var block = "uniform twinWaterUniforms {\n  float time;\n} twinWater;\n";
    var module = { name: "twinWater", vs: block, fs: block, uniformTypes: { time: "f32" } };
    var decl = [
      "in vec2 vTwinWaterXY;",
      "in vec3 vTwinWaterView;",
      // hash without sin(), so it stays stable for coordinates thousands of metres out
      "float twinHash(vec2 p) {",
      "  vec3 p3 = fract(vec3(p.xyx) * 0.1031);",
      "  p3 += dot(p3, p3.yzx + 33.33);",
      "  return fract((p3.x + p3.y) * p3.z);",
      "}",
      "float twinNoise(vec2 p) {",
      "  vec2 i = floor(p), f = fract(p);",
      "  vec2 u = f * f * (3.0 - 2.0 * f);",
      "  return mix(mix(twinHash(i), twinHash(i + vec2(1.0, 0.0)), u.x),",
      "             mix(twinHash(i + vec2(0.0, 1.0)), twinHash(i + vec2(1.0, 1.0)), u.x), u.y);",
      "}",
      "float twinWaves(vec2 p, float t) {",
      "  float h = 0.0, a = 0.5;",
      "  vec2 drift = vec2(0.55, 0.12) * t;",
      "  for (int i = 0; i < 5; i++) {",
      "    h += a * twinNoise(p + drift);",
      "    p = mat2(1.6, 1.2, -1.2, 1.6) * p;",
      "    drift = mat2(0.8, -0.6, 0.6, 0.8) * drift * 1.4;",
      "    a *= 0.5;",
      "  }",
      "  return h;",
      "}",
      ""
    ].join("\n");
    var fs = [
      "{",
      "  float t = twinWater.time;",
      "  vec2 p = vTwinWaterXY / " + Number(w.ripple).toFixed(2) + ";",
      "  float e = 0.06;",
      "  float h0 = twinWaves(p, t);",
      "  vec2 slope = vec2(twinWaves(p + vec2(e, 0.0), t) - h0, twinWaves(p + vec2(0.0, e), t) - h0) / e;",
      // calm the surface where a ripple would be smaller than a pixel
      "  float calm = clamp(1.4 - 1.6 * length(fwidth(p)), 0.0, 1.0);",
      "  vec3 n = normalize(vec3(-slope * 0.32 * calm, 1.0));",
      "  vec3 v = normalize(vTwinWaterView);",
      "  float ndv = clamp(dot(n, v), 0.0, 1.0);",
      "  float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);",
      "  vec3 r = reflect(-v, n);",
      "  vec3 sky = mix(" + glslVec3(w.horizon) + ", " + glslVec3(w.zenith) + ", pow(clamp(r.z, 0.0, 1.0), 0.6));",
      "  vec3 body = color.rgb * (0.85 + 0.3 * h0);",
      // The light sits low ahead of the viewer, across the water, so its glints
      // show whichever way the map is turned.
      "  vec3 l = normalize(vec3(-v.xy, " + Number(w.light_height).toFixed(3) + " * length(v.xy) + 0.001));",
      "  float rl = clamp(dot(r, l), 0.0, 1.0);",
      "  float glint = (pow(rl, 260.0) * 2.6 + pow(rl, 30.0) * 0.32 + pow(rl, 6.0) * 0.08) * calm * " + Number(w.glint_strength).toFixed(3) + ";",
      "  vec3 water = mix(body, sky, clamp(0.32 + fresnel * 1.2, 0.0, 0.94)) + " + glslVec3(w.glint) + " * glint;",
      "  color = vec4(water, min(1.0, color.a + glint * 0.6));",
      "}"
    ].join("\n");
    class WaterExtension extends D.LayerExtension {
      getShaders() {
        return { modules: [module], inject: {
          "vs:#decl": "out vec2 vTwinWaterXY;\nout vec3 vTwinWaterView;\n",
          // Metres east and north of the city centre (float precision is kept by
          // measuring from nearby rather than from 0,0).
          "vs:#main-end": "vTwinWaterXY = (geometry.worldPosition.xy - vec2(36.82, -1.29)) * vec2(111292.0, 110574.0);\n" +
                          "vTwinWaterView = project.cameraPosition - geometry.position.xyz;\n",
          "fs:#decl": decl,
          "fs:DECKGL_FILTER_COLOR": fs
        }};
      }
      draw() { this.setShaderModuleProps({ twinWater: { time: performance.now() / 1000 } }); }
    }
    WaterExtension.extensionName = "TwinWaterExtension";
    return new WaterExtension();
  }

  // ---- hover labels -------------------------------------------------------
  // Named buildings, indexed on a grid of ~200 m cells so a hover checks only
  // the few footprints nearby. The tiles carry no names, so the name is the
  // OpenStreetMap building whose footprint holds (or nearly touches) the point.
  var CELL = 0.002, names = null;
  function cellKey(lon, lat) { return Math.floor(lon / CELL) + ":" + Math.floor(lat / CELL); }
  function loadNames() {
    fetch(CITY.names).then(function (r) { return r.json(); }).then(function (data) {
      var index = {};
      data.buildings.forEach(function (b) {
        var lon0 = Infinity, lat0 = Infinity, lon1 = -Infinity, lat1 = -Infinity;
        b.ring.forEach(function (p) {
          lon0 = Math.min(lon0, p[0]); lon1 = Math.max(lon1, p[0]);
          lat0 = Math.min(lat0, p[1]); lat1 = Math.max(lat1, p[1]);
        });
        for (var x = Math.floor(lon0 / CELL); x <= Math.floor(lon1 / CELL); x++) {
          for (var y = Math.floor(lat0 / CELL); y <= Math.floor(lat1 / CELL); y++) {
            (index[x + ":" + y] = index[x + ":" + y] || []).push(b);
          }
        }
      });
      names = index;
    }).catch(function (e) { console.warn("Building names unavailable:", e); });
  }
  function inside(ring, lon, lat) {
    var hit = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      var xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }
  // Metres from the point to the footprint's outline (the pointer can rest on a
  // wall, just outside the footprint drawn on the ground).
  function edgeMetres(ring, lon, lat) {
    var best = Infinity, kx = 111320 * Math.cos(lat * Math.PI / 180), ky = 110574;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      var ax = (ring[j][0] - lon) * kx, ay = (ring[j][1] - lat) * ky;
      var bx = (ring[i][0] - lon) * kx, by = (ring[i][1] - lat) * ky;
      var dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
      var t = len ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len)) : 0;
      best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
    }
    return best;
  }
  function centroid(ring) {
    var x = 0, y = 0;
    ring.forEach(function (p) { x += p[0]; y += p[1]; });
    return [x / ring.length, y / ring.length];
  }
  // The outer ring of the hovered building, in longitude and latitude.
  function footprintOf(feature) {
    var g = feature && feature.geometry;
    if (!g || !g.coordinates) return null;
    var ring = g.type === "Polygon" ? g.coordinates[0] : (g.type === "MultiPolygon" ? g.coordinates[0][0] : null);
    return ring && ring.length > 2 && Math.abs(ring[0][0]) <= 180 ? ring : null;
  }
  // The named building matching the hovered footprint: one whose outline holds
  // the footprint's centre, or whose centre lies inside the footprint (tiles
  // can cut a building at their edge). Without a footprint, the nearest outline
  // to the point under the pointer.
  function nameFor(feature, coordinate) {
    if (!names) return null;
    var ring = footprintOf(feature);
    if (ring) {
      var c = centroid(ring), near = names[cellKey(c[0], c[1])] || [];
      for (var i = 0; i < near.length; i++) {
        var nc = centroid(near[i].ring);
        if (inside(near[i].ring, c[0], c[1]) || inside(ring, nc[0], nc[1])) return near[i].name;
      }
      return null;
    }
    if (!coordinate) return null;
    var lon = coordinate[0], lat = coordinate[1];
    var cand = names[cellKey(lon, lat)] || [], best = null, bestD = 10;
    for (var j = 0; j < cand.length; j++) {
      if (inside(cand[j].ring, lon, lat)) return cand[j].name;
      var d = edgeMetres(cand[j].ring, lon, lat);
      if (d < bestD) { bestD = d; best = cand[j].name; }
    }
    return best;
  }
  function esc(t) {
    return String(t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  var OPS_LAYERS = ["admin-boundaries", "rivers", "water-bodies", "infrastructure", "stations", "alert-areas"];
  function tooltip(info) {
    if (!info || !info.object) return null;
    var id = (info.layer && info.layer.id) || "";
    var p = info.object.properties || info.object;
    if (id === "flood-reports") {
      return { style: CITY.tooltip, html:
        "<div style='font-weight:600;font-size:13px;margin-bottom:2px'>" + esc(p.title) + "</div>" +
        "<div style='opacity:0.8'>" + esc(p.detail) + "</div>" +
        (p.note ? "<div style='opacity:0.75;margin-top:4px;font-style:italic'>\u201c" + esc(p.note) + "\u201d</div>" : "") };
    }
    if (OPS_LAYERS.indexOf(id) >= 0) {
      return { style: CITY.tooltip, html:
        "<div style='font-weight:600;font-size:13px;margin-bottom:2px'>" + esc(p.title) + "</div>" +
        "<div style='opacity:0.85'>" + esc(p.detail || "") + "</div>" +
        "<div style='opacity:0.55;font-size:11px;margin-top:4px'>" + esc(p.source || "") + "</div>" };
    }
    if (id === "past-floods") {
      var key = p.near
        ? "<span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:#facc15;margin-right:5px'></span>Reported flooded the week of this storm"
        : "<span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:#a855f7;margin-right:5px'></span>Past flood in the news, not a forecast";
      return { style: CITY.tooltip, html:
        "<div style='font-weight:600;font-size:13px;margin-bottom:2px'>" + esc(p.name) + "</div>" +
        "<div style='opacity:0.75;font-size:11.5px;margin-bottom:4px'>" + key + "</div>" +
        "<div>" + esc(p.title) + "</div>" +
        "<div style='opacity:0.8'>" + esc(p.detail) + "</div>" +
        (p.near ? "<div style='color:#facc15;margin-top:4px'>" + esc(p.near) + "</div>" : "") +
        "<div style='opacity:0.55;font-size:11px;margin-top:4px'>" + esc(p.source) + "</div>" };
    }
    if (id.indexOf("buildings-city") === 0) {
      var h = Math.round(+p.render_height || 0);
      var name = nameFor(info.object, info.coordinate);
      var floors = h >= 6 ? " · about " + Math.max(1, Math.round(h / 3.4)) + " floors" : "";
      return { style: CITY.tooltip, html:
        "<div style='font-weight:600;font-size:13px;margin-bottom:2px'>" + esc(name || "Building") + "</div>" +
        "<div style='opacity:0.75'>" + h + " m" + floors + "</div>" };
    }
    if (p.region === undefined) return null;
    return { style: CITY.tooltip, html:
      "<div style='font-weight:600;font-size:13px;margin-bottom:3px'>" + esc(p.region) + "</div>" +
      "<div style='opacity:0.85'>" + esc(p.name || "") + "</div>" +
      "<div style='opacity:0.7;font-size:11.5px;margin-top:2px'>" + esc(p.value_label || "") + "</div>" };
  }

  function useFallback(d) {
    fetch(CITY.fallback).then(function (r) { return r.json(); }).then(function (rows) {
      d.setProps({ layers: d.props.layers.map(function (layer) {
        return layer.id === "buildings" ? layer.clone({ data: rows, visible: true }) : layer;
      })});
    }).catch(function (e) { console.warn("Fallback buildings unavailable:", e); });
    d.setProps({ layers: d.props.layers.map(function (layer) {
      if (layer.id === "buildings-city") return layer.clone({ visible: false });
      return layer;
    })});
  }

  var tries = 0;
  (function install() {
    var d = findDeck(), D = window.deck;
    if (!d || !d.props || !d.props.layers || !D || !D.MVTLayer) {
      if (tries++ < 150) setTimeout(install, 100);
      return;
    }
    var failures = 0;
    var windows = null;
    try { windows = windowsExtension(D); } catch (e) { console.warn("Building windows unavailable:", e); }
    var city = new D.MVTLayer({
      id: "buildings-city",
      data: CITY.tiles,
      // Buildings are in the tiles from zoom 13; zoom 14 tiles are reused beyond.
      minZoom: 13,
      maxZoom: 14,
      loadOptions: { mvt: { layers: ["building"] } },
      extruded: true,
      filled: true,
      stroked: false,
      getElevation: function (f) { return Math.max(+f.properties.render_height || 0, 3); },
      getFillColor: CITY.color,
      material: CITY.material,
      extensions: windows ? [windows] : [],
      // Hovering a building names it (see tooltip) and lights it up.
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 46],
      // The tile set's description (TileJSON) failing to load reaches onError;
      // individual tiles failing reach onTileError.
      onError: function () { if (failures < 3) { failures = 3; useFallback(d); } return true; },
      onTileError: function () { if (++failures === 3) useFallback(d); }
    });
    // The flood fill (not its edge line) is drawn as water.
    var water = null;
    try { water = waterExtension(D); } catch (e) { console.warn("Water effect unavailable:", e); }
    // Under the routes and pins, which are drawn last so they stay on top.
    var layers = d.props.layers.map(function (layer) {
      if (water && (layer.id === "flood-main" || layer.id === "flood-halo")) {
        return layer.clone({ _subLayerProps: { "polygons-fill": { extensions: [water] } } });
      }
      return layer;
    });
    // The satellite flood map (src.reports.satellite): over the predicted water,
    // under the buildings, hidden until the switch on the page turns it on.
    var sat = CITY.satellite || {};
    var satellite = D.BitmapLayer ? new D.BitmapLayer({
      id: "satellite-flood",
      image: (sat.visible && sat.image) || sat.blank,
      bounds: sat.bounds,
      visible: !!(sat.visible && sat.image),
      pickable: false,
      parameters: { depthCompare: "always", depthWriteEnabled: false }
    }) : null;
    var at = layers.findIndex(function (l) { return l.id.indexOf("route-") === 0; });
    layers.splice(at < 0 ? layers.length : at, 0, city);
    if (satellite) layers.splice(at < 0 ? layers.length : at, 0, satellite);
    // No _animate: it redrew every frame (every building, lit and windowed) even
    // with no flooding on screen. The ripples are moved by the timer below.
    var baseDPR = Math.min(window.devicePixelRatio || 1, 1.5);
    d.setProps({ layers: layers, effects: [lighting(D)], getTooltip: tooltip, _animate: false,
                 useDevicePixels: baseDPR });
    loadNames();

    // ---- performance ------------------------------------------------------
    // "Lighter 3D" drops what costs most per frame: moving water, lit windows,
    // the extra lights, and high-density pixels. The page sends the viewer's
    // choice (auto / off / on); on auto, a short frame-rate probe decides.
    var perf = { mode: "auto", lite: false, fps: null };
    function floodOnScreen() {
      var l = d.props.layers.find(function (x) { return x.id === "flood-main"; });
      var data = l && l.props.data;
      return !!(data && data.features && data.features.length);
    }
    function applyPerf(lite) {
      perf.lite = lite;
      d.setProps({
        useDevicePixels: lite ? 1 : baseDPR,
        effects: lite ? [] : [lighting(D)],
        layers: d.props.layers.map(function (l) {
          return l.id === "buildings-city" ? l.clone({ extensions: (lite || !windows) ? [] : [windows] }) : l;
        })
      });
      post({ type: "perfState", mode: perf.mode, lite: lite, fps: perf.fps });
    }
    setInterval(function () {
      if (water && !perf.lite && !document.hidden && floodOnScreen()) d.redraw("water");
    }, 66);
    function probe() {
      // Frames the full effects can draw in 2.5 s of continuous redraws.
      var frames = 0, until = performance.now() + 2500;
      d.setProps({ onAfterRender: function () { frames++; } });
      (function tick() {
        if (performance.now() < until) { d.redraw("probe"); requestAnimationFrame(tick); return; }
        // deck.gl calls onAfterRender unguarded: null here broke every later frame.
        d.setProps({ onAfterRender: function () {} });
        perf.fps = Math.round(frames / 2.5);
        if (perf.mode === "auto") applyPerf(perf.fps < 28);
      })();
    }
    window.addEventListener("message", function (ev) {
      var msg = ev.data;
      if (!msg || msg.type !== "perf") return;
      perf.mode = msg.mode || "auto";
      if (perf.mode === "on") applyPerf(true);
      else if (perf.mode === "off") applyPerf(false);
      else if (perf.fps === null) setTimeout(probe, 1500);   // after the first tiles arrive
      else applyPerf(perf.fps < 28);
    });
    // Read-only view of the performance state, for checks and the test scripts.
    window.__twinPerf = function () {
      return { mode: perf.mode, lite: perf.lite, fps: perf.fps, water: !!water, flood: floodOnScreen(),
               hidden: !!document.hidden };
    };
    post({ type: "perfReady" });

    // The satellite switch on the page shows or hides the satellite layer.
    window.addEventListener("message", function (ev) {
      var msg = ev.data;
      if (!msg || msg.type !== "satellite") return;
      d.setProps({ layers: d.props.layers.map(function (layer) {
        if (layer.id !== "satellite-flood") return layer;
        return msg.visible && msg.image ? layer.clone({ image: msg.image, visible: true })
                                        : layer.clone({ visible: false });
      })});
    });
  })();
})();
</script>
"""
