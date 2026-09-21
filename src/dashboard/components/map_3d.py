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
        "flood": {
            1: ([240, 185, 63, 90],  [245, 205, 110, 110]),
            2: ([240, 138, 60, 120], [245, 170, 105, 125]),
            3: ([239, 68, 89, 155],  [245, 120, 135, 140]),
        },
        "halo": {1: [53, 194, 209, 26], 2: [35, 138, 205, 32], 3: [21, 82, 176, 38]},
        "buildings": ([53, 194, 209, 205], [68, 110, 130, 180], [90, 100, 115, 150]),
        "tooltip": {"backgroundColor": "#151d28", "color": "#eaf0f6", "border": "1px solid #202b38",
                    "boxShadow": "0 8px 20px rgba(0,0,0,0.4)"},
        "legend": {"bg": "rgba(16, 22, 31, 0.92)", "border": "#202b38", "text": "#eaf0f6",
                   "muted": "#5c6b7a", "shadow": "0 10px 28px rgba(0, 0, 0, 0.45)", "bldg": "#5a6473"},
        "pin_stroke": [16, 22, 31, 255],
    },
    "light": {
        "basemap": pdk.map_styles.CARTO_ROAD,
        "flood": {
            1: ([232, 160, 18, 125], [196, 128, 6, 170]),
            2: ([232, 112, 32, 150], [196, 84, 18, 185]),
            3: ([214, 40, 66, 170],  [176, 24, 48, 200]),
        },
        "halo": {1: [14, 143, 156, 22], 2: [20, 110, 190, 26], 3: [20, 70, 160, 30]},
        "buildings": ([14, 143, 156, 190], [132, 160, 174, 170], [176, 186, 198, 150]),
        "tooltip": {"backgroundColor": "#ffffff", "color": "#0f1a24", "border": "1px solid #d5dee5",
                    "boxShadow": "0 8px 20px rgba(15,26,36,0.14)"},
        "legend": {"bg": "rgba(255, 255, 255, 0.94)", "border": "#d5dee5", "text": "#0f1a24",
                   "muted": "#6b7a88", "shadow": "0 10px 28px rgba(15, 26, 36, 0.14)", "bldg": "#b0bac6"},
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

    # Same palette as RISK_COLORS in src/dashboard/callbacks.py, so a region card
    # reading CRITICAL in red corresponds to a red polygon rather than a blue one.
    # This trades the "water is blue" convention for agreement between the map and
    # the panel beside it; the map communicates risk, not the presence of a lake.
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
            for feat in geo_data.get("features", [])[:4000]:
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


#: Layer ids the live-update bridge can patch. Every one is created in every
#: document, empty if need be: a layer that does not exist cannot be updated by
#: id, so a route requested after the map loaded would have nowhere to go.
ROUTE_LAYER_IDS = ("route-fastest", "route-casing", "route-safe", "route-endpoints")


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
) -> pdk.Deck:
    """
    Construct Pydeck 3D Viewport with street-accurate flood overlay and crisp 3D buildings.

    `route_layers` maps each id in ROUTE_LAYER_IDS to its data rows, so a
    rebuilt document (theme or camera change) keeps an active route on screen.
    """
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
        get_line_width=1,
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 60],
    ))

    # 2. LOD2 Solid 3D Volumetric Building Footprint Extrusions
    building_data = _building_rows(theme)

    if building_data:
        layers.append(pdk.Layer(
            "PolygonLayer",
            building_data,
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
    no_depth = {"depthCompare": "always", "depthWriteEnabled": False}
    path_common = dict(get_path="path", get_color="color", get_width="width",
                       width_units="meters", cap_rounded=True, joint_rounded=True,
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
        get_radius=9, radius_units="pixels", pickable=False, parameters=no_depth,
    ))

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
            "style": {
                **palette["tooltip"],
                "fontFamily": "'IBM Plex Sans', 'Segoe UI', sans-serif",
                "fontSize": "12.5px",
                "borderRadius": "6px",
                "padding": "6px 10px",
            },
        },
    )
    # The side panel summarises the polygons that were actually drawn, so they
    # travel with the deck rather than being recomputed and risking divergence
    # between what the map shows and what the panel reports.
    deck._flood_features = flood_geojson.get("features", [])
    deck._halo_features = halo_geojson.get("features", [])
    deck._theme = theme
    return deck


def get_deck_html_with_embedded_legend(deck: pdk.Deck) -> str:
    """
    Generate Pydeck HTML string with an embedded floating Legend Pill.
    """
    base_html = deck.to_html(as_string=True)
    lg = MAP_THEMES.get(getattr(deck, "_theme", "dark"), MAP_THEMES["dark"])["legend"]

    # Matches the token system in src/dashboard/assets/custom.css. This HTML is
    # rendered inside the map's own iframe and cannot reach that stylesheet, so
    # the theme's colours are written in directly.
    legend_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        #pydeck-embedded-legend .sw {{ width:8px; height:8px; border-radius:2px; display:inline-block; }}
        #pydeck-embedded-legend .it {{ display:flex; align-items:center; gap:6px; }}
        #pydeck-embedded-legend .mu {{ color:{lg['muted']}; }}
        @media (max-width: 640px) {{ #pydeck-embedded-legend .wide {{ display:none; }} }}
    </style>
    <div id="pydeck-embedded-legend" style="
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background-color: {lg['bg']};
        backdrop-filter: blur(6px);
        border: 1px solid {lg['border']};
        border-radius: 10px;
        padding: 8px 16px;
        z-index: 99999;
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
        color: {lg['text']};
        font-size: 10.5px;
        letter-spacing: 0.2px;
        box-shadow: {lg['shadow']};
        pointer-events: none;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span class="mu" style="text-transform: uppercase; letter-spacing: 0.6px; font-size: 9.5px;">Flood Probability</span>
        <span class="it"><span class="sw" style="background:#f0b93f;"></span>Moderate <span class="mu">25&ndash;55%</span></span>
        <span class="it"><span class="sw" style="background:#f08a3c;"></span>High <span class="mu">55&ndash;82%</span></span>
        <span class="it"><span class="sw" style="background:#ef4459;"></span>Critical <span class="mu">&gt;82%</span></span>
        <span class="mu wide" style="font-size:9px;">Probability of flooding, not depth</span>
        <span class="it wide mu"><span class="sw" style="background:{lg['bldg']};"></span>3D Buildings</span>
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
      if (msg.fit) fitTo(d, msg.fit);
      post({ type: "layerUpdateAck", ok: true });
    } catch (e) {
      // Reported so the parent can force a full document rebuild rather than
      // silently leaving stale geometry on screen.
      post({ type: "layerUpdateAck", ok: false, error: String(e) });
    }
  });

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

    extra = legend_html + updater_js
    if "</body>" in base_html:
        return base_html.replace("</body>", f"{extra}\n</body>")
    return base_html + extra
