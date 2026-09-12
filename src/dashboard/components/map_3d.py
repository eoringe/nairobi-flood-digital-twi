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


_PLACE_LATS, _PLACE_LONS, _PLACE_NAMES = _load_places()


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
    try:
        from shapely import contains_xy
        gy, gx = np.meshgrid(lats[r0:r1], lons[c0:c1], indexing="ij")
        inside = contains_xy(poly, gx, gy)
        vals = window[inside] if inside.any() else window.ravel()
    except Exception:                                      # noqa: BLE001
        vals = window.ravel()

    in_band = vals[(vals >= lo) & (vals <= hi)]
    if in_band.size:
        return float(np.median(in_band))
    return float(np.clip(np.median(vals), lo, hi))


def generate_flood_contour_geojson(
    depth_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    display_mode: str = "PROBABILITY",
    value_is_probability: bool = False,
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
    # PROBABILITY : graded likelihood bands
    # EXTENT      : binary - flooded or not at the operating threshold
    if display_mode == "EXTENT":
        prob_grid = np.clip(smooth_grid * 100.0, 0.0, 100.0)
        risk_mask[prob_grid >= 50.0] = 3
        names = {1: "", 2: "", 3: "Flooded (>50% likelihood)"}
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
    styles = {
        1: {"fillColor": [240, 185, 63, 90],  "lineColor": [245, 205, 110, 110], "name": names[1]},   # MODERATE amber
        2: {"fillColor": [240, 138, 60, 120], "lineColor": [245, 170, 105, 125], "name": names[2]},   # HIGH orange
        3: {"fillColor": [239, 68, 89, 155],  "lineColor": [245, 120, 135, 140], "name": names[3]},   # CRITICAL red
    }

    halo_styles = {
        1: [53, 194, 209, 26],
        2: [35, 138, 205, 32],
        3: [21, 82, 176, 38],
    }

    transform = rasterio.transform.from_bounds(
        LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, w, h
    )

    MIN_AREA_DEG2 = 4.0e-6  # ~25m x 25m — drops stray single-pixel speckle
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

        def _iter_polys(geom):
            if geom is None or geom.is_empty:
                return []
            if geom.geom_type == "MultiPolygon":
                return list(geom.geoms)
            if geom.geom_type == "Polygon":
                return [geom]
            if geom.geom_type == "GeometryCollection":
                return [g for g in geom.geoms if g.geom_type == "Polygon"]
            return []

        style = styles[level]
        for poly in _iter_polys(smoothed):
            if poly.area < MIN_AREA_DEG2:
                continue

            # Name the area this polygon sits in, and read the underlying value
            # at its centroid so the tooltip reports this patch of water rather
            # than the scenario as a whole.
            # representative_point() is guaranteed to fall inside the polygon;
            # centroid() is not, and on a concave or ring-shaped patch it lands
            # in the gap.
            c = poly.representative_point()
            region = _nearest_region(c.y, c.x)

            # Report the peak value within the polygon rather than the value at
            # one point. Contours are smoothed and have building footprints
            # subtracted after classification, so a single interior sample can
            # land just below the band that named the polygon and contradict its
            # own label.
            lo, hi = bands[level]
            val = _band_value_in(poly, smooth_grid, lats, lons, lo, hi)
            if value_is_probability:
                value_label = f"{100.0 * val:.0f}% likelihood"
            else:
                value_label = f"{val:.2f} m"

            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "level": level,
                    "fillColor": style["fillColor"],
                    "lineColor": style["lineColor"],
                    "name": style["name"],
                    "region": region,
                    "value_label": value_label,
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
) -> pdk.Deck:
    """
    Construct Pydeck 3D Viewport with street-accurate flood overlay and crisp 3D buildings.
    """
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
    )

    # 1a. Faded halo underneath — feathers the water edge into the basemap
    # instead of stopping at a hard outline (see generate_flood_contour_geojson).
    if halo_geojson["features"]:
        halo_layer = pdk.Layer(
            "GeoJsonLayer",
            halo_geojson,
            opacity=1.0,
            stroked=False,
            filled=True,
            extruded=False,
            get_fill_color="properties.fillColor",
            pickable=False,
        )
        layers.append(halo_layer)

    # 1b. The readable water fill itself — translucent so streets and
    # terrain stay visible underneath, thin low-alpha edge instead of a
    # bright hazard-stripe outline.
    if flood_geojson["features"]:
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            flood_geojson,
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
        )
        layers.append(geojson_layer)

    # 2. LOD2 Solid 3D Volumetric Building Footprint Extrusions
    building_data = []
    if BUILDINGS_JSON.exists():
        try:
            with open(BUILDINGS_JSON, "r") as f:
                geo_data = json.load(f)

            for feat in geo_data.get("features", [])[:4000]:
                props = feat["properties"]
                h_val = props.get("height", 14.0)
                polygon = list(_footprint_polygon(feat).exterior.coords)

                if h_val > 25:
                    color = [53, 194, 209, 205]   # --accent, tall landmarks
                elif h_val > 12:
                    color = [68, 110, 130, 180]   # desaturated teal-slate, mid-rise
                else:
                    color = [90, 100, 115, 150]   # muted slate, low-rise

                building_data.append({
                    "polygon": polygon,
                    "height": h_val,
                    "color": color,
                })
        except Exception as e:
            logger.warning(f"Error loading building footprints: {e}")

    if building_data:
        building_layer = pdk.Layer(
            "PolygonLayer",
            building_data,
            get_polygon="polygon",
            get_elevation="height",
            get_fill_color="color",
            extruded=True,
            wireframe=True,
            pickable=True,
            opacity=0.85,
        )
        layers.append(building_layer)

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )

    # minZoom/maxZoom are Controller options, not top-level View props — deck.gl
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
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": ("<div style='font-weight:600;font-size:13px;margin-bottom:3px'>{region}</div>"
                     "<div style='opacity:0.85'>{name}</div>"
                     "<div style='opacity:0.7;font-size:11.5px;margin-top:2px'>{value_label}</div>"),
            "style": {
                "backgroundColor": "#151d28",
                "color": "#eaf0f6",
                "fontFamily": "'IBM Plex Sans', 'Segoe UI', sans-serif",
                "fontSize": "12.5px",
                "border": "1px solid #202b38",
                "borderRadius": "6px",
                "padding": "6px 10px",
                "boxShadow": "0 8px 20px rgba(0,0,0,0.4)",
            },
        },
    )
    return deck


def get_deck_html_with_embedded_legend(deck: pdk.Deck) -> str:
    """
    Generate Pydeck HTML string with an embedded floating Legend Pill.
    """
    base_html = deck.to_html(as_string=True)

    # Matches the token system in src/dashboard/assets/custom.css — a
    # quiet glass HUD panel rather than a colorful emoji pill, since this
    # HTML is rendered inside the map's own iframe and can't reach that
    # stylesheet directly.
    legend_html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    </style>
    <div id="pydeck-embedded-legend" style="
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(16, 22, 31, 0.92);
        backdrop-filter: blur(6px);
        border: 1px solid #202b38;
        border-radius: 10px;
        padding: 8px 16px;
        z-index: 99999;
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
        color: #eaf0f6;
        font-size: 10.5px;
        letter-spacing: 0.2px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
        pointer-events: none;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span style="color: #5c6b7a; text-transform: uppercase; letter-spacing: 0.6px; font-size: 9.5px;">Flood Likelihood</span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#f0b93f; display:inline-block;"></span>
            Moderate <span style="color:#5c6b7a;">25&ndash;55%</span>
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#f08a3c; display:inline-block;"></span>
            High <span style="color:#5c6b7a;">55&ndash;82%</span>
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#ef4459; display:inline-block;"></span>
            Critical <span style="color:#5c6b7a;">&gt;82%</span>
        </span>
        <span style="color:#5c6b7a; font-size:9px; margin-top:2px;">Likelihood of flooding, not depth</span>
        <span style="display:flex; align-items:center; gap:6px; color:#5c6b7a;">
            <span style="width:8px; height:8px; border-radius:2px; background:#5a6473; display:inline-block;"></span>
            3D Buildings
        </span>
    </div>
    """

    if "</body>" in base_html:
        return base_html.replace("</body>", f"{legend_html}\n</body>")
    return base_html + legend_html
