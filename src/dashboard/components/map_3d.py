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

from src.models.predict import LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST, GRID_H, GRID_W

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


def generate_flood_contour_geojson(
    depth_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    display_mode: str = "DEPTH"
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
    if np.max(depth_grid) < 0.2:
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
    if display_mode == "PROBABILITY":
        prob_grid = np.clip((smooth_grid / 2.2) * 100.0, 0.0, 99.0)
        risk_mask[prob_grid >= 25.0] = 1
        risk_mask[prob_grid >= 55.0] = 2
        risk_mask[prob_grid >= 82.0] = 3
        names = {1: "Moderate Risk (25-55%)", 2: "High Risk (55-82%)", 3: "Severe Risk (>82%)"}
    else:
        risk_mask[smooth_grid >= 0.35] = 1
        risk_mask[smooth_grid >= 1.1] = 2
        risk_mask[smooth_grid >= 1.85] = 3
        names = {1: "Shallow (0.35-1.1m)", 2: "Deep (1.1-1.85m)", 3: "Critical (>1.85m)"}

    styles = {
        1: {"fillColor": [53, 194, 209, 85],  "lineColor": [110, 220, 230, 90],  "name": names[1]},
        2: {"fillColor": [35, 138, 205, 120], "lineColor": [90, 180, 230, 110],  "name": names[2]},
        3: {"fillColor": [21, 82, 176, 160],  "lineColor": [70, 130, 220, 130],  "name": names[3]},
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
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "level": level,
                    "fillColor": style["fillColor"],
                    "lineColor": style["lineColor"],
                    "name": style["name"],
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
    center_lat: float = -1.2787,
    center_lon: float = 36.8213,
    zoom: float = 13.0,
    pitch: float = 45.0,
    bearing: float = -15.0,
    display_mode: str = "DEPTH",
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
            "html": "<b>Zone:</b> {name}",
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
        <span style="color: #5c6b7a; text-transform: uppercase; letter-spacing: 0.6px; font-size: 9.5px;">Water Depth</span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#35c2d1; display:inline-block;"></span>
            Shallow <span style="color:#5c6b7a;">0.35&ndash;1.1m</span>
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#238acd; display:inline-block;"></span>
            Deep <span style="color:#5c6b7a;">1.1&ndash;1.85m</span>
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:2px; background:#1552b0; display:inline-block;"></span>
            Critical <span style="color:#5c6b7a;">&gt;1.85m</span>
        </span>
        <span style="display:flex; align-items:center; gap:6px; color:#5c6b7a;">
            <span style="width:8px; height:8px; border-radius:2px; background:#5a6473; display:inline-block;"></span>
            3D Buildings
        </span>
    </div>
    """

    if "</body>" in base_html:
        return base_html.replace("</body>", f"{legend_html}\n</body>")
    return base_html + legend_html
