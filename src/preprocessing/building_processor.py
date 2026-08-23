"""
src.preprocessing.building_processor
======================================
Nairobi Urban Flood Digital Twin — 3D Volumetric Building Footprint Extractor

PURPOSE
-------
1. Read building footprint geometries from data/raw/assets/ (CSV / GeoJSON / Parquet)
2. Filter buildings within Nairobi County boundary (-1.45 <= Lat <= -1.15, 36.65 <= Lon <= 37.10)
3. Parse each building's real WKT footprint polygon (previously discarded in
   favor of a synthetic ~20m box around the centroid) and derive height from
   its real footprint area via a footprint-scaling heuristic (previously a
   uniform random value, since the source data has no height column) —
   larger footprints correlate with taller/commercial structures, matching
   the size distribution actually present in the Open Buildings-style
   source CSV (latitude, longitude, area_in_meters, confidence, geometry).
4. Save lightweight GeoJSON to data/processed/nairobi_buildings_3d.json

MEMORY CONTRACT
---------------
Processes CSV chunks (chunksize=20000) to keep RAM < 500 MB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
from loguru import logger

try:
    from shapely import wkt as shapely_wkt
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False

DEFAULT_ASSETS_DIR = Path("data/raw/assets")
DEFAULT_OUT_DIR = Path("data/processed")

NAIROBI_LAT_MIN, NAIROBI_LAT_MAX = -1.45, -1.15
NAIROBI_LON_MIN, NAIROBI_LON_MAX = 36.65, 37.10

#: Footprint-area-based height heuristic (no height column exists in the
#: source data). Calibrated so small residential/informal footprints
#: (~10-50 sq m) land in the 5-9 m range and large commercial complexes
#: (~1000+ sq m) land in the 25-45 m range.
HEIGHT_BASE_M = 4.0
HEIGHT_AREA_SCALE = 0.55
HEIGHT_MIN_M = 4.0
HEIGHT_MAX_M = 45.0


def _height_from_area(area_m2: float) -> float:
    height = HEIGHT_BASE_M + np.sqrt(max(area_m2, 0.0)) * HEIGHT_AREA_SCALE
    return float(np.clip(height, HEIGHT_MIN_M, HEIGHT_MAX_M))


def _parse_footprint_ring(wkt_str: str) -> list[list[float]] | None:
    """Parse a WKT POLYGON string into a [[lon, lat], ...] exterior ring."""
    if not _SHAPELY_OK or not isinstance(wkt_str, str):
        return None
    try:
        geom = shapely_wkt.loads(wkt_str)
        if geom.is_empty:
            return None
        poly = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda g: g.area)
        return [[round(x, 7), round(y, 7)] for x, y in poly.exterior.coords]
    except Exception:
        return None


def extract_nairobi_buildings(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_buildings: int = 25000,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nairobi_buildings_3d.json"

    if out_file.exists() and os.path.getsize(out_file) > 1000:
        logger.info(f"Using existing 3D buildings file at {out_file}")
        return out_file

    csv_gz_files = list(assets_dir.glob("*.csv.gz")) + list(assets_dir.glob("*.csv"))

    features = []

    if csv_gz_files and _SHAPELY_OK:
        csv_file = csv_gz_files[0]
        logger.info(f"Reading building footprints from {csv_file.name}")

        try:
            import pandas as pd
            chunksize = 20000
            n_no_geom = 0
            for chunk in pd.read_csv(csv_file, chunksize=chunksize, compression='gzip' if str(csv_file).endswith('.gz') else None):
                lat_col = [c for c in chunk.columns if 'lat' in c.lower()]
                lon_col = [c for c in chunk.columns if 'lon' in c.lower()]

                if not (lat_col and lon_col):
                    continue

                lat_name, lon_name = lat_col[0], lon_col[0]
                mask = (
                    (chunk[lat_name] >= NAIROBI_LAT_MIN) &
                    (chunk[lat_name] <= NAIROBI_LAT_MAX) &
                    (chunk[lon_name] >= NAIROBI_LON_MIN) &
                    (chunk[lon_name] <= NAIROBI_LON_MAX)
                )
                filtered = chunk[mask]

                for _, row in filtered.iterrows():
                    lat, lon = float(row[lat_name]), float(row[lon_name])
                    area_m2 = float(row.get('area_in_meters', 0.0) or 0.0)
                    confidence = float(row.get('confidence', 0.85))
                    height = _height_from_area(area_m2)

                    ring = _parse_footprint_ring(row.get('geometry'))
                    if ring is not None:
                        geometry = {"type": "Polygon", "coordinates": [ring]}
                    else:
                        n_no_geom += 1
                        geometry = {"type": "Point", "coordinates": [lon, lat]}

                    features.append({
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "height": round(height, 1),
                            "floors": max(1, int(height // 3)),
                            "area_m2": round(area_m2, 1),
                            "confidence": round(confidence, 2),
                        }
                    })

                    if len(features) >= max_buildings:
                        break
                if len(features) >= max_buildings:
                    break

            if n_no_geom:
                logger.warning(f"{n_no_geom} buildings had unparseable geometry — fell back to a point centroid.")

        except Exception as e:
            logger.warning(f"Error reading building CSV: {e}")

    elif csv_gz_files and not _SHAPELY_OK:
        logger.warning("shapely not installed — cannot parse real footprint polygons. pip install shapely>=2.0.0")

    # Fallback synthetic grid generator if raw file is missing/unreadable
    if not features:
        logger.info("Generating representative Nairobi 3D building vector grid (no source CSV / geometry available)...")
        np.random.seed(42)
        centers = [
            (-1.286389, 36.817222, "CBD", 15.0, 45.0),
            (-1.313333, 36.788889, "Kibera", 4.0, 10.0),
            (-1.317500, 36.862500, "Mukuru", 4.0, 12.0),
            (-1.266667, 36.800000, "Westlands", 12.0, 40.0),
        ]

        for lat_c, lon_c, name, min_h, max_h in centers:
            for _ in range(2500):
                lat = lat_c + np.random.normal(0, 0.015)
                lon = lon_c + np.random.normal(0, 0.015)
                height = float(np.random.uniform(min_h, max_h))
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(lon, 6), round(lat, 6)]
                    },
                    "properties": {
                        "height": round(height, 1),
                        "floors": max(1, int(height // 3)),
                        "zone": name
                    }
                })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_file, "w") as f:
        json.dump(geojson_data, f)

    n_polygon = sum(1 for f in features if f["geometry"]["type"] == "Polygon")
    logger.info(
        f"Successfully saved {len(features)} 3D building features to {out_file} "
        f"({n_polygon} with real footprint polygons, {len(features) - n_polygon} point fallback)."
    )
    return out_file


if __name__ == "__main__":
    extract_nairobi_buildings()
