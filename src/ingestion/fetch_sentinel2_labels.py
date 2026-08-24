"""
src.ingestion.fetch_sentinel2_labels
====================================
Fetch Sentinel-2 optical imagery over Nairobi (2015-2026) and derive water-flood
labels independent of SAR, using MNDWI (Modified Normalized Difference Water Index).

MNDWI = (GREEN - SWIR) / (GREEN + SWIR)
  MNDWI > 0.3  → open water
  MNDWI > 0.15 → wet soil / turbid water

Output: JSON file mapping each Sentinel-2 scene to:
  - Scene date
  - Cloud cover %
  - Water pixel count (MNDWI > 0.3)
  - Wet pixel count (MNDWI > 0.15)
  - Corresponding CHIRPS rainfall in preceding 7 days

This is the INDEPENDENT ground truth for model retraining. Completely separate
from the inverted SAR -16dB threshold problem.

USAGE
-----
    python -m src.ingestion.fetch_sentinel2_labels
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

YEARS = list(range(2015, 2027))
CLOUD_THRESH_PCT = 20  # Accept scenes with <= 20% cloud cover
OUT_FILE = "models/time_series/sentinel2_labels.json"


def _load_rain():
    """Load CHIRPS rainfall data keyed by date."""
    try:
        import json as json_mod
        rain = np.load("data/processed/arrays/rainfall_daily_mean.npy")
        raw = json_mod.load(open("data/processed/arrays/rainfall_dates.json"))
        dates = []
        for s in raw:
            y, _, doy = s.split("-")
            dates.append(date(int(y), 1, 1) + timedelta(days=int(doy) - 1))
        return {d: float(v) for d, v in zip(dates, rain)}
    except Exception as exc:
        print(f"[WARN ] Could not load CHIRPS: {exc}")
        return {}


def _antecedent_rain(rain_by_date: dict, d: date, days: int) -> float:
    """Sum of rainfall in preceding `days` days."""
    return sum(rain_by_date.get(d - timedelta(days=k), 0.0) for k in range(days))


def _init_gee():
    import ee
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ee.Initialize(project=os.environ.get("GEE_PROJECT_ID") or os.environ.get("GEE_PROJECT"))
    return ee


def main() -> None:
    ee = _init_gee()
    rain_by_date = _load_rain()
    geom = ee.Geometry.BBox(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)

    print(f"[GRID ] {LAT_SOUTH}..{LAT_NORTH}°N  {LON_WEST}..{LON_EAST}°E")
    print(f"[RAIN ] Loaded {len(rain_by_date)} daily rainfall records")
    print(f"[S2   ] Querying Sentinel-2 L2A MSI 2015-2026 over Nairobi")
    print(f"        Cloud threshold: <= {CLOUD_THRESH_PCT}%\n")

    scenes = []
    for year in YEARS:
        try:
            col = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geom)
                .filterDate(f"{year}-01-01", f"{year+1}-01-01")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESH_PCT))
            )
            count = col.size().getInfo()
            if count == 0:
                print(f"  {year}: 0 scenes with cloud < {CLOUD_THRESH_PCT}%")
                continue

            # Compute MNDWI for each scene
            def mndwi_img(img):
                b3 = img.select("B3")  # Green
                b11 = img.select("B11")  # SWIR
                mndwi = b3.subtract(b11).divide(b3.add(b11)).rename("mndwi")
                return img.addBands(mndwi)

            mndwi_col = col.map(mndwi_img)

            # Extract scene metadata
            dates_ms = mndwi_col.aggregate_array("system:time_start").getInfo()
            clouds = mndwi_col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            water_counts = []
            wet_counts = []

            for img in col.toList(col.size()).getInfo():
                try:
                    # This is slow but necessary for per-scene stats
                    img_obj = ee.Image(img)
                    mndwi_img_obj = img_obj.select("B3").subtract(img_obj.select("B11")).divide(
                        img_obj.select("B3").add(img_obj.select("B11"))
                    )
                    water = mndwi_img_obj.gt(0.3).multiply(ee.Image.pixelArea()).reduceRegion(
                        ee.Reducer.sum(), geom, 10, maxPixels=1e10
                    )
                    wet = mndwi_img_obj.gt(0.15).multiply(ee.Image.pixelArea()).reduceRegion(
                        ee.Reducer.sum(), geom, 10, maxPixels=1e10
                    )
                    water_counts.append((water.get("B3").getInfo() or 0) / 1e6)  # km²
                    wet_counts.append((wet.get("B3").getInfo() or 0) / 1e6)
                except Exception:
                    water_counts.append(0.0)
                    wet_counts.append(0.0)

            for ms, cloud, w_km2, wet_km2 in zip(dates_ms, clouds, water_counts, wet_counts):
                d = date.fromtimestamp(ms / 1000)
                rain_7d = _antecedent_rain(rain_by_date, d, 7)
                scenes.append({
                    "date": d.isoformat(),
                    "cloud_pct": float(cloud),
                    "water_km2": w_km2,
                    "wet_km2": wet_km2,
                    "rain_7d_mm": rain_7d,
                })
            print(f"  {year}: {count} scenes, {len(water_counts)} with water stats")
        except Exception as exc:
            print(f"  {year}: ERROR {str(exc)[:60]}")

    if not scenes:
        print("[FATAL] No Sentinel-2 scenes found.")
        sys.exit(1)

    # Sort by date
    scenes.sort(key=lambda s: s["date"])

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"\n[SAVE ] {len(scenes)} scenes → {OUT_FILE}")
    print(f"        Date range: {scenes[0]['date']} to {scenes[-1]['date']}")
    w = np.array([s["water_km2"] for s in scenes])
    print(f"        Water detection: mean={w.mean():.2f} km², max={w.max():.2f} km²")
    r = np.array([s["rain_7d_mm"] for s in scenes])
    print(f"        Preceding rainfall: mean={r.mean():.1f}mm, max={r.max():.1f}mm")


if __name__ == "__main__":
    main()
