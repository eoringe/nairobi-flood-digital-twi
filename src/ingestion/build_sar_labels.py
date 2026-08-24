"""
src.ingestion.build_sar_labels
==============================
Build training labels from Sentinel-1 SAR change detection paired with rainfall.

Instead of Sentinel-2 optical detection (which can't see urban street flooding),
we use SAR's own change-detection capability: compare wet-season backscatter
against dry-season baseline to identify where backscatter DROPPED (indicating water).

METHODOLOGY
-----------
1. For each year, compute median VV backscatter from DRY season (Jan-Feb)
2. For each WET-season acquisition, compute per-pixel backscatter DROP vs. baseline
3. Pair with CHIRPS rainfall: high rainfall + high drop = flood likely
4. Create binary labels: flood_likely = (drop > DROP_THRESH_DB) AND (rain_7d > RAIN_THRESH_MM)

KEY INSIGHT
-----------
SAR change detection (drop in backscatter) is the CORRECT method for flood mapping.
The inverted-label problem came from using absolute -16 dB threshold on a median
composite (which smooths out transient signal). Change detection against a stable
dry-season baseline works because:
  - Each pixel's own dry state is the reference (surface type cancels out)
  - Water causes large specular drops (~5-10 dB) independent of terrain
  - Anchors detection to actual recent events (not seasonal average)

PARAMETERS
----------
DROP_THRESH_DB = 3.0 dB   # Backscatter drop required to call a pixel flooded
RAIN_THRESH_MM = 30.0 mm  # Antecedent 7-day rainfall to pair with drops
RELATIVE_ORBIT = 57       # ASCENDING orbit (consistent 2015-2026)

USAGE
-----
    python -m src.ingestion.build_sar_labels
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

YEARS = list(range(2015, 2027))
RELATIVE_ORBIT = 57
DROP_THRESH_DB = 3.0
RAIN_THRESH_MM = 30.0
OUT_FILE = Path("models/time_series/sar_change_labels.json")


def _load_rain():
    """Load CHIRPS daily rainfall keyed by date."""
    try:
        rain = np.load("data/processed/arrays/rainfall_daily_mean.npy")
        raw = json.load(open("data/processed/arrays/rainfall_dates.json"))
        dates = []
        for s in raw:
            y, _, doy = s.split("-")
            dates.append(date(int(y), 1, 1) + timedelta(days=int(doy) - 1))
        return {d: float(v) for d, v in zip(dates, rain)}
    except Exception as exc:
        print(f"[WARN ] Could not load CHIRPS: {exc}")
        return {}


def _antecedent_rain(rain_by_date: dict, d: date, days: int) -> float:
    """Sum rainfall in preceding `days` days."""
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
    print(f"[S1   ] Sentinel-1 change detection: orbit {RELATIVE_ORBIT} ASCENDING")
    print(f"        Drop threshold: {DROP_THRESH_DB} dB  |  Rain threshold: {RAIN_THRESH_MM} mm (7-day)\n")

    scenes = []

    # Base collection filter (orbit + geometry)
    base_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", RELATIVE_ORBIT))
        .select("VV")
    )

    print("Building per-year dry-season baselines (Jan-Feb)...")
    for year in YEARS:
        try:
            # Dry baseline: Jan-Feb, median across scenes
            baseline_col = base_col.filterDate(f"{year}-01-01", f"{year}-03-01")
            n_base = baseline_col.size().getInfo()
            if n_base == 0:
                print(f"  {year}: 0 baseline scenes")
                continue

            baseline_db = baseline_col.median()
            print(f"  {year}: {n_base} baseline scenes → median composite")

            # Wet season: MAM and OND
            wet_col = base_col.filterDate(f"{year}-03-01", f"{year}-06-01").merge(
                base_col.filterDate(f"{year}-10-01", f"{year}-12-31")
            )
            n_wet = wet_col.size().getInfo()
            if n_wet == 0:
                print(f"           0 wet-season scenes")
                continue

            # Compute backscatter drop for each wet scene
            def compute_drop(img):
                drop_db = baseline_db.subtract(img)  # Positive = backscatter decreased
                return img.set("drop_db_mean", drop_db.reduceRegion(
                    ee.Reducer.mean(), geom, 10, maxPixels=1e10
                ).get("VV"))

            scored = wet_col.map(compute_drop)

            # Extract per-scene metadata
            dates_ms = scored.aggregate_array("system:time_start").getInfo()
            drops = scored.aggregate_array("drop_db_mean").getInfo()

            for ms, drop_db in zip(dates_ms, drops):
                d = date.fromtimestamp(ms / 1000)
                rain_7d = _antecedent_rain(rain_by_date, d, 7)

                # Binary label: flood likely if high drop AND high rain
                flood_likely = (drop_db is not None and drop_db > DROP_THRESH_DB
                               and rain_7d > RAIN_THRESH_MM)

                scenes.append({
                    "date": d.isoformat(),
                    "drop_db": float(drop_db) if drop_db is not None else 0.0,
                    "rain_7d_mm": rain_7d,
                    "flood_likely": bool(flood_likely),
                    "reason": (
                        "drop high + rain high" if flood_likely else
                        f"drop={drop_db:.1f}dB rain={rain_7d:.0f}mm"
                    ),
                })
            print(f"           {n_wet} wet-season scenes processed")

        except Exception as exc:
            print(f"  {year}: ERROR {str(exc)[:60]}")

    if not scenes:
        print("[FATAL] No SAR scenes found.")
        sys.exit(1)

    # Sort by date
    scenes.sort(key=lambda s: s["date"])

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(scenes, f, indent=2)

    # Summary statistics
    floods = sum(1 for s in scenes if s["flood_likely"])
    drops = np.array([s["drop_db"] for s in scenes])
    rains = np.array([s["rain_7d_mm"] for s in scenes])

    print(f"\n[SAVE ] {len(scenes)} scenes → {OUT_FILE}")
    print(f"        Date range: {scenes[0]['date']} to {scenes[-1]['date']}")
    print(f"        Labeled as 'flood likely': {floods}/{len(scenes)} ({100*floods/len(scenes):.1f}%)")
    print(f"        Backscatter drop: mean={drops.mean():.2f}dB, max={drops.max():.2f}dB, min={drops.min():.2f}dB")
    print(f"        Preceding rainfall: mean={rains.mean():.1f}mm, max={rains.max():.1f}mm, min={rains.min():.1f}mm")

    # Correlation check
    from scipy.stats import pearsonr, spearmanr
    r_drop_rain = pearsonr(drops, rains)
    r_flood_rain = pearsonr([1.0 if s["flood_likely"] else 0.0 for s in scenes], rains)
    print(f"\n        Correlation: backscatter drop vs rain: r={r_drop_rain[0]:+.3f} (p={r_drop_rain[1]:.4f})")
    print(f"        Correlation: flood label vs rain: r={r_flood_rain[0]:+.3f} (p={r_flood_rain[1]:.4f})")


if __name__ == "__main__":
    main()
