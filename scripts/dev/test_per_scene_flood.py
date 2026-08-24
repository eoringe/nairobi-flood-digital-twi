"""
scripts.dev.test_per_scene_flood
=================================
Decisive test: does per-SCENE change-detection flood area respond to the
rainfall that actually fell in the days before that acquisition?

Seasonal totals are the wrong predictor — a season can be wet overall
while a given Sentinel-1 pass happens to fall on a dry day two weeks
after the storm. Urban flash floods in Nairobi recede within ~1-3 days,
so the physically meaningful covariate is ANTECEDENT rainfall in a short
window before each acquisition.

If detected area correlates positively with antecedent rainfall, the
change-detection labels carry real flood signal and can be used to build
a per-scene training set (which also lifts N from 23 season-composites to
several hundred acquisitions).

USAGE
-----
    python -m scripts.dev.test_per_scene_flood
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

DROP_THRESH_DB = 3.0
ABS_MAX_DB = -14.0
RELATIVE_ORBIT = 57
YEARS = list(range(2015, 2026))


def _load_rain():
    rain = np.load("data/processed/arrays/rainfall_daily_mean.npy")
    raw = json.load(open("data/processed/arrays/rainfall_dates.json"))
    dates = []
    for s in raw:
        y, _, doy = s.split("-")
        dates.append(date(int(y), 1, 1) + timedelta(days=int(doy) - 1))
    return {d: float(v) for d, v in zip(dates, rain)}


def _antecedent(rain_by_date: dict, d: date, days: int) -> float:
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

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", RELATIVE_ORBIT))
        .select("VV")
    )

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    permanent = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    slope_deg = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    plausible = merit.select("hnd").lt(15).And(slope_deg.lt(5)).And(permanent.lt(20))

    # One dry-season baseline per year, same orbit — the pixel's own
    # "normal" state, so surface type cancels out of the difference.
    print("Building per-year dry-season baselines (Jan-Feb, same orbit)...")
    rows = []
    for year in YEARS:
        baseline = col.filterDate(f"{year}-01-01", f"{year}-03-01")
        if baseline.size().getInfo() == 0:
            continue
        base_db = baseline.median()

        # All wet-season acquisitions this year (MAM + OND)
        season = col.filterDate(f"{year}-03-01", f"{year}-06-01").merge(
            col.filterDate(f"{year}-10-01", f"{year}-12-31")
        )

        def _area(img):
            drop = base_db.subtract(img)
            flooded = drop.gt(DROP_THRESH_DB).And(img.lt(ABS_MAX_DB)).And(plausible)
            a = (
                flooded.multiply(ee.Image.pixelArea())
                .reduceRegion(ee.Reducer.sum(), geom, 30, maxPixels=1e10)
                .get("VV")
            )
            return img.set("flood_m2", a)

        scored = season.map(_area)
        try:
            dates_ms = scored.aggregate_array("system:time_start").getInfo()
            areas = scored.aggregate_array("flood_m2").getInfo()
        except Exception as exc:
            print(f"  {year}: FAILED {str(exc)[:60]}")
            continue

        for ms, a in zip(dates_ms, areas):
            d = date.fromtimestamp(ms / 1000)
            rows.append({
                "date": d,
                "flood_km2": (a or 0) / 1e6,
                "rain_1d": _antecedent(rain_by_date, d, 1),
                "rain_3d": _antecedent(rain_by_date, d, 3),
                "rain_7d": _antecedent(rain_by_date, d, 7),
                "rain_14d": _antecedent(rain_by_date, d, 14),
            })
        print(f"  {year}: {len(dates_ms)} wet-season scenes")

    if len(rows) < 10:
        print("Too few scenes to evaluate.")
        return

    from scipy.stats import pearsonr, spearmanr
    a = np.array([r["flood_km2"] for r in rows])
    print(f"\n{len(rows)} acquisitions | flood area: mean={a.mean():.3f} "
          f"median={np.median(a):.3f} max={a.max():.3f} km2")
    print("\nDetected flood area vs ANTECEDENT rainfall:")
    print(f"{'window':<10}{'Pearson r':>12}{'p':>9}{'Spearman r':>13}{'p':>9}")
    for w in ["rain_1d", "rain_3d", "rain_7d", "rain_14d"]:
        x = np.array([r[w] for r in rows])
        pr, pp = pearsonr(x, a)
        sr, sp = spearmanr(x, a)
        print(f"{w:<10}{pr:>+12.3f}{pp:>9.4f}{sr:>+13.3f}{sp:>9.4f}")

    print("\nTop 10 scenes by detected flood area:")
    for r in sorted(rows, key=lambda r: -r["flood_km2"])[:10]:
        print(f"  {r['date']}  area={r['flood_km2']:.3f} km2  "
              f"rain3d={r['rain_3d']:.1f}mm  rain7d={r['rain_7d']:.1f}mm")

    out = "models/time_series/per_scene_flood_test.json"
    with open(out, "w") as f:
        json.dump([{**r, "date": r["date"].isoformat()} for r in rows], f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
