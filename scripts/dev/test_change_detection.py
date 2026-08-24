"""
scripts.dev.test_change_detection
==================================
Proof-of-concept for CORRECT Sentinel-1 flood mapping over Nairobi.

WHY
---
The existing pipeline (src.ingestion.fetch_sentinel_targets) thresholds
absolute VV backscatter at -16 dB and takes a seasonal median. Validation
showed the resulting "flood" area is ANTI-correlated with seasonal
rainfall (Spearman r = -0.74, p < 0.001): the wettest season on record
(2024 long rains, 838 mm, the catastrophic Apr-May 2024 Nairobi floods)
produced the FEWEST detections, while the driest (2017 long rains,
186 mm) produced the most.

Cause: a fixed low-backscatter threshold selects SMOOTH DRY surfaces
(bare soil, asphalt, airfields). Wetting soil RAISES its backscatter, so
in genuinely wet seasons those pixels climb above the threshold and
disappear. The labels were measuring dryness.

CORRECT METHOD (UN-SPIDER / Copernicus EMS recommended practice)
-----------------------------------------------------------------
Change detection against the same pixel's own dry-season baseline:

    drop_dB = baseline_VV_dB - flood_VV_dB
    flood   = (drop_dB > DROP_THRESH) AND (flood_VV_dB < ABS_MAX_DB)

Referencing each pixel to itself cancels out permanent surface type — a
smooth road is smooth in both epochs and produces no drop, whereas
standing water on it causes a large specular drop. Physical plausibility
masks (permanent water, steep slope, high HAND) then remove residual
speckle.

ACID TEST
---------
Detected flood area must correlate POSITIVELY with rainfall.

USAGE
-----
    python -m scripts.dev.test_change_detection
"""
from __future__ import annotations

import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

# Backscatter drop (dB) required to call a pixel newly-flooded.
DROP_THRESH_DB = 3.0
# Flood-epoch backscatter must also be low in absolute terms (open water is dark).
ABS_MAX_DB = -14.0

# Storm windows: (label, dry-baseline start/end, flood-epoch start/end, seasonal rain mm)
# Flood epochs bracket the documented peak of each season's flooding.
EVENTS = [
    ("2024_long_rains", "2024-01-01", "2024-03-01", "2024-04-15", "2024-05-15", 837.8),
    ("2018_long_rains", "2018-01-01", "2018-03-01", "2018-04-15", "2018-05-15", 732.1),
    ("2020_long_rains", "2020-01-01", "2020-03-01", "2020-04-15", "2020-05-15", 514.6),
    ("2025_long_rains", "2025-01-01", "2025-03-01", "2025-04-15", "2025-05-15", 441.3),
    ("2015_long_rains", "2015-01-01", "2015-03-01", "2015-04-15", "2015-05-15", 344.9),
    ("2016_long_rains", "2016-01-01", "2016-03-01", "2016-04-15", "2016-05-15", 324.7),
    ("2021_long_rains", "2021-01-01", "2021-03-01", "2021-04-15", "2021-05-15", 324.1),
    ("2023_long_rains", "2023-01-01", "2023-03-01", "2023-04-15", "2023-05-15", 283.6),
    ("2022_long_rains", "2022-01-01", "2022-03-01", "2022-04-15", "2022-05-15", 203.4),
    ("2017_long_rains", "2017-01-01", "2017-03-01", "2017-04-15", "2017-05-15", 185.9),
    ("2019_long_rains", "2019-01-01", "2019-03-01", "2019-04-15", "2019-05-15", 188.7),
]


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
    geom = ee.Geometry.BBox(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)

    # Change detection is only valid between scenes sharing the SAME viewing
    # geometry — backscatter depends strongly on incidence angle, so mixing
    # orbits injects apparent "drops" that are pure geometry. Availability
    # check over Nairobi: ASCENDING relative orbits 57 and 130 are present in
    # every year 2017-2025 (15-16 scenes/season), whereas DESCENDING orbit 79
    # is missing entirely for 2022-2024. Pin to ASCENDING orbit 57.
    RELATIVE_ORBIT = 57
    base_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", RELATIVE_ORBIT))
        .select("VV")
    )

    # Physical plausibility masks — water cannot stand on a cliff, and
    # permanent rivers/dams are not "new" flooding.
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hand = merit.select("hnd")
    permanent = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    slope_deg = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    plausible = hand.lt(15).And(slope_deg.lt(5)).And(permanent.lt(20))

    print(f"Change detection: drop > {DROP_THRESH_DB} dB AND flood VV < {ABS_MAX_DB} dB")
    print("Physical mask: HAND < 15 m, slope < 5 deg, JRC permanent water < 20%\n")
    print(f"{'event':<20}{'rain_mm':>9}{'base_n':>8}{'flood_n':>8}{'flood_km2':>11}")

    rows = []
    for label, b0, b1, f0, f1, rain_mm in EVENTS:
        baseline = base_col.filterDate(b0, b1)
        flood = base_col.filterDate(f0, f1)
        nb = baseline.size().getInfo()
        nf = flood.size().getInfo()
        if nb == 0 or nf == 0:
            print(f"{label:<20}{rain_mm:>9.1f}{nb:>8}{nf:>8}  (insufficient scenes)")
            continue

        # Median in dB space; S1_GRD is already log-scaled.
        base_db = baseline.median()
        flood_db = flood.median()
        drop = base_db.subtract(flood_db)

        flooded = drop.gt(DROP_THRESH_DB).And(flood_db.lt(ABS_MAX_DB)).And(plausible)

        area_km2 = (
            flooded.multiply(ee.Image.pixelArea())
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, scale=30, maxPixels=1e10)
            .get("VV")
            .getInfo()
        )
        area_km2 = (area_km2 or 0) / 1e6
        rows.append((label, rain_mm, area_km2))
        print(f"{label:<20}{rain_mm:>9.1f}{nb:>8}{nf:>8}{area_km2:>11.3f}")

    if len(rows) >= 3:
        from scipy.stats import pearsonr, spearmanr
        r = np.array([x[1] for x in rows])
        a = np.array([x[2] for x in rows])
        print()
        print("ACID TEST — detected flood area vs seasonal rainfall:")
        print("  Pearson  r=%+.3f p=%.3f" % pearsonr(r, a))
        print("  Spearman r=%+.3f p=%.3f" % spearmanr(r, a))
        print("  (old absolute-threshold pipeline: Spearman r=-0.738 — inverted)")


if __name__ == "__main__":
    main()
