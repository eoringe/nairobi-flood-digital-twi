"""
src.ingestion.build_population_grid
===================================
Population per prediction-grid cell, from WorldPop, for "people at risk".

WHY
---
The dashboard estimated exposure as flooded km2 x 6,300 people/km2, the
county-wide average. Nairobi's density varies by more than two orders of
magnitude: an informal settlement along the Nairobi River can hold tens of
thousands of people per km2, an industrial estate almost none. Flooding
concentrates along exactly the dense river corridors, so the average
systematically understates exposure where it matters most.

SOURCE
------
WorldPop Global2 R2025A, Kenya, 2025, 100 m, *constrained*: people are
allocated only to cells with mapped buildings, which suits an urban area.
https://www.worldpop.org  (CC BY 4.0). The 2025 figure is a projection from the
most recent census (2019), not a count.

METHOD
------
WorldPop pixels (~92 m) are larger than prediction cells (~67 m x 79 m), so the
raster is resampled by area-weighted averaging of *density* onto the grid, then
multiplied by cell area. Summing pixels by centre instead would leave many grid
cells empty and double others. The total inside the grid is printed as a check.

OUTPUT
------
data/processed/arrays/population_grid_worldpop2025.npy   (GRID_H, GRID_W) float32, people per cell

USAGE
-----
    python -m src.ingestion.build_population_grid
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from loguru import logger

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/KEN/v1/100m/"
       "constrained/ken_pop_2025_CN_100m_R2025A_v1.tif")
RAW = Path("data/raw/worldpop/ken_pop_2025_CN_100m_R2025A_v1.tif")
OUT = Path("data/processed/arrays/population_grid_worldpop2025.npy")
META = OUT.with_suffix(".json")


def main() -> None:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading WorldPop Kenya 2025 (~70 MB) -> {RAW}")
        urllib.request.urlretrieve(URL, RAW)

    dst_transform = from_bounds(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, GRID_W, GRID_H)
    with rasterio.open(RAW) as src:
        # Read a window a little larger than the grid, with nodata as zero
        # people: in a constrained product nodata means "no buildings here".
        window = rasterio.windows.from_bounds(LON_WEST - 0.01, LAT_SOUTH - 0.01,
                                              LON_EAST + 0.01, LAT_NORTH + 0.01, src.transform)
        people = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float64)
        if src.nodata is not None:
            people[people == src.nodata] = 0.0
        people[~np.isfinite(people) | (people < 0)] = 0.0
        src_transform = src.window_transform(window)
        src_crs = src.crs

    src_px_area = abs(src_transform.a * src_transform.e)            # deg2 per source pixel
    dst_px_area = abs(dst_transform.a * dst_transform.e)            # deg2 per grid cell

    density = people / src_px_area                                   # people per deg2
    dst = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    reproject(density, dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs="EPSG:4326", resampling=Resampling.average)
    grid = (dst * dst_px_area).astype(np.float32)

    # Check: the resampled total should match the raw pixels inside the grid.
    inside = rasterio.windows.from_bounds(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, src_transform)
    r0, c0 = int(round(inside.row_off)), int(round(inside.col_off))
    raw_total = float(people[r0:r0 + int(round(inside.height)), c0:c0 + int(round(inside.width))].sum())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, grid)
    cell_km2 = 0.067 * 0.079
    META.write_text(json.dumps({
        "source": "WorldPop Global2 R2025A, Kenya 2025, 100 m constrained (CC BY 4.0)",
        "url": URL,
        "units": "people per grid cell",
        "grid_total_people": float(grid.sum()),
        "raw_pixels_total_people": raw_total,
        "max_density_per_km2": float(grid.max() / cell_km2),
    }, indent=2))
    logger.info(f"Population grid: {grid.sum():,.0f} people in the grid "
                f"(raw pixels in same window: {raw_total:,.0f}); densest cell "
                f"{grid.max() / cell_km2:,.0f} people/km2; populated cells {(grid > 0).mean():.0%} -> {OUT}")


if __name__ == "__main__":
    main()
