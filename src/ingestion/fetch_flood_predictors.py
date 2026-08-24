"""
src.ingestion.fetch_flood_predictors
=====================================
Fetch the physically-meaningful flood predictors that the original
terrain stack (DEM / slope / TWI) was missing.

WHY THIS EXISTS
---------------
Diagnostic work on the trained surrogate found the model's input carried
NO spatial information beyond a single static terrain stack: the rainfall
channel was one scalar broadcast across all 198x252 pixels (CHIRPS p25 is
0.25deg ~ 28 km, so the whole of Nairobi fell inside a single rainfall
pixel), and the terrain channels were byte-identical across every sample.
The model could therefore only ever express `fixed_pattern x f(rainfall)`
and was structurally incapable of learning WHERE floods occur.

This module adds the predictors that actually drive urban flooding:

  hand        MERIT Hydro Height Above Nearest Drainage (metres).
              THE standard terrain flood-susceptibility predictor -- low
              HAND means "close to, and barely above, a drainage line",
              which is where water goes. Replaces the hand-placed
              Gaussian confluence bumps in src.models.predict with a real,
              globally-validated hydrographic quantity.
  upa         MERIT Hydro upstream drainage area (km^2). Real flow
              accumulation: how much catchment drains through this pixel.
  permanent   JRC Global Surface Water occurrence (%). Lets the model (and
              the evaluation) separate genuine flooding from rivers/dams
              that are wet in every single scene -- previously the largest
              source of false "flood" pixels.
  built       ESA WorldCover built-up fraction. Urban flood response is
              dominated by impervious surface; this was entirely absent.

All outputs are written on the EXACT src.grid_config prediction grid so
they can be stacked directly with the existing terrain features.

USAGE
-----
    python -m src.ingestion.fetch_flood_predictors
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

OUT_DIR = Path("data/processed/arrays")
NATIVE_SCALE_M = 90  # MERIT Hydro native resolution; JRC/WorldCover are finer and get aggregated


def _init_gee():
    import ee
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    project = os.environ.get("GEE_PROJECT_ID") or os.environ.get("GEE_PROJECT")
    ee.Initialize(project=project)
    return ee


def _download_to_grid(ee, image, band_name: str, scale_m: int) -> np.ndarray:
    """Download one GEE band over the Nairobi window and resample onto the prediction grid."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    geom = ee.Geometry.BBox(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)
    url = image.getDownloadURL({
        "region": geom, "scale": scale_m, "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    dst = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    dst_transform = from_bounds(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, GRID_W, GRID_H)
    with MemoryFile(resp.content) as mem, mem.open() as src:
        # Resampling.average: these are continuous/fractional quantities, and
        # averaging preserves sparse positive signal when downsampling, which
        # nearest-neighbour silently discards.
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
    return np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    ee = _init_gee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[GRID ] {GRID_H}x{GRID_W}  lat {LAT_SOUTH}..{LAT_NORTH}  lon {LON_WEST}..{LON_EAST}")

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    layers: dict[str, tuple] = {
        # Height Above Nearest Drainage — primary flood-susceptibility predictor
        "hand": (merit.select("hnd"), NATIVE_SCALE_M),
        # Upstream drainage area (km^2) — real flow accumulation
        "upa": (merit.select("upa"), NATIVE_SCALE_M),
        # JRC permanent-water occurrence 0-100%
        "permanent_water": (
            ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0), 30),
        # ESA WorldCover class 50 == Built-up; average of the binary mask
        # over the coarser grid cell gives a built-up FRACTION per pixel.
        "built_up": (
            ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(50).toFloat(), 30),
    }

    results: dict[str, np.ndarray] = {}
    for name, (img, scale_m) in layers.items():
        print(f"[FETCH] {name} @ {scale_m}m ...", flush=True)
        try:
            arr = _download_to_grid(ee, img, name, scale_m)
        except Exception as exc:
            print(f"[FAIL ] {name}: {exc}")
            continue
        results[name] = arr
        finite = arr[np.isfinite(arr)]
        print(f"[OK   ] {name}: min={finite.min():.3f} max={finite.max():.3f} "
              f"mean={finite.mean():.3f} nonzero={100 * (arr != 0).mean():.1f}%")

    if not results:
        print("[FATAL] no predictor layers fetched.")
        sys.exit(1)

    for name, arr in results.items():
        out = OUT_DIR / f"predictor_{name}.npy"
        np.save(out, arr)
        print(f"[SAVE ] {out}")

    print(f"\nFetched {len(results)}/{len(layers)} predictor layers.")


if __name__ == "__main__":
    main()
