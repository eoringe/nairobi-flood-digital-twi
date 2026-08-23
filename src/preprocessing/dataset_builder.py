"""
src.preprocessing.dataset_builder
===================================
Nairobi Urban Flood Digital Twin — Dataset Builder & Matrix Assembler

PURPOSE
-------
1. Crop the SRTM mosaic (DEM/Slope/TWI) from its full extent down to the
   Nairobi prediction window defined in src.grid_config, THEN resample
   to the fixed (GRID_H, GRID_W) grid — previously this resized the
   *entire* ~2deg x 2deg mosaic straight to (198, 252), which put terrain
   for a region stretching well beyond Nairobi into the same pixel
   indices src.models.predict treats as the tight Nairobi core window.
2. Build real, spatiotemporally diverse training targets from the
   Sentinel-1 SAR water-mask composites already downloaded by
   src.ingestion.fetch_sentinel_targets (previously unused) instead of a
   purely synthetic rainfall*TWI formula. Real observed flood extent
   gates *where* water is predicted; the terrain physics formula (same
   family used in src.models.predict) supplies *how deep* within that
   observed extent, scaled by the real CHIRPS rainfall for that date.
3. Hold out one independent extreme-scenario test case built from the
   ICPAC 100-year flood consensus raster — never used for training or
   hyperparameter selection, only for final evaluation under an extreme
   return-period scenario (src.models.train reports metrics on it).

MEMORY CONTRACT
---------------
Uses float32 numpy arrays. Peak RAM stays low because the SRTM mosaic is
cropped to the small Nairobi window before any per-sample work begins.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine, from_bounds
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

DEFAULT_PROCESSED_DIR = Path("data/processed/arrays")
DEFAULT_S1_DIR = Path("data/raw/vectors/nairobi_s1_temporal_targets")
DEFAULT_ICPAC_TIF = Path("data/raw/vectors/nairobi_flood_100yr.tif")

TARGET_HEIGHT = GRID_H
TARGET_WIDTH = GRID_W
SEQ_LEN = 7

#: Storm season month windows — must match src.ingestion.fetch_sentinel_targets
#: STORM_WINDOWS so filenames like s1_water_mask_2022_long_rains.tif resolve
#: to the same calendar window the composite was actually built from.
STORM_WINDOWS = {
    "long_rains": (3, 5),    # March - May
    "short_rains": (10, 12), # October - December
}

#: Reference rate the physics-shape scale is normalized against. Pre-existing
#: minor mismatch left as-is (changing it would rescale the training target
#: and require a retrain): the dashboard's "10-Year" preset button actually
#: sets the slider to 40mm/day (src.dashboard.callbacks), not 45.
RAIN_REF_MM_DAY = 45.0
DEPTH_CAP_M = 4.5
DEPTH_ZERO_FLOOR_M = 0.2

DST_TRANSFORM = from_bounds(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH, GRID_W, GRID_H) if _RASTERIO_OK else None
DST_CRS = "EPSG:4326"


# ============================================================================
# GRID RESAMPLING HELPERS
# ============================================================================

def _resample_array_to_grid(
    array: np.ndarray,
    src_transform: "Affine",
    resampling: "Resampling",
    src_nodata: float | None = None,
) -> np.ndarray:
    """Reproject/resample an in-memory array onto the shared Nairobi grid."""
    dst = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
    reproject(
        source=array.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=DST_CRS,
        src_nodata=src_nodata,
        dst_transform=DST_TRANSFORM,
        dst_crs=DST_CRS,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def _resample_file_to_grid(path: Path, resampling: "Resampling") -> np.ndarray | None:
    """Reproject/resample band 1 of a GeoTIFF onto the shared Nairobi grid."""
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        dst = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=DST_TRANSFORM,
            dst_crs=DST_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def _load_and_crop_terrain(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the full-mosaic DEM/Slope/TWI arrays to the Nairobi window and
    resample onto (GRID_H, GRID_W), then min-max normalize each to [0, 1].
    """
    dem_path = processed_dir / "dem_mosaic.npy"
    profile_path = processed_dir / "dem_mosaic_profile.npz"
    slope_path = processed_dir / "slope_nairobi.npy"
    twi_path = processed_dir / "twi_nairobi.npy"

    if not dem_path.exists() or not profile_path.exists():
        raise FileNotFoundError(f"{dem_path} / {profile_path} not found. Run srtm_mosaic first!")

    profile = np.load(profile_path, allow_pickle=True)
    src_transform = Affine(*[float(v) for v in profile["transform"][:6]])

    dem_full = np.load(dem_path, mmap_mode="r")
    logger.info(f"Full mosaic shape: {dem_full.shape} (covers well beyond Nairobi — cropping now)")

    dem_grid = _resample_array_to_grid(np.asarray(dem_full), src_transform, Resampling.average, src_nodata=np.nan)

    if slope_path.exists():
        slope_full = np.load(slope_path, mmap_mode="r")
        slope_grid = _resample_array_to_grid(np.asarray(slope_full), src_transform, Resampling.average, src_nodata=np.nan)
    else:
        slope_grid = np.zeros_like(dem_grid)

    if twi_path.exists():
        twi_full = np.load(twi_path, mmap_mode="r")
        twi_grid = _resample_array_to_grid(np.asarray(twi_full), src_transform, Resampling.average, src_nodata=np.nan)
    else:
        twi_grid = np.zeros_like(dem_grid)

    # Any residual NaN (nodata slivers at the crop edge) filled with the local mean
    for grid in (dem_grid, slope_grid, twi_grid):
        if np.isnan(grid).any():
            grid[np.isnan(grid)] = np.nanmean(grid)

    def _norm(a: np.ndarray) -> np.ndarray:
        return ((a - a.min()) / (a.max() - a.min() + 1e-6)).astype(np.float32)

    logger.info(
        f"Cropped Nairobi terrain window: DEM range [{dem_grid.min():.0f}, {dem_grid.max():.0f}] m "
        f"(bounds lat [{LAT_SOUTH},{LAT_NORTH}] lon [{LON_WEST},{LON_EAST}])"
    )
    return _norm(dem_grid), _norm(slope_grid), _norm(twi_grid)


# ============================================================================
# RAINFALL SERIES (real calendar dates)
# ============================================================================

def _parse_rainfall_dates(raw_dates: list[str]) -> list[date]:
    """
    Parse the date labels written by src.ingestion.fetch_chirps.

    xarray is not installed in this environment, so that script's netCDF4
    fallback path runs, which labels each day '{year}-day-{ordinal}' instead
    of an ISO date. Day-of-year counts per year are still correct (365/366),
    so real calendar dates are reconstructed here rather than treating the
    series as unordered.
    """
    parsed = []
    for d in raw_dates:
        m = re.match(r"^(\d{4})-day-(\d{1,3})$", d)
        if m:
            year, doy = int(m.group(1)), int(m.group(2))
            parsed.append(date(year, 1, 1) + timedelta(days=doy - 1))
        else:
            parsed.append(date.fromisoformat(d))
    return parsed


def _load_rainfall_series(processed_dir: Path) -> tuple[list[date], np.ndarray]:
    rain_path = processed_dir / "rainfall_daily_mean.npy"
    dates_path = processed_dir / "rainfall_dates.json"

    if not rain_path.exists() or not dates_path.exists():
        logger.warning("No real rainfall series found — synthetic fallback will be used.")
        rng = np.random.default_rng(42)
        values = rng.exponential(scale=12.0, size=2 * 365).astype(np.float32)
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
        return dates, values

    values = np.load(rain_path).astype(np.float32)
    raw_dates = json.load(open(dates_path))
    dates = _parse_rainfall_dates(raw_dates)
    return dates, values


# ============================================================================
# SENTINEL-1 EVENT DISCOVERY
# ============================================================================

def _discover_s1_events(s1_dir: Path) -> list[dict]:
    """Find s1_water_mask_<year>_<season>.tif files and resolve their season window."""
    events = []
    if not s1_dir.exists():
        return events
    for tif in sorted(s1_dir.glob("s1_water_mask_*.tif")):
        m = re.match(r"s1_water_mask_(\d{4})_(long_rains|short_rains)\.tif$", tif.name)
        if not m:
            continue
        year, season = int(m.group(1)), m.group(2)
        month_start, month_end = STORM_WINDOWS[season]
        season_start = date(year, month_start, 1)
        # end-of-month for month_end
        if month_end == 12:
            season_end = date(year, 12, 31)
        else:
            season_end = date(year, month_end + 1, 1) - timedelta(days=1)
        events.append({
            "year": year, "season": season, "path": tif,
            "start": season_start, "end": season_end,
        })
    return events


# ============================================================================
# PHYSICS-SHAPE (spatial magnitude prior — no hardcoded confluence points)
# ============================================================================

def _physics_shape(dem_norm: np.ndarray, slope_norm: np.ndarray, twi_norm: np.ndarray) -> np.ndarray:
    """
    Same TWI/slope/DEM family used by src.models.predict's ensemble physics
    term, deliberately WITHOUT the hardcoded confluence-node blending —
    training targets should come from real observed extent + terrain, not
    from the same manual coordinates the model is meant to learn to
    reproduce. Normalized to [0, 1].
    """
    slope_safe = np.maximum(slope_norm, 0.02)
    raw = (twi_norm ** 1.2) / (slope_safe ** 0.5) * np.exp(-2.0 * dem_norm)
    peak = raw.max()
    return (raw / peak).astype(np.float32) if peak > 0 else raw.astype(np.float32)


# ============================================================================
# SAMPLE ASSEMBLY
# ============================================================================

def _rain_window(dates: list[date], values: np.ndarray, end_day: date, seq_len: int) -> np.ndarray | None:
    """Return the seq_len-day rainfall sequence ending on end_day, or None if unavailable."""
    try:
        end_idx = dates.index(end_day)
    except ValueError:
        return None
    start_idx = end_idx - seq_len + 1
    if start_idx < 0:
        return None
    return values[start_idx:end_idx + 1]


def _build_event_samples(
    event: dict,
    extent_mask: np.ndarray,
    phys_shape: np.ndarray,
    dates: list[date],
    rain_values: np.ndarray,
    stride_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Slide a SEQ_LEN-day rainfall window across the event's storm season,
    pairing each real 7-day rainfall sequence with a target depth built from
    the (fixed, real) observed extent for that season and a magnitude scaled
    by that window's actual rainfall intensity.
    """
    samples = []
    day = event["start"]
    while day <= event["end"]:
        rain_seq = _rain_window(dates, rain_values, day, SEQ_LEN)
        if rain_seq is not None:
            effective_rate = float(np.mean(rain_seq))  # mm/day over the trailing week
            scale = (max(effective_rate, 0.0) / RAIN_REF_MM_DAY) ** 0.9
            depth = phys_shape * 2.5 * scale * extent_mask
            depth = np.clip(depth, 0.0, DEPTH_CAP_M).astype(np.float32)
            depth[depth < DEPTH_ZERO_FLOOR_M] = 0.0
            samples.append((rain_seq.astype(np.float32), depth))
        day += timedelta(days=stride_days)
    return samples


def build_training_dataset(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    s1_dir: Path = DEFAULT_S1_DIR,
    icpac_tif: Path = DEFAULT_ICPAC_TIF,
    seq_len: int = SEQ_LEN,
    stride_days: int = 3,
) -> None:
    if not _RASTERIO_OK:
        logger.error("rasterio is required to build the dataset. pip install rasterio>=1.4")
        return

    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Dataset Assembly Pipeline (real Sentinel-1 / ICPAC targets)...")

    dem_norm, slope_norm, twi_norm = _load_and_crop_terrain(processed_dir)
    static_features = np.stack([dem_norm, slope_norm, twi_norm], axis=0).astype(np.float32)
    np.save(processed_dir / "static_terrain_features.npy", static_features)
    logger.info(f"Saved cropped static terrain features: shape={static_features.shape}")

    phys_shape = _physics_shape(dem_norm, slope_norm, twi_norm)

    dates, rain_values = _load_rainfall_series(processed_dir)

    events = _discover_s1_events(s1_dir)
    if not events:
        logger.error(
            f"No Sentinel-1 water-mask GeoTIFFs found in {s1_dir}. "
            "Run `python -m src.ingestion.fetch_sentinel_targets` first."
        )
        return
    logger.info(f"Discovered {len(events)} real Sentinel-1 storm-season events.")

    X_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    sample_event_ids: list[str] = []
    event_log = []

    for event in events:
        event_id = f"{event['year']}_{event['season']}"
        mask_raw = _resample_file_to_grid(event["path"], Resampling.average)
        if mask_raw is None:
            logger.warning(f"Could not read {event['path']}, skipping.")
            continue
        extent_mask = np.nan_to_num(mask_raw, nan=0.0).astype(np.float32)
        extent_mask = np.clip(extent_mask, 0.0, 1.0)

        event_samples = _build_event_samples(event, extent_mask, phys_shape, dates, rain_values, stride_days)
        for rain_seq, target_depth in event_samples:
            seq_frames = []
            for r_val in rain_seq:
                rain_grid = np.full((1, GRID_H, GRID_W), r_val / 150.0, dtype=np.float32)
                seq_frames.append(np.concatenate([rain_grid, static_features], axis=0))
            X_samples.append(np.stack(seq_frames, axis=0))
            y_samples.append(target_depth)
            sample_event_ids.append(event_id)

        event_log.append({
            "year": event["year"], "season": event["season"],
            "n_samples": len(event_samples),
            "wet_pixel_pct": round(float((extent_mask > 0.1).mean() * 100), 2),
        })
        logger.info(
            f"  {event['year']} {event['season']}: {len(event_samples)} samples, "
            f"{event_log[-1]['wet_pixel_pct']}% wet extent"
        )

    if not X_samples:
        logger.error("No training samples could be assembled from the Sentinel-1 events.")
        return

    # Shuffle sample order (events are contiguous otherwise). event_ids.npy
    # is saved alongside so src.models.train can split by EVENT rather than
    # by sample — many samples share the same underlying S1 extent (only
    # their rainfall window differs), so a random per-sample split would
    # leak the same spatial pattern across train/val/test.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(X_samples))
    X_arr = np.array(X_samples, dtype=np.float32)[order]
    y_arr = np.array(y_samples, dtype=np.float32)[order]
    event_ids_arr = np.array(sample_event_ids)[order]

    np.save(processed_dir / "X_train.npy", X_arr)
    np.save(processed_dir / "y_train.npy", y_arr)
    np.save(processed_dir / "event_ids.npy", event_ids_arr)
    logger.info(f"Saved X_train {X_arr.shape} / y_train {y_arr.shape} from {len(events)} real S1 events.")

    # ---- Held-out extreme-scenario test case (ICPAC 100-yr consensus) -----
    # The WCS layer's documented "0-6 multi-model agreement" scale doesn't
    # match what the service actually returns (observed range up to ~80,
    # varying by request) — normalizing by the resampled tile's own max
    # is robust to that regardless of the true underlying unit.
    icpac_grid = _resample_file_to_grid(icpac_tif, Resampling.average)
    if icpac_grid is not None and np.isfinite(icpac_grid).any() and np.nanmax(icpac_grid) > 0:
        agreement = np.nan_to_num(icpac_grid, nan=0.0)
        confidence = np.clip(agreement / agreement.max(), 0.0, 1.0)
        rain_100yr = np.linspace(135.0 * 0.2, 135.0, seq_len, dtype=np.float32)  # 100-yr preset, 135mm/day
        scale = (135.0 / RAIN_REF_MM_DAY) ** 0.9
        depth_100yr = np.clip(phys_shape * 2.5 * scale * confidence, 0.0, DEPTH_CAP_M).astype(np.float32)
        depth_100yr[depth_100yr < DEPTH_ZERO_FLOOR_M] = 0.0

        seq_frames = [
            np.concatenate([np.full((1, GRID_H, GRID_W), r / 150.0, dtype=np.float32), static_features], axis=0)
            for r in rain_100yr
        ]
        X_100yr = np.array([np.stack(seq_frames, axis=0)], dtype=np.float32)
        y_100yr = np.array([depth_100yr], dtype=np.float32)
        np.save(processed_dir / "X_test_100yr.npy", X_100yr)
        np.save(processed_dir / "y_test_100yr.npy", y_100yr)
        logger.info(
            f"Saved held-out 100-year extreme-scenario test case from ICPAC consensus raster "
            f"(never used in training or HPO). Wet extent: {(depth_100yr > 0).mean() * 100:.1f}%"
        )
    else:
        logger.warning(
            f"ICPAC raster at {icpac_tif} has no valid data in the Nairobi window (upstream WCS "
            f"issue — see MEMORY_CONSTRAINTS/§1.7.1's external-API dependency limitation) — "
            f"skipping the extreme-scenario test case."
        )

    grid_meta = {
        "target_height": GRID_H,
        "target_width": GRID_W,
        "sequence_length": seq_len,
        "num_samples": int(X_arr.shape[0]),
        "channels": ["rain", "dem", "slope", "twi"],
        "grid_bounds": {"lat_north": LAT_NORTH, "lat_south": LAT_SOUTH, "lon_west": LON_WEST, "lon_east": LON_EAST},
        "target_source": "sentinel1_water_mask_extent x terrain_physics_depth x chirps_rainfall_scale",
        "contributing_events": event_log,
        "held_out_test_case": "icpac_100yr_flood_consensus" if icpac_grid is not None else None,
    }
    with open(processed_dir / "dataset_metadata.json", "w") as f:
        json.dump(grid_meta, f, indent=2)

    logger.info(f"Dataset assembly complete. {X_arr.shape[0]} real-event samples across {len(events)} storm seasons.")


if __name__ == "__main__":
    build_training_dataset()
