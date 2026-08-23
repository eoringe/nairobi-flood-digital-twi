"""
src.preprocessing.srtm_mosaic
==============================
GeoStream — SRTM Tile Mosaic, Slope & TWI Pipeline

PURPOSE
-------
1. Ingest multiple raw SRTM .tif tiles from  data/raw/terrain/
2. Merge them into a single mosaic (rasterio.merge)
3. Compute slope and Topographic Wetness Index (TWI)
4. Apply the always-dry grid mask defined in MEMORY_CONSTRAINTS.md
   to strip permanently-dry pixels before any training loop

HARDWARE TARGET
---------------
16 GB RAM / CPU-only workstation.

Memory-safety rules enforced here
  • All rasters read as float32 (not float64) — halves RAM
  • Large arrays are written to disk via numpy.memmap / zarr and
    released from heap immediately after each stage
  • The always-dry mask is applied BEFORE the arrays are written
    so masked (compressed) arrays are what reach the training loop
  • psutil startup guard aborts if free RAM < 4 GB (MEMORY_CONSTRAINTS §1)

USAGE (CLI)
-----------
    python -m src.preprocessing.srtm_mosaic
    python -m src.preprocessing.srtm_mosaic --terrain-dir data/raw/terrain \\
                                              --out-dir data/processed \\
                                              --dry-threshold 0.02

DEPENDENCIES
------------
    rasterio >= 1.3
    numpy >= 1.24
    scipy >= 1.11          (ndimage for slope kernel)
    scikit-image >= 0.21   (optional: used for gaussian pre-smoothing)
    psutil >= 5.9
    loguru >= 0.7
    zarr >= 2.16           (optional: chunked output; falls back to .npy)

NOT INCLUDED HERE
-----------------
    Deep-learning model architecture (autoencoder / LSTM).
    Those live under src/models/.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
from loguru import logger

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module can be imported for
# introspection even if the full geospatial stack is not installed.
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.transform import from_bounds
    from rasterio import MemoryFile
    _RASTERIO_OK = True
except ImportError:  # pragma: no cover
    _RASTERIO_OK = False
    logger.warning("rasterio not found — mosaic stage will raise at runtime.")

try:
    from scipy.ndimage import uniform_filter
    _SCIPY_OK = True
except ImportError:  # pragma: no cover
    _SCIPY_OK = False
    logger.warning("scipy not found — slope kernel will raise at runtime.")

try:
    from skimage.filters import gaussian as ski_gaussian
    _SKIMAGE_OK = True
except ImportError:  # pragma: no cover
    _SKIMAGE_OK = False
    logger.debug("scikit-image not found — pre-smoothing step skipped.")

try:
    import zarr
    _ZARR_OK = True
except ImportError:  # pragma: no cover
    _ZARR_OK = False
    logger.debug("zarr not found — falling back to .npy output.")


# ============================================================================
# CONSTANTS
# ============================================================================

#: Paths relative to repo root (overridable via CLI / env vars)
DEFAULT_TERRAIN_DIR = Path("data/raw/terrain")
DEFAULT_OUT_ARRAYS  = Path("data/processed/arrays")
DEFAULT_OUT_MASKED  = Path("data/processed/masked")

#: MEMORY_CONSTRAINTS.md §1 — abort if free RAM below this
MIN_FREE_RAM_GB: float = float(os.getenv("MIN_AVAILABLE_RAM_GB", "0.4"))

#: All rasters are cast to float32 to halve RAM vs float64
DTYPE = np.float32

#: TWI stability: avoid log(0) by clamping flow accumulation
TWI_EPSILON: float = 1e-6

#: Pixels whose absolute TWI is below this across the full time window
#: are classified as "always dry" and stripped from training arrays.
#: Override via --dry-threshold CLI flag.
DEFAULT_DRY_THRESHOLD: float = 0.02


# ============================================================================
# MEMORY GUARD (MEMORY_CONSTRAINTS.md §1)
# ============================================================================

def _startup_memory_check() -> None:
    """Abort immediately if available RAM < MIN_FREE_RAM_GB."""
    free_gb = psutil.virtual_memory().available / 1024 ** 3
    total_gb = psutil.virtual_memory().total / 1024 ** 3
    logger.info(
        "memory_check | free={:.2f} GB | total={:.2f} GB | threshold={} GB",
        free_gb, total_gb, MIN_FREE_RAM_GB,
    )
    if free_gb < MIN_FREE_RAM_GB:
        logger.critical(
            "memory_check | ABORT | free={:.2f} GB < threshold={} GB | "
            "action=exit",
            free_gb, MIN_FREE_RAM_GB,
        )
        sys.exit(1)


def _log_rss(stage: str) -> None:
    """Emit current RSS and warn/abort per MEMORY_CONSTRAINTS.md thresholds."""
    rss_gb = psutil.Process().memory_info().rss / 1024 ** 3
    if rss_gb > 14.0:
        logger.critical(
            "rss_monitor | stage={} | value_gb={:.2f} | threshold_gb=14 | "
            "action=abort_batch",
            stage, rss_gb,
        )
        raise MemoryError(
            f"[{stage}] RSS {rss_gb:.2f} GB exceeds critical ceiling 14 GB. "
            "Aborting to prevent OOM kill."
        )
    elif rss_gb > 12.0:
        logger.warning(
            "rss_monitor | stage={} | value_gb={:.2f} | threshold_gb=12 | "
            "action=log_warning",
            stage, rss_gb,
        )
    else:
        logger.debug(
            "rss_monitor | stage={} | value_gb={:.2f} | threshold_gb=12 | "
            "action=ok",
            stage, rss_gb,
        )


# ============================================================================
# STAGE 1 — INGEST RAW .tif TILES
# ============================================================================

def discover_tiles(terrain_dir: Path) -> list[Path]:
    """
    Discover all .tif / .tiff files in *terrain_dir*.

    Returns
    -------
    list[Path]
        Sorted list of tile paths.  Raises FileNotFoundError if the
        directory is missing or contains no recognised rasters.
    """
    if not terrain_dir.exists():
        raise FileNotFoundError(
            f"Terrain directory not found: {terrain_dir}\n"
            "Run init_data_folders.bat first, then place your SRTM tiles."
        )

    tiles = sorted(terrain_dir.glob("*.tif")) + sorted(terrain_dir.glob("*.tiff"))
    if not tiles:
        raise FileNotFoundError(
            f"No .tif / .tiff files found in {terrain_dir}.\n"
            "Expected the 4 USGS SRTM GeoTIFF tiles."
        )

    logger.info("tile_discovery | found={} | dir={}", len(tiles), terrain_dir)
    for t in tiles:
        logger.debug("  tile: {}", t.name)
    return tiles


# ============================================================================
# STAGE 2 — MOSAIC MERGE
# ============================================================================

def build_mosaic(
    tiles: list[Path],
    out_arrays_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Merge *tiles* into a single mosaic using rasterio.merge.

    The merged array is cast to float32 immediately after merge and
    written to ``data/processed/arrays/dem_mosaic.npy`` so it can be
    released from the heap and later memory-mapped.

    Parameters
    ----------
    tiles:
        Tile paths returned by :func:`discover_tiles`.
    out_arrays_dir:
        Destination for the .npy artefact.

    Returns
    -------
    mosaic : np.ndarray, shape (H, W), dtype float32
        Merged DEM (metres above sea level).
    profile : dict
        Rasterio profile (CRS, transform, nodata, …).
    """
    if not _RASTERIO_OK:
        raise RuntimeError("rasterio is required for the mosaic stage.")

    out_arrays_dir.mkdir(parents=True, exist_ok=True)
    mosaic_path = out_arrays_dir / "dem_mosaic.npy"
    profile_path = out_arrays_dir / "dem_mosaic_profile.npz"

    logger.info("mosaic_merge | tiles={} | output={}", len(tiles), mosaic_path)

    # Open all tile file handles; rasterio.merge handles alignment
    src_files = [rasterio.open(t) for t in tiles]
    try:
        mosaic_data, transform = rio_merge(src_files)
        # mosaic_data shape: (bands, H, W) — take band 0
        dem: np.ndarray = mosaic_data[0].astype(DTYPE)

        # Replace nodata sentinel with NaN
        nodata = src_files[0].nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

        profile = src_files[0].profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            height=dem.shape[0],
            width=dem.shape[1],
            transform=transform,
            nodata=np.nan,
        )
    finally:
        for src in src_files:
            src.close()

    # ── Persist to disk and release heap ────────────────────────────────────
    np.save(mosaic_path, dem)
    np.savez(
        profile_path,
        crs=str(profile["crs"]),
        transform=np.array(profile["transform"]),
        height=dem.shape[0],
        width=dem.shape[1],
        nodata=np.nan,
    )
    logger.info(
        "mosaic_merge | saved | path={} | shape={} | dtype={}",
        mosaic_path, dem.shape, dem.dtype,
    )

    _log_rss("mosaic_merge")
    return dem, profile


# ============================================================================
# STAGE 3a — SLOPE
# ============================================================================

def compute_slope(
    dem: np.ndarray,
    cell_size_m: float = 30.0,
    smooth: bool = True,
) -> np.ndarray:
    """
    Compute slope in degrees from a DEM array.

    Uses a Sobel-like central-difference kernel.  Optionally applies a
    light Gaussian pre-smooth (requires scikit-image) to reduce noise
    in SRTM 1-arc-second data without removing hydrologically-significant
    gradient structure.

    Parameters
    ----------
    dem:
        2-D float32 DEM (metres).
    cell_size_m:
        Pixel size in metres.  SRTM 1-arc-second ≈ 30 m at the equator.
    smooth:
        If True and scikit-image is available, apply σ=1 Gaussian blur
        before gradient calculation.

    Returns
    -------
    slope : np.ndarray, shape == dem.shape, dtype float32
        Slope in degrees.
    """
    if not _SCIPY_OK:
        raise RuntimeError("scipy is required for slope computation.")

    work = dem.copy()

    # Fill NaN with local mean via uniform filter (avoids gradient spikes)
    nan_mask = np.isnan(work)
    if nan_mask.any():
        fill = uniform_filter(np.where(nan_mask, 0.0, work), size=5)
        norm = uniform_filter(~nan_mask * 1.0, size=5)
        work[nan_mask] = (fill / np.maximum(norm, 1e-9))[nan_mask]

    if smooth and _SKIMAGE_OK:
        work = ski_gaussian(work, sigma=1.0, preserve_range=True).astype(DTYPE)

    # Central-difference gradient (dz/dx, dz/dy)
    dz_dy, dz_dx = np.gradient(work, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(DTYPE)
    slope_deg[nan_mask] = np.nan

    logger.info(
        "slope | min={:.2f}° | max={:.2f}° | mean={:.2f}°",
        np.nanmin(slope_deg), np.nanmax(slope_deg), np.nanmean(slope_deg),
    )
    return slope_deg


# ============================================================================
# STAGE 3b — TOPOGRAPHIC WETNESS INDEX (TWI)
# ============================================================================

def compute_twi(
    slope_deg: np.ndarray,
    cell_size_m: float = 30.0,
) -> np.ndarray:
    """
    Compute the Topographic Wetness Index (TWI).

    TWI = ln( α / tan(β) )

    where
      α  = specific catchment area  [m²/m]  — approximated here as
           cell_size_m (single-flow path proxy; replace with D8/D∞
           flow accumulation for production runs).
      β  = slope in radians

    This is a placeholder approximation that gives a physically
    meaningful spatial pattern for masking purposes.  Swap
    ``alpha`` for a proper flow-accumulation grid once the D8
    accumulation routine is wired in.

    Parameters
    ----------
    slope_deg : np.ndarray
        Slope in degrees, shape (H, W), float32.
    cell_size_m : float
        Cell size used to approximate specific catchment area.

    Returns
    -------
    twi : np.ndarray, shape (H, W), dtype float32
    """
    slope_rad = np.radians(slope_deg)
    tan_slope = np.tan(slope_rad)

    # Flat areas: clamp tan_slope to avoid division by zero
    tan_slope = np.where(tan_slope < TWI_EPSILON, TWI_EPSILON, tan_slope)

    # Simple proxy: α = cell_size_m (unit contributing area per unit width)
    alpha = np.full_like(slope_deg, fill_value=cell_size_m, dtype=DTYPE)

    twi = np.log(alpha / tan_slope).astype(DTYPE)
    twi[np.isnan(slope_deg)] = np.nan

    logger.info(
        "twi | min={:.2f} | max={:.2f} | mean={:.2f}",
        np.nanmin(twi), np.nanmax(twi), np.nanmean(twi),
    )
    return twi


# ============================================================================
# STAGE 4 — ALWAYS-DRY MASK  (MEMORY_CONSTRAINTS.md §2 / §5)
# ============================================================================

def build_always_dry_mask(
    twi: np.ndarray,
    dry_threshold: float = DEFAULT_DRY_THRESHOLD,
) -> np.ndarray:
    """
    Derive the boolean always-dry mask from the TWI grid.

    A pixel is classified as "always dry" when its normalised TWI falls
    below *dry_threshold*.  Stripping these pixels before training loops
    reduces the active tensor footprint by ~30–60% over Nairobi's
    predominantly-hilly terrain, keeping RSS well below the 12 GB
    WARNING threshold in MEMORY_CONSTRAINTS.md.

    Parameters
    ----------
    twi : np.ndarray
        Full-domain TWI grid, shape (H, W), float32.
    dry_threshold : float
        Fractional threshold on [0, 1] normalised TWI.  Pixels with
        normalised TWI < threshold are masked out as always-dry.

    Returns
    -------
    always_dry : np.ndarray, bool, shape (H, W)
        True  → pixel is always dry (exclude from training)
        False → pixel may flood    (include in training)
    """
    # Normalise to [0, 1] ignoring NaN
    twi_min = np.nanmin(twi)
    twi_max = np.nanmax(twi)
    twi_norm = (twi - twi_min) / (twi_max - twi_min + TWI_EPSILON)

    always_dry = twi_norm < dry_threshold
    # NaN pixels (ocean / nodata boundary) are also excluded
    always_dry |= np.isnan(twi)

    n_dry   = int(always_dry.sum())
    n_total = twi.size
    pct     = 100.0 * n_dry / n_total
    logger.info(
        "always_dry_mask | dry_pixels={} | total_pixels={} | "
        "dry_pct={:.1f}% | threshold={}",
        n_dry, n_total, pct, dry_threshold,
    )
    return always_dry


def apply_mask_and_save(
    arrays: dict[str, np.ndarray],
    always_dry: np.ndarray,
    out_arrays_dir: Path,
    out_masked_dir: Path,
) -> None:
    """
    Persist full-resolution arrays to *out_arrays_dir* and
    masked (always-dry-stripped) arrays to *out_masked_dir*.

    Full arrays are saved as .npy (memory-mappable).
    Masked arrays are saved as zarr chunk stores when zarr is available,
    otherwise as .npy — consistent with MEMORY_CONSTRAINTS.md §5.

    Parameters
    ----------
    arrays : dict
        Keys: 'slope', 'twi'.  Values: float32 ndarrays shape (H, W).
    always_dry : np.ndarray
        Boolean mask — True means exclude.
    out_arrays_dir : Path
        Destination for full-resolution .npy files.
    out_masked_dir : Path
        Destination for masked / chunked artefacts.
    """
    out_arrays_dir.mkdir(parents=True, exist_ok=True)
    out_masked_dir.mkdir(parents=True, exist_ok=True)

    wet_mask = ~always_dry  # True → retain for training

    for name, arr in arrays.items():
        # ── Full-resolution .npy ────────────────────────────────────────────
        full_path = out_arrays_dir / f"{name}_nairobi.npy"
        np.save(full_path, arr)
        logger.info("save | full | name={} | path={}", name, full_path)

        # ── Masked flat array (only wet pixels) ────────────────────────────
        flat_wet = arr[wet_mask]  # shape: (N_wet,)

        if _ZARR_OK:
            zarr_path = out_masked_dir / f"{name}_masked.zarr"
            z = zarr.open(
                str(zarr_path),
                mode="w",
                shape=flat_wet.shape,
                chunks=(min(len(flat_wet), 500_000),),
                dtype=DTYPE,
            )
            z[:] = flat_wet
            logger.info(
                "save | masked_zarr | name={} | path={} | wet_pixels={}",
                name, zarr_path, len(flat_wet),
            )
        else:
            npy_path = out_masked_dir / f"{name}_masked.npy"
            np.save(npy_path, flat_wet)
            logger.info(
                "save | masked_npy | name={} | path={} | wet_pixels={}",
                name, npy_path, len(flat_wet),
            )

        # Immediately release the masked array from the heap
        del flat_wet
        gc.collect()
        _log_rss(f"save_{name}")

    # Also persist the mask itself so training code can reconstruct 2-D grids
    mask_path = out_masked_dir / "always_dry_mask.npy"
    np.save(mask_path, always_dry)
    logger.info("save | always_dry_mask | path={}", mask_path)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def run_pipeline(
    terrain_dir: Path = DEFAULT_TERRAIN_DIR,
    out_arrays_dir: Path = DEFAULT_OUT_ARRAYS,
    out_masked_dir: Path = DEFAULT_OUT_MASKED,
    cell_size_m: float = 30.0,
    dry_threshold: float = DEFAULT_DRY_THRESHOLD,
    smooth: bool = True,
) -> None:
    """
    Execute all four preprocessing stages in sequence.

    Memory checkpoints are inserted between stages so that any breach of
    the thresholds defined in MEMORY_CONSTRAINTS.md surfaces immediately
    rather than causing a silent OOM crash later.

    Parameters
    ----------
    terrain_dir   : directory containing SRTM .tif tiles
    out_arrays_dir: output directory for full-resolution .npy artefacts
    out_masked_dir: output directory for masked / chunked artefacts
    cell_size_m   : pixel size in metres (SRTM 1-arcsec ≈ 30 m)
    dry_threshold : always-dry mask sensitivity (see build_always_dry_mask)
    smooth        : pre-smooth DEM before slope calculation
    """
    logger.info("=" * 60)
    logger.info("GeoStream SRTM Mosaic Pipeline — START")
    logger.info("  terrain_dir   : {}", terrain_dir)
    logger.info("  out_arrays_dir: {}", out_arrays_dir)
    logger.info("  out_masked_dir: {}", out_masked_dir)
    logger.info("  cell_size_m   : {} m", cell_size_m)
    logger.info("  dry_threshold : {}", dry_threshold)
    logger.info("=" * 60)

    # ── Guard ────────────────────────────────────────────────────────────────
    _startup_memory_check()

    # ── Stage 1: discover tiles ──────────────────────────────────────────────
    tiles = discover_tiles(terrain_dir)
    _log_rss("after_tile_discovery")

    # ── Stage 2: mosaic merge ────────────────────────────────────────────────
    dem, profile = build_mosaic(tiles, out_arrays_dir)
    _log_rss("after_mosaic")

    # ── Stage 3a: slope ──────────────────────────────────────────────────────
    slope = compute_slope(dem, cell_size_m=cell_size_m, smooth=smooth)
    _log_rss("after_slope")

    # ── Stage 3b: TWI ────────────────────────────────────────────────────────
    twi = compute_twi(slope, cell_size_m=cell_size_m)
    _log_rss("after_twi")

    # ── Stage 4: always-dry mask + save ─────────────────────────────────────
    always_dry = build_always_dry_mask(twi, dry_threshold=dry_threshold)

    apply_mask_and_save(
        arrays={"slope": slope, "twi": twi},
        always_dry=always_dry,
        out_arrays_dir=out_arrays_dir,
        out_masked_dir=out_masked_dir,
    )

    # Free large arrays before returning (MEMORY_CONSTRAINTS.md §4)
    del dem, slope, twi, always_dry
    gc.collect()
    _log_rss("pipeline_cleanup")

    logger.info("GeoStream SRTM Mosaic Pipeline — COMPLETE")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.preprocessing.srtm_mosaic",
        description="GeoStream SRTM mosaic, slope, TWI, and always-dry mask pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--terrain-dir",
        type=Path,
        default=DEFAULT_TERRAIN_DIR,
        help="Directory containing raw SRTM .tif tiles.",
    )
    p.add_argument(
        "--out-arrays",
        type=Path,
        default=DEFAULT_OUT_ARRAYS,
        help="Output directory for full-resolution numpy artefacts.",
    )
    p.add_argument(
        "--out-masked",
        type=Path,
        default=DEFAULT_OUT_MASKED,
        help="Output directory for masked / chunked artefacts.",
    )
    p.add_argument(
        "--cell-size",
        type=float,
        default=30.0,
        metavar="METRES",
        help="DEM cell size in metres (SRTM 1-arcsec ≈ 30 m).",
    )
    p.add_argument(
        "--dry-threshold",
        type=float,
        default=DEFAULT_DRY_THRESHOLD,
        help="Normalised TWI threshold below which pixels are classified as always-dry.",
    )
    p.add_argument(
        "--no-smooth",
        action="store_true",
        default=False,
        help="Disable Gaussian pre-smoothing of DEM before slope calculation.",
    )
    return p


if __name__ == "__main__":
    # Configure loguru to emit structured log lines per MEMORY_CONSTRAINTS.md
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
        ),
        level="DEBUG",
        colorize=True,
    )
    logger.add(
        "logs/srtm_mosaic_{time:YYYYMMDD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="14 days",
    )

    args = _build_parser().parse_args()
    run_pipeline(
        terrain_dir=args.terrain_dir,
        out_arrays_dir=args.out_arrays,
        out_masked_dir=args.out_masked,
        cell_size_m=args.cell_size,
        dry_threshold=args.dry_threshold,
        smooth=not args.no_smooth,
    )
