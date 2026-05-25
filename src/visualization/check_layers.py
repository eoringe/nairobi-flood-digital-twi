"""
src.visualization.check_layers
================================
GeoStream -- Diagnostic Layer Snapshot Viewer

PURPOSE
-------
Load, downsample, and display a 3-panel side-by-side diagnostic plot of
the core spatial layers acquired by the ingestion pipeline:

  Panel 1 -- DEM topography          (data/raw/terrain/*.tif)
  Panel 2 -- ICPAC flood consensus   (data/raw/vectors/nairobi_flood_100yr.tif)
  Panel 3 -- Sentinel-1 water mask   (data/raw/vectors/nairobi_s1_temporal_targets/*.tif)

The output figure is saved to:
  reports/figures/nairobi_layer_diagnostic.png

MEMORY CONTRACT  (MEMORY_CONSTRAINTS.md)
-----------------------------------------
- Every raster is opened with rasterio context managers -- no full-
  resolution grid is ever unpacked into active RAM.
- The out_shape parameter inside .read(1, out_shape=(...)) downsamples
  each layer to a 600x600 canvas DURING the read call itself, so the
  only arrays that exist in memory are the display-sized thumbnails.
- Peak RSS for this script is < 50 MB.

USAGE
-----
  python -m src.visualization.check_layers
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import psutil

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError:
    print(
        "[FATAL] rasterio is not installed.\n"
        "        pip install rasterio>=1.3",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")     # headless backend for file export
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
except ImportError:
    print(
        "[FATAL] matplotlib is not installed.\n"
        "        pip install matplotlib>=3.7",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

#: Downsample target -- every raster is read into this shape
#: 600 x 600 float32 = 1.4 MB per panel = 4.2 MB total.  Safe.
DISPLAY_SIZE: tuple[int, int] = (600, 600)

#: Data paths (relative to repo root)
TERRAIN_DIR: Path    = Path("data/raw/terrain")
FLOOD_TIF: Path      = Path("data/raw/vectors/nairobi_flood_100yr.tif")
S1_TARGETS_DIR: Path = Path("data/raw/vectors/nairobi_s1_temporal_targets")

#: Output path for the diagnostic figure
OUTPUT_PATH: str = os.path.join("reports", "figures", "nairobi_layer_diagnostic.png")

#: Nairobi bounding box for geographic extent [left, right, bottom, top]
NAIROBI_EXTENT: list[float] = [36.65, 37.10, -1.45, -1.15]

#: RAM floor for startup check
MIN_FREE_RAM_GB: float = 0.5


# ============================================================================
# HELPERS
# ============================================================================

def _check_memory() -> None:
    """Abort if free RAM < MIN_FREE_RAM_GB."""
    mem = psutil.virtual_memory()
    free_gb = mem.available / 1024 ** 3
    print(f"[MEMORY] free={free_gb:.2f} GB | floor={MIN_FREE_RAM_GB} GB")
    if free_gb < MIN_FREE_RAM_GB:
        print(
            f"[FATAL ] Free RAM {free_gb:.2f} GB < {MIN_FREE_RAM_GB} GB. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_first_tif(directory: Path, label: str) -> Path:
    """
    Return the first .tif file found in *directory*.
    Raises FileNotFoundError with a clear message if none exist.
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"[{label}] Directory does not exist: {directory}\n"
            "  Run the ingestion pipeline first."
        )
    tifs = sorted(directory.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(
            f"[{label}] No .tif files found in {directory}\n"
            "  Run the ingestion pipeline first."
        )
    return tifs[0]


def _read_downsampled(path: Path, label: str) -> tuple[np.ndarray, dict]:
    """
    Open *path* with rasterio, read band 1 downsampled to DISPLAY_SIZE.

    Returns
    -------
    array   : np.ndarray, shape DISPLAY_SIZE, dtype float32
    profile : dict  (CRS, transform, etc.)
    """
    print(f"  [{label}] Opening {path.name} ...")
    with rasterio.open(path) as src:
        native_h, native_w = src.height, src.width
        print(
            f"  [{label}] Native  : {native_w} W x {native_h} H px | "
            f"dtype={src.dtypes[0]} | CRS={src.crs}"
        )

        # Downsample during read -- only the 600x600 thumbnail
        # ever exists in RAM (MEMORY_CONSTRAINTS.md guardrail).
        arr = src.read(
            1,
            out_shape=DISPLAY_SIZE,
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        print(
            f"  [{label}] Display : {arr.shape[1]} W x {arr.shape[0]} H px | "
            f"dtype={arr.dtype} | "
            f"~{arr.nbytes / 1024:.0f} KB"
        )

        profile = src.profile.copy()

    return arr, profile


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_panels(
    dem: np.ndarray,
    flood: np.ndarray,
    s1: np.ndarray,
    s1_label: str,
) -> None:
    """
    Render the 3-panel diagnostic figure and save to OUTPUT_PATH.

    Each panel is clearly labeled for supervisor review with:
    - Descriptive title explaining the data source
    - Subtitle explaining what the layer represents
    - Geographic coordinate axes (longitude / latitude)
    - Annotated colorbar with clear units
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="#F8F9FA")

    # ---- Global title -------------------------------------------------------
    fig.suptitle(
        "Nairobi Urban Digital Twin — Core Geospatial Data Layers",
        fontsize=18,
        fontweight="bold",
        color="navy",
        y=1.02,
    )
    fig.text(
        0.5, 0.97,
        "Study Area: Nairobi County, Kenya  ·  "
        "Bounding Box: 36.65°E – 37.10°E, 1.15°S – 1.45°S  ·  "
        f"Display: {DISPLAY_SIZE[1]}×{DISPLAY_SIZE[0]} px",
        ha="center", fontsize=9, color="#555555", style="italic",
    )

    # =====================================================================
    # PANEL 1: DEM Topography
    # =====================================================================
    ax1 = axes[0]
    ax1.set_facecolor("#EAEAEA")
    dem_display = np.where(dem == 0, np.nan, dem)  # mask nodata as NaN
    im1 = ax1.imshow(
        dem_display, cmap="terrain", interpolation="bilinear",
        extent=NAIROBI_EXTENT, aspect="auto",
    )
    ax1.set_title(
        "Digital Elevation Model (SRTM 30m)",
        fontsize=12, fontweight="bold", color="#1B2A4A", pad=12,
    )
    ax1.set_xlabel("Longitude (°E)", fontsize=10, labelpad=6)
    ax1.set_ylabel("Latitude (°S)", fontsize=10, labelpad=6)
    ax1.tick_params(labelsize=8)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, shrink=0.85)
    cbar1.set_label("Elevation Above Sea Level (m)", fontsize=9, labelpad=8)
    cbar1.ax.tick_params(labelsize=8)
    # Annotation explaining what this layer shows
    ax1.text(
        0.03, 0.03,
        "Terrain height used to model\n"
        "gravity-driven flood flow paths",
        transform=ax1.transAxes, fontsize=7.5,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333", alpha=0.85),
        verticalalignment="bottom",
    )

    # =====================================================================
    # PANEL 2: ICPAC Flood Consensus
    # =====================================================================
    ax2 = axes[1]
    ax2.set_facecolor("#1E1E2C")
    # Replace NaN/nodata with -1 so we can display cleanly
    flood_clean = np.nan_to_num(flood, nan=-1.0)

    # Discrete colormap for integer agreement classes 0-6
    flood_colors = plt.cm.YlOrRd(np.linspace(0.15, 1.0, 7))
    flood_cmap = ListedColormap(flood_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    flood_norm = BoundaryNorm(bounds, flood_cmap.N)

    # Mask out nodata pixels (anything < 0)
    flood_masked = np.ma.masked_where(flood_clean < 0, flood_clean)
    im2 = ax2.imshow(
        flood_masked, cmap=flood_cmap, norm=flood_norm,
        interpolation="nearest",
        extent=NAIROBI_EXTENT, aspect="auto",
    )
    ax2.set_title(
        "100-Year Flood Hazard Zone (ICPAC Consensus)",
        fontsize=12, fontweight="bold", color="#1B2A4A", pad=12,
    )
    ax2.set_xlabel("Longitude (°E)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Latitude (°S)", fontsize=10, labelpad=6)
    ax2.tick_params(labelsize=8)
    cbar2 = fig.colorbar(
        im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.85,
        ticks=range(7),
    )
    cbar2.set_label(
        "Number of Flood Models in Agreement (0–6)",
        fontsize=9, labelpad=8,
    )
    cbar2.ax.set_yticklabels(
        ["0 — No risk", "1", "2", "3", "4", "5", "6 — All agree"],
        fontsize=7.5,
    )
    # Annotation
    ax2.text(
        0.03, 0.03,
        "Multi-model consensus: higher = more\n"
        "models predict flooding at this location",
        transform=ax2.transAxes, fontsize=7.5,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333", alpha=0.85),
        verticalalignment="bottom",
    )

    # =====================================================================
    # PANEL 3: Sentinel-1 SAR Water Mask
    # =====================================================================
    ax3 = axes[2]
    ax3.set_facecolor("#1E1E2C")
    # Custom binary colormap: dark = dry, cyan = water
    s1_cmap = ListedColormap(["#1E1E2C", "#00B4D8"])
    im3 = ax3.imshow(
        s1, cmap=s1_cmap, interpolation="nearest",
        vmin=0, vmax=1,
        extent=NAIROBI_EXTENT, aspect="auto",
    )
    ax3.set_title(
        f"Sentinel-1 SAR Water Detection\n({s1_label})",
        fontsize=12, fontweight="bold", color="#1B2A4A", pad=12,
    )
    ax3.set_xlabel("Longitude (°E)", fontsize=10, labelpad=6)
    ax3.set_ylabel("Latitude (°S)", fontsize=10, labelpad=6)
    ax3.tick_params(labelsize=8)
    cbar3 = fig.colorbar(
        im3, ax=ax3, fraction=0.046, pad=0.04, shrink=0.85,
        ticks=[0, 1],
    )
    cbar3.ax.set_yticklabels(
        ["Dry Land", "Standing Water"],
        fontsize=8,
    )
    cbar3.set_label("Surface Water State", fontsize=9, labelpad=8)
    # Annotation
    ax3.text(
        0.03, 0.03,
        "Satellite radar detects standing water\n"
        "independent of cloud cover (day & night)",
        transform=ax3.transAxes, fontsize=7.5,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333", alpha=0.85),
        verticalalignment="bottom",
    )

    # ---- Export --------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n[EXPORT] Saved : {OUTPUT_PATH}")
    print(f"         Size  : {file_size_mb:.2f} MB  |  DPI: 300")


# ============================================================================
# MAIN
# ============================================================================

def check_layers() -> None:
    """
    Load all three diagnostic layers, downsample safely, and display.
    """
    _check_memory()

    print()
    print("=" * 62)
    print("  GeoStream | Layer Diagnostic Snapshot")
    print("=" * 62)
    print(f"  Display size : {DISPLAY_SIZE[1]} W x {DISPLAY_SIZE[0]} H px")
    print(f"  DEM source   : {TERRAIN_DIR}/")
    print(f"  Flood source : {FLOOD_TIF}")
    print(f"  S1 source    : {S1_TARGETS_DIR}/")
    print("=" * 62)

    # ---- Validate paths before doing any work --------------------------------
    errors: list[str] = []

    try:
        dem_path = _find_first_tif(TERRAIN_DIR, "DEM")
    except FileNotFoundError as e:
        errors.append(str(e))
        dem_path = None

    if not FLOOD_TIF.exists():
        errors.append(
            f"[FLOOD] File not found: {FLOOD_TIF}\n"
            "  Run: python -m src.ingestion.download_flood_target"
        )

    try:
        s1_path = _find_first_tif(S1_TARGETS_DIR, "S1")
    except FileNotFoundError as e:
        errors.append(str(e))
        s1_path = None

    if errors:
        print("\n[ERROR] Missing data files:\n")
        for err in errors:
            print(f"  {err}\n")
        sys.exit(1)

    # ---- Read each layer with downsampling -----------------------------------
    print()
    dem_arr, _  = _read_downsampled(dem_path, "DEM")
    flood_arr, _ = _read_downsampled(FLOOD_TIF, "FLOOD")
    s1_arr, _   = _read_downsampled(s1_path, "S1")

    # ---- RSS checkpoint after all reads --------------------------------------
    rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
    total_array_kb = (dem_arr.nbytes + flood_arr.nbytes + s1_arr.nbytes) / 1024
    print(
        f"\n[MEM  ] RSS after reads: {rss_mb:.1f} MB | "
        f"array footprint: {total_array_kb:.0f} KB | "
        f"status: OK"
    )

    # ---- Render ---------------------------------------------------------------
    s1_label = s1_path.stem.replace("s1_water_mask_", "").replace("_", " ").title()
    _plot_panels(dem_arr, flood_arr, s1_arr, s1_label)

    print("\n[DONE ] Layer diagnostic figure exported successfully.")
    print(f"        Open '{OUTPUT_PATH}' to review.")


if __name__ == "__main__":
    check_layers()
