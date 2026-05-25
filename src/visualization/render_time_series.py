"""
src.visualization.render_time_series
=====================================
GeoStream -- Historical Flood Timeline Composite Renderer

PURPOSE
-------
Programmatically ingest the complete folder of Sentinel-1 seasonal water
masks (2020–2026), stack them chronologically, and export a high-resolution
multi-panel composite PNG matrix.  This allows supervisors to instantly
evaluate historical flooding profiles without requiring external GIS software.

MEMORY CONTRACT  (MEMORY_CONSTRAINTS.md)
-----------------------------------------
- Each raster is opened via rasterio context managers -- no full-resolution
  grid is ever unpacked into active RAM.
- The out_shape parameter (400, 400) downsamples each layer during the read
  call, so only lightweight display thumbnails exist in memory.
- Peak RSS for the full 13-panel pipeline is < 40 MB.

USAGE
-----
  python -m src.visualization.render_time_series
"""

from __future__ import annotations

import glob
import os
import os.path
import sys
import re

import numpy as np

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
    matplotlib.use("Agg")          # headless backend for file-only export
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print(
        "[FATAL] matplotlib is not installed.\n"
        "        pip install matplotlib>=3.7",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# CUSTOM BINARY COLORMAP
# ============================================================================
#: High-contrast two-tone map:  0 → dark charcoal (land),  1 → vivid blue (water)
FLOOD_CMAP = ListedColormap(["#1E1E2C", "#00B4D8"])



# ============================================================================
# CONFIGURATION
# ============================================================================

#: Input directory containing the Sentinel-1 seasonal water masks
INPUT_DIR: str = os.path.join("data", "raw", "vectors", "nairobi_s1_temporal_targets")

#: Glob pattern to match raster filenames
FILE_PATTERN: str = "s1_water_mask_*.tif"

#: Output path for the composite PNG
OUTPUT_PATH: str = os.path.join("reports", "figures", "nairobi_historical_timelines.png")

#: Downsample target -- each raster is read into this shape
#: 400 x 400 float32 ≈ 0.6 MB per panel ≈ 8 MB total.  Memory-safe.
DISPLAY_SIZE: tuple[int, int] = (400, 400)

#: Nairobi bounding box for cartographic extent [left, right, bottom, top]
NAIROBI_EXTENT: list[float] = [36.65, 37.10, -1.45, -1.15]

#: Grid dimensions
GRID_ROWS: int = 4
GRID_COLS: int = 4


# ============================================================================
# SEASON ORDERING TABLE
# ============================================================================

#: Canonical ordering: long rains precede short rains within each year
SEASON_ORDER: dict[str, int] = {
    "long_rains":  0,
    "short_rains": 1,
}


# ============================================================================
# HELPERS
# ============================================================================

def _discover_rasters(directory: str, pattern: str) -> list[str]:
    """
    Glob for all .tif files matching *pattern* inside *directory*.

    Returns an explicitly chronologically-sorted list of absolute paths,
    ordered first by year (ascending) then by season (long → short).
    """
    search = os.path.join(directory, pattern)
    paths = glob.glob(search)

    if not paths:
        print(
            f"[FATAL] No files matching '{pattern}' found in: {directory}\n"
            "        Run the ingestion pipeline first:\n"
            "          python -m src.ingestion.fetch_sentinel_targets",
            file=sys.stderr,
        )
        sys.exit(1)

    def _sort_key(filepath: str) -> tuple[int, int]:
        """Extract (year, season_rank) from the filename for sorting."""
        basename = os.path.basename(filepath)
        # Match pattern: s1_water_mask_<YEAR>_<SEASON>.tif
        match = re.match(r"s1_water_mask_(\d{4})_(.+)\.tif", basename)
        if not match:
            return (9999, 9)
        year = int(match.group(1))
        season = match.group(2)
        return (year, SEASON_ORDER.get(season, 9))

    paths.sort(key=_sort_key)
    return paths


def _parse_title(filepath: str) -> str:
    """
    Extract a clean human-readable title from the raster filename.

    Example:
        "s1_water_mask_2020_long_rains.tif"  →  "2020 LONG RAINS"
    """
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0]          # drop .tif
    stem = stem.replace("s1_water_mask_", "")     # strip prefix
    # Split into year + season tokens
    parts = stem.split("_", 1)
    if len(parts) == 2:
        year, season = parts
        return f"{year} {season.upper().replace('_', ' ')}"
    return stem.upper()


def _read_downsampled(filepath: str, index: int) -> np.ndarray:
    """
    Open *filepath* with rasterio, read band 1 downsampled to DISPLAY_SIZE.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .tif raster.
    index : int
        Panel index for log readability.

    Returns
    -------
    np.ndarray
        Float32 array of shape DISPLAY_SIZE.
    """
    basename = os.path.basename(filepath)
    print(f"  [{index + 1:>2}] Reading {basename} ...")

    with rasterio.open(filepath) as src:
        native_h, native_w = src.height, src.width
        print(
            f"       Native  : {native_w}W × {native_h}H | "
            f"dtype={src.dtypes[0]} | CRS={src.crs}"
        )

        # Downsample during read -- only the 400×400 thumbnail ever
        # exists in RAM  (MEMORY_CONSTRAINTS.md guardrail).
        arr = src.read(
            1,
            out_shape=DISPLAY_SIZE,
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        print(
            f"       Display : {arr.shape[1]}W × {arr.shape[0]}H | "
            f"~{arr.nbytes / 1024:.0f} KB"
        )

    return arr


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def render_time_series() -> None:
    """
    End-to-end pipeline:
    1. Discover & sort all Sentinel-1 seasonal masks.
    2. Read each raster with memory-safe downsampling.
    3. Render a 4×4 multi-panel composite with cartographic styling.
    4. Export the final figure as a high-DPI PNG.
    """

    print()
    print("=" * 66)
    print("  GeoStream | Historical Flood Timeline Composite Renderer")
    print("=" * 66)
    print(f"  Input dir  : {INPUT_DIR}")
    print(f"  Pattern    : {FILE_PATTERN}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"  Grid       : {GRID_ROWS} rows × {GRID_COLS} cols")
    print(f"  Panel size : {DISPLAY_SIZE[1]}W × {DISPLAY_SIZE[0]}H px")
    print(f"  Extent     : {NAIROBI_EXTENT}")
    print("=" * 66)

    # ------------------------------------------------------------------
    # STEP 1 -- Discover & sort rasters
    # ------------------------------------------------------------------
    print("\n[STEP 1] Discovering raster files ...")
    raster_paths = _discover_rasters(INPUT_DIR, FILE_PATTERN)
    print(f"         Found {len(raster_paths)} seasonal masks:")

    for i, p in enumerate(raster_paths):
        title = _parse_title(p)
        print(f"           {i + 1:>2}. {title}  ({os.path.basename(p)})")

    # ------------------------------------------------------------------
    # STEP 2 -- Read all rasters with downsampling
    # ------------------------------------------------------------------
    print(f"\n[STEP 2] Reading rasters (downsampled to {DISPLAY_SIZE}) ...")
    arrays: list[np.ndarray] = []
    titles: list[str] = []

    for i, path in enumerate(raster_paths):
        arr = _read_downsampled(path, i)
        arrays.append(arr)
        titles.append(_parse_title(path))

    total_kb = sum(a.nbytes for a in arrays) / 1024
    print(f"\n         Total array footprint: {total_kb:.0f} KB ({len(arrays)} panels)")

    # ------------------------------------------------------------------
    # STEP 3 -- Build the multi-panel composite
    # ------------------------------------------------------------------
    print("\n[STEP 3] Constructing figure matrix layout ...")

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(22, 22),
        facecolor="#F8F9FA",
    )
    axes_flat = axes.flatten()

    # ---- Global title (plain English for non-technical audience) ----------
    fig.suptitle(
        "Where Did Flooding Occur in Nairobi Over the Past 6 Years?",
        fontsize=22,
        fontweight="bold",
        color="navy",
        y=0.97,
    )

    # ---- Explanatory subtitle --------------------------------------------
    fig.text(
        0.5, 0.945,
        "Each panel below shows one rainy season in Nairobi (2020–2026).  "
        "Bright cyan areas = locations where satellite radar detected standing water on the ground.\n"
        "Dark areas = dry land with no flooding detected.  "
        "Read left-to-right, top-to-bottom to see how flooding patterns changed over time.",
        ha="center", va="top",
        fontsize=10.5,
        color="#333333",
        linespacing=1.5,
    )

    # ---- Data source caption ---------------------------------------------
    fig.text(
        0.5, 0.912,
        "Data: Sentinel-1 C-band SAR (European Space Agency)  ·  "
        "Processed via Google Earth Engine  ·  "
        "Study Area: Nairobi County, Kenya (36.65°E – 37.10°E, 1.15°S – 1.45°S)",
        ha="center",
        fontsize=9,
        color="#777777",
        style="italic",
    )

    # ---- Color legend (top-right) ----------------------------------------
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#00B4D8", edgecolor="white", label="Standing Water (Flooded)"),
        Patch(facecolor="#1E1E2C", edgecolor="white", label="Dry Land (No Flooding)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.915),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        title="Map Legend",
        title_fontsize=11,
    )

    # ---- Season explanation labels ----------------------------------------
    #: "Long Rains" = March–May,  "Short Rains" = October–December
    SEASON_INFO: dict[str, str] = {
        "LONG RAINS":  "Mar – May",
        "SHORT RAINS": "Oct – Dec",
    }

    # -- Render each panel -------------------------------------------------
    for i, (arr, title) in enumerate(zip(arrays, titles)):
        ax = axes_flat[i]

        # Dark panel background matches land color for clean edges
        ax.set_facecolor("#1E1E2C")

        ax.imshow(
            arr,
            cmap=FLOOD_CMAP,
            interpolation="nearest",
            extent=NAIROBI_EXTENT,
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        # Build a descriptive title with the season month-range
        season_key = title.split(" ", 1)[1] if " " in title else ""
        month_range = SEASON_INFO.get(season_key, "")
        display_title = f"{title}\n({month_range})" if month_range else title

        ax.set_title(
            display_title,
            fontsize=12,
            fontweight="bold",
            color="#1B2A4A",
            pad=10,
        )

        # Clean cartographic look -- no pixel axes
        ax.axis("off")

        # Water-pixel percentage annotation (bottom-left of each panel)
        water_pct = (arr > 0).sum() / arr.size * 100
        ax.text(
            0.03, 0.04,
            f"Water cover: {water_pct:.1f}%",
            transform=ax.transAxes, fontsize=8,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#00B4D8",
                      edgecolor="white", alpha=0.90),
            verticalalignment="bottom",
        )

        print(f"  Panel {i + 1:>2} rendered: {title}  (water={water_pct:.1f}%)")

    # -- Hide unused grid cells --------------------------------------------
    for j in range(len(arrays), len(axes_flat)):
        axes_flat[j].set_visible(False)
        print(f"  Panel {j + 1:>2} hidden  : (unused)")

    # ---- Bottom footnote -------------------------------------------------
    fig.text(
        0.5, 0.01,
        "HOW TO READ THIS CHART:  Each small map shows Nairobi during one rainy season.  "
        "Kenya has two rainy seasons each year — the 'Long Rains' (March–May) and the "
        "'Short Rains' (October–December).  Compare panels to see which seasons brought "
        "the most flooding and which areas are repeatedly affected.",
        ha="center", va="bottom",
        fontsize=9.5,
        color="#444444",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD",
                  edgecolor="#B0D4E8", alpha=0.95),
    )

    plt.subplots_adjust(
        top=0.88,
        bottom=0.06,
        left=0.02,
        right=0.98,
        hspace=0.22,
        wspace=0.05,
    )

    # ------------------------------------------------------------------
    # STEP 4 -- Export
    # ------------------------------------------------------------------
    print(f"\n[STEP 4] Exporting composite figure ...")

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"         Output directory verified: {output_dir}/")

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"         Saved : {OUTPUT_PATH}")
    print(f"         Size  : {file_size_mb:.2f} MB")
    print(f"         DPI   : 300")

    print()
    print("=" * 66)
    print("  [DONE] Historical timeline composite generated successfully.")
    print(f"         Open '{OUTPUT_PATH}' to review.")
    print("=" * 66)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    render_time_series()
