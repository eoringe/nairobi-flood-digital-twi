"""
src.visualization.render_rainfall
===================================
GeoStream -- CHIRPS Rainfall Time-Series Dashboard

PURPOSE
-------
Visualize the CHIRPS daily rainfall NetCDF files (2020–2026) to show
historical precipitation patterns over Nairobi.  This helps the supervisor
understand *when* heavy rain occurred and how it correlates with flood
events detected by the Sentinel-1 water masks.

CHIRPS = Climate Hazards Group InfraRed Precipitation with Station data
  - 0.25° grid resolution (~25km)
  - Daily precipitation estimates in mm/day
  - Coverage: 2020–2026

OUTPUT
------
  reports/figures/nairobi_rainfall_dashboard.png

USAGE
-----
  python -m src.visualization.render_rainfall
"""

from __future__ import annotations

import glob
import os
import os.path
import sys
import re

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print(
        "[FATAL] matplotlib is not installed.\n"
        "        pip install matplotlib>=3.7",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import netCDF4
except ImportError:
    print(
        "[FATAL] netCDF4 is not installed.\n"
        "        pip install netCDF4>=1.6",
        file=sys.stderr,
    )
    sys.exit(1)

from datetime import datetime, timedelta


# ============================================================================
# CONFIGURATION
# ============================================================================

CLIMATE_DIR: str = os.path.join("data", "raw", "climate")
OUTPUT_PATH: str = os.path.join("reports", "figures", "nairobi_rainfall_dashboard.png")

#: Nairobi approximate center coordinates for grid extraction
NAIROBI_LAT: float = -1.30
NAIROBI_LON: float = 36.85

#: Season date ranges (month, day)
LONG_RAINS_START  = (3, 1)   # March
LONG_RAINS_END    = (5, 31)  # May
SHORT_RAINS_START = (10, 1)  # October
SHORT_RAINS_END   = (12, 31) # December


# ============================================================================
# DATA LOADING
# ============================================================================

def _find_nearest_idx(arr: np.ndarray, value: float) -> int:
    """Find the index of the nearest value in a 1D array."""
    return int(np.abs(arr - value).argmin())


def _load_chirps_year(filepath: str) -> tuple[list[datetime], list[float]]:
    """
    Load one CHIRPS NetCDF file, extract the Nairobi grid cell time series.

    Returns (dates, daily_precip_mm).
    """
    basename = os.path.basename(filepath)
    print(f"  Reading {basename} ...")

    ds = netCDF4.Dataset(filepath, "r")

    # Get coordinate arrays
    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]

    # Find nearest grid cell to Nairobi
    lat_idx = _find_nearest_idx(lats, NAIROBI_LAT)
    lon_idx = _find_nearest_idx(lons, NAIROBI_LON)

    actual_lat = float(lats[lat_idx])
    actual_lon = float(lons[lon_idx])
    print(f"    Grid cell: lat={actual_lat:.2f}, lon={actual_lon:.2f} "
          f"(target: {NAIROBI_LAT}, {NAIROBI_LON})")

    # Extract time and precipitation
    time_var = ds.variables["time"]
    time_units = time_var.units       # e.g. "days since 1980-1-1 0:0:0"
    time_vals = time_var[:]

    # Parse the reference date from units string
    ref_match = re.search(r"since\s+(\d{4})-(\d{1,2})-(\d{1,2})", time_units)
    if ref_match:
        ref_date = datetime(
            int(ref_match.group(1)),
            int(ref_match.group(2)),
            int(ref_match.group(3)),
        )
    else:
        ref_date = datetime(1980, 1, 1)

    dates = [ref_date + timedelta(days=float(t)) for t in time_vals]

    # Extract precipitation for the Nairobi pixel
    precip_var = ds.variables["precip"]
    precip = precip_var[:, lat_idx, lon_idx]

    # Handle masked/fill values
    if hasattr(precip, "filled"):
        precip = precip.filled(0.0)
    precip = np.array(precip, dtype=np.float32)
    precip[precip < 0] = 0.0

    ds.close()

    print(f"    Days: {len(dates)} | "
          f"Range: {dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')} | "
          f"Max precip: {precip.max():.1f} mm/day")

    return dates, precip.tolist()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def render_rainfall() -> None:
    """Build the multi-panel CHIRPS rainfall dashboard."""

    print()
    print("=" * 66)
    print("  GeoStream | CHIRPS Rainfall Dashboard")
    print("=" * 66)

    # -- Discover NetCDF files ---------------------------------------------
    pattern = os.path.join(CLIMATE_DIR, "chirps-v2.0.*.nc")
    nc_files = sorted(glob.glob(pattern))

    if not nc_files:
        print(f"[FATAL] No CHIRPS NetCDF files found in: {CLIMATE_DIR}",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(nc_files)} CHIRPS annual files.")

    # -- Load all years ----------------------------------------------------
    all_dates: list[datetime] = []
    all_precip: list[float] = []

    yearly_totals: dict[int, float] = {}
    monthly_avgs: dict[int, list[float]] = {m: [] for m in range(1, 13)}

    for fp in nc_files:
        dates, precip = _load_chirps_year(fp)
        all_dates.extend(dates)
        all_precip.extend(precip)

        # Yearly total
        year = dates[0].year
        yearly_totals[year] = sum(precip)

        # Monthly averages
        for d, p in zip(dates, precip):
            monthly_avgs[d.month].append(p)

    print(f"\n  Total data points: {len(all_dates)}")

    # -- Compute seasonal totals per year ----------------------------------
    years = sorted(yearly_totals.keys())
    long_rains_totals = []
    short_rains_totals = []

    for y in years:
        lr_total = sum(
            p for d, p in zip(all_dates, all_precip)
            if d.year == y
            and LONG_RAINS_START <= (d.month, d.day) <= LONG_RAINS_END
        )
        sr_total = sum(
            p for d, p in zip(all_dates, all_precip)
            if d.year == y
            and SHORT_RAINS_START <= (d.month, d.day) <= SHORT_RAINS_END
        )
        long_rains_totals.append(lr_total)
        short_rains_totals.append(sr_total)

    # =====================================================================
    # BUILD FIGURE -- 2x2 dashboard
    # =====================================================================
    print("\n[RENDER] Building rainfall dashboard ...")

    fig, axes = plt.subplots(2, 2, figsize=(22, 14), facecolor="#F8F9FA")

    fig.suptitle(
        "How Much Rain Falls on Nairobi, and When?",
        fontsize=22, fontweight="bold", color="navy", y=0.98,
    )
    fig.text(
        0.5, 0.945,
        "Data: CHIRPS v2.0 Daily Precipitation (0.25° grid)  ·  "
        "Climate Hazards Group, UC Santa Barbara  ·  "
        f"Grid Cell: {NAIROBI_LAT}°S, {NAIROBI_LON}°E (Nairobi)",
        ha="center", fontsize=10, color="#777", style="italic",
    )

    # ---- Panel 1: Full daily time series ---------------------------------
    ax1 = axes[0, 0]
    ax1.bar(all_dates, all_precip, width=1.0, color="#2196F3",
            alpha=0.7, edgecolor="none", zorder=3)
    ax1.set_ylabel("Daily Rainfall (mm)", fontsize=11)
    ax1.set_title(
        "Daily Rainfall Over Nairobi (2020–2026)\n"
        "Each bar = one day's rainfall amount",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_xlim(min(all_dates), max(all_dates))

    # Shade rainy seasons
    for y in years:
        for start_m, end_m, color, label in [
            (3, 5, "#FF9800", "Long Rains"),
            (10, 12, "#4CAF50", "Short Rains"),
        ]:
            try:
                s = datetime(y, start_m, 1)
                e = datetime(y, end_m, 28)
                ax1.axvspan(s, e, alpha=0.08, color=color, zorder=1)
            except ValueError:
                pass

    ax1.text(
        0.02, 0.95,
        "Orange shading = Long Rains (Mar–May)\n"
        "Green shading = Short Rains (Oct–Dec)",
        transform=ax1.transAxes, fontsize=8, color="#555",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # ---- Panel 2: Annual totals ------------------------------------------
    ax2 = axes[0, 1]
    bar_colors = ["#1565C0" if t > np.mean(list(yearly_totals.values()))
                  else "#90CAF9" for t in yearly_totals.values()]
    bars = ax2.bar(
        [str(y) for y in years],
        [yearly_totals[y] for y in years],
        color=bar_colors, edgecolor="white", zorder=3,
    )
    ax2.set_ylabel("Total Annual Rainfall (mm)", fontsize=11)
    ax2.set_title(
        "Total Rainfall Per Year\n"
        "Dark blue = above average, light blue = below average",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    # Average line
    avg = np.mean(list(yearly_totals.values()))
    ax2.axhline(avg, color="#E53935", linewidth=2, linestyle="--", zorder=4,
                label=f"Average: {avg:.0f} mm")
    ax2.legend(fontsize=10)
    # Value labels
    for bar, val in zip(bars, [yearly_totals[y] for y in years]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 10,
                 f"{val:.0f}", ha="center", fontsize=9, fontweight="bold",
                 color="#333")

    # ---- Panel 3: Monthly rainfall cycle (climatology) -------------------
    ax3 = axes[1, 0]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_means = [np.mean(monthly_avgs[m]) for m in range(1, 13)]

    bar_month_colors = []
    for m in range(1, 13):
        if m in (3, 4, 5):
            bar_month_colors.append("#FF9800")     # Long rains
        elif m in (10, 11, 12):
            bar_month_colors.append("#4CAF50")     # Short rains
        else:
            bar_month_colors.append("#90CAF9")     # Dry season

    ax3.bar(month_names, month_means, color=bar_month_colors,
            edgecolor="white", zorder=3)
    ax3.set_ylabel("Average Daily Rainfall (mm/day)", fontsize=11)
    ax3.set_title(
        "Nairobi's Typical Rainfall Pattern Throughout the Year\n"
        "The city experiences two rainy seasons annually",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    # Legend patches
    from matplotlib.patches import Patch
    ax3.legend(
        handles=[
            Patch(facecolor="#FF9800", label="Long Rains (Mar–May)"),
            Patch(facecolor="#4CAF50", label="Short Rains (Oct–Dec)"),
            Patch(facecolor="#90CAF9", label="Dry Season"),
        ],
        fontsize=9, loc="upper right",
    )

    # ---- Panel 4: Seasonal comparison ------------------------------------
    ax4 = axes[1, 1]
    x = np.arange(len(years))
    width = 0.35
    ax4.bar(x - width / 2, long_rains_totals, width,
            label="Long Rains (Mar–May)", color="#FF9800",
            edgecolor="white", zorder=3)
    ax4.bar(x + width / 2, short_rains_totals, width,
            label="Short Rains (Oct–Dec)", color="#4CAF50",
            edgecolor="white", zorder=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(y) for y in years], fontsize=10)
    ax4.set_ylabel("Seasonal Total Rainfall (mm)", fontsize=11)
    ax4.set_title(
        "Comparing Long Rains vs Short Rains Each Year\n"
        "Which rainy season brought more water?",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3, zorder=0)

    # ---- Bottom footnote -------------------------------------------------
    fig.text(
        0.5, 0.005,
        "HOW TO READ: CHIRPS rainfall data shows how much rain falls on Nairobi each day.  "
        "Kenya has two rainy seasons: 'Long Rains' (March–May) and 'Short Rains' "
        "(October–December).  Heavy rainfall during these periods often triggers the "
        "flooding events detected by our Sentinel-1 satellite imagery.",
        ha="center", va="bottom", fontsize=9.5, color="#444", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                  edgecolor="#A5D6A7", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    # ---- Export ----------------------------------------------------------
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(
        OUTPUT_PATH, dpi=300, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    plt.close(fig)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n[EXPORT] Saved : {OUTPUT_PATH}")
    print(f"         Size  : {file_size_mb:.2f} MB  |  DPI: 300")

    print()
    print("=" * 66)
    print("  [DONE] Rainfall dashboard exported successfully.")
    print(f"         Open '{OUTPUT_PATH}' to review.")
    print("=" * 66)


if __name__ == "__main__":
    render_rainfall()
