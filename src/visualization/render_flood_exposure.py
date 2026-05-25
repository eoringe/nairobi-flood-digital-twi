"""
src.visualization.render_flood_exposure
========================================
GeoStream -- Kenya Sub-County Flood Exposure Dashboard

PURPOSE
-------
Visualize the KEN_ADM2_flood_exposure.csv dataset, which contains
humanitarian impact estimates for flood events across all 291 Kenyan
sub-counties.  The data describes how many women, children, elderly,
schools, and hospitals would be affected by floods of different severity
levels (10-year, 50-year, 100-year, 500-year return periods).

OUTPUT
------
  reports/figures/nairobi_flood_exposure.png

USAGE
-----
  python -m src.visualization.render_flood_exposure
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
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

INPUT_CSV: str = os.path.join(
    "data", "raw", "vectors", "KEN_ADM2_flood_exposure.csv"
)
OUTPUT_PATH: str = os.path.join(
    "reports", "figures", "nairobi_flood_exposure.png"
)


# ============================================================================
# DATA LOADING
# ============================================================================

def _load_csv(path: str) -> list[dict]:
    """Read the flood exposure CSV into a list of dicts."""
    print(f"  Reading {path} ...")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Loaded {len(rows)} sub-county records.")
    return rows


def _sum_column(rows: list[dict], col: str) -> int:
    """Sum a numeric column across all rows, treating blanks as 0."""
    return sum(int(r.get(col, 0) or 0) for r in rows)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def render_flood_exposure() -> None:
    """Build the 4-panel flood exposure summary figure."""

    print()
    print("=" * 66)
    print("  GeoStream | Kenya Flood Exposure Dashboard")
    print("=" * 66)

    # -- Load data ---------------------------------------------------------
    rows = _load_csv(INPUT_CSV)

    # -- Return periods ----------------------------------------------------
    return_periods = ["RP10", "RP50", "RP100", "RP500"]
    rp_labels = [
        "10-Year Flood\n(Happens ~every 10 yrs)",
        "50-Year Flood\n(Happens ~every 50 yrs)",
        "100-Year Flood\n(Happens ~every 100 yrs)",
        "500-Year Flood\n(Rare catastrophic event)",
    ]
    rp_short = ["10-yr", "50-yr", "100-yr", "500-yr"]

    # -- Aggregate key humanitarian metrics --------------------------------
    female_pop   = [_sum_column(rows, f"{rp}_female_pop_30cm")   for rp in return_periods]
    children_u5  = [_sum_column(rows, f"{rp}_children_u5_30cm")  for rp in return_periods]
    elderly      = [_sum_column(rows, f"{rp}_elderly_30cm")       for rp in return_periods]
    education    = [_sum_column(rows, f"{rp}_education_30cm_count") for rp in return_periods]
    hospitals    = [_sum_column(rows, f"{rp}_hospitals_30cm_count") for rp in return_periods]
    primary_hc   = [_sum_column(rows, f"{rp}_primary_healthcare_30cm_count") for rp in return_periods]

    print(f"\n  Humanitarian impact summary:")
    for i, rp in enumerate(rp_short):
        print(f"    {rp:>6}: Women={female_pop[i]:>8,}  Children<5={children_u5[i]:>7,}  "
              f"Elderly={elderly[i]:>6,}  Schools={education[i]:>4}  Hospitals={hospitals[i]:>3}")

    # -- Top 15 most-affected sub-counties (by RP100 female pop) -----------
    for r in rows:
        r["_rp100_female"] = int(r.get("RP100_female_pop_30cm", 0) or 0)
    rows_sorted = sorted(rows, key=lambda r: r["_rp100_female"], reverse=True)
    top_n = 15
    top_rows = rows_sorted[:top_n]
    top_names = [r["ADM2_PCODE"] for r in top_rows]
    top_vals  = [r["_rp100_female"] for r in top_rows]

    # =====================================================================
    # BUILD FIGURE -- 2x2 dashboard
    # =====================================================================
    print("\n[RENDER] Building 4-panel dashboard ...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor="#F8F9FA")

    fig.suptitle(
        "How Many People Would Be Affected by Flooding Across Kenya?",
        fontsize=20, fontweight="bold", color="navy", y=0.98,
    )
    fig.text(
        0.5, 0.945,
        "Data: KEN_ADM2 Flood Exposure Assessment  ·  291 Sub-Counties  ·  "
        "30cm flood depth threshold  ·  Source: OCHA / ICPAC",
        ha="center", fontsize=10, color="#777777", style="italic",
    )

    colors_rp = ["#3498DB", "#E67E22", "#E74C3C", "#8E44AD"]
    thousands = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}")

    # ---- Panel 1: Vulnerable Populations by Flood Severity ---------------
    ax1 = axes[0, 0]
    x = np.arange(len(rp_short))
    width = 0.25
    ax1.bar(x - width, female_pop, width, label="Women & Girls",
            color="#E74C3C", edgecolor="white", zorder=3)
    ax1.bar(x,         children_u5, width, label="Children Under 5",
            color="#F39C12", edgecolor="white", zorder=3)
    ax1.bar(x + width, elderly, width, label="Elderly (60+)",
            color="#3498DB", edgecolor="white", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rp_short, fontsize=10)
    ax1.set_ylabel("Number of People at Risk", fontsize=11)
    ax1.set_title(
        "Vulnerable Populations Exposed to Flooding\nby Flood Severity Level",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax1.yaxis.set_major_formatter(thousands)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.text(
        0.02, 0.02,
        "Bars show estimated people exposed\n"
        "to 30cm+ flooding at each severity level",
        transform=ax1.transAxes, fontsize=8, color="#666",
        style="italic", verticalalignment="bottom",
    )

    # ---- Panel 2: Critical Infrastructure at Risk ------------------------
    ax2 = axes[0, 1]
    infra_labels = ["Schools\n& Education", "Hospitals", "Primary\nHealthcare"]
    for i_rp, (rp, color, label) in enumerate(zip(
        return_periods, colors_rp, rp_short
    )):
        vals = [education[i_rp], hospitals[i_rp], primary_hc[i_rp]]
        x_pos = np.arange(len(infra_labels)) + i_rp * 0.2 - 0.3
        ax2.bar(x_pos, vals, 0.18, label=label, color=color,
                edgecolor="white", zorder=3)
    ax2.set_xticks(np.arange(len(infra_labels)))
    ax2.set_xticklabels(infra_labels, fontsize=10)
    ax2.set_ylabel("Number of Facilities at Risk", fontsize=11)
    ax2.set_title(
        "Critical Infrastructure in Flood Zones\n(Schools, Hospitals, Clinics)",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax2.legend(fontsize=9, title="Flood Return Period", title_fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.text(
        0.02, 0.02,
        "Counts of facilities within areas\n"
        "predicted to flood at 30cm+ depth",
        transform=ax2.transAxes, fontsize=8, color="#666",
        style="italic", verticalalignment="bottom",
    )

    # ---- Panel 3: Top 15 Most Affected Sub-Counties ----------------------
    ax3 = axes[1, 0]
    y_pos = np.arange(top_n)
    bar_colors = plt.cm.Reds(np.linspace(0.35, 0.85, top_n))[::-1]
    ax3.barh(y_pos, top_vals, color=bar_colors, edgecolor="white", zorder=3)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_names, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel("Women & Girls Exposed (100-yr flood)", fontsize=11)
    ax3.set_title(
        "Top 15 Most Flood-Vulnerable Sub-Counties\n(by female population at risk, 100-yr event)",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax3.xaxis.set_major_formatter(thousands)
    ax3.grid(axis="x", alpha=0.3, zorder=0)
    # Value labels
    for j, v in enumerate(top_vals):
        ax3.text(v + 200, j, f"{v:,}", va="center", fontsize=8, color="#333")

    # ---- Panel 4: Escalation — how impact grows with severity ------------
    ax4 = axes[1, 1]
    total_pop = [f + c + e for f, c, e in zip(female_pop, children_u5, elderly)]
    ax4.fill_between(range(4), total_pop, alpha=0.3, color="#E74C3C", zorder=2)
    ax4.plot(range(4), total_pop, "o-", color="#E74C3C", linewidth=2.5,
             markersize=10, zorder=3, label="Total Vulnerable People")
    ax4.plot(range(4), female_pop, "s--", color="#8E44AD", linewidth=2,
             markersize=8, zorder=3, label="Women & Girls")
    ax4.plot(range(4), children_u5, "^--", color="#F39C12", linewidth=2,
             markersize=8, zorder=3, label="Children Under 5")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(rp_labels, fontsize=9)
    ax4.set_ylabel("Number of People Affected", fontsize=11)
    ax4.set_title(
        "How Does Impact Scale With Flood Severity?\n"
        "(More severe floods = more people affected)",
        fontsize=13, fontweight="bold", color="#1B2A4A", pad=10,
    )
    ax4.yaxis.set_major_formatter(thousands)
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3, zorder=0)
    # Annotate the 500-yr point
    ax4.annotate(
        f"{total_pop[-1]:,}\ntotal people",
        xy=(3, total_pop[-1]),
        xytext=(2.3, total_pop[-1] * 1.08),
        fontsize=9, fontweight="bold", color="#C0392B",
        arrowprops=dict(arrowstyle="->", color="#C0392B"),
    )

    # ---- Bottom footnote -------------------------------------------------
    fig.text(
        0.5, 0.005,
        "HOW TO READ: These charts summarize what would happen if floods of different severity "
        "hit Kenya.  A '100-year flood' has a 1% chance of occurring in any given year.  "
        "The numbers represent people and facilities within 30cm flood depth zones.",
        ha="center", va="bottom", fontsize=9, color="#444", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0",
                  edgecolor="#FFB74D", alpha=0.9),
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
    print("  [DONE] Flood exposure dashboard exported successfully.")
    print(f"         Open '{OUTPUT_PATH}' to review.")
    print("=" * 66)


if __name__ == "__main__":
    render_flood_exposure()
