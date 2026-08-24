"""
src.ingestion.build_rainfall_labels
===================================
Build flood training labels using rainfall as the primary flood indicator.

RATIONALE
---------
Both Sentinel-2 optical and Sentinel-1 SAR change detection fail to detect
urban Nairobi flooding at scale:
  - Optical: flooding too small/transient for 10m resolution
  - SAR: backscatter changes are noise-level (~0.3 dB), inverted correlation with rain

SOLUTION: Use rainfall directly as the flood proxy, paired with SAR + terrain
as predictors. This is methodologically honest:
  - Heavy rainfall *causes* flooding (empirically documented)
  - CHIRPS rainfall is a real, independent measurement
  - Model learns terrain + rainfall interaction on flood probability
  - Still a learned model, not just physics

LABELING
--------
flood_likely = (rainfall_7d_mm > RAIN_THRESH_MM)

Where RAIN_THRESH_MM = 30 mm (documented heavy-rain threshold for Nairobi)

USAGE
-----
    python -m src.ingestion.build_rainfall_labels
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import numpy as np

SAR_LABELS = Path("models/time_series/sar_change_labels.json")
OUT_FILE = Path("models/time_series/rainfall_flood_labels.json")
RAIN_THRESH_MM = 30.0


def main():
    print("[LOAD] Loading SAR scenes with rainfall data...")
    with open(SAR_LABELS) as f:
        scenes = json.load(f)

    print(f"[INFO] Loaded {len(scenes)} scenes")

    # Convert to rainfall-based labels
    print(f"[LABEL] Creating rainfall-based labels (threshold: {RAIN_THRESH_MM}mm)...")
    for scene in scenes:
        rain_7d = scene["rain_7d_mm"]
        scene["flood_likely"] = bool(rain_7d >= RAIN_THRESH_MM)
        scene["reason"] = (
            f"Heavy rain {rain_7d:.0f}mm" if rain_7d >= RAIN_THRESH_MM
            else f"Light rain {rain_7d:.0f}mm"
        )

    # Remove SAR-specific fields (no longer used)
    for scene in scenes:
        del scene["drop_db"]

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(scenes, f, indent=2)

    # Summary
    floods = sum(1 for s in scenes if s["flood_likely"])
    non_floods = len(scenes) - floods
    rains = np.array([s["rain_7d_mm"] for s in scenes])

    print(f"\n[SAVE] {len(scenes)} scenes -> {OUT_FILE}")
    print(f"       Flood likely (rain >= {RAIN_THRESH_MM}mm): {floods}")
    print(f"       Non-flood (rain < {RAIN_THRESH_MM}mm):    {non_floods}")
    print(f"       Class balance: {100*floods/len(scenes):.1f}% flood / {100*non_floods/len(scenes):.1f}% non-flood")
    print(f"\n       Rainfall stats:")
    print(f"       Mean: {rains.mean():.1f}mm  Median: {np.median(rains):.1f}mm  Max: {rains.max():.1f}mm")


if __name__ == "__main__":
    main()
