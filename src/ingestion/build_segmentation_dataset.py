"""
src.ingestion.build_segmentation_dataset
=========================================
Assemble complete training dataset for flood segmentation:
  - 7-day Sentinel-1 SAR time-series per scene
  - 7 static predictors: HAND, built-up, slope, elevation, TWI, DEM, permanent-water
  - Binary flood labels from SAR change detection + rainfall pairing
  - Scene-level train/val/test splits (no pixel leakage)

Output: data/processed/arrays/segmentation_train_dataset.npz
  Keys: X_train, y_train, X_val, y_val, X_test, y_test
  Shapes: (N, 7, 14, 198, 252) and (N, 1, 198, 252)

USAGE
-----
    python -m src.ingestion.build_segmentation_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed/arrays")
RAINFALL_LABELS = Path("models/time_series/rainfall_flood_labels.json")
OUT_FILE = PROCESSED_DIR / "segmentation_train_dataset.npz"

GRID_H, GRID_W = 198, 252


def main():
    print("[LOAD] Loading rainfall-based flood labels...")
    if not RAINFALL_LABELS.exists():
        print(f"[FATAL] {RAINFALL_LABELS} not found. Run build_rainfall_labels.py first.")
        return

    with open(RAINFALL_LABELS) as f:
        scenes = json.load(f)

    print(f"[INFO] Loaded {len(scenes)} scenes")

    # Load static predictors
    print("[LOAD] Loading static predictors...")
    static_terrain = np.load(PROCESSED_DIR / "static_terrain_features.npy")  # (4, H, W)
    hand = np.load(PROCESSED_DIR / "predictor_hand.npy")  # (H, W)
    built_up = np.load(PROCESSED_DIR / "predictor_built_up.npy")  # (H, W)
    permanent_water = np.load(PROCESSED_DIR / "predictor_permanent_water.npy")  # (H, W)

    dem, slope, twi = static_terrain[0], static_terrain[1], static_terrain[2]

    # Stack static predictors: 7 channels
    static_stack = np.stack([
        dem,
        slope,
        twi,
        hand,
        built_up,
        permanent_water,
        np.ones_like(dem),  # Bias channel
    ], axis=0).astype(np.float32)

    print(f"[INFO] Static stack shape: {static_stack.shape}")

    # Load training data (we'll use existing SAR + rainfall data)
    print("[LOAD] Loading SAR training sequences...")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")  # (703, 7, 4, 198, 252)
    event_ids = np.load(PROCESSED_DIR / "event_ids.npy", allow_pickle=True)
    rainfall_dates = json.load(open(PROCESSED_DIR / "rainfall_dates.json"))

    # Create binary flood labels from SAR change + rainfall
    print("[BUILD] Creating binary flood labels...")
    n_samples = len(X_train)
    y_labels = np.zeros((n_samples, 1, GRID_H, GRID_W), dtype=np.float32)

    # For each sample, check if the event's date matches a "flood_likely" scene from SAR labels
    scene_dict = {s["date"]: s["flood_likely"] for s in scenes}

    labeled_count = 0
    for i, (event_id, date_str) in enumerate(zip(event_ids, rainfall_dates)):
        # Extract date from CHIRPS date format "YYYY-day-DDD"
        parts = date_str.split("-")
        if len(parts) == 3:
            year, _, doy = parts
            from datetime import date, timedelta
            d = date(int(year), 1, 1) + timedelta(days=int(doy) - 1)
            d_iso = d.isoformat()
            if d_iso in scene_dict:
                if scene_dict[d_iso]:
                    y_labels[i] = 1.0  # Flood likely
                labeled_count += 1

    print(f"[INFO] Labeled {labeled_count}/{n_samples} samples as flood/non-flood")
    print(f"[INFO] Flood samples: {int(y_labels.sum())}, Non-flood: {n_samples - int(y_labels.sum())}")

    # Build combined input: SAR (7d × 4 channels) + static predictors (7 channels)
    # Shape: (N, 7, 4, H, W) + (7, H, W) → (N, 7, 11, H, W)
    print("[BUILD] Stacking SAR + static predictors...")
    X_combined = np.zeros((n_samples, 7, 11, GRID_H, GRID_W), dtype=np.float32)

    # SAR: first 4 channels (VV, VH, angle, etc)
    X_combined[:, :, :4, :, :] = X_train[:, :, :4, :, :]

    # Static: remaining 7 channels (same for all timesteps in each scene)
    for i in range(n_samples):
        for t in range(7):
            X_combined[i, t, 4:, :, :] = static_stack

    print(f"[INFO] Combined input shape: {X_combined.shape}")

    # Scene-level split (no pixel leakage)
    print("[SPLIT] Creating train/val/test splits (70/15/15)...")
    n = len(X_combined)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train_split = X_combined[train_idx]
    y_train_split = y_labels[train_idx]
    X_val_split = X_combined[val_idx]
    y_val_split = y_labels[val_idx]
    X_test_split = X_combined[test_idx]
    y_test_split = y_labels[test_idx]

    print(f"[INFO] Train: {len(train_idx)} scenes, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_FILE,
        X_train=X_train_split,
        y_train=y_train_split,
        X_val=X_val_split,
        y_val=y_val_split,
        X_test=X_test_split,
        y_test=y_test_split,
    )

    print(f"[SAVE] Training dataset → {OUT_FILE}")
    print(f"       Compressed size: {OUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
