"""
src.models.cross_validate
==========================
Nairobi Urban Flood Digital Twin — K-Fold Cross-Validation Across Real Events

WHY THIS EXISTS
---------------
src.models.train's single event-aware train/val/test split draws its
held-out test set from only 2-3 of the real Sentinel-1 storm-season
events. Comparing two single-split runs on this project showed the
reported test wet-region R2 swinging from -0.26 to -2.42 purely because a
different, harder trio of events happened to land in test the second
time — and within just ONE of those runs, validation wet-region R2 (on a
FIXED set of validation events) swung from -11.5 to -3.4 across 30 epochs
of ordinary training noise (std ~1.6). At this sample size (23 independent
events), which events land in which split dominates the reported number
more than genuine model quality does. A single split is not a reliable
enough signal to defend.

WHAT THIS DOES INSTEAD
-----------------------
Standard K-fold cross-validation across events: split the 23 events into
K roughly-equal groups. For each fold, hold out that group as the test
set, train fresh on the rest (with its own internal train/val split for
checkpoint selection), and evaluate once on the held-out fold. Every one
of the 23 events ends up evaluated exactly once, as genuinely unseen data,
across the K folds combined — so the POOLED metric across all folds'
held-out predictions is a much more trustworthy generalization estimate
than any single fold's (or single split's) number.

The terrain-reconstruction autoencoder (Stage 1) is pretrained ONCE,
shared across all folds — it never sees flood labels or event-specific
data at all (only random patches of the single static terrain grid), so
reusing it introduces no leakage between folds and saves K-1 redundant
pretraining passes.

USAGE
-----
    python -m src.models.cross_validate
    python -m src.models.cross_validate --folds 5 --epochs-lstm 15
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.models.train import (
    PROCESSED_DIR, AUTOENCODER_DIR, TIME_SERIES_DIR,
    fit_convlstm, pretrain_terrain_autoencoder, _rmse_mae_r2, _wet_region_metrics,
    fit_linear_calibration, apply_calibration,
)
from src.utils.memory_check import MemoryGuard

CV_RESULTS_PATH = TIME_SERIES_DIR / "cross_validation_results.json"
CV_FOLD_DIR = TIME_SERIES_DIR / "cv_folds"


def make_event_folds(event_ids: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Split the unique source events into k roughly-equal, non-overlapping groups."""
    unique_events = np.unique(event_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_events)
    return [np.array(f) for f in np.array_split(shuffled, k)]


def run_cross_validation(
    k: int = 5,
    epochs_ae_pretrain: int = 25,
    epochs_lstm: int = 15,
    val_fraction_of_remaining: float = 0.15,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 42,
) -> dict | None:
    TIME_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    CV_FOLD_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    guard = MemoryGuard()

    x_path = PROCESSED_DIR / "X_train.npy"
    y_path = PROCESSED_DIR / "y_train.npy"
    event_ids_path = PROCESSED_DIR / "event_ids.npy"
    terrain_path = PROCESSED_DIR / "static_terrain_features.npy"

    if not x_path.exists() or not event_ids_path.exists():
        logger.error("Dataset files (or event_ids.npy) missing. Run dataset_builder first!")
        return None

    X_data = np.load(x_path)
    y_data = np.load(y_path)
    event_ids = np.load(event_ids_path)
    static_terrain = np.load(terrain_path) if terrain_path.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cross-validation: {len(np.unique(event_ids))} events, {len(X_data)} samples, device={device}")

    # ---- Shared Stage-1 pretraining (fold-independent — no leakage) --------
    ae_encoder = None
    if static_terrain is not None:
        logger.info("--- Pretraining shared terrain autoencoder (used to seed every fold) ---")
        ae_model = pretrain_terrain_autoencoder(static_terrain, epochs=epochs_ae_pretrain, device=device)
        ae_encoder = ae_model.encoder
        torch.save(ae_model.state_dict(), AUTOENCODER_DIR / "spatial_autoencoder.pth")
    else:
        logger.warning("No static_terrain_features.npy found — every fold's encoder trains from scratch.")

    folds = make_event_folds(event_ids, k=k, seed=seed)
    logger.info(f"{k} folds: " + " | ".join(f"fold{i}={sorted(f)}" for i, f in enumerate(folds)))

    fold_results = []
    pooled_preds, pooled_preds_calibrated, pooled_targets = [], [], []

    for fold_i, test_events in enumerate(folds):
        logger.info(f"\n{'=' * 70}\nFOLD {fold_i + 1}/{k} — held-out test events: {sorted(test_events)}\n{'=' * 70}")

        remaining_events = np.array([e for e in np.unique(event_ids) if e not in set(test_events)])
        rng = np.random.default_rng(seed + fold_i)
        shuffled_remaining = rng.permutation(remaining_events)
        n_val_ev = max(1, round(val_fraction_of_remaining * len(remaining_events)))
        val_events = set(shuffled_remaining[:n_val_ev])
        train_events = set(shuffled_remaining[n_val_ev:])

        train_idx = np.where(np.isin(event_ids, list(train_events)))[0]
        val_idx = np.where(np.isin(event_ids, list(val_events)))[0]
        test_idx = np.where(np.isin(event_ids, list(test_events)))[0]
        logger.info(
            f"  train={len(train_idx)} samples/{len(train_events)} events | "
            f"val={len(val_idx)} samples/{len(val_events)} events | "
            f"test={len(test_idx)} samples/{len(test_events)} events"
        )

        fold_checkpoint = CV_FOLD_DIR / f"fold_{fold_i}_conv_lstm.pth"
        model, best_val_wet_rmse, history = fit_convlstm(
            X_data, y_data, train_idx, val_idx,
            ae_encoder=ae_encoder, epochs_lstm=epochs_lstm, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, dropout=dropout, device=device,
            checkpoint_path=fold_checkpoint, log_prefix=f"Fold{fold_i}", guard=guard,
        )

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(X_data[val_idx]).float().to(device)).cpu().numpy()
            test_preds = model(torch.from_numpy(X_data[test_idx]).float().to(device)).cpu().numpy()
        val_targets = y_data[val_idx][:, None, :, :]
        test_targets = y_data[test_idx][:, None, :, :]

        # Calibration fit on THIS fold's validation set only, applied to
        # THIS fold's test set — never fit on data being evaluated. Raw
        # network output is systematically under-scaled (see
        # fit_linear_calibration's docstring); this corrects it per-fold.
        cal_scale, cal_bias = fit_linear_calibration(val_preds, val_targets)
        test_preds_calibrated = apply_calibration(test_preds, cal_scale, cal_bias)

        test_metrics = _rmse_mae_r2(test_preds, test_targets)
        test_wet_metrics = _wet_region_metrics(test_preds, test_targets)
        test_wet_metrics_calibrated = _wet_region_metrics(test_preds_calibrated, test_targets)
        logger.info(
            f"FOLD {fold_i + 1} TEST — whole-grid MAE={test_metrics['mae']:.4f}m | "
            + (f"wet-region raw MAE={test_wet_metrics['mae']:.4f}m R2={test_wet_metrics['r2']:.4f} | "
               f"calibrated MAE={test_wet_metrics_calibrated['mae']:.4f}m R2={test_wet_metrics_calibrated['r2']:.4f}"
               if test_wet_metrics else "wet-region n/a")
        )

        fold_results.append({
            "fold": fold_i,
            "test_events": sorted(test_events.tolist()),
            "val_events": sorted(val_events),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "best_val_wet_rmse": best_val_wet_rmse,
            "calibration": {"scale": cal_scale, "bias": cal_bias},
            "test_metrics_wholegrid": test_metrics,
            "test_metrics_wet_region_raw": test_wet_metrics,
            "test_metrics_wet_region_calibrated": test_wet_metrics_calibrated,
        })
        pooled_preds.append(test_preds)
        pooled_preds_calibrated.append(test_preds_calibrated)
        pooled_targets.append(test_targets)

        del model
        gc.collect()

    # ---- Pooled metric: every event's held-out prediction, combined --------
    # More robust than averaging each fold's R2 independently — a fold whose
    # 4-5 test events happen to have low target variance can otherwise
    # produce a wildly unstable per-fold R2 that skews a simple average.
    all_preds = np.concatenate(pooled_preds, axis=0)
    all_preds_calibrated = np.concatenate(pooled_preds_calibrated, axis=0)
    all_targets = np.concatenate(pooled_targets, axis=0)
    pooled_wholegrid = _rmse_mae_r2(all_preds, all_targets)
    pooled_wet = _wet_region_metrics(all_preds, all_targets)
    pooled_wet_calibrated = _wet_region_metrics(all_preds_calibrated, all_targets)

    per_fold_wet_rmse = [f["test_metrics_wet_region_raw"]["rmse"] for f in fold_results if f["test_metrics_wet_region_raw"]]
    per_fold_wet_r2 = [f["test_metrics_wet_region_raw"]["r2"] for f in fold_results if f["test_metrics_wet_region_raw"]]
    per_fold_wet_r2_calibrated = [f["test_metrics_wet_region_calibrated"]["r2"] for f in fold_results if f["test_metrics_wet_region_calibrated"]]

    summary = {
        "k_folds": k,
        "epochs_lstm_per_fold": epochs_lstm,
        "total_events": int(len(np.unique(event_ids))),
        "pooled_test_metrics_wholegrid": pooled_wholegrid,
        "pooled_test_metrics_wet_region_raw": pooled_wet,
        "pooled_test_metrics_wet_region_calibrated": pooled_wet_calibrated,
        "per_fold_wet_rmse_mean": float(np.mean(per_fold_wet_rmse)) if per_fold_wet_rmse else None,
        "per_fold_wet_rmse_std": float(np.std(per_fold_wet_rmse)) if per_fold_wet_rmse else None,
        "per_fold_wet_r2_mean": float(np.mean(per_fold_wet_r2)) if per_fold_wet_r2 else None,
        "per_fold_wet_r2_std": float(np.std(per_fold_wet_r2)) if per_fold_wet_r2 else None,
        "per_fold_wet_r2_calibrated_mean": float(np.mean(per_fold_wet_r2_calibrated)) if per_fold_wet_r2_calibrated else None,
        "per_fold_wet_r2_calibrated_std": float(np.std(per_fold_wet_r2_calibrated)) if per_fold_wet_r2_calibrated else None,
        "folds": fold_results,
    }
    with open(CV_RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 70}\nCROSS-VALIDATION COMPLETE ({k} folds, every event held out exactly once)\n{'=' * 70}")
    logger.info(
        f"POOLED wet-region metrics (all {len(np.unique(event_ids))} events combined, each evaluated once as held-out):"
    )
    if pooled_wet:
        logger.info(f"  RAW:        MAE={pooled_wet['mae']:.4f} m | RMSE={pooled_wet['rmse']:.4f} m | R2={pooled_wet['r2']:.4f}")
    if pooled_wet_calibrated:
        logger.info(
            f"  CALIBRATED: MAE={pooled_wet_calibrated['mae']:.4f} m | RMSE={pooled_wet_calibrated['rmse']:.4f} m | "
            f"R2={pooled_wet_calibrated['r2']:.4f}"
        )
    if per_fold_wet_rmse:
        logger.info(
            f"Per-fold wet-region RMSE (raw): mean={summary['per_fold_wet_rmse_mean']:.4f} m, "
            f"std={summary['per_fold_wet_rmse_std']:.4f} m across {k} folds"
        )
    if per_fold_wet_r2_calibrated:
        logger.info(
            f"Per-fold wet-region R2 (calibrated): mean={summary['per_fold_wet_r2_calibrated_mean']:.4f}, "
            f"std={summary['per_fold_wet_r2_calibrated_std']:.4f} across {k} folds"
        )
    logger.info(f"Full results: {CV_RESULTS_PATH}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.models.cross_validate")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs-ae-pretrain", type=int, default=25)
    p.add_argument("--epochs-lstm", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_cross_validation(
        k=args.folds, epochs_ae_pretrain=args.epochs_ae_pretrain,
        epochs_lstm=args.epochs_lstm, seed=args.seed,
    )
