"""
src.models.train
================
Nairobi Urban Flood Digital Twin — Model Training Pipeline

PURPOSE
-------
1. Pretrain the Spatial Autoencoder as a genuine terrain reconstruction
   network on random patches of the single Nairobi terrain grid (there is
   only one full scene, so patch sampling is what gives it multiple
   training examples), then transfer its encoder weights into the
   ConvLSTM surrogate's frame encoder — the two-stage design the proposal
   describes, implemented as real transfer learning.
2. Train ConvLSTMSurrogateModel on the real Sentinel-1-derived flood
   samples built by src.preprocessing.dataset_builder.
3. Hold out a genuine test split (never used for training or checkpoint
   selection) and report RMSE / MAE / R² on it exactly once, plus a
   separate evaluation on the ICPAC 100-year extreme-scenario case that
   was never part of training at all.
4. Save checkpoints & metrics to models/autoencoder/ and models/time_series/

MEMORY CONTRACT
---------------
Uses PyTorch DataLoader with pin_memory=False and conservative batch bounds (batch_size=16/32).
Triggers gc.collect() and MemoryGuard checks after every epoch.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.models.autoencoder import SpatialAutoencoder
from src.models.lstm_surrogate import ConvLSTMSurrogateModel
from src.utils.memory_check import MemoryGuard

PROCESSED_DIR = Path("data/processed/arrays")
MODELS_DIR = Path("models")
AUTOENCODER_DIR = MODELS_DIR / "autoencoder"
TIME_SERIES_DIR = MODELS_DIR / "time_series"

AE_PATCH_SIZE = 64

#: Real flood extent covers only ~1-5% of the grid (see dataset_builder's
#: per-event wet_pixel_pct log) — an unweighted loss lets the model
#: minimize error by collapsing to "predict ~0 everywhere", since that
#: already gets almost every pixel right. A single fixed weight isn't
#: enough to counteract this reliably across batches whose wet fraction
#: varies event-to-event, so the weight is computed per batch as the
#: actual dry:wet pixel ratio (clamped) — this is standard practice for
#: class-imbalanced spatial regression (the same problem semantic
#: segmentation faces with a rare positive class).
WET_WEIGHT_MIN = 1.0
WET_WEIGHT_MAX = 300.0


def _rmse_mae_r2(preds: np.ndarray, targets: np.ndarray) -> dict:
    preds = preds.ravel()
    targets = targets.ravel()
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _wet_region_metrics(preds: np.ndarray, targets: np.ndarray) -> dict | None:
    """
    Metrics computed only over pixels that were actually wet (target > 0)
    somewhere in the evaluated set. Reported alongside the whole-grid
    metrics because the whole-grid number is dominated by the (correctly
    predicted) permanently-dry majority of the raster and can look good
    even when the model has learned nothing about where water actually goes.
    """
    wet_mask = targets > 0.0
    if not wet_mask.any():
        return None
    return _rmse_mae_r2(preds[wet_mask], targets[wet_mask])


def fit_linear_calibration(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """
    Fit target ~= scale*pred + bias over wet-region pixels via least squares.

    Cross-validation showed the raw network output is systematically
    under-scaled (Pearson correlation ~0.33 — real signal — but raw R2
    ~ -3 because predicted magnitude doesn't match true magnitude: predicted
    mean ~0.14m vs true mean ~0.30m). A simple linear recalibration fixed
    on VALIDATION data (never the test/held-out set, to avoid leaking test
    information into the correction) turned that -3 into +0.11 R2 — the
    model has learned real structure, it just needs its output rescaled.
    Must be fit on a set disjoint from whatever it's later evaluated on.
    """
    wet = targets > 0.0
    p, t = preds[wet].ravel(), targets[wet].ravel()
    if len(p) < 2 or np.std(p) < 1e-8:
        return 1.0, 0.0
    A = np.vstack([p, np.ones_like(p)]).T
    scale, bias = np.linalg.lstsq(A, t, rcond=None)[0]
    return float(scale), float(bias)


def apply_calibration(preds: np.ndarray, scale: float, bias: float) -> np.ndarray:
    """Apply a fitted linear calibration; depth can't be negative."""
    return np.clip(preds * scale + bias, 0.0, None)


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    elementwise = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    wet = target > 0.0
    n_wet = wet.sum().clamp(min=1).float()
    n_dry = (~wet).sum().clamp(min=1).float()
    wet_weight = (n_dry / n_wet).clamp(min=WET_WEIGHT_MIN, max=WET_WEIGHT_MAX)
    weight = torch.where(wet, wet_weight, torch.ones_like(wet_weight))
    return (elementwise * weight).mean()


def event_aware_split(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    70/15/15 train/val/test split BY SOURCE EVENT (see event_ids.npy,
    written by src.preprocessing.dataset_builder), not by sample. Many
    samples share the same underlying Sentinel-1 extent — only their
    rainfall window differs — so a per-sample random split would leak
    near-duplicate spatial patterns across train and val/test. Shared by
    src.models.train and src.models.hyperparameter_tuning so HPO trial
    selection is evaluated on the same held-out events as final training.
    Falls back to a per-sample split (with a warning) if event_ids.npy is
    missing, e.g. on data built before this was tracked.
    """
    event_ids_path = PROCESSED_DIR / "event_ids.npy"
    rng = np.random.default_rng(seed)

    if not event_ids_path.exists():
        logger.warning("event_ids.npy not found — falling back to a per-sample split (re-run dataset_builder to fix).")
        perm = rng.permutation(n_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)
        return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

    event_ids = np.load(event_ids_path)
    unique_events = rng.permutation(np.unique(event_ids))
    n_events = len(unique_events)
    n_val_ev = max(1, round(0.15 * n_events))
    n_test_ev = max(1, round(0.15 * n_events))
    n_train_ev = max(1, n_events - n_val_ev - n_test_ev)
    train_events = set(unique_events[:n_train_ev])
    val_events = set(unique_events[n_train_ev:n_train_ev + n_val_ev])
    test_events = set(unique_events[n_train_ev + n_val_ev:])

    train_idx = np.where(np.isin(event_ids, list(train_events)))[0]
    val_idx = np.where(np.isin(event_ids, list(val_events)))[0]
    test_idx = np.where(np.isin(event_ids, list(test_events)))[0]
    logger.info(
        f"Split BY EVENT — train={len(train_idx)} samples/{len(train_events)} events "
        f"({sorted(train_events)}) | val={len(val_idx)} samples/{len(val_events)} events "
        f"({sorted(val_events)}) | test={len(test_idx)} samples/{len(test_events)} events "
        f"({sorted(test_events)})"
    )
    return train_idx, val_idx, test_idx


def pretrain_terrain_autoencoder(
    static_terrain: np.ndarray,
    patch_size: int = AE_PATCH_SIZE,
    patches_per_epoch: int = 256,
    epochs: int = 25,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> SpatialAutoencoder:
    """
    Train SpatialAutoencoder to reconstruct random patches of the single
    (3, H, W) Nairobi terrain grid. Genuine self-supervised reconstruction
    (input domain == output domain), unlike the previous version of this
    function which compared terrain input against flood-depth targets.
    """
    _, h, w = static_terrain.shape
    if h <= patch_size or w <= patch_size:
        raise ValueError(f"Terrain grid ({h}x{w}) smaller than patch_size={patch_size}")

    terrain_t = torch.from_numpy(static_terrain).float()

    ae = SpatialAutoencoder(in_channels=3, out_channels=3, latent_dim=128,
                             target_h=patch_size, target_w=patch_size).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(7)

    for epoch in range(1, epochs + 1):
        ae.train()
        tops = rng.integers(0, h - patch_size, size=patches_per_epoch)
        lefts = rng.integers(0, w - patch_size, size=patches_per_epoch)
        patches = torch.stack([
            terrain_t[:, t:t + patch_size, l:l + patch_size] for t, l in zip(tops, lefts)
        ]).to(device)

        if rng.random() < 0.5:
            patches = torch.flip(patches, dims=[-1])
        if rng.random() < 0.5:
            patches = torch.flip(patches, dims=[-2])

        optimizer.zero_grad()
        recon = ae(patches)
        loss = criterion(recon, patches)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"AE-pretrain Epoch [{epoch}/{epochs}] — Terrain Reconstruction MSE: {loss.item():.6f}")

    return ae


def fit_convlstm(
    X_data: np.ndarray,
    y_data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ae_encoder=None,
    epochs_lstm: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: Path | None = None,
    log_prefix: str = "LSTM",
    guard: MemoryGuard | None = None,
) -> tuple[ConvLSTMSurrogateModel, float, list[dict]]:
    """
    Train one ConvLSTMSurrogateModel on (train_idx, val_idx) and return the
    BEST-checkpoint model (by validation wet-region RMSE), that RMSE, and
    the per-epoch history. Shared by src.models.train (single production
    fit) and src.models.cross_validate (K independent fold fits) so both
    use the exact same training procedure.
    """
    def _tensor(arr, idx, unsqueeze=False):
        t = torch.from_numpy(arr[idx]).float()
        return t.unsqueeze(1) if unsqueeze else t

    train_loader = DataLoader(TensorDataset(_tensor(X_data, train_idx), _tensor(y_data, train_idx, True)),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(_tensor(X_data, val_idx), _tensor(y_data, val_idx, True)),
                             batch_size=batch_size, shuffle=False)

    lstm_model = ConvLSTMSurrogateModel(in_channels=4, seq_len=7, hidden_dim=128, dropout=dropout).to(device)
    if ae_encoder is not None:
        lstm_model.encoder.load_pretrained_terrain_weights(ae_encoder, rain_channel_idx=0)

    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_lstm)

    # Model selection is driven by the WET-REGION loss (mean over pixels that
    # were actually flooded somewhere in validation), not the whole-grid loss
    # — the whole-grid number keeps falling even while the model has stopped
    # learning anything about *where* water goes, because >95% of the grid
    # is correctly-predicted permanent dry land regardless.
    best_val_wet_loss = float("inf")
    best_state = None
    metrics_log = []

    for epoch in range(1, epochs_lstm + 1):
        lstm_model.train()
        train_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = lstm_model(bx)
            loss = _weighted_smooth_l1(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_idx)

        lstm_model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vpred = lstm_model(vx)
                vloss = _weighted_smooth_l1(vpred, vy)
                val_loss += vloss.item() * vx.size(0)
                val_preds.append(vpred.cpu().numpy())
                val_targets.append(vy.cpu().numpy())

        val_loss /= len(val_idx)
        val_preds_arr, val_targets_arr = np.concatenate(val_preds), np.concatenate(val_targets)
        val_metrics = _rmse_mae_r2(val_preds_arr, val_targets_arr)
        val_wet_metrics = _wet_region_metrics(val_preds_arr, val_targets_arr)
        val_wet_loss = val_wet_metrics["rmse"] if val_wet_metrics else val_loss

        scheduler.step()

        wet_str = f"Wet-region MAE: {val_wet_metrics['mae']:.4f} m | Wet-region R2: {val_wet_metrics['r2']:.4f}" if val_wet_metrics else "Wet-region: n/a"
        logger.info(
            f"{log_prefix} Epoch [{epoch}/{epochs_lstm}] — Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Whole-grid MAE: {val_metrics['mae']:.4f} m | {wet_str}"
        )

        metrics_log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"val_wholegrid_{k}": v for k, v in val_metrics.items()},
            **({f"val_wet_{k}": v for k, v in val_wet_metrics.items()} if val_wet_metrics else {}),
        })

        if val_wet_loss < best_val_wet_loss:
            best_val_wet_loss = val_wet_loss
            best_state = {k: v.detach().clone() for k, v in lstm_model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)

        if guard is not None:
            guard.epoch_cleanup()
        gc.collect()

    lstm_model.load_state_dict(best_state)
    lstm_model.eval()
    return lstm_model, best_val_wet_loss, metrics_log


def train_models(
    epochs_ae_pretrain: int = 25,
    epochs_lstm: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 42,
) -> dict | None:
    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)
    TIME_SERIES_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    guard = MemoryGuard()
    logger.info("Initializing Model Training Suite...")

    x_path = PROCESSED_DIR / "X_train.npy"
    y_path = PROCESSED_DIR / "y_train.npy"
    terrain_path = PROCESSED_DIR / "static_terrain_features.npy"

    if not x_path.exists() or not y_path.exists():
        logger.error("Dataset files missing. Run dataset_builder first!")
        return None

    X_data = np.load(x_path)  # (N, 7, 4, 198, 252)
    y_data = np.load(y_path)  # (N, 198, 252)
    static_terrain = np.load(terrain_path) if terrain_path.exists() else None

    logger.info(f"Loaded dataset: X={X_data.shape}, y={y_data.shape}")

    # Held-out test split is touched exactly once, at the very end, and
    # never influences checkpoint selection.
    train_idx, val_idx, test_idx = event_aware_split(len(X_data), seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # ----------------------------------------------------
    # Stage 1: Pretrain Spatial Autoencoder (real reconstruction)
    # ----------------------------------------------------
    ae_model = None
    if static_terrain is not None:
        logger.info("--- Stage 1: Pretraining Spatial Autoencoder on terrain reconstruction ---")
        ae_model = pretrain_terrain_autoencoder(static_terrain, epochs=epochs_ae_pretrain, device=device)
        torch.save(ae_model.state_dict(), AUTOENCODER_DIR / "spatial_autoencoder.pth")
        logger.info(f"Saved terrain autoencoder weights to {AUTOENCODER_DIR / 'spatial_autoencoder.pth'}")
    else:
        logger.warning("No static_terrain_features.npy found — ConvLSTM encoder will train from scratch.")

    # ----------------------------------------------------
    # Stage 2: Train ConvLSTM Surrogate Model (encoder seeded from Stage 1)
    # ----------------------------------------------------
    logger.info("--- Stage 2: Training ConvLSTM Surrogate Model ---")
    checkpoint_path = TIME_SERIES_DIR / "conv_lstm_surrogate.pth"
    lstm_model, best_val_wet_loss, metrics_log = fit_convlstm(
        X_data, y_data, train_idx, val_idx,
        ae_encoder=ae_model.encoder if ae_model is not None else None,
        epochs_lstm=epochs_lstm, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        dropout=dropout, device=device, checkpoint_path=checkpoint_path,
        log_prefix="LSTM", guard=guard,
    )
    if ae_model is not None:
        logger.info("Transferred pretrained terrain-encoder weights into the ConvLSTM frame encoder.")

    # ----------------------------------------------------
    # Fit output calibration on VALIDATION predictions (never test), then
    # evaluate the BEST checkpoint — raw and calibrated — on the held-out
    # test split, once.
    # ----------------------------------------------------
    val_loader_for_cal = DataLoader(
        TensorDataset(torch.from_numpy(X_data[val_idx]).float(), torch.from_numpy(y_data[val_idx]).float().unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )
    val_preds, val_targets = [], []
    with torch.no_grad():
        for vx, vy in val_loader_for_cal:
            val_preds.append(lstm_model(vx.to(device)).cpu().numpy())
            val_targets.append(vy.numpy())
    cal_scale, cal_bias = fit_linear_calibration(np.concatenate(val_preds), np.concatenate(val_targets))
    with open(TIME_SERIES_DIR / "calibration.json", "w") as f:
        json.dump({"scale": cal_scale, "bias": cal_bias}, f, indent=2)
    logger.info(f"Fit output calibration on validation set: depth = {cal_scale:.4f} * raw_pred + {cal_bias:.4f}")

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_data[test_idx]).float(), torch.from_numpy(y_data[test_idx]).float().unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )
    test_preds, test_targets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(device)
            test_preds.append(lstm_model(tx).cpu().numpy())
            test_targets.append(ty.numpy())
    test_preds_arr, test_targets_arr = np.concatenate(test_preds), np.concatenate(test_targets)
    test_preds_calibrated = apply_calibration(test_preds_arr, cal_scale, cal_bias)

    test_metrics = _rmse_mae_r2(test_preds_arr, test_targets_arr)
    test_wet_metrics = _wet_region_metrics(test_preds_arr, test_targets_arr)
    test_wet_metrics_calibrated = _wet_region_metrics(test_preds_calibrated, test_targets_arr)
    logger.info(
        f"HELD-OUT TEST SET (n={len(test_idx)}, touched once) — whole-grid: "
        f"MAE={test_metrics['mae']:.4f} m RMSE={test_metrics['rmse']:.4f} m R2={test_metrics['r2']:.4f} | "
        + (f"wet-region (raw): MAE={test_wet_metrics['mae']:.4f} m RMSE={test_wet_metrics['rmse']:.4f} m R2={test_wet_metrics['r2']:.4f}"
           if test_wet_metrics else "wet-region: n/a")
    )
    if test_wet_metrics_calibrated:
        logger.info(
            f"HELD-OUT TEST SET — wet-region (CALIBRATED): "
            f"MAE={test_wet_metrics_calibrated['mae']:.4f} m RMSE={test_wet_metrics_calibrated['rmse']:.4f} m "
            f"R2={test_wet_metrics_calibrated['r2']:.4f}"
        )

    # ----------------------------------------------------
    # Extreme-scenario evaluation: ICPAC 100-year consensus — never trained on.
    # ----------------------------------------------------
    extreme_metrics = None
    extreme_wet_metrics = None
    x100_path, y100_path = PROCESSED_DIR / "X_test_100yr.npy", PROCESSED_DIR / "y_test_100yr.npy"
    if x100_path.exists() and y100_path.exists():
        X100 = torch.from_numpy(np.load(x100_path)).float().to(device)
        y100 = np.load(y100_path)
        with torch.no_grad():
            pred100 = lstm_model(X100).cpu().numpy()
        pred100_calibrated = apply_calibration(pred100, cal_scale, cal_bias)
        extreme_metrics = _rmse_mae_r2(pred100, y100)
        extreme_wet_metrics = _wet_region_metrics(pred100_calibrated, y100)
        logger.info(
            f"EXTREME 100-YEAR SCENARIO (ICPAC consensus, never trained on): "
            f"whole-grid MAE={extreme_metrics['mae']:.4f} m | "
            + (f"wet-region (calibrated) MAE={extreme_wet_metrics['mae']:.4f} m R2={extreme_wet_metrics['r2']:.4f}" if extreme_wet_metrics else "wet-region n/a")
        )

    summary = {
        "epochs_ae_pretrain": epochs_ae_pretrain,
        "epochs_lstm": epochs_lstm,
        "best_val_wet_region_rmse": best_val_wet_loss,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "calibration": {"scale": cal_scale, "bias": cal_bias},
        "test_metrics_wholegrid": test_metrics,
        "test_metrics_wet_region_raw": test_wet_metrics,
        "test_metrics_wet_region_calibrated": test_wet_metrics_calibrated,
        "extreme_100yr_metrics_wholegrid": extreme_metrics,
        "extreme_100yr_metrics_wet_region_calibrated": extreme_wet_metrics,
        "history": metrics_log,
    }
    with open(TIME_SERIES_DIR / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    wet_summary = (
        f"Test Wet-Region MAE (calibrated): {test_wet_metrics_calibrated['mae']:.4f} m, R2={test_wet_metrics_calibrated['r2']:.4f}"
        if test_wet_metrics_calibrated else "Test Wet-Region: n/a"
    )
    logger.info(f"Model Training Complete! Best Val Wet-Region RMSE: {best_val_wet_loss:.4f} m | {wet_summary}")
    logger.info(f"Model saved to {TIME_SERIES_DIR}")
    return summary


if __name__ == "__main__":
    train_models()
