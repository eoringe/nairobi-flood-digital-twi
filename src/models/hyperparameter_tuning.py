"""
src.models.hyperparameter_tuning
================================
Hyperparameter Optimization (HPO) Suite for ConvLSTM Flood Surrogate Model

Performs systematic grid search / trial evaluation across:
1. Learning rates (1e-4, 5e-4, 1e-3, 3e-3)
2. Hidden dimensions (64, 128, 256)
3. Optimizers (Adam vs AdamW with weight decay)
4. Dropout
5. LR Schedulers (CosineAnnealingLR vs ReduceLROnPlateau)

Uses ONLY the train/val split — the held-out test split and the ICPAC
100-year extreme-scenario case managed by src.models.train are never
touched here, so final test-set numbers are not influenced by trial
selection. Trial ranking and the loss function itself both use the same
wet-region-weighted objective as src.models.train (see `_weighted_smooth_l1`
there) since the real flood extent covers only ~1-5% of the grid and an
unweighted loss rewards trials that just predict close to zero everywhere.

Saves detailed trial metrics and identifies the optimal hyperparameter combination.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.models.lstm_surrogate import ConvLSTMSurrogateModel
from src.models.train import _rmse_mae_r2, _wet_region_metrics, _weighted_smooth_l1, event_aware_split

PROCESSED_DIR = Path("data/processed/arrays")
MODELS_DIR = Path("models")
HPO_RESULTS_PATH = MODELS_DIR / "hpo_results.json"


def run_hyperparameter_search(seed: int = 42):
    logger.info("Initializing ConvLSTM Hyperparameter Optimization (HPO) Suite...")

    x_path = PROCESSED_DIR / "X_train.npy"
    y_path = PROCESSED_DIR / "y_train.npy"

    if not x_path.exists() or not y_path.exists():
        logger.error("Dataset missing. Run dataset_builder first!")
        return

    X_data = np.load(x_path)  # (N, 7, 4, 198, 252)
    y_data = np.load(y_path)  # (N, 198, 252)

    # Same event-aware split as src.models.train, same seed — this never
    # sees the events held out as test there, and val/train never share an
    # event's underlying Sentinel-1 extent.
    train_idx, val_idx, _test_idx = event_aware_split(len(X_data), seed=seed)

    X_train = torch.from_numpy(X_data[train_idx]).float()
    y_train = torch.from_numpy(y_data[train_idx]).float().unsqueeze(1)
    X_val = torch.from_numpy(X_data[val_idx]).float()
    y_val = torch.from_numpy(y_data[val_idx]).float().unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"HPO compute device: {device} | train={len(train_idx)} val={len(val_idx)}")

    # Hyperparameter Grid
    param_grid = [
        {"hidden_dim": 64,  "lr": 1e-3, "optimizer": "Adam",  "weight_decay": 0.0,  "dropout": 0.0},
        {"hidden_dim": 128, "lr": 1e-3, "optimizer": "Adam",  "weight_decay": 0.0,  "dropout": 0.1},
        {"hidden_dim": 128, "lr": 5e-4, "optimizer": "AdamW", "weight_decay": 1e-4, "dropout": 0.1},
        {"hidden_dim": 128, "lr": 1e-3, "optimizer": "AdamW", "weight_decay": 1e-4, "dropout": 0.2},
        {"hidden_dim": 256, "lr": 5e-4, "optimizer": "AdamW", "weight_decay": 1e-4, "dropout": 0.2},
    ]

    batch_size = 16
    epochs_per_trial = 6

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    results = []
    best_trial = None
    best_val_wet_rmse = float("inf")

    for idx, params in enumerate(param_grid, 1):
        logger.info(f"\n--- HPO Trial [{idx}/{len(param_grid)}] Params: {params} ---")
        t0 = time.time()

        model = ConvLSTMSurrogateModel(
            in_channels=4, seq_len=7,
            hidden_dim=params["hidden_dim"], dropout=params["dropout"],
        ).to(device)

        if params["optimizer"] == "AdamW":
            opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        else:
            opt = torch.optim.Adam(model.parameters(), lr=params["lr"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_per_trial)

        trial_metrics = []
        val_wet_rmse = float("inf")
        for epoch in range(1, epochs_per_trial + 1):
            model.train()
            t_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                pred = model(bx)
                loss = _weighted_smooth_l1(pred, by)
                loss.backward()
                opt.step()
                t_loss += loss.item() * bx.size(0)
            t_loss /= len(train_idx)

            model.eval()
            v_loss = 0.0
            v_preds, v_targets = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vpred = model(vx)
                    v_loss += _weighted_smooth_l1(vpred, vy).item() * vx.size(0)
                    v_preds.append(vpred.cpu().numpy())
                    v_targets.append(vy.cpu().numpy())
            v_loss /= len(val_idx)
            v_preds_arr, v_targets_arr = np.concatenate(v_preds), np.concatenate(v_targets)
            v_metrics = _rmse_mae_r2(v_preds_arr, v_targets_arr)
            v_wet = _wet_region_metrics(v_preds_arr, v_targets_arr)
            val_wet_rmse = v_wet["rmse"] if v_wet else v_loss
            scheduler.step()

            trial_metrics.append({
                "epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                "val_wholegrid_mae": v_metrics["mae"],
                "val_wet_mae": v_wet["mae"] if v_wet else None,
                "val_wet_r2": v_wet["r2"] if v_wet else None,
            })
            logger.info(
                f"  Epoch {epoch} — Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f} | "
                f"Wet MAE: {v_wet['mae']:.4f}m" if v_wet else f"  Epoch {epoch} — Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}"
            )

        duration = time.time() - t0
        trial_summary = {
            "trial_id": idx,
            "params": params,
            "final_val_loss": v_loss,
            "final_val_wet_rmse": val_wet_rmse,
            "duration_sec": round(duration, 2),
            "history": trial_metrics,
        }
        results.append(trial_summary)

        if val_wet_rmse < best_val_wet_rmse:
            best_val_wet_rmse = val_wet_rmse
            best_trial = trial_summary
            torch.save(model.state_dict(), MODELS_DIR / "time_series" / "conv_lstm_surrogate.pth")
            logger.info(f"  * New Best Trial! Val Wet-Region RMSE: {val_wet_rmse:.4f}m")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HPO_RESULTS_PATH, "w") as f:
        json.dump({"best_trial": best_trial, "all_trials": results}, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"HPO Complete! Best Trial [{best_trial['trial_id']}] Parameters:")
    logger.info(f"  Params: {best_trial['params']}")
    logger.info(f"  Val Wet-Region RMSE: {best_trial['final_val_wet_rmse']:.4f}m")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_hyperparameter_search()
