"""
src.models.train_segmentation
=============================
Training loop for flood segmentation model.

Usage:
    python -m src.models.train_segmentation

Expects training data to be prepared in:
    data/processed/arrays/segmentation_train_dataset.npz
        Keys: X_train, y_train, X_val, y_val, X_test, y_test
        Shapes: (N, 7, 14, 198, 252) for X [7 timesteps, 14 channels]
                (N, 1, 198, 252) for y [binary flood probability]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.segmentation import create_segmentation_model

# Config
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DATA = Path("data/processed/arrays/segmentation_train_dataset.npz")
OUT_MODEL = Path("models/time_series/segmentation_model.pth")
OUT_METRICS = Path("models/time_series/segmentation_metrics.json")


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute IoU, F1, precision, recall."""
    pred = (pred > 0.5).astype(int).ravel()
    target = target.astype(int).ravel()
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target.astype(bool)).sum())
    fn = int(np.logical_and(~pred.astype(bool), target.astype(bool)).sum())
    tn = int(np.logical_and(~pred.astype(bool), ~target.astype(bool)).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main():
    print("[INIT] Creating model and loading data...")
    model, criterion = create_segmentation_model(DEVICE)
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    if not TRAIN_DATA.exists():
        print(f"[FATAL] Training data not found at {TRAIN_DATA}")
        print("[INFO] Run: python -m src.ingestion.build_segmentation_dataset")
        return

    data = np.load(TRAIN_DATA)
    X_train = torch.from_numpy(data["X_train"]).float().to(DEVICE)
    y_train = torch.from_numpy(data["y_train"]).float().to(DEVICE)
    X_val = torch.from_numpy(data["X_val"]).float().to(DEVICE)
    y_val = torch.from_numpy(data["y_val"]).float().to(DEVICE)
    X_test = torch.from_numpy(data["X_test"]).float().to(DEVICE)
    y_test = torch.from_numpy(data["y_test"]).float().to(DEVICE)

    print(f"[DATA] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[DATA] X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"[DATA] X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer with cosine annealing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print(f"\n[TRAIN] Starting training for {EPOCHS} epochs...")
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(float(train_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_metrics = compute_metrics(val_pred.cpu().numpy(), y_val.cpu().numpy())

        history["val_loss"].append(float(val_loss))
        history["val_iou"].append(float(val_metrics["iou"]))
        history["val_f1"].append(float(val_metrics["f1"]))

        # Learning rate schedule
        scheduler.step()

        # Checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"[CKPT] Saved best model (F1={best_val_f1:.4f}) to {OUT_MODEL}")

        if (epoch + 1) % 10 == 0:
            print(f"[E{epoch+1:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_iou={val_metrics['iou']:.4f} val_f1={val_metrics['f1']:.4f}")

    # Test evaluation
    print(f"\n[TEST] Evaluating on held-out test set...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_metrics = compute_metrics(test_pred.cpu().numpy(), y_test.cpu().numpy())

    print(f"[RESULT] Test IoU={test_metrics['iou']:.4f}, F1={test_metrics['f1']:.4f}, "
          f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")

    # Save results
    history["test_metrics"] = test_metrics
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[SAVE] Training history → {OUT_METRICS}")


if __name__ == "__main__":
    main()
