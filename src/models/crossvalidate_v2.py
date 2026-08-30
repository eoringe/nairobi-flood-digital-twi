"""
src.models.crossvalidate_v2
===========================
Event-aware k-fold cross-validation over the storm seasons.

WHY
---
The single held-out test split contains only 37 storm-positive samples drawn
from 4 storm seasons. That is thin enough that the headline F1 depends heavily
on which seasons happen to land in the test fold -- visible in the single-run
result, where validation F1 (0.25) and test F1 (0.17) disagreed substantially.

Rotating every season through the test fold gives a mean and a spread instead of
one fragile number, so the reported figure reflects the model rather than the
luck of a split.

FOLDING
-------
Folds are formed over EVENTS (storm seasons), never over samples. Consecutive
samples share overlapping rainfall windows and identical terrain, so a
sample-level split would place near-duplicates in both train and test and
inflate the score. Within each fold a few training seasons are further held out
as validation, used only for checkpoint selection.

USAGE
-----
    python -m src.models.crossvalidate_v2 --data <npz> --tag forecast
    python -m src.models.crossvalidate_v2 --data <npz> --tag nwp --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.train_segmentation_v2 import GpuDataset, OUT_DIR, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="data/processed/arrays/segmentation_dataset_v2_forecast.npz")
    ap.add_argument("--tag", type=str, default="forecast")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-events", type=int, default=3,
                    help="training seasons held out per fold for checkpoint selection")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = np.load(args.data, allow_pickle=False)
    params = json.loads(str(npz["params"][0]))
    events = npz["event_ids"]
    uniq = np.array(sorted(set(events)))

    rng = np.random.default_rng(42)
    shuffled = uniq[rng.permutation(len(uniq))]
    folds = np.array_split(shuffled, args.folds)

    print("=" * 72)
    print(f"EVENT-AWARE {args.folds}-FOLD CROSS-VALIDATION   mode={params['mode']}")
    print("=" * 72)
    print(f"  {len(uniq)} storm seasons, {len(events)} samples, device={device}")
    print(f"  every season is held out for test exactly once\n")

    results = []
    for k, test_events in enumerate(folds):
        rest = np.array([e for e in shuffled if e not in set(test_events)])
        val_events = set(rest[:args.val_events])
        train_events = set(rest[args.val_events:])
        test_set = set(test_events)

        pick = lambda S: np.array([i for i, e in enumerate(events) if e in S], dtype=np.int64)
        tr_i, va_i, te_i = pick(train_events), pick(val_events), pick(test_set)

        train = GpuDataset(npz, tr_i, device)
        val = GpuDataset(npz, va_i, device)
        test = GpuDataset(npz, te_i, device)

        n_pos = int(sum(1 for j in te_i if npz["y"][j].any()))
        print(f"--- fold {k+1}/{args.folds} "
              f"| test seasons: {', '.join(sorted(test_set))}")
        print(f"    train {train.n:4d} | val {val.n:4d} | test {test.n:4d} samples "
              f"({n_pos} storm-positive)")

        best_val, test_m, _, _ = run_training(
            train, val, test, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            base=args.base,
            ckpt_path=OUT_DIR / f"cv_{args.tag}_fold{k+1}.pth",
            verbose=False,
        )
        print(f"    -> test F1 {test_m['f1']:.4f}  IoU {test_m['iou']:.4f}  "
              f"prec {test_m['precision']:.4f}  rec {test_m['recall']:.4f}"
              f"   (best val F1 {best_val:.4f})\n")
        results.append({"fold": k + 1, "test_seasons": sorted(test_set),
                        "n_test_positive": n_pos, "best_val_f1": best_val, **test_m})

    print("=" * 72)
    print(f"SUMMARY over {args.folds} folds")
    print("=" * 72)
    print(f"{'metric':<12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 48)
    summary = {}
    for key in ("f1", "iou", "precision", "recall"):
        v = np.array([r[key] for r in results])
        summary[key] = {"mean": float(v.mean()), "std": float(v.std()),
                        "min": float(v.min()), "max": float(v.max())}
        print(f"{key:<12}{v.mean():>9.4f}{v.std():>9.4f}{v.min():>9.4f}{v.max():>9.4f}")
    print("=" * 72)
    f1 = summary["f1"]
    print(f"\n  Report as: F1 = {f1['mean']:.3f} +/- {f1['std']:.3f} "
          f"({args.folds}-fold, event-aware)")
    print(f"  Spread across folds ({f1['min']:.3f} to {f1['max']:.3f}) reflects how much")
    print(f"  the single-split number depends on which seasons are held out.")

    out = OUT_DIR / f"crossval_v2_{args.tag}.json"
    out.write_text(json.dumps({
        "label_params": params, "config": vars(args),
        "folds": results, "summary": summary,
    }, indent=2))
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
