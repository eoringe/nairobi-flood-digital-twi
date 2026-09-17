"""
src.models.calibrate_v2
=======================
Probability calibration and per-season evaluation for the v2 U-Nets.

WHY CALIBRATE
-------------
LIMITATIONS.md section 10 measured that mid-range outputs are overconfident: a
cell the model calls 85% floods about 62% of the time. The interface shows these
numbers as probabilities, so they should mean what they say.

METHOD
------
Isotonic regression, fitted on the VALIDATION seasons only and judged on the
held-out TEST seasons, so the improvement is not measured on the data that
produced it. Isotonic regression learns a monotone mapping raw -> calibrated.
Being monotone, it never reorders cells: the map's ranking of where flooding is
most likely is unchanged, only the stated percentages move.

Cells are accumulated into 1,000 fine bins of raw probability (counts of cells
and of flooded cells per bin) rather than held individually, which keeps ~14
million validation cells in a few kilobytes and makes the fit exact up to the
0.001 bin width.

WHAT CALIBRATION CANNOT FIX
---------------------------
Calibration is against the training LABELS, which are constructed from rainfall
and drainage susceptibility, not observed floods (LIMITATIONS.md section 1).
After this step "70%" means "70% of such cells carry a flood label" - better
than an overconfident number, but still not a verified real-world frequency.

PER-SEASON BREAKDOWN
--------------------
The single test split holds 4 storm seasons. Reporting F1 per season shows how
much the headline figure depends on which seasons landed in the test set, and a
season-level bootstrap gives a (crude, 4-cluster) interval. Proper error bars
need the event-aware k-fold run in src/models/crossvalidate_v2.py, which retrains
the network and needs a GPU.

USAGE
-----
    python -m src.models.calibrate_v2 --tag nwp_drainage              # fit + evaluate (deployed model)
    python -m src.models.calibrate_v2 --tag forecast_drainage --no-fit # per-season evaluation only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from src.models.train_segmentation_v2 import GpuDataset, UNet

MODELS_DIR = Path("models/time_series")
DATA_DIR = Path("data/processed/arrays")
N_BINS = 1000
#: The reliability bands used in LIMITATIONS.md section 10.
BANDS = np.linspace(0.0, 1.0, 11)


def _infer(model, npz, idx, batch=8, thresholds=(0.5,), events=None):
    """
    Run the model over samples, returning fine-bin histograms and per-season
    confusion counts at each raw-probability threshold.
    """
    ds = GpuDataset(npz, idx, torch.device("cpu"))
    total = np.zeros(N_BINS, np.float64)
    pos = np.zeros(N_BINS, np.float64)
    psum = np.zeros(N_BINS, np.float64)
    conf: dict = {}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, ds.n, batch):
            sel = torch.arange(i, min(i + batch, ds.n))
            x, y = ds.batch(sel)
            p = torch.sigmoid(model(x))[:, 0].numpy().astype(np.float64)
            yy = y[:, 0].numpy().astype(bool)
            b = np.minimum((p * N_BINS).astype(np.int64), N_BINS - 1)
            total += np.bincount(b.ravel(), minlength=N_BINS)
            pos += np.bincount(b.ravel(), weights=yy.ravel(), minlength=N_BINS)
            psum += np.bincount(b.ravel(), weights=p.ravel(), minlength=N_BINS)
            if events is not None:
                for k in range(len(sel)):
                    ev = str(events[idx[i + k]])
                    for t in thresholds:
                        pred = p[k] > t
                        c = conf.setdefault((ev, t), [0.0, 0.0, 0.0, 0])
                        c[0] += float((pred & yy[k]).sum())
                        c[1] += float((pred & ~yy[k]).sum())
                        c[2] += float((~pred & yy[k]).sum())
                        c[3] += int(yy[k].any())
            done = min(i + batch, ds.n)
            if done % 80 == 0 or done == ds.n:
                print(f"    {done}/{ds.n} samples  ({time.time() - t0:.0f}s)", flush=True)
    return total, pos, psum, conf


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall}


def _reliability(total, pos, pred_value):
    """
    Reliability over the 10 bands. `pred_value` is the (raw or calibrated)
    probability assigned to each fine bin; cells are grouped by that value.
    """
    rows = []
    band = np.minimum(np.searchsorted(BANDS, pred_value, side="right") - 1, 9)
    for k in range(10):
        m = band == k
        n = total[m].sum()
        if n == 0:
            continue
        rows.append({
            "band": f"{BANDS[k]:.1f}-{BANDS[k + 1]:.1f}",
            "cells": int(n),
            "share_pct": 100.0 * n / total.sum(),
            "observed_pct": 100.0 * pos[m].sum() / n,
            "predicted_pct": 100.0 * (total[m] * pred_value[m]).sum() / n,
        })
    for r in rows:
        r["gap_pts"] = r["predicted_pct"] - r["observed_pct"]
    gaps = [abs(r["gap_pts"]) / 100 for r in rows]
    mid = [abs(r["gap_pts"]) / 100 for r in rows if r["band"] not in ("0.0-0.1", "0.9-1.0")]
    brier = float(((pos * (1 - pred_value) ** 2) + ((total - pos) * pred_value ** 2)).sum() / total.sum())
    return rows, {
        "ece_unweighted": float(np.mean(gaps)) if gaps else 0.0,
        "ece_mid_bands": float(np.mean(mid)) if mid else 0.0,
        "brier": brier,
    }


def _print_table(title, rows, summary):
    print(f"\n  {title}")
    print(f"  {'band':<9}{'share':>9}{'observed':>10}{'predicted':>11}{'gap':>7}")
    for r in rows:
        print(f"  {r['band']:<9}{r['share_pct']:>8.3f}%{r['observed_pct']:>9.1f}%"
              f"{r['predicted_pct']:>10.1f}%{r['gap_pts']:>+7.0f}")
    print(f"  unweighted ECE {summary['ece_unweighted']:.3f} | mid-band ECE {summary['ece_mid_bands']:.3f}"
          f" | Brier {summary['brier']:.5f}")


def _season_report(conf, threshold, n_boot=10000, seed=0):
    seasons = sorted({ev for ev, t in conf if t == threshold})
    per = {}
    for ev in seasons:
        tp, fp, fn, npos = conf[(ev, threshold)]
        per[ev] = {**_metrics(tp, fp, fn), "storm_positive_samples": npos}
    counts = np.array([conf[(ev, threshold)][:3] for ev in seasons])
    pooled = _metrics(*counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        c = counts[rng.integers(0, len(seasons), len(seasons))].sum(axis=0)
        boots.append(_metrics(*c)["f1"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return per, pooled, {"f1_ci95_season_bootstrap": [float(lo), float(hi)], "n_seasons": len(seasons)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nwp_drainage")
    ap.add_argument("--no-fit", action="store_true", help="per-season evaluation only")
    args = ap.parse_args()

    npz = np.load(DATA_DIR / f"segmentation_dataset_v2_{args.tag}.npz", allow_pickle=False)
    model = UNet(in_ch=npz["rain_seq"].shape[1] + npz["static"].shape[0], base=32)
    model.load_state_dict(torch.load(MODELS_DIR / f"segmentation_model_v2_{args.tag}.pth", map_location="cpu"))
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    events = npz["event_ids"]
    report: dict = {"tag": args.tag}

    thresholds = [0.5]
    calib = None
    if not args.no_fit:
        print(f"[1/2] validation seasons: {sorted(set(map(str, events[npz['val_idx']])))}")
        v_total, v_pos, v_psum, _ = _infer(model, npz, npz["val_idx"])
        keep = v_total > 0
        x = v_psum[keep] / v_total[keep]
        y = v_pos[keep] / v_total[keep]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso.fit(x, y, sample_weight=v_total[keep])
        grid = np.linspace(0.0, 1.0, N_BINS + 1)
        cal = np.clip(iso.predict(grid), 0.0, 1.0)
        # Raw probability at which the calibrated value first reaches 0.5, so the
        # flooded mask can be evaluated in calibrated terms.
        above = np.flatnonzero(cal >= 0.5)
        raw_at_half = float(grid[above[0]]) if len(above) else 1.0
        thresholds.append(raw_at_half)
        calib = {"x": grid.round(4).tolist(), "y": cal.round(5).tolist(),
                 "raw_threshold_for_calibrated_0_5": raw_at_half,
                 "fitted_on": "validation seasons " + ", ".join(sorted(set(map(str, events[npz["val_idx"]])))),
                 "method": "isotonic regression on 1000 fine bins of raw probability"}

    print(f"[{'2/2' if calib else '1/1'}] test seasons: {sorted(set(map(str, events[npz['test_idx']])))}")
    t_total, t_pos, t_psum, conf = _infer(model, npz, npz["test_idx"], thresholds=thresholds, events=events)
    bin_mean = np.where(t_total > 0, t_psum / np.maximum(t_total, 1), (np.arange(N_BINS) + 0.5) / N_BINS)

    rows_raw, sum_raw = _reliability(t_total, t_pos, bin_mean)
    _print_table("TEST reliability, raw model output", rows_raw, sum_raw)
    report["test_reliability_raw"] = {"bands": rows_raw, **sum_raw}

    per, pooled, boot = _season_report(conf, 0.5)
    report["test_raw_threshold_0_5"] = {"pooled": pooled, "per_season": per, **boot}

    if calib:
        cal_value = np.interp(bin_mean, calib["x"], calib["y"])
        rows_cal, sum_cal = _reliability(t_total, t_pos, cal_value)
        _print_table("TEST reliability, calibrated (fitted on validation only)", rows_cal, sum_cal)
        report["test_reliability_calibrated"] = {"bands": rows_cal, **sum_cal}
        per_c, pooled_c, boot_c = _season_report(conf, calib["raw_threshold_for_calibrated_0_5"])
        report["test_calibrated_threshold_0_5"] = {"pooled": pooled_c, "per_season": per_c, **boot_c}
        calib["test_ece_unweighted_before_after"] = [sum_raw["ece_unweighted"], sum_cal["ece_unweighted"]]
        calib["test_ece_mid_bands_before_after"] = [sum_raw["ece_mid_bands"], sum_cal["ece_mid_bands"]]
        calib["test_brier_before_after"] = [sum_raw["brier"], sum_cal["brier"]]
        out = MODELS_DIR / f"calibration_v2_{args.tag}.json"
        out.write_text(json.dumps(calib))
        print(f"\n[SAVE] {out}")

    print("\n  PER TEST SEASON (raw output, threshold 0.5)")
    print(f"  {'season':<20}{'storm samples':>14}{'F1':>8}{'IoU':>8}{'prec':>8}{'rec':>8}")
    for ev, m in per.items():
        print(f"  {ev:<20}{m['storm_positive_samples']:>14}{m['f1']:>8.3f}{m['iou']:>8.3f}"
              f"{m['precision']:>8.3f}{m['recall']:>8.3f}")
    print(f"  {'pooled':<20}{'':>14}{pooled['f1']:>8.3f}{pooled['iou']:>8.3f}"
          f"{pooled['precision']:>8.3f}{pooled['recall']:>8.3f}")
    print(f"  season-bootstrap 95% interval for F1: {boot['f1_ci95_season_bootstrap'][0]:.3f}"
          f" - {boot['f1_ci95_season_bootstrap'][1]:.3f}  (only {boot['n_seasons']} seasons: crude)")
    if calib:
        pc = report["test_calibrated_threshold_0_5"]["pooled"]
        print(f"  calibrated 0.5 threshold (raw {calib['raw_threshold_for_calibrated_0_5']:.3f}): "
              f"F1 {pc['f1']:.3f} IoU {pc['iou']:.3f} prec {pc['precision']:.3f} rec {pc['recall']:.3f}")

    out = MODELS_DIR / f"evaluation_v2_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    main()
