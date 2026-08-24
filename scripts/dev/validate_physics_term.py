"""
scripts.dev.validate_physics_term
==================================
Validates the DEPLOYED physics hydrology term (src.models.predict's
hydro_mask, WITH the 10 hand-placed confluence points) against real
Sentinel-1 observed flood extent for all 23 historical events.

WHY THIS IS A FAIR TEST, NOT A CIRCULAR ONE
--------------------------------------------
Training targets (y_train.npy) were built as:
    depth = phys_shape * 2.5 * scale * extent_mask
where `phys_shape` is DELIBERATELY built WITHOUT the confluence points
(src.preprocessing.dataset_builder._physics_shape's docstring: "training
targets should come from real observed extent + terrain, not from the
same manual coordinates the model is meant to learn to reproduce").

So the DEPTH VALUES in y_train.npy are partly circular (same formula
family the model uses), but `extent_mask` — WHERE Sentinel-1 actually
observed water for that event — is real, independent satellite
observation, not model output. And `hydro_mask` (confluence-aware) is a
different, richer computation than the confluence-free `phys_shape` used
to build the target. Comparing hydro_mask's predicted WET/DRY pixels
against the real extent_mask (recovered as `y > 0`, since phys_shape > 0
almost everywhere and the 0.2m floor is the only thing zeroing target
pixels within the true extent) is therefore a genuine, non-circular
check of whether the deployed model predicts flooding in the right
PLACES, even though we can't independently verify the depth VALUES this
way.

USAGE
-----
    python -m scripts.dev.validate_physics_term
"""
from __future__ import annotations

import numpy as np

from src.models.predict import FloodSurrogatePredictor

PROCESSED_DIR = "data/processed/arrays"
DEPTH_ZERO_FLOOR_M = 0.2
RAIN_REF_MM_DAY = 45.0


def _physics_shape(dem_norm, slope_norm, twi_norm) -> np.ndarray:
    """Exact copy of dataset_builder._physics_shape — the confluence-free
    formula used to build the (partly-circular) training-target depths."""
    slope_safe = np.maximum(slope_norm, 0.02)
    raw = (twi_norm ** 1.2) / (slope_safe ** 0.5) * np.exp(-2.0 * dem_norm)
    peak = raw.max()
    return (raw / peak).astype(np.float32) if peak > 0 else raw.astype(np.float32)


def _extent_metrics(pred_wet: np.ndarray, real_wet: np.ndarray) -> dict:
    pred_wet = pred_wet.astype(bool)
    real_wet = real_wet.astype(bool)
    tp = np.logical_and(pred_wet, real_wet).sum()
    fp = np.logical_and(pred_wet, ~real_wet).sum()
    fn = np.logical_and(~pred_wet, real_wet).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "real_wet_px": int(real_wet.sum()), "pred_wet_px": int(pred_wet.sum())}


def main():
    X = np.load(f"{PROCESSED_DIR}/X_train.npy")
    y = np.load(f"{PROCESSED_DIR}/y_train.npy")
    event_ids = np.load(f"{PROCESSED_DIR}/event_ids.npy", allow_pickle=True)
    static_terrain = np.load(f"{PROCESSED_DIR}/static_terrain_features.npy")

    dem, slope, twi = static_terrain[0], static_terrain[1], static_terrain[2]
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
    twi_norm = (twi - twi.min()) / (twi.max() - twi.min() + 1e-6)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
    phys_shape = _physics_shape(dem_norm, slope_norm, twi_norm)

    predictor = FloodSurrogatePredictor()
    hydro_mask = predictor.hydro_mask  # the DEPLOYED, confluence-aware term

    hydro_rows, physshape_rows = [], []

    for ev in np.unique(event_ids):
        idxs = np.where(event_ids == ev)[0]
        # extent_mask is identical across all of this event's samples
        # (only the rain-driven magnitude varies) — take the first sample.
        y_sample = y[idxs[0]]
        real_wet = y_sample > 0
        if real_wet.sum() < 5:
            continue  # no meaningful observed flooding for this event — skip

        # Recover the implied rainfall SCALE this specific sample was built
        # with, straight from the saved depth values (avoids any ambiguity
        # about how X_train.npy's rain channel is normalized).
        ratio = y_sample[real_wet] / (phys_shape[real_wet] * 2.5 + 1e-9)
        scale_recovered = float(np.median(ratio))

        pred_depth_hydro = hydro_mask * 2.5 * scale_recovered
        pred_wet_hydro = pred_depth_hydro >= DEPTH_ZERO_FLOOR_M
        hydro_rows.append((str(ev), _extent_metrics(pred_wet_hydro, real_wet)))

        pred_depth_ps = phys_shape * 2.5 * scale_recovered
        pred_wet_ps = pred_depth_ps >= DEPTH_ZERO_FLOOR_M
        physshape_rows.append((str(ev), _extent_metrics(pred_wet_ps, real_wet)))

    def _summarize(rows, label):
        ious = [r[1]["iou"] for r in rows if not np.isnan(r[1]["iou"])]
        precs = [r[1]["precision"] for r in rows if not np.isnan(r[1]["precision"])]
        recs = [r[1]["recall"] for r in rows if not np.isnan(r[1]["recall"])]
        f1s = [r[1]["f1"] for r in rows if not np.isnan(r[1]["f1"])]
        print(f"\n=== {label} — {len(rows)} events ===")
        for ev_name, m in rows:
            print(f"  {ev_name:<20} IoU={m['iou']:.3f}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                  f"(real={m['real_wet_px']}px pred={m['pred_wet_px']}px)")
        print(f"  --- MEAN across events: IoU={np.mean(ious):.3f}  "
              f"Precision={np.mean(precs):.3f}  Recall={np.mean(recs):.3f}  F1={np.mean(f1s):.3f}")
        return {"mean_iou": float(np.mean(ious)), "mean_precision": float(np.mean(precs)),
                "mean_recall": float(np.mean(recs)), "mean_f1": float(np.mean(f1s)),
                "n_events": len(rows)}

    hydro_summary = _summarize(hydro_rows, "DEPLOYED hydro_mask (with confluence points)")
    ps_summary = _summarize(physshape_rows, "BASELINE phys_shape (DEM/slope/TWI only, no confluences)")

    print("\n=== HEADLINE ===")
    print(f"Deployed model:  mean IoU={hydro_summary['mean_iou']:.3f}, mean F1={hydro_summary['mean_f1']:.3f}")
    print(f"DEM-only baseline: mean IoU={ps_summary['mean_iou']:.3f}, mean F1={ps_summary['mean_f1']:.3f}")

    import json
    from pathlib import Path
    out = {"hydro_mask_with_confluences": hydro_summary, "physics_shape_baseline": ps_summary,
           "per_event_hydro_mask": [{"event": e, **m} for e, m in hydro_rows],
           "per_event_physics_shape": [{"event": e, **m} for e, m in physshape_rows]}
    out_path = Path("models/time_series/physics_term_validation.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
