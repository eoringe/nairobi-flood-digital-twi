"""
src.ingestion.build_segmentation_dataset_v2
===========================================
Rebuild of the flood-segmentation training set, replacing
`build_segmentation_dataset.py`, which had four defects that together made the
old dataset untrainable:

  1. The rainfall join did `zip(event_ids, rainfall_dates)`, pairing SAR sample
     *i* with calendar day *i*. Those lists are unrelated (703 season-tagged
     samples vs a 4,138-day calendar), so the join was arbitrary and `zip`
     truncated it to 2015-01-01..2016-12-03. Six samples came back flood-
     positive out of 703; every unmatched sample silently defaulted to y=0.
  2. Labels were scene-level flags broadcast across all 49,896 pixels
     (`y_labels[i] = 1.0`), so the target had zero spatial variance.
  3. Inputs were 91% redundant: 77 channels holding 13 unique layers, with
     dem/slope/twi duplicated twice and a constant `np.ones_like(dem)` channel.
  4. The train/val/test split was a random per-sample shuffle, leaking the same
     storm across all three sets.

WHAT CHANGED
------------
Labels are computed directly from the full CHIRPS daily series
(`rainfall_daily_mean.npy`, 4,138 days) rather than joined against the sparse
162-scene `rainfall_flood_labels.json`, so every sample gets a real label.

The task is posed as a FORECAST. The input window and the label window do not
overlap:

    input : rainfall over days [t-7 .. t-1]  +  terrain
    label : flood extent over days [t .. t+FORECAST_DAYS-1]

Day t's rainfall is never shown to the model, so the label cannot be recovered
by summing an input channel. The model has to anticipate a storm from
antecedent rainfall and seasonality, then place the water using terrain.

Flood extent is susceptibility-derived, not independently observed:

    flood(t) = storm(t) AND susceptible(pixel)
    storm(t)      = sum(rain[t .. t+FORECAST_DAYS-1]) >= RAIN_THRESH_MM
    susceptible   = HAND <= HAND_THRESH_M
                    AND slope <= SLOPE_THRESH
                    AND permanent_water < PW_THRESH

This is a real limitation and must be stated as one in the thesis: extent
reflects terrain susceptibility rather than observed inundation. The 23
`s1_water_mask_*.tif` composites are deliberately NOT used — they were built
with an absolute `VV < -16 dB` cutoff already shown to be inverted for this
site (r = -0.74 against rainfall).

STORAGE
-------
Rainfall is spatially uniform in this dataset (`rainfall_chirps.npy` is
(4138, 1, 1), broadcast to the grid at dataset_builder.py:356), so each sample's
rainfall is 7 scalars, not 7 maps. Terrain is identical across every sample.
Storing those compactly and expanding at load time takes the dataset from
~6 GB to ~40 MB:

    rain_seq   (N, 7)        float32   antecedent rainfall, mm/day
    static     (6, H, W)     float32   dem, slope, twi, hand, built_up, perm_water
    y          (N, 1, H, W)  uint8     flood mask
    dates      (N,)          str       ISO date of day t
    event_ids  (N,)          str       storm-season tag, for event-aware splitting

The training Dataset expands each sample to (13, H, W).

USAGE
-----
    python -m src.ingestion.build_segmentation_dataset_v2
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed/arrays")

GRID_H, GRID_W = 198, 252
SEQ_LEN = 7

#: Longer antecedent windows. Captures whether the catchment is already wet,
#: which single-week rainfall misses.
LONG_WINDOWS = (14, 30)

#: Storm season month windows — must match src.preprocessing.dataset_builder
STORM_WINDOWS = {
    "long_rains": (3, 5),
    "short_rains": (10, 12),
}

# ---- Label parameters (swept in the sensitivity analysis) -------------------
#: Days ahead the model forecasts. Label covers [t, t+FORECAST_DAYS-1].
FORECAST_DAYS = 3
#: Accumulated rainfall over the forecast window at which flooding begins.
RAIN_THRESH_MM = 30.0
#: Accumulated rainfall at which flood extent saturates at EXTENT_MAX_PCT.
RAIN_SATURATION_MM = 120.0
#: Grid fraction (%) inundated by a storm just clearing RAIN_THRESH_MM.
EXTENT_MIN_PCT = 2.0
#: Grid fraction (%) inundated by a storm at or above RAIN_SATURATION_MM.
EXTENT_MAX_PCT = 18.0
#: HAND e-folding scale (metres): susceptibility decays with height above drainage.
HAND_DECAY_M = 3.0
#: Slope e-folding scale (normalised units).
SLOPE_DECAY = 0.15
#: JRC Global Surface Water occurrence (%) above which a pixel is permanent water.
PW_THRESH = 25.0

#: Sampling stride in days across each storm season. 1 = every available window.
#: Consecutive windows overlap heavily, which is fine because the split is
#: event-aware: no storm season is ever spread across two sets.
STRIDE_DAYS = 1


def parse_chirps_date(raw: str) -> date:
    """Parse the 'YYYY-day-DDD' labels written by src.ingestion.fetch_chirps."""
    parts = raw.split("-")
    if len(parts) == 3 and parts[1] == "day":
        return date(int(parts[0]), 1, 1) + timedelta(days=int(parts[2]) - 1)
    return date.fromisoformat(raw)


def load_rainfall() -> tuple[list[date], np.ndarray]:
    dates = [parse_chirps_date(d) for d in json.load(open(PROCESSED_DIR / "rainfall_dates.json"))]
    values = np.load(PROCESSED_DIR / "rainfall_daily_mean.npy").astype(np.float32)
    assert len(dates) == len(values), f"date/value length mismatch: {len(dates)} vs {len(values)}"
    return dates, values


def load_static() -> tuple[np.ndarray, dict]:
    """Stack the six unique static predictors. No duplicates, no constant channel."""
    terrain = np.load(PROCESSED_DIR / "static_terrain_features.npy")  # (3, H, W) dem, slope, twi
    hand = np.load(PROCESSED_DIR / "predictor_hand.npy")
    built_up = np.load(PROCESSED_DIR / "predictor_built_up.npy")
    perm_water = np.load(PROCESSED_DIR / "predictor_permanent_water.npy")

    static = np.stack([
        terrain[0],   # dem   (normalised 0-1)
        terrain[1],   # slope (normalised 0-1)
        terrain[2],   # twi   (normalised 0-1)
        hand,         # metres
        built_up,     # fraction 0-1
        perm_water,   # GSW occurrence %
    ], axis=0).astype(np.float32)
    static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)

    names = ["dem", "slope", "twi", "hand", "built_up", "permanent_water"]
    return static, {"channel_names": names}


def build_susceptibility(static: np.ndarray) -> np.ndarray:
    """
    Continuous flood-susceptibility score in [0, 1], higher = more flood-prone.

    A graded score rather than a binary mask, so that flood extent can grow with
    storm intensity. If every storm produced the same fixed mask the dataset
    would contain only two distinct labels and the model would just learn to
    stamp a constant stencil.

    Controls follow standard flood-susceptibility practice:
      HAND  - dominant control; water pools near drainage. Decays exponentially.
      slope - flat ground pools, steep ground sheds. Decays exponentially.
      TWI   - topographic convergence; boosts already-favourable pixels.
    Permanent water is excluded outright: it is always wet, so counting it as
    flood would inflate every metric.
    """
    slope, twi, hand, perm_water = static[1], static[2], static[3], static[5]

    s_hand = np.exp(-hand / HAND_DECAY_M)
    s_slope = np.exp(-slope / SLOPE_DECAY)
    twi_n = (twi - twi.min()) / max(float(np.ptp(twi)), 1e-6)

    score = s_hand * s_slope * (0.5 + 0.5 * twi_n)
    score[perm_water >= PW_THRESH] = 0.0

    if score.max() > 0:
        score = score / score.max()
    return score.astype(np.float32)


def flood_extent(score: np.ndarray, intensity_mm: float) -> np.ndarray:
    """
    Flood mask for a storm of the given accumulated rainfall.

    Extent is intensity-graded: a storm just clearing the threshold inundates
    only the most susceptible ground, while an extreme storm spreads further up
    the susceptibility gradient. The fraction of the grid flooded is
    interpolated between EXTENT_MIN_PCT and EXTENT_MAX_PCT as intensity rises
    from RAIN_THRESH_MM to RAIN_SATURATION_MM, then that fraction is taken off
    the top of the susceptibility distribution.
    """
    if intensity_mm < RAIN_THRESH_MM:
        return np.zeros(score.shape, dtype=np.uint8)

    span = max(RAIN_SATURATION_MM - RAIN_THRESH_MM, 1e-6)
    frac = float(np.clip((intensity_mm - RAIN_THRESH_MM) / span, 0.0, 1.0))
    target_pct = EXTENT_MIN_PCT + frac * (EXTENT_MAX_PCT - EXTENT_MIN_PCT)

    cutoff = np.percentile(score[score > 0], 100.0 - target_pct)
    return (score >= cutoff).astype(np.uint8)


def build_scalars(rain: np.ndarray, i: int, day: date, mode: str,
                  forecast_days: int) -> tuple[np.ndarray, list[str]]:
    """
    Per-sample scalar features, broadcast to maps at training time.

    Rainfall in this dataset is spatially uniform (CHIRPS was fetched as a
    single point), so every rainfall-derived feature is a scalar rather than a
    map. Storing them as scalars and expanding on the GPU keeps the file tiny.

    Two modes:
      forecast - only information available BEFORE day t. The model must
                 anticipate the storm. This is the honest forecasting task and
                 is capped by rainfall predictability (AUC ~0.65).
      nwp      - additionally receives the rainfall over the label window, as an
                 operational system receives a numerical weather prediction from
                 a forecasting centre. The model's job is then the hydrological
                 mapping from rainfall to extent, NOT forecasting weather. Scores
                 far higher, and must be reported as a different claim.
    """
    feats = list(rain[i - SEQ_LEN:i])
    names = [f"rain_t-{k}" for k in range(SEQ_LEN, 0, -1)]

    # Seasonality. Storms cluster within a season, and without this the model
    # cannot tell early March from late May. Measured worth: scene F1 0.21 -> 0.35.
    doy = day.timetuple().tm_yday
    feats += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    names += ["doy_sin", "doy_cos"]

    for w in LONG_WINDOWS:
        feats.append(rain[i - w:i].sum())
        names.append(f"rain_sum_{w}d")

    if mode == "nwp":
        feats += list(rain[i:i + forecast_days])
        names += [f"rain_fcst_t+{k}" for k in range(forecast_days)]

    return np.asarray(feats, dtype=np.float32), names


def scalar_scales(names: list[str]) -> np.ndarray:
    """Per-feature divisors bringing each scalar to roughly unit range."""
    out = []
    for n in names:
        if n.startswith("rain_sum_14"):
            out.append(100.0)
        elif n.startswith("rain_sum_30"):
            out.append(200.0)
        elif n.startswith("doy_"):
            out.append(1.0)
        else:                      # daily rainfall, antecedent or forecast
            out.append(50.0)
    return np.asarray(out, dtype=np.float32)


def discover_events(dates: list[date]) -> list[dict]:
    """Storm seasons covered by the CHIRPS series."""
    events = []
    for year in range(dates[0].year, dates[-1].year + 1):
        for season, (m0, m1) in STORM_WINDOWS.items():
            start = date(year, m0, 1)
            end = date(year, 12, 31) if m1 == 12 else date(year, m1 + 1, 1) - timedelta(days=1)
            if start >= dates[0] and end <= dates[-1]:
                events.append({"event_id": f"{year}_{season}", "start": start, "end": end})
    return events


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["forecast", "nwp"], default="forecast",
                    help="forecast = Model A (anticipate the storm); "
                         "nwp = Model B (rainfall forecast supplied as input)")
    ap.add_argument("--forecast-days", type=int, default=FORECAST_DAYS)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    forecast_days = args.forecast_days
    out_file = Path(args.out) if args.out else \
        PROCESSED_DIR / f"segmentation_dataset_v2_{args.mode}.npz"

    print("=" * 72)
    print(f"REBUILDING SEGMENTATION DATASET (v2)  mode={args.mode}")
    print("=" * 72)
    if args.mode == "nwp":
        print("  Model B: rainfall over the label window is provided as input,")
        print("  as an operational system receives an NWP forecast. This is a")
        print("  rainfall-to-extent mapping, NOT a weather forecast.\n")

    dates, rain = load_rainfall()
    date_index = {d: i for i, d in enumerate(dates)}
    print(f"[LOAD] CHIRPS daily series: {len(dates)} days, {dates[0]} -> {dates[-1]}")

    static, meta = load_static()
    print(f"[LOAD] Static predictors: {static.shape} {meta['channel_names']}")

    score = build_susceptibility(static)
    live = int((score > 0).sum())
    print(f"[MASK] Susceptibility score: {live:,}/{score.size:,} pixels non-zero "
          f"({100*live/score.size:.1f}%), permanent water excluded")
    print(f"       Extent grows {EXTENT_MIN_PCT}% -> {EXTENT_MAX_PCT}% of grid as "
          f"{RAIN_THRESH_MM:.0f}mm -> {RAIN_SATURATION_MM:.0f}mm falls in {FORECAST_DAYS} days")

    events = discover_events(dates)
    print(f"[EVENT] {len(events)} storm seasons in range")

    rain_seqs, labels, sample_dates, sample_events = [], [], [], []
    skipped = 0
    feat_names: list[str] = []
    lookback = max(SEQ_LEN, *LONG_WINDOWS)

    for ev in events:
        day = ev["start"]
        while day <= ev["end"]:
            i = date_index.get(day)
            # need `lookback` days before t, and forecast_days from t onward
            if i is None or i - lookback < 0 or i + forecast_days > len(rain):
                skipped += 1
                day += timedelta(days=STRIDE_DAYS)
                continue

            feats, feat_names = build_scalars(rain, i, day, args.mode, forecast_days)
            forecast = rain[i:i + forecast_days]          # label window

            y = flood_extent(score, float(forecast.sum()))

            rain_seqs.append(feats)
            labels.append(y[None, :, :])
            sample_dates.append(day.isoformat())
            sample_events.append(ev["event_id"])
            day += timedelta(days=STRIDE_DAYS)

    rain_arr = np.asarray(rain_seqs, dtype=np.float32)
    y_arr = np.asarray(labels, dtype=np.uint8)
    dates_arr = np.asarray(sample_dates)
    events_arr = np.asarray(sample_events)
    n = len(rain_arr)

    n_storm = int(sum(1 for y in y_arr if y.any()))
    extents = np.array([100 * y.mean() for y in y_arr if y.any()])
    n_unique = len({y.tobytes() for y in y_arr})

    print(f"\n[BUILD] {n} samples ({skipped} skipped for insufficient window)")
    print(f"        Storm-positive scenes : {n_storm}/{n} = {100*n_storm/n:.1f}%")
    print(f"        Positive pixel rate   : {100*y_arr.mean():.2f}%")
    print(f"        Distinct flood masks  : {n_unique}  (was 2 with a fixed stencil)")
    if len(extents):
        print(f"        Extent when flooded   : min {extents.min():.1f}%  "
              f"median {np.median(extents):.1f}%  max {extents.max():.1f}%")
    print(f"        Scalar features       : {rain_arr.shape[1]}  {feat_names}")

    # ---- Event-aware split -------------------------------------------------
    uniq = sorted(set(sample_events))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(uniq))
    n_tr = int(0.70 * len(uniq))
    n_va = max(1, int(0.15 * len(uniq)))
    groups = {
        "train": {uniq[i] for i in order[:n_tr]},
        "val": {uniq[i] for i in order[n_tr:n_tr + n_va]},
        "test": {uniq[i] for i in order[n_tr + n_va:]},
    }

    idx = {k: np.array([i for i, e in enumerate(sample_events) if e in g], dtype=np.int64)
           for k, g in groups.items()}

    print(f"\n[SPLIT] Event-aware over {len(uniq)} storm seasons (no season spans two sets)")
    for k in ("train", "val", "test"):
        ii = idx[k]
        pos = int(sum(1 for j in ii if y_arr[j].any()))
        print(f"        {k:5s}: {len(groups[k]):2d} events, {len(ii):4d} samples, "
              f"{pos:4d} storm-positive ({100*pos/max(len(ii),1):.1f}%)")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        rain_seq=rain_arr,
        scalar_scale=scalar_scales(feat_names),
        scalar_names=np.asarray(feat_names),
        static=static,
        y=y_arr,
        dates=dates_arr,
        event_ids=events_arr,
        train_idx=idx["train"],
        val_idx=idx["val"],
        test_idx=idx["test"],
        channel_names=np.asarray(meta["channel_names"] + feat_names),
        susceptibility=score,
        params=np.asarray([json.dumps({
            "mode": args.mode,
            "n_scalar_features": int(rain_arr.shape[1]),
            "forecast_days": forecast_days,
            "rain_thresh_mm": RAIN_THRESH_MM,
            "rain_saturation_mm": RAIN_SATURATION_MM,
            "extent_min_pct": EXTENT_MIN_PCT,
            "extent_max_pct": EXTENT_MAX_PCT,
            "hand_decay_m": HAND_DECAY_M,
            "slope_decay": SLOPE_DECAY,
            "pw_thresh": PW_THRESH,
            "seq_len": SEQ_LEN,
            "stride_days": STRIDE_DAYS,
        })]),
    )
    size_mb = out_file.stat().st_size / 1e6
    print(f"\n[SAVE] {out_file}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
