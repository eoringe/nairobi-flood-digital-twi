"""
src.forecast.rain_bias_correction
=================================
Put Open-Meteo forecast rainfall on the CHIRPS scale the model was trained on.

THE PROBLEM
-----------
The U-Net learned what CHIRPS rainfall totals do. Live, it is fed Open-Meteo
forecasts, which are systematically wetter on dry days and far drier in storms:

    3-day totals, Nairobi, 2022-2026     CHIRPS     Open-Meteo forecast
    median                               0.0 mm     2.5 mm
    99th percentile                     91.4 mm    42.7 mm
    April 2024 flood, max 3-day        103.4 mm    58.0 mm
    March 2026 flood, max 3-day        140.1 mm    58.1 mm

Fed raw, the system saw only a quarter of the flood-producing storms. That is
an input-scale mismatch, not a model error, and no amount of retraining fixes it.

THE FIX
-------
Empirical quantile mapping of 3-day totals: a forecast total at the p-th
percentile of the forecast climatology is replaced by the p-th percentile of the
CHIRPS climatology. It is monotone (a wetter forecast always stays wetter) and
is the standard way to bias-correct model rainfall for a downstream model.

It corrects the distribution, not day-to-day skill: the two series correlate at
only r = 0.62, so some forecast storms will still be missed or false. The
evaluation below measures that trade-off on years the mapping never saw.

USAGE
-----
    python -m src.forecast.rain_bias_correction        # fetch, evaluate, fit, save

    from src.forecast.rain_bias_correction import correct_3day_total
    correct_3day_total(25.0)   # -> CHIRPS-scale mm
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

OUT = Path("data/processed/rainfall_bias_correction.json")
CHIRPS = Path("data/processed/arrays/rainfall_chirps.npy")
CHIRPS_DATES = Path("data/processed/arrays/rainfall_dates.json")
LAT, LON = -1.2864, 36.8172
#: The same feed live mode uses: Open-Meteo's archive of its own forecasts.
FEED_URL = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&start_date={start}&end_date={end}&daily=precipitation_sum&timezone=Africa%2FNairobi")
FEED_START = "2022-01-01"            # the historical-forecast archive begins in 2022
QUANTILES = np.linspace(0.0, 1.0, 401)
STORM_THRESHOLDS = (30.0, 40.0, 60.0)


# ------------------------------------------------------------------ apply --
@lru_cache(maxsize=1)
def _table() -> dict | None:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        logger.warning(f"{OUT} missing - forecast rainfall is NOT bias-corrected. "
                       f"Run `python -m src.forecast.rain_bias_correction`.")
        return None


def _map(x: np.ndarray, q_raw: np.ndarray, q_ref: np.ndarray, top_ratio: float) -> np.ndarray:
    y = np.interp(x, q_raw, q_ref)
    hi = x > q_raw[-1]
    y[hi] = q_ref[-1] + (x[hi] - q_raw[-1]) * top_ratio
    return y


def correct_3day_total(mm: float | np.ndarray):
    """CHIRPS-scale equivalent of an Open-Meteo 3-day (72 h) rainfall total."""
    t = _table()
    arr = np.atleast_1d(np.asarray(mm, dtype=np.float64))
    if t is None:
        out = arr
    else:
        out = _map(arr, np.asarray(t["q_raw"]), np.asarray(t["q_chirps"]), t["top_ratio"])
    return float(out[0]) if np.ndim(mm) == 0 else out


def is_available() -> bool:
    return _table() is not None


# -------------------------------------------------------------------- fit --
def _fit(raw: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    q_raw = np.quantile(raw, QUANTILES)
    q_ref = np.quantile(ref, QUANTILES)
    # Beyond the largest fitted forecast total, continue at the ratio between the
    # two tails rather than flattening, so a record storm is not capped.
    top_ratio = float(q_ref[-5:].mean() / max(q_raw[-5:].mean(), 1e-6))
    return q_raw, q_ref, top_ratio


def _detect(pred: np.ndarray, truth: np.ndarray, thr: float) -> dict:
    p, t = pred >= thr, truth >= thr
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"storm_windows": int(t.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def _load_chirps() -> dict[str, float]:
    raw_dates = json.loads(CHIRPS_DATES.read_text())
    dates = [date(int(s[:4]), 1, 1) + timedelta(days=int(s.split("-")[-1]) - 1) for s in raw_dates]
    values = np.load(CHIRPS).reshape(len(dates))
    return {d.isoformat(): float(v) for d, v in zip(dates, values)}


def _three_day(series: dict[str, float | None], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([series.get(k) if series.get(k) is not None else np.nan for k in keys], float)
    total = np.convolve(np.nan_to_num(v), np.ones(3), "valid")
    complete = np.convolve(np.isnan(v).astype(float), np.ones(3), "valid") == 0
    return total, complete


def main() -> None:
    chirps = _load_chirps()
    end = max(chirps)
    url = FEED_URL.format(lat=LAT, lon=LON, start=FEED_START, end=end)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nairobi-flood-digital-twin/1.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=120).read())
            break
        except Exception as exc:                              # noqa: BLE001
            logger.warning(f"fetch attempt {attempt + 1} failed: {exc}")
            time.sleep(20)
    else:
        raise RuntimeError("Open-Meteo historical-forecast API unreachable")
    feed = dict(zip(data["daily"]["time"], data["daily"]["precipitation_sum"]))

    keys = sorted(k for k in feed if k in chirps)
    raw3, ok1 = _three_day(feed, keys)
    ref3, ok2 = _three_day(chirps, keys)
    ok = ok1 & ok2
    ends = np.array(keys[2:])[ok]
    raw3, ref3 = raw3[ok], ref3[ok]
    logger.info(f"{len(raw3)} complete 3-day windows, {ends[0]} to {ends[-1]}")

    # Two-fold temporal evaluation: fit on one half of the record, test on the
    # other, both ways round, so no reported number uses data the fit has seen.
    mid = ends[len(ends) // 2]
    evaluation = []
    for fit_mask, label in ((ends < mid, f"fit before {mid}, test after"),
                            (ends >= mid, f"fit from {mid}, test before")):
        q_raw, q_ref, top = _fit(raw3[fit_mask], ref3[fit_mask])
        test = ~fit_mask
        mapped = _map(raw3[test], q_raw, q_ref, top)
        fold = {"split": label, "test_windows": int(test.sum()), "thresholds": {}}
        for thr in STORM_THRESHOLDS:
            fold["thresholds"][f"{thr:.0f}mm"] = {"raw": _detect(raw3[test], ref3[test], thr),
                                                 "corrected": _detect(mapped, ref3[test], thr)}
        evaluation.append(fold)

    print(f"\n  Storm detection on held-out halves (truth: CHIRPS 3-day total >= threshold)")
    print(f"  {'split':<34}{'thr':>6}{'n':>5}   {'raw P/R/F1':>18}   {'corrected P/R/F1':>18}")
    for fold in evaluation:
        for thr, m in fold["thresholds"].items():
            r, c = m["raw"], m["corrected"]
            print(f"  {fold['split']:<34}{thr:>6}{r['storm_windows']:>5}   "
                  f"{r['precision']:.2f}/{r['recall']:.2f}/{r['f1']:.2f}      "
                  f"{c['precision']:.2f}/{c['recall']:.2f}/{c['f1']:.2f}")

    q_raw, q_ref, top = _fit(raw3, ref3)
    OUT.write_text(json.dumps({
        "method": "empirical quantile mapping of 3-day rainfall totals, Open-Meteo forecast -> CHIRPS",
        "feed": "Open-Meteo historical-forecast API (the archive of the live forecast feed)",
        "point": [LAT, LON],
        "fitted_period": [str(ends[0]), str(ends[-1])],
        "n_windows": int(len(raw3)),
        "q_raw": np.round(q_raw, 3).tolist(),
        "q_chirps": np.round(q_ref, 3).tolist(),
        "top_ratio": round(top, 4),
        "correlation_3day": round(float(np.corrcoef(raw3, ref3)[0, 1]), 3),
        "evaluation_two_fold": evaluation,
    }, indent=1), encoding="utf-8")
    _table.cache_clear()
    print("\n  final mapping (all data): " + ", ".join(
        f"{x:.0f}->{correct_3day_total(float(x)):.0f}" for x in (5, 10, 15, 20, 25, 30, 40, 50, 60)))
    logger.info(f"Saved {OUT}")


if __name__ == "__main__":
    main()
