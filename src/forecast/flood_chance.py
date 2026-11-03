"""
src.forecast.flood_chance
=========================
"What is the chance flooding is reported in Nairobi in the next three days?",
learned from real floods rather than from the rainfall rule the U-Net's labels
were built with, and a call made at 95% confidence.

WHY
---
The U-Net reproduces its constructed labels almost exactly (F1 0.94), so more
training against them cannot make it more accurate about real floods. The
warning trigger it implies (flooding from a 3-day total of about 30 mm) was
chosen, not fitted. Groundsource gives 158 days from 2015 to 2025 on which the
news reported flooding inside the model grid; this module fits the trigger to
them instead.

THE MODEL
---------
Logistic regression on three CHIRPS-scale rainfall totals for an outlook issued
on day d:

    rain_3d    rain over d, d+1, d+2 (the forecast window)
    ante_14d   rain over the 14 days before d
    ante_30d   rain over the 30 days before d

Wet ground floods more easily, which is what the antecedent totals capture. A
richer model (14 features, ERA5 soil moisture, hourly intensity, gradient
boosting) was tried and did no better on held-out years; three inputs that the
live feed supplies were kept.

CALIBRATION AND THE 95% CALL
----------------------------
Live, the inputs come from the Open-Meteo forecast feed, bias-corrected to the
CHIRPS scale. The fitted score is therefore re-calibrated (Platt scaling) on the
feed's own archive, 2022 onward, and the 95% call is set by class-conditional
(Mondrian) split conformal prediction on that archive:

    "flooding expected"     the score is too high for a dry day: at most 5% of
                            dry days get this call
    "no flooding expected"  the score is too low for a flood day: at most 5% of
                            flood days get this call
    "flooding possible"     neither can be ruled out at 95%

The guarantee is about coverage over many days, under the assumption that
future days resemble the calibration years. It is checked here on years the
calibration did not see: the feed archive is split into 2022-23 and 2024-25,
each half calibrated on the other.

Groundsource is news, about 60% right in place and time, and coverage grew
sharply after 2021. "Flood reported" is the event being predicted, so the
chance is a lower bound on the chance of flooding.

USAGE
-----
    python -m src.forecast.flood_chance          # fit, evaluate, save

    from src.forecast.flood_chance import chance_from_hourly
    chance_from_hourly(series, as_of)            # -> {"p": 0.28, "call": "possible", ...} or None
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from src.forecast.rain_bias_correction import correct_3day_total

OUT = Path("models/time_series/flood_chance.json")
CHIRPS = Path("data/processed/arrays/rainfall_daily_mean.npy")
CHIRPS_DATES = Path("data/processed/arrays/rainfall_dates.json")
LAT, LON = -1.2864, 36.8172
FEED_URL = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&start_date={start}&end_date={end}&daily=precipitation_sum&timezone=Africa%2FNairobi")
FEED_START = date(2022, 1, 1)
FIT_END_YEAR = 2021            # the held-out evaluation tests 2022 onward
LAST_YEAR = 2025               # Groundsource coverage ends late 2025
FEATURES = ("rain_3d", "ante_14d", "ante_30d")
ANTECEDENT_DAYS = 30
CONFIDENCE = 0.95
CALLS = {
    "likely": "Flooding expected",
    "possible": "Flooding possible",
    "unlikely": "No flooding expected",
}


# ---------------------------------------------------------------- features --
def features(window: list[float], before: list[float], feed: bool = False) -> list[float]:
    """
    Model inputs from daily rainfall: `window` is the 3 outlook days, `before`
    the 30 days preceding them, most recent first. Feed rainfall is put on the
    CHIRPS scale in 3-day blocks, the unit the bias correction was fitted on.
    """
    if not feed:
        return [sum(window), sum(before[:14]), sum(before[:30])]
    blocks = [correct_3day_total(sum(before[i:i + 3])) for i in range(0, ANTECEDENT_DAYS, 3)]
    return [correct_3day_total(sum(window)), sum(blocks[:5]) * 14 / 15, sum(blocks)]


def _daily(series: dict[datetime, float]) -> dict[date, float]:
    """Calendar-day totals from hour-ending hourly rainfall."""
    out: dict[date, float] = {}
    for t, v in series.items():
        d = (t - timedelta(hours=1)).date()
        out[d] = out.get(d, 0.0) + v
    return out


# ------------------------------------------------------------------- apply --
@lru_cache(maxsize=1)
def _params() -> dict | None:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        logger.warning(f"{OUT} missing - no flood chance. Run `python -m src.forecast.flood_chance`.")
        return None


def chance(x: list[float]) -> dict | None:
    """Calibrated probability and 95% call for one set of CHIRPS-scale inputs."""
    p = _params()
    if p is None:
        return None
    z = (np.asarray(x, float) - p["mean"]) / p["scale"]
    score = float(z @ np.asarray(p["coef"]) + p["intercept"])
    prob = float(1.0 / (1.0 + np.exp(-(p["platt"][0] * score + p["platt"][1]))))
    flood_ok, dry_ok = prob >= p["t_flood"], prob <= p["t_dry"]
    call = "likely" if flood_ok and not dry_ok else ("unlikely" if dry_ok and not flood_ok else "possible")
    return {"p": round(prob, 3), "call": call, "call_text": CALLS[call],
            "inputs_mm": dict(zip(FEATURES, (round(v, 1) for v in x)))}


def chance_from_hourly(series: dict[datetime, float], as_of: datetime) -> dict | None:
    """
    Flood chance for the outlook issued at `as_of` from the hourly forecast feed:
    today and the next two days, after the 30 days before today. None when the
    series does not cover them.
    """
    daily = _daily(series)
    today = as_of.date()
    window = [today + timedelta(days=k) for k in range(3)]
    before = [today - timedelta(days=k) for k in range(1, ANTECEDENT_DAYS + 1)]
    if not all(d in daily for d in window + before):
        return None
    return chance(features([daily[d] for d in window], [daily[d] for d in before], feed=True))


# -------------------------------------------------------------------- fit --
def _chirps() -> dict[date, float]:
    from src.validation.validate_documented_events import parse_chirps_date
    dates = [parse_chirps_date(s) for s in json.loads(CHIRPS_DATES.read_text())]
    return dict(zip(dates, np.load(CHIRPS).astype(float).ravel()))


def _feed(end: date) -> dict[date, float]:
    url = FEED_URL.format(lat=LAT, lon=LON, start=FEED_START, end=end)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nairobi-flood-digital-twin/1.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return {date.fromisoformat(t): (v or 0.0) for t, v in
                    zip(data["daily"]["time"], data["daily"]["precipitation_sum"])}
        except Exception as exc:                              # noqa: BLE001
            logger.warning(f"fetch attempt {attempt + 1} failed: {exc}")
            time.sleep(20)
    raise RuntimeError("Open-Meteo historical-forecast API unreachable")


def _table(rain: dict[date, float], flood: set[date], start: date, end: date, feed: bool):
    X, y, days = [], [], []
    d = start
    while d <= end:
        window = [d + timedelta(days=k) for k in range(3)]
        before = [d - timedelta(days=k) for k in range(1, ANTECEDENT_DAYS + 1)]
        if all(k in rain for k in window + before):
            X.append(features([rain[k] for k in window], [rain[k] for k in before], feed))
            y.append(any(k in flood for k in window))
            days.append(d)
        d += timedelta(days=1)
    return np.array(X), np.array(y), days


def _logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def _platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e3).fit(score.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _sigmoid(a: float, b: float, s: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * s + b)))


def _thresholds(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mondrian split-conformal cut-offs: flood possible if p >= t_flood, dry if p <= t_dry."""
    alpha = 1.0 - CONFIDENCE
    t_flood = float(np.quantile(prob[y], alpha, method="lower"))
    t_dry = float(np.quantile(prob[~y], 1.0 - alpha, method="higher"))
    return t_flood, t_dry


def _calls(prob: np.ndarray, t_flood: float, t_dry: float) -> np.ndarray:
    f, d = prob >= t_flood, prob <= t_dry
    return np.where(f & ~d, "likely", np.where(d & ~f, "unlikely", "possible"))


def _warned(days: list[date], warn: np.ndarray, flood: set[date]) -> tuple[int, int]:
    """Flood days covered by a warning issued on that day or up to two days before."""
    issued = {d for d, w in zip(days, warn) if w}
    covered = set(days) | {d + timedelta(days=k) for d in days for k in (1, 2)}
    fl = [e for e in flood if e in covered]
    return sum(any(e - timedelta(days=k) in issued for k in range(3)) for e in fl), len(fl)


def _score_against_rule(name: str, X: np.ndarray, y: np.ndarray, days: list[date],
                        prob: np.ndarray, flood: set[date]) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    rule = X[:, 0] >= 30.0
    k = int(rule.sum())
    model = prob >= np.sort(prob)[::-1][k - 1]
    rc, n = _warned(days, rule, flood)
    mc, _ = _warned(days, model, flood)
    out = {
        "outlook_days": len(y), "flood_outlooks": int(y.sum()),
        "auc": {"rule_3day_total": round(roc_auc_score(y, X[:, 0]), 3), "model": round(roc_auc_score(y, prob), 3)},
        "at_rule_warning_count": {
            "warnings": k,
            "rule": {"flood_days_warned": rc, "of": n, "precision": round(float(y[rule].mean()), 3)},
            "model": {"flood_days_warned": mc, "of": n, "precision": round(float(y[model].mean()), 3)},
        },
        "brier": {"model": round(brier_score_loss(y, prob), 4),
                  "climatology": round(brier_score_loss(y, np.full(len(y), y.mean())), 4)},
    }
    a = out["at_rule_warning_count"]
    print(f"\n  {name}: {len(y)} outlooks, {y.sum()} followed by a reported flood")
    print(f"    AUC: 3-day rule {out['auc']['rule_3day_total']:.3f}, model {out['auc']['model']:.3f}")
    print(f"    at the rule's {k} warnings: rule warned {rc}/{n} flood days at precision "
          f"{a['rule']['precision']:.0%}; model {mc}/{n} at {a['model']['precision']:.0%}")
    return out


def _reliability(prob: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)):
        s = (prob >= lo) & (prob < hi)
        if s.any():
            rows.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(s.sum()),
                         "predicted": round(float(prob[s].mean()), 3), "observed": round(float(y[s].mean()), 3)})
    return rows


def _call_summary(calls: np.ndarray, y: np.ndarray) -> dict:
    return {
        "flood_days_not_called_unlikely": round(float((calls[y] != "unlikely").mean()), 3),
        "dry_days_not_called_likely": round(float((calls[~y] != "likely").mean()), 3),
        "calls": {c: {"days": int((calls == c).sum()), "share": round(float((calls == c).mean()), 3),
                      "flood_reported": round(float(y[calls == c].mean()), 3) if (calls == c).any() else None}
                  for c in CALLS},
    }


def main() -> None:
    from src.reports.history import load_events

    flood = {e["start"] for e in load_events() if date(2015, 1, 1) <= e["start"] <= date(LAST_YEAR, 12, 31)}
    chirps = _chirps()
    feed = _feed(date(LAST_YEAR, 12, 31))
    print(f"\n  {len(flood)} days with a news-reported flood in the grid, 2015-{LAST_YEAR}")

    # 1. Held-out years: fit on 2015-2021 CHIRPS, test on 2022 onward.
    Xa, ya, da = _table(chirps, flood, date(2015, 1, 1), date(FIT_END_YEAR, 12, 31), feed=False)
    held = _logistic().fit(Xa, ya)
    evaluation = {}
    Xc, yc, dc = _table(chirps, flood, FEED_START, date(LAST_YEAR, 12, 28), feed=False)
    evaluation["chirps_perfect_rain"] = _score_against_rule(
        "2022-2025, CHIRPS rainfall (a perfect forecast)", Xc, yc, dc, held.predict_proba(Xc)[:, 1], flood)
    Xf, yf, df = _table(feed, flood, FEED_START, date(LAST_YEAR, 12, 28), feed=True)
    sf = held.decision_function(Xf)
    evaluation["operational_feed"] = _score_against_rule(
        "2022-2025, live forecast feed, bias-corrected", Xf, yf, df, _sigmoid(1.0, 0.0, sf), flood)

    # 2. Calibration and the 95% call on the feed, each half judged by the other.
    late = np.array([d.year >= 2024 for d in df])
    prob = np.zeros(len(yf))
    calls = np.empty(len(yf), dtype=object)
    for test in (late, ~late):
        a, b = _platt(sf[~test], yf[~test])
        t_flood, t_dry = _thresholds(_sigmoid(a, b, sf[~test]), yf[~test])
        prob[test] = _sigmoid(a, b, sf[test])
        calls[test] = _calls(prob[test], t_flood, t_dry)
    evaluation["feed_calibration_cross_fitted"] = {
        "split": "2022-2023 calibrates 2024-2025 and vice versa",
        "reliability": _reliability(prob, yf), "call_95": _call_summary(calls, yf),
    }
    cs = evaluation["feed_calibration_cross_fitted"]["call_95"]
    print(f"\n  95% call on unseen halves of the feed archive:")
    print(f"    flood days not called 'no flooding': {cs['flood_days_not_called_unlikely']:.1%} (target >= 95%)")
    print(f"    dry days not called 'flooding expected': {cs['dry_days_not_called_likely']:.1%} (target >= 95%)")
    for c, v in cs["calls"].items():
        print(f"    {CALLS[c]:<22} {v['days']:5d} days ({v['share']:.0%}), flood reported after {v['flood_reported']:.0%}")
    for r in evaluation["feed_calibration_cross_fitted"]["reliability"]:
        print(f"    p {r['band']}: n {r['n']:4d}, predicted {r['predicted']:.2f}, observed {r['observed']:.2f}")

    # 3. Deployed: fit on all CHIRPS years, calibrate and set the call on the whole feed archive.
    Xall, yall, _ = _table(chirps, flood, date(2015, 1, 1), date(LAST_YEAR, 12, 28), feed=False)
    final = _logistic().fit(Xall, yall)
    sfin = final.decision_function(Xf)
    a, b = _platt(sfin, yf)
    t_flood, t_dry = _thresholds(_sigmoid(a, b, sfin), yf)
    scaler, lr = final[0], final[-1]
    OUT.write_text(json.dumps({
        "method": "logistic regression on CHIRPS-scale rainfall totals, Platt-calibrated on the bias-corrected "
                  "forecast feed; 95% call by class-conditional split conformal prediction",
        "target": "a flood reported in the news (Groundsource) inside the model grid on any of the 3 outlook days",
        "features": list(FEATURES),
        "fitted": {"model": f"CHIRPS 2015-{LAST_YEAR}", "calibration": f"Open-Meteo forecast feed {FEED_START}-{LAST_YEAR}"},
        "mean": scaler.mean_.round(4).tolist(), "scale": scaler.scale_.round(4).tolist(),
        "coef": lr.coef_[0].round(5).tolist(), "intercept": round(float(lr.intercept_[0]), 5),
        "platt": [round(a, 5), round(b, 5)],
        "confidence": CONFIDENCE, "t_flood": round(t_flood, 5), "t_dry": round(t_dry, 5),
        "evaluation": evaluation,
    }, indent=1), encoding="utf-8")
    _params.cache_clear()
    logger.info(f"Saved {OUT}")


if __name__ == "__main__":
    main()
