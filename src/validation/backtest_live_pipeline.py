"""
src.validation.backtest_live_pipeline
=====================================
Backtest of the deployed warning chain, 2022-2026: forecast rainfall -> bias
correction -> U-Net -> city warning level.

WHAT IT ANSWERS
---------------
RESULTS.md reports how well the U-Net maps rainfall to flood extent. That is not
the question a user cares about, which is: "had this system been running, would
it have warned before Nairobi's floods, and stayed quiet otherwise?" The live
system never sees CHIRPS; it sees Open-Meteo forecasts. This script runs the
real input feed through the real chain for every day since the feed's archive
begins (2022) and scores it three ways:

1. Documented Nairobi floods inside the period (validate_documented_events.py).
2. Dry-season control periods: any warning there is a false alarm.
3. Every day against CHIRPS storm days (3-day total >= 30 mm), the rainfall
   definition the model was trained on.

HONESTY OF THE CORRECTED RUN
----------------------------
The bias correction is refitted inside this script on one half of the period and
applied to the other half, both ways round, so no day is scored with a correction
that saw it. The deployed correction uses all the data.

RESOLUTION
----------
Daily (3-day totals ending each day), because the archive's daily totals are
complete and reliable; the hourly outlook adds lead time within a day, which the
replays demonstrate. A day with a corrected total under 15 mm is scored as no
warning without running the model: the deployed model produces no flooded cells
below ~20 mm.

USAGE
-----
    python -m src.validation.backtest_live_pipeline
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.forecast.nowcast import MIN_CELLS, _place_levels
from src.forecast.rain_bias_correction import _fit, _load_chirps, _map
from src.models.predict_v2 import FloodPredictor
from src.routing.flood_router import place_raster
from src.validation.validate_documented_events import DOCUMENTED_FLOODS as EVENTS, DRY_CONTROLS as CONTROLS

FEED_FILE = Path("data/raw/openmeteo_hist_forecast_daily.json")
OUT = Path("models/time_series/backtest_live_pipeline.json")
NO_FLOOD_BELOW_MM = 15.0
LEVEL_LABEL = {1: "moderate", 2: "high", 3: "critical"}


def main() -> None:
    feed = json.loads(FEED_FILE.read_text())["daily"]
    chirps = _load_chirps()
    days = sorted(k for k in feed if k in chirps)

    def total3(series, i):
        ks = days[i - 2:i + 1]
        vals = [series.get(k) for k in ks]
        return None if any(v is None for v in vals) else float(sum(vals))

    idx = [i for i in range(2, len(days)) if total3(feed, i) is not None]
    end = [days[i] for i in idx]
    raw = np.array([total3(feed, i) for i in idx])
    ref = np.array([total3(chirps, i) for i in idx])

    # Out-of-fold correction.
    mid = end[len(end) // 2]
    first = np.array([d < mid for d in end])
    corrected = np.empty_like(raw)
    for fit_mask in (first, ~first):
        q_raw, q_ref, top = _fit(raw[fit_mask], ref[fit_mask])
        corrected[~fit_mask] = _map(raw[~fit_mask], q_raw, q_ref, top)

    predictor = FloodPredictor()
    place_idx, names = place_raster()
    inside = place_idx[:-1]

    def city_level(mm: float, d: str) -> tuple[int, int]:
        """(worst band anywhere, number of places at moderate or worse)."""
        if mm < NO_FLOOD_BELOW_MM:
            return 0, 0
        prob = predictor.probability_for(mm, scenario_date=date.fromisoformat(d))
        lv = _place_levels(prob, inside, len(names))
        return int(lv.max()), int((lv >= 1).sum())

    runs = {}
    for label, series in (("raw", raw), ("corrected", corrected)):
        levels, n_places = [], []
        for mm, d in zip(series, end):
            lv, n = city_level(float(mm), d)
            levels.append(lv); n_places.append(n)
        runs[label] = (np.array(levels), np.array(n_places))
        print(f"  {label}: model runs done")

    report = {"period": [end[0], end[-1]], "days": len(end), "min_cells_per_place": MIN_CELLS}
    day_index = {d: i for i, d in enumerate(end)}

    print(f"\n  BACKTEST {end[0]} to {end[-1]} ({len(end)} days)")
    print("\n  Documented Nairobi floods in the period")
    print(f"  {'event':<42}{'CHIRPS max':>11}{'feed max':>10}{'raw':>14}{'corrected':>16}")
    report["events"] = []
    for ev in EVENTS:
        w0, w1 = ev["window"]
        sel = [day_index[d.isoformat()] for d in (w0 + timedelta(n) for n in range((w1 - w0).days + 1))
               if d.isoformat() in day_index]
        if not sel:
            continue
        row = {"event": ev["name"], "window": [str(w0), str(w1)],
               "chirps_max_3day": float(ref[sel].max()), "feed_max_3day": float(raw[sel].max()),
               "corrected_max_3day": float(corrected[sel].max())}
        cells = []
        for label in ("raw", "corrected"):
            lv = runs[label][0][sel]
            flagged = int((lv >= 1).sum())
            first_day = next((end[s] for s in sel if runs[label][0][s] >= 1), None)
            row[label] = {"detected": flagged > 0, "days_warned": flagged, "window_days": len(sel),
                          "max_level": int(lv.max()), "first_warning": first_day}
            cells.append(f"{'yes' if flagged else 'NO '} {flagged:>2}/{len(sel):<2} L{int(lv.max())}")
        report["events"].append(row)
        print(f"  {ev['name']:<42}{row['chirps_max_3day']:>9.0f}mm{row['feed_max_3day']:>8.0f}mm"
              f"{cells[0]:>14}{cells[1]:>16}")

    print("\n  Dry-season controls (any warning is a false alarm)")
    report["controls"] = []
    for c in CONTROLS:
        w0, w1 = c["window"]
        sel = [day_index[d.isoformat()] for d in (w0 + timedelta(n) for n in range((w1 - w0).days + 1))
               if d.isoformat() in day_index]
        if not sel:
            continue
        row = {"control": c["name"], "days": len(sel)}
        for label in ("raw", "corrected"):
            row[label] = int((runs[label][0][sel] >= 1).sum())
        report["controls"].append(row)
        print(f"  {c['name']:<42} days {len(sel):>3}   false-alarm days raw {row['raw']:>2}   corrected {row['corrected']:>2}")

    print("\n  Every day vs CHIRPS storm days (3-day total >= 30 mm)")
    truth = ref >= 30.0
    report["daily_vs_chirps_storms"] = {"storm_days": int(truth.sum())}
    for label in ("raw", "corrected"):
        warn = runs[label][0] >= 1
        tp, fp, fn = int((warn & truth).sum()), int((warn & ~truth).sum()), int((~warn & truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        far = fp / max(int((~truth).sum()), 1)
        report["daily_vs_chirps_storms"][label] = {"warning_days": int(warn.sum()), "precision": prec,
                                                  "recall": rec, "f1": f1, "false_alarm_rate": far}
        print(f"  {label:<10} warned {int(warn.sum()):>4} days | precision {prec:.2f} recall {rec:.2f} "
              f"F1 {f1:.2f} | false alarms on {100 * far:.1f}% of non-storm days")

    # Operating points: what counts as a city-wide warning. A single small place
    # at moderate is a weak signal; requiring several places, or a higher band,
    # trades missed storms for fewer false alarms.
    print("\n  Warning rule sensitivity (corrected feed)")
    print(f"  {'rule':<30}{'warn days':>10}{'prec':>7}{'rec':>7}{'F1':>7}{'false alarms':>14}{'events':>8}{'dry FA':>8}")
    levels, n_places = runs["corrected"]
    ev_sel = []
    for ev in EVENTS:
        w0, w1 = ev["window"]
        s = [day_index[d.isoformat()] for d in (w0 + timedelta(n) for n in range((w1 - w0).days + 1))
             if d.isoformat() in day_index]
        if s:
            ev_sel.append(s)
    dry_sel = []
    for c in CONTROLS:
        w0, w1 = c["window"]
        dry_sel += [day_index[d.isoformat()] for d in (w0 + timedelta(n) for n in range((w1 - w0).days + 1))
                    if d.isoformat() in day_index]
    report["rule_sensitivity"] = []
    for min_level, min_places in ((1, 1), (1, 3), (1, 5), (1, 10), (2, 1), (2, 3), (3, 1)):
        warn = (levels >= min_level) & (n_places >= min_places)
        tp, fp, fn = int((warn & truth).sum()), int((warn & ~truth).sum()), int((~warn & truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        far = fp / max(int((~truth).sum()), 1)
        detected = sum(1 for s in ev_sel if warn[s].any())
        dry_fa = int(warn[dry_sel].sum())
        rule = f"band>={LEVEL_LABEL[min_level]}, places>={min_places}"
        report["rule_sensitivity"].append({"min_level": min_level, "min_places": min_places,
                                           "precision": prec, "recall": rec, "f1": f1,
                                           "false_alarm_rate": far, "events_detected": detected,
                                           "events_total": len(ev_sel), "dry_control_false_alarm_days": dry_fa})
        print(f"  {rule:<30}{int(warn.sum()):>10}{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}{100 * far:>13.1f}%"
              f"{detected:>5}/{len(ev_sel)}{dry_fa:>8}")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
