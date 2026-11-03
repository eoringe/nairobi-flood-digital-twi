"""
src.validation.validate_groundsource
====================================
Validation against floods reported in the news: Google's Groundsource dataset.

THE GAP THIS ADDRESSES
----------------------
validate_documented_events.py checks timing on six hand-picked storms and ends
with the caveat that spatial extent "remains unverified -- reports name affected
settlements but give no inundation polygons". validate_spatial_neighbourhoods.py
checks place with 34 approximate centroids.

Groundsource (Google Research, 2026) supplies what both lacked: 630 dated,
neighbourhood-sized flood polygons inside the model grid, 2015 to February 2026,
extracted from news articles by Gemini. None of it was used to build the labels
or train the model, so it is independent evidence on both questions:

  1. TIMING  Did the deployed warning chain warn when floods were reported?
  2. PLACE   On those days, did it point at the neighbourhoods reported flooded?

METHOD
------
* Every day of the period runs the deployed chain: 3-day rainfall -> U-Net ->
  per-place warning level (src.forecast.nowcast), a warning being any place at
  "moderate" or worse. Rainfall is the CHIRPS total that actually fell, i.e. a
  perfect forecast, so TIMING is an upper bound on what the live system (which
  uses a rainfall forecast) achieves; backtest_live_pipeline.py measures the
  forecast's cost.
* A reported flood on day e counts as warned if the outlook issued on e-2, e-1
  or e flagged it (each outlook covers three days).
* PLACE scores each event polygon by AUC: the chance that a cell inside the
  reported area ranks above a cell outside it (0.5 is no skill). Compared with
  two baselines: population density (where more people live, more floods get
  reported), and the drainage susceptibility map the labels came from.

CAUTIONS (stated in the output too)
-----------------------------------
* Google's review found 60% of events right in both place and timing.
* News coverage is uneven: a day or place with no report is not a dry one, so
  "false alarms" here are an upper bound.
* Many events repeat the same neighbourhood polygon; uncertainty is therefore
  bootstrapped over places, not events.

USAGE
-----
    python -m src.validation.validate_groundsource      (~2 min on a laptop CPU)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from src.forecast.nowcast import _place_levels
from src.ingestion.build_segmentation_dataset_v2 import STORM_WINDOWS
from src.models.predict_v2 import DEFAULT_DATASET, FloodPredictor
from src.reports.history import cell_mask, load_events
from src.routing.flood_router import place_raster
from src.validation.validate_documented_events import parse_chirps_date

PROCESSED_DIR = Path("data/processed/arrays")
POP_GRID = PROCESSED_DIR / "population_grid_worldpop2025.npy"
OUT = Path("models/time_series/validation_groundsource.json")

FORECAST_DAYS = 3
#: As in backtest_live_pipeline: the deployed model floods no cells below ~20 mm.
NO_FLOOD_BELOW_MM = 15.0
#: Polygons up to this size are used for PLACE. Larger ones cover a big share of
#: the 266 km2 grid and say little about where.
SPATIAL_MAX_KM2 = 25.0
BOOTSTRAP = 2000


def _season(d: date) -> str | None:
    for name, (m0, m1) in STORM_WINDOWS.items():
        if m0 <= d.month <= m1:
            return f"{d.year}_{name}"
    return None


def _auc(ranks: np.ndarray, inside: np.ndarray) -> float:
    """Mann-Whitney AUC of cells inside vs outside, from precomputed ranks."""
    n1 = int(inside.sum())
    n0 = inside.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[inside].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_within(score: np.ndarray, strata: np.ndarray, inside: np.ndarray) -> float:
    """AUC computed within each stratum (e.g. population decile) and averaged,
    weighted by the event's cells in it: does the score still separate the
    reported area from places with a similar population?"""
    total, weight = 0.0, 0
    for s in np.unique(strata[inside]):
        sel = strata == s
        a = _auc(rankdata(score[sel]), inside[sel])
        if not np.isnan(a):
            n = int(inside[sel].sum())
            total += a * n
            weight += n
    return total / weight if weight else float("nan")


def _rates(warn_days: set, flood_days: set, period: list[date], weights: Counter) -> dict:
    """Hit rate over reported flood days (and over events), and precision of warnings."""
    caught = {e for e in flood_days if any(e - timedelta(days=k) in warn_days for k in range(FORECAST_DAYS))}
    followed = {d for d in warn_days if any(d + timedelta(days=k) in flood_days for k in range(FORECAST_DAYS))}
    n_ev = sum(weights[e] for e in flood_days)
    return {
        "flood_days": len(flood_days),
        "flood_days_warned": len(caught),
        "hit_rate_days": len(caught) / max(len(flood_days), 1),
        "hit_rate_events": sum(weights[e] for e in caught) / max(n_ev, 1),
        "warning_days": len(warn_days),
        "warning_share_of_days": len(warn_days) / max(len(period), 1),
        "precision": len(followed) / max(len(warn_days), 1),
    }


def _boot(values: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float]:
    """95% interval of the mean, resampling whole places."""
    uniq = np.unique(groups)
    per = {g: values[groups == g] for g in uniq}
    means = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(uniq, size=len(uniq))
        means.append(np.concatenate([per[g] for g in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    events = load_events()
    if not events:
        raise SystemExit("data/processed/nairobi_groundsource.json is missing: "
                         "run `python -m src.ingestion.fetch_groundsource` first.")

    dates = [parse_chirps_date(s) for s in json.load(open(PROCESSED_DIR / "rainfall_dates.json"))]
    rain = np.load(PROCESSED_DIR / "rainfall_daily_mean.npy").astype(np.float64)
    last_report = max(e["start"] for e in events)
    period = [d for i, d in enumerate(dates)
              if i >= 30 and i + FORECAST_DAYS <= len(rain) and d <= last_report]
    index = {d: i for i, d in enumerate(dates)}
    in_period = [e for e in events if period[0] + timedelta(days=FORECAST_DAYS - 1) <= e["start"] <= period[-1]]

    npz = np.load(DEFAULT_DATASET, allow_pickle=False)
    test_seasons = {str(s) for s in npz["event_ids"][npz["test_idx"]]}

    print("=" * 78)
    print("VALIDATION AGAINST NEWS-REPORTED FLOODS (Groundsource, Google Research)")
    print("=" * 78)
    print(f"  Period          : {period[0]} to {period[-1]} ({len(period)} days)")
    print(f"  Reported floods : {len(in_period)} events on {len({e['start'] for e in in_period})} days, "
          f"{len({e['wkt'] for e in in_period})} distinct places")
    print("  Rainfall        : CHIRPS, as fallen (a perfect 3-day forecast: an upper bound)\n")

    # ---- run the deployed chain on every day ---------------------------------
    predictor = FloodPredictor()
    place_idx, names = place_raster()
    inside_grid = place_idx[:-1]
    flood_days = {e["start"] for e in in_period}
    keep_prob = {e["start"] for e in in_period}
    warn_days, prob_on, totals = set(), {}, {}
    for d in period:
        i = index[d]
        mm = float(rain[i:i + FORECAST_DAYS].sum())
        totals[d] = mm
        if mm < NO_FLOOD_BELOW_MM:
            continue
        prob = predictor.predict(rainfall_mm=float(round(mm)), scenario_date=d)["probability_grid"]
        if _place_levels(prob, inside_grid, len(names)).max() >= 1:
            warn_days.add(d)
        if d in keep_prob:
            prob_on[d] = prob
    print(f"  Model runs      : {sum(1 for m in totals.values() if m >= NO_FLOOD_BELOW_MM)} days at or above "
          f"{NO_FLOOD_BELOW_MM:.0f} mm; {len(warn_days)} warning days\n")

    # ---- 1. TIMING --------------------------------------------------------------
    weights = Counter(e["start"] for e in in_period)
    in_season = [d for d in period if _season(d)]
    climatology = set(in_season)
    rows = {
        "model": _rates(warn_days, flood_days, period, weights),
        "every storm-season day": _rates(climatology, flood_days, period, weights),
    }
    # A random warner with the model's warning rate: its precision is the share of
    # days followed by a report within the outlook window.
    followed_any = sum(1 for d in period
                       if any(d + timedelta(days=k) in flood_days for k in range(FORECAST_DAYS)))
    random_precision = followed_any / len(period)

    test_flood = {d for d in flood_days if _season(d) in test_seasons}
    test_period = [d for d in period if _season(d) in test_seasons]
    rows["model, test seasons only"] = _rates({d for d in warn_days if _season(d) in test_seasons},
                                              test_flood, test_period, weights)
    big = {d for d in flood_days if weights[d] >= 3}
    rows["model, days with 3+ reports"] = _rates(warn_days, big, period, weights)

    print("-" * 78)
    print("  1. TIMING: was there a warning when flooding was reported?\n")
    print(f"  {'':<30}{'flood days warned':>19}{'events warned':>15}{'warning days':>14}{'precision':>11}")
    print("  " + "-" * 87)
    for label, r in rows.items():
        print(f"  {label:<30}{r['flood_days_warned']:>8}/{r['flood_days']:<5} {100*r['hit_rate_days']:>4.0f}%"
              f"{100*r['hit_rate_events']:>14.0f}%{r['warning_days']:>14}{100*r['precision']:>10.0f}%")
    print(f"\n  A random warner at any rate has precision {100*random_precision:.0f}% "
          f"(share of days followed by a report).")
    lift = rows["model"]["precision"] / max(random_precision, 1e-9)
    print(f"  Model warnings are followed by a reported flood {lift:.1f}x as often as chance.")

    # Missed days, worst first, with the rain that fell: misses below the label
    # threshold are floods rainfall alone does not explain.
    missed = sorted((d for d in flood_days if d not in
                     {e for e in flood_days if any(e - timedelta(days=k) in warn_days for k in range(FORECAST_DAYS))}),
                    key=lambda d: -weights[d])
    print("\n  Largest missed days (reports, then the most rain in any covering outlook):")
    for d in missed[:6]:
        mm = max(totals.get(d - timedelta(days=k), 0.0) for k in range(FORECAST_DAYS))
        print(f"    {d}  {weights[d]:>3} reports   {mm:5.1f} mm")
    dry_misses = sum(1 for d in missed
                     if max(totals.get(d - timedelta(days=k), 0.0) for k in range(FORECAST_DAYS)) < 30.0)
    print(f"  {dry_misses}/{len(missed)} missed days had under 30 mm in every covering outlook.")

    # ---- 2. PLACE ---------------------------------------------------------------
    pop = np.load(POP_GRID) if POP_GRID.exists() else None
    susc = predictor.susceptibility
    # The day's map floods only 2-18% of the grid and ties everything else at ~0,
    # so it is also scored by the model's full ranking: its raw output for a
    # storm at the label saturation depth, before calibration flattens it.
    saved_cal, predictor.calibration = predictor.calibration, None
    ranking = predictor.predict(rainfall_mm=predictor.label_params["rain_saturation_mm"],
                                scenario_date=date(2024, 4, 24))["probability_grid"]
    predictor.calibration = saved_cal
    static_ranks = {
        "model's ranking (heavy storm, raw output)": rankdata(ranking.ravel()).reshape(ranking.shape),
        "drainage susceptibility (label source)": rankdata(susc.ravel()).reshape(susc.shape),
    }
    if pop is not None:
        static_ranks["population density (baseline)"] = rankdata(pop.ravel()).reshape(pop.shape)
        deciles = np.digitize(pop, np.percentile(pop, np.arange(10, 100, 10))).ravel()

    caught_days = {e for e in flood_days if any(e - timedelta(days=k) in warn_days for k in range(FORECAST_DAYS))}
    masks = {}
    model_auc, model_lift, groups = [], [], []
    static_auc = {k: [] for k in static_ranks}
    matched_key = "model's ranking, population-matched"
    if pop is not None:
        static_auc[matched_key] = []
    static_groups = []
    for e in in_period:
        if e["area_km2"] > SPATIAL_MAX_KM2:
            continue
        m = masks.get(e["wkt"])
        if m is None:
            m = masks[e["wkt"]] = cell_mask(e["geom"])
        if not m.any():
            continue
        for k, r in static_ranks.items():
            static_auc[k].append(_auc(r, m))
        if pop is not None:
            static_auc[matched_key].append(_auc_within(ranking.ravel(), deciles, m.ravel()))
        static_groups.append(e["wkt"])
        if e["start"] in caught_days and e["start"] in prob_on:
            p = prob_on[e["start"]]
            model_auc.append(_auc(rankdata(p.ravel()).reshape(p.shape), m))
            flooded = p > 0.5
            model_lift.append(flooded[m].mean() / max(flooded.mean(), 1e-9))
            groups.append(e["wkt"])

    rng = np.random.default_rng(0)
    print()
    print("-" * 78)
    print("  2. PLACE: did the map point at the neighbourhoods reported flooded?\n")
    print(f"  Event polygons up to {SPATIAL_MAX_KM2:.0f} km2. AUC 0.5 = no better than chance; "
          "intervals resample places.\n")
    print(f"  {'field':<44}{'events':>8}{'places':>8}{'mean AUC':>10}{'95% interval':>17}")
    print("  " + "-" * 87)
    place_report = {}
    g = np.array(groups)
    if model_auc:
        a = np.array(model_auc)
        lo, hi = _boot(a, g, rng)
        place_report["model on warned days"] = {"events": len(a), "places": len(set(groups)),
                                                "auc": float(a.mean()), "ci95": [lo, hi],
                                                "lift": float(np.mean(model_lift))}
        print(f"  {'model, on the days it warned':<44}{len(a):>8}{len(set(groups)):>8}{a.mean():>10.3f}"
              f"{f'{lo:.3f}-{hi:.3f}':>17}")
    sg = np.array(static_groups)
    for k, vals in static_auc.items():
        a = np.array(vals)
        ok = ~np.isnan(a)
        a = a[ok]
        lo, hi = _boot(a, sg[ok], rng)
        place_report[k] = {"events": len(a), "places": len(set(static_groups)),
                           "auc": float(a.mean()), "ci95": [lo, hi]}
        print(f"  {k:<44}{len(a):>8}{len(set(static_groups)):>8}{a.mean():>10.3f}{f'{lo:.3f}-{hi:.3f}':>17}")
    if model_lift:
        print(f"\n  On warned days, a reported area was predicted flooded "
              f"{np.mean(model_lift):.1f}x as often as the city as a whole.")

    # ---- verdict, computed rather than asserted ------------------------------
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    m = rows["model"]
    print(f"  TIMING: warned on {100*m['hit_rate_days']:.0f}% of reported flood days "
          f"({100*m['hit_rate_events']:.0f}% of reports), warning on "
          f"{100*m['warning_share_of_days']:.0f}% of all days;")
    print(f"          warnings followed by a report {lift:.1f}x as often as chance.")
    def _say(r: dict) -> str:
        return (f"AUC {r['auc']:.2f} ({r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}), "
                + ("above chance" if r["ci95"][0] > 0.5 else "NOT distinguishable from chance"))
    mp = place_report.get("model on warned days")
    mr = place_report.get("model's ranking (heavy storm, raw output)")
    pp = place_report.get("population density (baseline)")
    pm = place_report.get(matched_key)
    if mp:
        print(f"  PLACE : the day's map: {_say(mp)};")
    if mr:
        print(f"          the model's ranking: {_say(mr)}"
              + (f"; population alone {pp['auc']:.2f}." if pp else "."))
    if pm:
        print(f"          among places of similar population: {_say(pm)} -- "
              + ("the model adds information beyond where people (and reporters) are."
                 if pm["ci95"][0] > 0.5 else
                 "the model's spatial skill cannot be told apart from news-coverage bias."))
    print("\n  Cautions: Google found 60% of events right in place and timing; news covers")
    print("  some areas far more than others, so unreported days and places are not")
    print("  dry ones and the precision above is a lower bound.")

    OUT.write_text(json.dumps({
        "source": "Groundsource (Google Research, 2026), doi:10.5281/zenodo.18647053, CC BY 4.0",
        "period": [period[0].isoformat(), period[-1].isoformat()],
        "events": len(in_period),
        "rainfall": "CHIRPS as fallen (perfect forecast; upper bound)",
        "timing": rows,
        "random_precision": random_precision,
        "precision_lift": lift,
        "missed_days_under_30mm": [dry_misses, len(missed)],
        "place": place_report,
        "spatial_max_km2": SPATIAL_MAX_KM2,
    }, indent=2))
    print(f"\n  Saved {OUT}")


if __name__ == "__main__":
    main()
