"""
src.validation.validate_flood_prone
===================================
Does knowing where floods were reported before help rank where they will be
reported next?

Held out in time: reports up to 2021 are the history, reports from 2022 to 2025
the test. Each test event polygon (<= 25 km2) is scored by AUC - the chance
that a cell inside the reported area ranks above a cell outside it, 0.5 being
no skill - and the mean is taken over events, with an interval that resamples
places (many events repeat the same neighbourhood).

Fields compared:
    drainage susceptibility      the field the U-Net's labels were built from
    population                   news reports follow people
    reported before (to 2021)    history.report_density
    combined                     history.flood_prone_rank, the field the
                                 outlook uses to order places

A machine-learned cell model on terrain, built-up land and population, trained
on the 2015-2021 reports, was also tried and did not beat susceptibility (it
learned the places already reported); it is not kept.

USAGE
-----
    python -m src.validation.validate_flood_prone
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from src.reports import history
from src.validation.validate_groundsource import _auc, _auc_within, _boot

ARRAYS = Path("data/processed/arrays")
OUT = Path("models/time_series/validation_flood_prone.json")
HISTORY_UNTIL = date(2021, 12, 31)
TEST = (date(2022, 1, 1), date(2025, 12, 31))


def main() -> None:
    susc = np.load(ARRAYS / "segmentation_dataset_v2_nwp_drainage.npz")["susceptibility"].astype(np.float64)
    pop = np.load(ARRAYS / "population_grid_worldpop2025.npy").astype(np.float64)
    events = [e for e in history.load_events()
              if e["area_km2"] <= history.PRONE_MAX_KM2 and TEST[0] <= e["start"] <= TEST[1]]
    masks: dict[str, np.ndarray] = {}
    for e in events:
        if e["wkt"] not in masks:
            masks[e["wkt"]] = history.cell_mask(e["geom"]).ravel().astype(bool)
    groups = np.array([e["wkt"] for e in events])
    strata = np.digitize(pop.ravel(), np.percentile(pop, np.arange(10, 100, 10)))
    fields = {
        "drainage susceptibility": susc,
        "population": pop,
        "reported before (to 2021)": history.report_density(HISTORY_UNTIL),
        "combined (deployed)": history.flood_prone_rank(susc, HISTORY_UNTIL),
    }
    rng = np.random.default_rng(0)
    print(f"\n  {len(events)} reported floods 2022-2025 in {len(masks)} places; history: reports to {HISTORY_UNTIL}")
    print(f"  {'field':<28}{'AUC (95% interval)':>22}{'population-matched':>24}")
    report = {}
    for name, f in fields.items():
        r = rankdata(f.ravel())
        plain = np.array([_auc(r, masks[e["wkt"]]) for e in events])
        matched = np.array([_auc_within(f.ravel(), strata, masks[e["wkt"]]) for e in events])
        lo, hi = _boot(plain, groups, rng)
        mlo, mhi = _boot(matched, groups, rng)
        report[name] = {"auc": round(float(plain.mean()), 3), "interval": [round(lo, 3), round(hi, 3)],
                        "population_matched": round(float(matched.mean()), 3),
                        "population_matched_interval": [round(mlo, 3), round(mhi, 3)]}
        print(f"  {name:<28}{plain.mean():>8.3f} ({lo:.2f}-{hi:.2f}){matched.mean():>12.3f} ({mlo:.2f}-{mhi:.2f})")
    OUT.write_text(json.dumps({
        "question": "do past news reports help rank where floods are reported next?",
        "history": f"Groundsource reports up to {HISTORY_UNTIL}", "test": f"reports {TEST[0]} to {TEST[1]}",
        "events": len(events), "places": len(masks), "fields": report,
    }, indent=1), encoding="utf-8")
    print(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
