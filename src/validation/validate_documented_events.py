"""
src.validation.validate_documented_events
=========================================
External validation against independently documented Nairobi flood events.

WHY THIS MATTERS
----------------
Every metric elsewhere in this project measures agreement with labels that were
themselves constructed from rainfall and terrain (LIMITATIONS.md section 1).
Model B reaching F1 0.94 shows it reproduces that labelling rule, not that the
floods are real. This script is the only check that compares the pipeline
against events reported by people on the ground.

It validates the TEMPORAL component only: did a storm the labels call
flood-producing actually coincide with reported flooding, and did quiet periods
stay quiet. Spatial extent cannot be validated this way -- press and situation
reports name affected settlements but do not give inundation polygons.

EVENTS
------
Dates from Copernicus EMS, ReliefWeb situation reports and contemporaneous
reporting. Each entry records what was reported, so a reader can audit the
source rather than trust this file.

USAGE
-----
    python -m src.validation.validate_documented_events
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed/arrays")

#: Independently documented flood events affecting Nairobi.
#: `peak` is the reported worst day; `window` bounds the reported episode.
DOCUMENTED_FLOODS = [
    {
        "name": "April 2024 long rains",
        "peak": date(2024, 4, 24),
        "window": (date(2024, 4, 23), date(2024, 4, 30)),
        "reported": "Floods swept Mathare 24 Apr; >7,000 displaced in Mathare alone; "
                    "Nairobi County ~147,061 affected, 20,968 families displaced",
        "source": "Copernicus EMS / ReliefWeb / Save the Children",
    },
    {
        "name": "November 2023 El Nino short rains",
        "peak": date(2023, 11, 19),
        "window": (date(2023, 11, 11), date(2023, 11, 25)),
        "reported": "Nairobi rivers burst banks, informal settlements flooded; "
                    "~36,000 displaced and 46 killed nationally by 16 Nov",
        "source": "ReliefWeb rapid needs assessment / Xinhua / EastAfrican",
    },
]

#: Control periods: mid-dry-season, when no flooding should be flagged.
DRY_CONTROLS = [
    {"name": "Jan-Feb 2024 dry season", "window": (date(2024, 1, 10), date(2024, 2, 20))},
    {"name": "Jul-Aug 2023 cool dry", "window": (date(2023, 7, 10), date(2023, 8, 20))},
    {"name": "Jan-Feb 2022 dry season", "window": (date(2022, 1, 10), date(2022, 2, 20))},
]

RAIN_THRESH_MM = 30.0
FORECAST_DAYS = 3


def parse_chirps_date(raw: str) -> date:
    p = raw.split("-")
    if len(p) == 3 and p[1] == "day":
        return date(int(p[0]), 1, 1) + timedelta(days=int(p[2]) - 1)
    return date.fromisoformat(raw)


def main() -> None:
    dates = [parse_chirps_date(s) for s in json.load(open(PROCESSED_DIR / "rainfall_dates.json"))]
    rain = np.load(PROCESSED_DIR / "rainfall_daily_mean.npy").astype(np.float64)
    index = {d: i for i, d in enumerate(dates)}

    def accum(day: date, n: int) -> float | None:
        i = index.get(day)
        if i is None or i + n > len(rain):
            return None
        return float(rain[i:i + n].sum())

    def flagged_days(w0: date, w1: date) -> tuple[int, int]:
        """(days flagged flood-positive, days evaluated) across a window."""
        hit = tot = 0
        d = w0
        while d <= w1:
            a = accum(d, FORECAST_DAYS)
            if a is not None:
                tot += 1
                hit += a >= RAIN_THRESH_MM
            d += timedelta(days=1)
        return hit, tot

    print("=" * 78)
    print("EXTERNAL VALIDATION: documented Nairobi flood events")
    print("=" * 78)
    print(f"  Label rule: >= {RAIN_THRESH_MM:.0f} mm accumulated over {FORECAST_DAYS} days")
    print(f"  CHIRPS series: {dates[0]} to {dates[-1]}\n")

    detected = 0
    for ev in DOCUMENTED_FLOODS:
        w0, w1 = ev["window"]
        peak_3d = accum(ev["peak"], FORECAST_DAYS)
        peak_7d = accum(ev["peak"] - timedelta(days=6), 7)
        hit, tot = flagged_days(w0, w1)
        ok = hit > 0
        detected += ok

        print(f"  {ev['name']}")
        print(f"    reported : {ev['reported']}")
        print(f"    source   : {ev['source']}")
        print(f"    window   : {w0} to {w1}   peak {ev['peak']}")
        if peak_3d is None:
            print("    RAINFALL : outside CHIRPS series\n")
            continue
        print(f"    rainfall : {peak_3d:6.1f} mm over 3 days from peak, "
              f"{peak_7d:6.1f} mm over the preceding week")
        print(f"    labels   : {hit}/{tot} days in window flagged flood-positive")
        print(f"    VERDICT  : {'DETECTED' if ok else 'MISSED'}\n")

    print("-" * 78)
    print("  Dry-season controls (labels should stay quiet)\n")
    false_alarm_days = control_days = 0
    for c in DRY_CONTROLS:
        hit, tot = flagged_days(*c["window"])
        false_alarm_days += hit
        control_days += tot
        print(f"    {c['name']:<28} {hit:3d}/{tot:3d} days flagged   "
              f"{'CLEAN' if hit == 0 else 'FALSE ALARMS'}")

    print()
    print("=" * 78)
    print(f"  Documented events detected : {detected}/{len(DOCUMENTED_FLOODS)}")
    print(f"  Dry-control false alarms   : {false_alarm_days}/{control_days} days "
          f"({100*false_alarm_days/max(control_days,1):.1f}%)")
    print("=" * 78)

    # ---- what each model could actually have known, on real events ----------
    print()
    print("=" * 78)
    print("  WHY MODEL A FAILS, ON REAL EVENTS")
    print("=" * 78)
    print("  Model A sees only the 7 days BEFORE the flood and must anticipate it.")
    print("  Model B is given the rainfall that actually fell.\n")
    print(f"  {'event':<32}{'antecedent':>12}{'what fell':>12}   {'A can see it?':<14}")
    print("  " + "-" * 72)
    for ev in DOCUMENTED_FLOODS:
        i = index.get(ev["peak"])
        if i is None or i - 7 < 0:
            continue
        ante = float(rain[i - 7:i].sum())
        fell = accum(ev["peak"], FORECAST_DAYS)
        # could antecedent rain alone have signalled the storm?
        plausible = "plausibly" if ante >= RAIN_THRESH_MM else "NO - looked dry"
        print(f"  {ev['name']:<32}{ante:>9.1f} mm{fell:>9.1f} mm   {plausible:<14}")
    # narrate the starkest case using the computed figures, never hard-coded ones
    worst = None
    for ev in DOCUMENTED_FLOODS:
        i = index.get(ev["peak"])
        if i is None or i - 7 < 0:
            continue
        ante = float(rain[i - 7:i].sum())
        fell = accum(ev["peak"], FORECAST_DAYS) or 0.0
        if worst is None or ante < worst[1]:
            worst = (ev["name"], ante, fell)

    if worst:
        name, ante, fell = worst
        print()
        print(f"  The {name} event is the clearest case: only {ante:.1f} mm fell in")
        print(f"  the preceding week, so the antecedent record looked like a dry spell")
        print(f"  right up until {fell:.1f} mm arrived. No model restricted to rainfall")
        print("  history could have called it. This is what the F1 0.17 ceiling looks")
        print("  like on a real flood, and why an operational system needs a")
        print("  meteorological forecast rather than an extrapolation of past rainfall.")
    print("\n  Caveat: this validates the TEMPORAL trigger only. Whether the predicted")
    print("  spatial extent matches where flooding actually occurred remains unverified")
    print("  -- reports name affected settlements but give no inundation polygons.")


if __name__ == "__main__":
    main()
