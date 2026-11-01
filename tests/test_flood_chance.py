"""
tests.test_flood_chance
=======================
The 3-day flood chance learned from news-reported floods, its 95% call, and
the flood-prone ranking the outlook orders places by.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.forecast import flood_chance as fc

AS_OF = datetime(2024, 4, 24, 10)


def _hourly(daily_mm: dict[int, float], days_back: int = 31) -> dict[datetime, float]:
    """Hour-ending series; `daily_mm` maps a day offset from AS_OF's date to its total."""
    start = datetime.combine(AS_OF.date(), datetime.min.time()) - timedelta(days=days_back)
    out = {}
    for h in range(1, (days_back + 3) * 24 + 1):
        t = start + timedelta(hours=h)
        out[t] = daily_mm.get(((t - timedelta(hours=1)).date() - AS_OF.date()).days, 0.0) / 24
    return out


def test_fitted_parameters_are_present():
    assert fc._params() is not None, "run `python -m src.forecast.flood_chance`"


def test_the_95_percent_call_held_on_years_it_was_not_calibrated_on():
    calls = fc._params()["evaluation"]["feed_calibration_cross_fitted"]["call_95"]
    assert calls["flood_days_not_called_unlikely"] >= 0.93
    assert calls["dry_days_not_called_likely"] >= 0.93


def test_trained_on_real_floods_it_beats_the_rainfall_rule_on_held_out_years():
    ev = fc._params()["evaluation"]["operational_feed"]
    assert ev["auc"]["model"] > ev["auc"]["rule_3day_total"]
    at = ev["at_rule_warning_count"]
    assert at["model"]["precision"] > at["rule"]["precision"]


def test_more_rain_and_wetter_ground_raise_the_chance():
    base = fc.chance([20.0, 20.0, 40.0])["p"]
    assert fc.chance([60.0, 20.0, 40.0])["p"] > base, "a wetter forecast"
    assert fc.chance([20.0, 120.0, 200.0])["p"] > base, "wetter ground before it"


def test_calls_follow_the_conformal_cut_offs():
    assert fc.chance([0.0, 0.0, 0.0])["call"] == "unlikely"
    assert fc.chance([150.0, 150.0, 250.0])["call"] == "likely"
    p = fc._params()
    assert p["t_flood"] < p["t_dry"], "the 'possible' band must exist between the two cut-offs"


def test_chance_from_hourly_needs_the_30_days_before_today():
    assert fc.chance_from_hourly(_hourly({}, days_back=10), AS_OF) is None
    out = fc.chance_from_hourly(_hourly({0: 30.0, 1: 30.0, -5: 20.0}), AS_OF)
    assert out is not None and 0.0 < out["p"] < 1.0 and out["call_text"] in fc.CALLS.values()


def test_daily_totals_respect_hour_ending_timestamps():
    t0 = datetime(2024, 4, 24)
    daily = fc._daily({t0: 5.0, t0 + timedelta(hours=1): 2.0})
    assert daily == {t0.date() - timedelta(days=1): 5.0, t0.date(): 2.0}, "00:00 closes the previous day"


def test_feed_features_are_bias_corrected_in_three_day_blocks():
    raw = fc.features([10.0, 10.0, 10.0], [3.0] * 30, feed=False)
    assert raw == [30.0, 42.0, 90.0]
    corrected = fc.features([10.0, 10.0, 10.0], [3.0] * 30, feed=True)
    assert corrected[0] == pytest.approx(fc.correct_3day_total(30.0))
    assert corrected[2] == pytest.approx(10 * fc.correct_3day_total(9.0))


# ------------------------------------------------------- flood-prone ranking --
def test_flood_prone_rank_lifts_places_reported_before(monkeypatch):
    from src.reports import history
    susc = np.tile(np.linspace(0, 1, 10), (10, 1))          # rises west to east
    density = np.zeros((10, 10)); density[:, 0] = 5.0       # reports only on the west edge
    monkeypatch.setattr(history, "report_density", lambda until=None: density)
    r = history.flood_prone_rank(susc)
    assert r.shape == susc.shape and 0.0 <= r.min() and r.max() <= 1.0
    assert r[:, 0].mean() > r[:, 1].mean(), "past reports lift the least susceptible column"

    monkeypatch.setattr(history, "report_density", lambda until=None: np.zeros((10, 10)))
    plain = history.flood_prone_rank(susc)
    assert np.corrcoef(plain.ravel(), susc.ravel())[0, 1] > 0.99, "with no reports it is susceptibility"


def test_a_replay_never_ranks_places_by_the_floods_it_replays(monkeypatch):
    from src.forecast import nowcast
    seen = {}

    def fake_rank(susc, until=None):
        seen["until"] = until
        return np.zeros(susc.shape)

    class Predictor:
        susceptibility = np.ones((2, 2))

    monkeypatch.setattr(nowcast, "flood_prone_rank", fake_rank)
    nowcast._place_order(Predictor(), np.array([0, 0, 1, 1]), np.array([2, 2]), until=AS_OF.date() - timedelta(days=1))
    assert seen["until"] < AS_OF.date()
