"""
tests.test_forecast
===================
Rainfall bias correction, the 12-hour outlook, and behaviour when the forecast
service is unreachable.

Rainfall is supplied as synthetic hourly series through monkeypatch, so these
tests are deterministic and never touch the network.

Traceability: TC-03, TC-09, TC-12 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.forecast import nowcast as nc_mod
from src.forecast import rain_bias_correction as bias
from tests.conftest import needs_model

AS_OF = datetime(2024, 4, 24, 10)


def _series(values_by_hour: dict[int, float], hours_back: int = 96) -> dict[datetime, float]:
    """Hourly rainfall series around AS_OF; keys are hour offsets from it."""
    series = {AS_OF + timedelta(hours=h): 0.0 for h in range(-hours_back, 25)}
    for h, mm in values_by_hour.items():
        series[AS_OF + timedelta(hours=h)] = mm
    return series


# ----------------------------------------------------------- bias correction --
def test_bias_correction_table_is_available():
    assert bias.is_available(), "run `python -m src.forecast.rain_bias_correction`"


def test_correction_scales_up_storm_totals():
    """TC-12: the forecast feed reports about half of CHIRPS in storms."""
    corrected = bias.correct_3day_total(58.0)
    assert corrected > 100.0, f"58 mm forecast corrected to only {corrected:.0f} mm"


def test_correction_is_monotone_and_keeps_dry_dry():
    values = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0])
    corrected = bias.correct_3day_total(values)
    assert np.all(np.diff(corrected) >= 0), "correction must never reorder totals"
    assert corrected[0] < 1.0, "no rain must stay no rain"


# ----------------------------------------------------------------- outlook --
def test_trend_reports_rain_arriving_later_in_the_window():
    """Regression: a storm in hour 4+ was described as 'little or no rain'."""
    series = _series({1: 0.0, 2: 0.0, 3: 0.0, 5: 4.0, 6: 6.0, 7: 5.0})
    assert nc_mod._trend(series, AS_OF) == "arriving"


def test_trend_reports_intensifying_and_dry():
    assert nc_mod._trend(_series({1: 3.0, 2: 4.0, 3: 5.0}), AS_OF) == "intensifying"
    assert nc_mod._trend(_series({}), AS_OF) == "dry"


@needs_model
def test_place_levels_flag_a_flooded_area(predictor, grid_60):
    """A place is only warned when enough of its cells reach a risk band."""
    place_idx, names = nc_mod.place_raster()
    levels = nc_mod._place_levels(grid_60, place_idx[:-1], len(names))
    assert levels.max() >= 2, "a 60 mm storm should reach at least the high band somewhere"
    assert (levels > 0).sum() >= 5, "several places should be flagged"

    dry = np.zeros_like(grid_60)
    assert nc_mod._place_levels(dry, place_idx[:-1], len(names)).max() == 0


@needs_model
def test_outlook_warns_before_flooding_starts(monkeypatch, predictor):
    """
    TC-03: rain that pushes the 72-hour total past the flooding level within the
    window must produce a warning that says when, not merely that it is raining.
    """
    rising = {h: 3.0 for h in range(1, 13)}          # steady heavy rain ahead
    rising.update({-h: 0.35 for h in range(0, 72)})  # some antecedent rain
    series = _series(rising)
    monkeypatch.setattr(nc_mod, "_hourly_series",
                        lambda source, as_of: (series, AS_OF, "synthetic test feed", None))
    monkeypatch.setitem(nc_mod.SOURCES, "test-event", {"label": "test", "as_of": AS_OF})

    out = nc_mod.build_nowcast("test-event", predictor)

    assert out.error is None
    assert out.acc72_mm[0] < out.acc72_mm[-1], "72-hour total should rise across the window"
    assert out.bias_corrected is True
    assert out.warnings, "no warning produced for a storm crossing the flooding level"
    first = out.warnings[0]
    assert first["places"], "a warning must name places"
    assert "flooding" in first["text"].lower()
    assert 0 <= first["hour"] <= nc_mod.HORIZON_H


@needs_model
def test_outlook_stays_quiet_when_it_barely_rains(monkeypatch, predictor):
    series = _series({h: 0.02 for h in range(1, 13)})
    monkeypatch.setattr(nc_mod, "_hourly_series",
                        lambda source, as_of: (series, AS_OF, "synthetic test feed", None))
    monkeypatch.setitem(nc_mod.SOURCES, "test-dry", {"label": "test", "as_of": AS_OF})

    out = nc_mod.build_nowcast("test-dry", predictor)

    assert out.warnings == []
    assert "No flooding expected" in out.headline


# ------------------------------------------------------- offline resilience --
def test_last_forecast_is_reused_when_the_service_is_down(monkeypatch, tmp_path):
    """TC-09: an unreachable forecast API must not blank the outlook."""
    issued = datetime.now(nc_mod.EAT).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    times = [(issued - timedelta(hours=k)).isoformat() for k in range(72, -1, -1)]
    cache = tmp_path / "live_rainfall_cache.json"
    cache.write_text(json.dumps({
        "as_of": issued.isoformat(), "provider": "cached test feed",
        "data": {"hourly": {"time": times, "precipitation": [0.1] * len(times)}},
    }), encoding="utf-8")

    monkeypatch.setattr(nc_mod, "LIVE_CACHE_FILE", cache)
    monkeypatch.setattr(nc_mod, "_get_json", lambda *a, **k: (_ for _ in ()).throw(OSError("network down")))

    series, as_of, provider, notice = nc_mod._hourly_series("live", None)

    assert series, "cached rainfall should have been returned"
    assert provider == "cached test feed"
    assert notice and notice.startswith("Offline")


def test_cache_write_failure_does_not_raise(monkeypatch, tmp_path):
    """A read-only or root-owned volume must not break the forecast (Railway)."""
    target = tmp_path / "missing-dir" / "cache.json"
    monkeypatch.setattr(nc_mod.Path, "mkdir",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("read-only volume")))
    nc_mod._save_json(target, {"a": 1})       # must log a warning, not raise
    assert not target.exists()
