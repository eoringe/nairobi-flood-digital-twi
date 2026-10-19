"""
tests.test_model
================
The deployed flood model: output shape and range, behaviour as rainfall rises,
calibration, caching and inference latency.

Traceability: TC-01, TC-02, TC-08 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.grid_config import GRID_H, GRID_W
from tests.conftest import SCENARIO_DATE, needs_model

CELL_KM2 = 0.067 * 0.079
pytestmark = needs_model


def flooded_km2(grid: np.ndarray, threshold: float = 0.5) -> float:
    return float((grid > threshold).sum()) * CELL_KM2


def test_output_is_a_probability_grid_of_the_right_size(grid_60):
    """TC-01: one probability per grid cell, every value a valid probability."""
    assert grid_60.shape == (GRID_H, GRID_W)
    assert grid_60.min() >= 0.0
    assert grid_60.max() <= 1.0
    assert np.isfinite(grid_60).all()


def test_sixty_millimetres_floods_a_plausible_area(grid_60):
    """TC-01: a 60 mm storm floods a realistic share of the city, not all or nothing."""
    area = flooded_km2(grid_60)
    assert 8.0 < area < 20.0, f"flooded area {area:.1f} km2 outside the expected range"


def test_no_rain_produces_no_flooding(predictor):
    """TC-02: with no rainfall, no cell is reported flooded."""
    dry = predictor.probability_for(0.0, scenario_date=SCENARIO_DATE)
    assert flooded_km2(dry) == 0.0


def test_flooded_area_grows_with_rainfall(predictor):
    """More rain must never flood less ground."""
    areas = [flooded_km2(predictor.probability_for(mm, scenario_date=SCENARIO_DATE))
             for mm in (30.0, 60.0, 100.0)]
    assert areas[0] < areas[1] < areas[2], f"areas not increasing: {areas}"


def test_flooding_concentrates_on_susceptible_ground(predictor, grid_60):
    """A prediction that ignored terrain would spread evenly; this one must not."""
    check = predictor.validate_spatial_plausibility(grid_60)
    assert check["plausible"] is True
    assert check["concentration_ratio"] > 1.5


def test_probabilities_are_calibrated(predictor):
    """The deployed model applies the isotonic calibration fitted on validation seasons."""
    assert predictor.calibration is not None, "calibration file missing; run src.models.calibrate_v2"
    assert "calibration" in predictor.model_version
    x = np.asarray(predictor.calibration["x"])
    y = np.asarray(predictor.calibration["y"])
    assert np.all(np.diff(y) >= -1e-9), "calibration must be monotone, or it would reorder risk"
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert len(x) == len(y)


def test_repeated_predictions_are_cached_and_identical(predictor):
    """Caching must return the same grid, not merely a similar one."""
    first = predictor.probability_for(45.0, scenario_date=SCENARIO_DATE)
    start = time.perf_counter()
    second = predictor.probability_for(45.0, scenario_date=SCENARIO_DATE)
    elapsed = time.perf_counter() - start
    assert np.array_equal(first, second)
    assert elapsed < 0.05, f"cached lookup took {elapsed:.3f}s"


def test_inference_latency_is_acceptable(predictor):
    """TC-08: an uncached prediction stays within the interactive budget."""
    start = time.perf_counter()
    predictor.predict(rainfall_mm=77.0, scenario_date=SCENARIO_DATE)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"inference took {elapsed:.2f}s on this machine"


def test_result_declares_its_units_and_model(predictor):
    """Guards against a depth being displayed: the result must say what it is."""
    result = predictor.predict(rainfall_mm=60.0, scenario_date=SCENARIO_DATE)
    assert "probability" in result["units"].lower()
    assert "not a depth" in result["units"].lower()
    assert result["model_version"].startswith("segmentation_model_v2")
    assert result["assumptions"]["forecast_days"] == 3
