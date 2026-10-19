"""
Shared fixtures for the test suite.

The model, road graph and search index are expensive to load (seconds each), so
they are built once per session and shared. Tests never call the internet:
anything that would fetch rainfall or geocode a name is either restricted to the
offline path or fed synthetic data through monkeypatch.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

#: Fixed date so day-of-year inputs, and therefore model output, are repeatable.
SCENARIO_DATE = date(2024, 4, 26)

DATA = Path("data/processed")
MODEL_FILE = Path("models/time_series/segmentation_model_v2_nwp_drainage.pth")

needs_model = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason=f"{MODEL_FILE} not present; copy the trained weights before running model tests",
)
needs_roads = pytest.mark.skipif(
    not (DATA / "nairobi_roads.json.gz").exists(),
    reason="road network not present; run `python -m src.ingestion.fetch_road_network`",
)


@pytest.fixture(scope="session")
def predictor():
    from src.models.predict_v2 import FloodPredictor
    return FloodPredictor()


@pytest.fixture(scope="session")
def grid_60(predictor):
    """Flood probabilities for a 60 mm three-day storm: widespread but not extreme."""
    return predictor.probability_for(60.0, scenario_date=SCENARIO_DATE)


@pytest.fixture(scope="session")
def router():
    from src.routing.flood_router import FloodAwareRouter
    return FloodAwareRouter()


@pytest.fixture(scope="session")
def search():
    from src.routing.geocoder import LocationSearch
    return LocationSearch()


@pytest.fixture(scope="session")
def places() -> dict[str, tuple[float, float]]:
    raw = json.loads((DATA / "nairobi_places.json").read_text(encoding="utf-8"))["places"]
    out: dict[str, tuple[float, float]] = {}
    for p in raw:
        out.setdefault(p["name"], (p["lat"], p["lon"]))
    return out
