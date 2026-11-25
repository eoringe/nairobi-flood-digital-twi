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

import os

import pytest

# The app loads .env when imported (src.dashboard.app), and .env may hold the
# deployed DATABASE_URL. Tests must never write there: unless a disposable
# PostgreSQL test database is chosen explicitly (TWIN_TEST_POSTGRES=1, below),
# DATABASE_URL is blanked before anything is imported. load_dotenv() does not
# override a variable that is already set, even to "".
if os.environ.get("TWIN_TEST_POSTGRES") != "1":
    os.environ["DATABASE_URL"] = ""

#: Fixed date so day-of-year inputs, and therefore model output, are repeatable.
SCENARIO_DATE = date(2024, 4, 26)

DATA = Path("data/processed")
MODEL_FILE = Path("models/time_series/segmentation_model_v2_nwp_drainage.pth")

needs_model = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason=f"{MODEL_FILE} not present; copy the trained weights before running model tests",
)
#: Tests that open or fake a SQLite file directly (older databases, a read-only
#: disk); they do not apply when the stores run on PostgreSQL.
sqlite_only = pytest.mark.skipif(
    __import__("os").environ.get("TWIN_TEST_POSTGRES") == "1",
    reason="checks the SQLite file itself; PostgreSQL is covered by tests/test_postgres_backend.py",
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


# ------------------------------------------------------------- PostgreSQL --
# With TWIN_TEST_POSTGRES=1 and DATABASE_URL pointing at a disposable test
# database, every store runs on PostgreSQL instead of SQLite, and each test
# starts from empty schemas, as it would from a fresh temporary SQLite file:
#
#   TWIN_TEST_POSTGRES=1 DATABASE_URL=postgresql://user@localhost:5432/twin_test \
#       python -m pytest tests/test_auth.py tests/test_reports.py ...
#
# Never point it at a database whose contents matter: the schemas are dropped.
import os as _os

if _os.environ.get("TWIN_TEST_POSTGRES") == "1":
    @pytest.fixture(autouse=True)
    def _fresh_postgres_schemas():
        import psycopg
        from src.persistence import backend
        assert backend.using_postgres(), "TWIN_TEST_POSTGRES=1 needs a postgresql:// DATABASE_URL"
        backend.close_pools()
        backend._IDENTITY.clear()
        with psycopg.connect(backend.database_url(), autocommit=True) as c:
            for schema in ("accounts", "operations", "reports", "scenarios"):
                c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        yield
        backend.close_pools()
