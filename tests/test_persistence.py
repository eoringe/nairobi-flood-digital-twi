"""
tests.test_persistence
======================
Scenario history and the database schema from Chapter 4's ERD.

Both use temporary files, so a test run never touches the real database.

Traceability: TC-11 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import numpy as np

from src.persistence import scenario_store
from src.persistence.database import DatabaseManager

SAMPLE_RISKS = {
    "Mathare": {"risk_level": "CRITICAL", "peak_probability_pct": 91.0, "flooded_pct": 62.0},
    "Kibera": {"risk_level": "MODERATE", "peak_probability_pct": 38.0, "flooded_pct": 9.0},
}


def test_scenario_run_is_saved_and_listed(monkeypatch, tmp_path):
    """TC-11: a saved scenario survives and comes back in the history list."""
    monkeypatch.setattr(scenario_store, "DB_PATH", tmp_path / "scenarios.db")
    scenario_store.init_db()

    row_id = scenario_store.save_scenario_run(
        rainfall_mm_day=60.0, time_hour=0.0, display_mode="PROBABILITY",
        max_depth_m=4.3, flooded_area_km2=13.5, est_affected_pop=366_100,
        region_risks=SAMPLE_RISKS,
    )
    assert row_id is not None

    rows = scenario_store.list_recent_scenarios(limit=5)
    assert len(rows) == 1
    assert rows[0]["rainfall_mm_day"] == 60.0
    assert rows[0]["critical_zone_count"] == 1          # CRITICAL and HIGH count
    assert rows[0]["run_at_utc"]


def test_saving_never_raises_when_storage_is_unavailable(monkeypatch, tmp_path):
    """A container without a writable volume must keep serving (Railway)."""
    monkeypatch.setattr(scenario_store, "DB_PATH", tmp_path / "no-such-dir" / "x.db")
    monkeypatch.setattr(scenario_store.Path, "mkdir",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("read-only")))
    assert scenario_store.save_scenario_run(
        rainfall_mm_day=30.0, time_hour=0.0, display_mode="PROBABILITY",
        max_depth_m=1.0, flooded_area_km2=1.0, est_affected_pop=10, region_risks={},
    ) is None


def test_database_schema_creates_every_table(tmp_path):
    """The ERD from Chapter 4 is implemented as 19 tables."""
    db = DatabaseManager(database_url=f"sqlite:///{tmp_path / 'twin.db'}")
    assert db.initialise() is True

    tables = db.list_tables()
    assert len(tables) >= 19, f"expected the full schema, found {len(tables)}: {tables}"
    for expected in ("users", "rainfall_observations", "topographic_grid",
                     "model_versions", "simulations", "simulation_results", "audit_log"):
        assert expected in tables
    db.close()


def test_model_version_can_be_registered_and_activated(tmp_path):
    """Predictions must be traceable to the model that produced them."""
    db = DatabaseManager(database_url=f"sqlite:///{tmp_path / 'twin.db'}")
    db.initialise()

    model_id = db.register_model(
        artefact_path="models/time_series/segmentation_model_v2_nwp_drainage.pth",
        metrics={"f1": 0.937, "iou": 0.881},
        dataset_version_id=None, architecture="U-Net", activate=True,
    )
    assert model_id is not None

    active = db.get_active_model()
    assert active and active["architecture"] == "U-Net"
    db.close()


def test_simulation_and_result_are_stored_together(tmp_path):
    db = DatabaseManager(database_url=f"sqlite:///{tmp_path / 'twin.db'}")
    db.initialise()

    sim_id = db.store_simulation(rainfall_mm=60.0, exec_time_sec=1.2)
    assert sim_id is not None

    probability = np.zeros((198, 252), dtype=np.float32)
    probability[100:110, 100:110] = 0.9
    stored = db.store_result(sim_id, probability, threshold=0.5,
                             result_path=str(tmp_path / "result.npy"))
    assert stored is not False
    assert db.get_user_simulations(limit=5)
    db.close()
