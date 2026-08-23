"""
tests.test_pipeline
====================
Unit tests for data ingestion, feature preprocessing, and memory guardrails.
"""

import numpy as np
import pytest
from pathlib import Path

from src.utils.memory_check import MemoryGuard

ARRAYS_DIR = Path("data/processed/arrays")

# The multi-GB processed arrays are gitignored (MEMORY_CONSTRAINTS.md-scale
# artefacts, not checked into source control) and only exist locally after
# running the ingestion + dataset_builder pipeline. Skip rather than fail
# in a fresh checkout / CI runner that hasn't run that pipeline, so CI still
# catches real regressions in code that doesn't require the data.
requires_processed_arrays = pytest.mark.skipif(
    not (ARRAYS_DIR / "static_terrain_features.npy").exists(),
    reason="data/processed/arrays not present — run the ingestion + dataset_builder pipeline first",
)


def test_memory_guard_initialization():
    """Verify MemoryGuard initializes without error."""
    guard = MemoryGuard()
    rss = guard.check_rss()
    assert rss > 0.0
    assert rss < 14.0


@requires_processed_arrays
def test_processed_arrays_exist():
    """Verify preprocessed arrays exist in data/processed/arrays/."""
    assert (ARRAYS_DIR / "dem_mosaic.npy").exists()
    assert (ARRAYS_DIR / "static_terrain_features.npy").exists()
    assert (ARRAYS_DIR / "X_train.npy").exists()
    assert (ARRAYS_DIR / "y_train.npy").exists()


@requires_processed_arrays
def test_processed_arrays_shapes():
    """Verify static terrain features matrix shape is (3, 198, 252)."""
    terrain = np.load(ARRAYS_DIR / "static_terrain_features.npy")
    assert terrain.shape == (3, 198, 252)
    assert terrain.dtype == np.float32
