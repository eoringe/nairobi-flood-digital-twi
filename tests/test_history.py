"""
tests.test_history
==================
Past floods from Groundsource news reports: the map layer and the validation's
scoring helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import numpy as np
import pytest

from src.reports import history

# Two reports of the same Mathare polygon and one of a small Kibera one.
MATHARE = "POLYGON ((36.85 -1.27, 36.87 -1.27, 36.87 -1.25, 36.85 -1.25, 36.85 -1.27))"
KIBERA = "POLYGON ((36.783 -1.3135, 36.7835 -1.3135, 36.7835 -1.313, 36.783 -1.313, 36.783 -1.3135))"


@pytest.fixture
def events(tmp_path, monkeypatch):
    path = tmp_path / "groundsource.json"
    path.write_text(json.dumps({"events": [
        {"uuid": "a", "start": "2023-11-19", "end": "2023-11-20", "area_km2": 4.9, "wkt": MATHARE},
        {"uuid": "b", "start": "2024-04-24", "end": "2024-04-25", "area_km2": 4.9, "wkt": MATHARE},
        {"uuid": "c", "start": "2024-04-24", "end": "2024-04-24", "area_km2": 0.003, "wkt": KIBERA},
    ]}))
    monkeypatch.setattr(history, "EVENTS_FILE", path)
    for fn in (history.load_events, history.places):
        fn.cache_clear()
    yield
    for fn in (history.load_events, history.places):
        fn.cache_clear()


def test_one_area_per_place_with_its_report_days(events):
    rows = history.map_rows()
    assert len(rows) == 2
    mathare = next(r for r in rows if "2 days" in r["title"])
    assert "most recently 24 Apr 2024" in mathare["detail"]
    assert all(r["near"] == "" and r["source"] == history.ATTRIBUTION for r in rows)


def test_a_replayed_storm_picks_out_that_weeks_places(events):
    rows = history.map_rows(around=date(2024, 4, 23))
    assert all(r["near"].startswith("In the news that week: 24 Apr 2024") for r in rows)
    assert rows[0]["fill"][:3] == history.FILL_NEAR
    s = history.summary(around=date(2024, 4, 23))
    assert (s["events"], s["places"], s["shown"], s["near_events"], s["near_places"]) == (3, 2, 2, 2, 2)
    assert history.summary(around=date(2025, 1, 10))["near_events"] == 0


def test_cell_mask_covers_the_polygon_and_never_drops_a_tiny_one(events):
    ev = {e["uuid"]: e for e in history.load_events()}
    big = history.cell_mask(ev["a"]["geom"])
    assert 500 < big.sum() < 1500          # ~2.2 x 2.2 km at ~70 m cells
    tiny = history.cell_mask(ev["c"]["geom"])
    assert tiny.sum() == 1, "a polygon smaller than a cell still marks its cell"


def test_areas_larger_than_a_neighbourhood_are_not_drawn(events, monkeypatch):
    monkeypatch.setattr(history, "MAP_MAX_KM2", 1.0)
    rows = history.map_rows()
    assert len(rows) == 1 and "1 day" in rows[0]["title"]
    assert history.summary()["shown"] == 1


def test_missing_file_means_no_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "EVENTS_FILE", tmp_path / "absent.json")
    history.load_events.cache_clear(); history.places.cache_clear()
    assert history.map_rows() == [] and history.summary() == {"events": 0, "places": 0}
    history.load_events.cache_clear(); history.places.cache_clear()


# ------------------------------------------------------- validation helpers --
def test_auc_is_one_half_for_no_skill_and_one_for_perfect():
    from scipy.stats import rankdata
    from src.validation.validate_groundsource import _auc, _auc_within
    score = np.arange(100, dtype=float)
    inside = score >= 90
    assert _auc(rankdata(score), inside) == 1.0
    assert _auc(rankdata(np.ones(100)), inside) == 0.5
    # Perfect within each stratum even though the strata themselves are reversed.
    strata = (score < 50).astype(int)
    within = np.where(strata == 1, score + 100, score)
    inside2 = (score % 50) >= 45
    assert _auc_within(within, strata, inside2) == 1.0


def test_a_warning_covers_the_three_days_of_its_outlook():
    from src.validation.validate_groundsource import _rates
    d = date(2024, 4, 20)
    flood = {date(2024, 4, 22), date(2024, 4, 30)}
    period = [date(2024, 4, k) for k in range(1, 31)]
    r = _rates({d}, flood, period, Counter({date(2024, 4, 22): 3, date(2024, 4, 30): 1}))
    assert r["flood_days_warned"] == 1 and r["hit_rate_events"] == 0.75
    assert r["precision"] == 1.0
