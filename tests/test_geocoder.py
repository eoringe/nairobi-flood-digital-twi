"""
tests.test_geocoder
===================
Location search for the trip planner. Only the offline index is exercised
(`online=False`), so no request leaves the machine.

Traceability: TC-06, TC-07 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import time

from src.routing.geocoder import decode_value, encode_value


def test_index_covers_thousands_of_locations(search):
    assert len(search.names) > 10_000


def test_finds_a_landmark_by_name(search):
    """TC-06: a well-known Nairobi landmark is found offline."""
    results = search.search("kencom", online=False)
    assert results, "no match for 'kencom'"
    assert "kencom" in results[0]["label"].lower()
    assert decode_value(results[0]["value"]) is not None


def test_finds_a_road_and_a_neighbourhood(search):
    assert any("ngong road" in r["label"].lower() for r in search.search("ngong road", online=False))
    assert any("kibera" in r["label"].lower() for r in search.search("kibera", online=False))


def test_offline_search_is_fast(search):
    """TC-06: search runs while the user types, so it must not block."""
    start = time.perf_counter()
    search.search("kariobangi", online=False)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"offline search took {elapsed:.3f}s"


def test_coordinates_can_be_typed_directly(search):
    results = search.search("-1.2921, 36.8219", online=False)
    assert len(results) == 1
    lat, lon, _name = decode_value(results[0]["value"])
    assert abs(lat + 1.2921) < 1e-4 and abs(lon - 36.8219) < 1e-4


def test_points_outside_the_mapped_area_are_offered_but_disabled(search):
    """TC-07: a miss is explained rather than looking like a broken search."""
    results = search.search("-1.10, 36.80", online=False)
    assert len(results) == 1
    assert results[0].get("disabled") is True
    assert "outside the mapped area" in results[0]["label"]


def test_short_queries_return_nothing(search):
    assert search.search("k", online=False) == []
    assert search.search("", online=False) == []


def test_option_values_round_trip():
    value = encode_value(-1.2921, 36.8219, "Test Place")
    lat, lon, name = decode_value(value)
    assert (round(lat, 4), round(lon, 4), name) == (-1.2921, 36.8219, "Test Place")
