"""
tests.test_routing
==================
Flood-aware routing over the OpenStreetMap road network.

Traceability: TC-04, TC-05 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.routing.flood_router import LEVEL_NAMES, level_of, place_name_at, smooth_for_routing
from tests.conftest import needs_model, needs_roads

pytestmark = needs_roads


def test_road_graph_covers_nairobi(router):
    assert router.n_nodes > 100_000
    assert router.n_edges > 200_000
    # Every routable node must belong to one connected component, or a route
    # could be requested between points that cannot reach each other.
    assert len(router.routable) > 0.9 * router.n_nodes


def test_route_between_two_places_on_dry_ground(router, places):
    result = router.route(places["Eastleigh"], places["Kibera"], None)
    safe, fast = result["safe"], result["fastest"]
    assert safe is not None and fast is not None
    assert safe.node_path == fast.node_path, "with no flooding both routes should agree"
    assert 5.0 < safe.distance_km < 30.0
    assert safe.duration_min > 0
    assert len(safe.coordinates) > 10
    assert safe.worst_level == 0


@needs_model
def test_flood_aware_route_avoids_flooding_the_fastest_route_crosses(router, places, grid_60):
    """TC-04: the recommended route trades time for lower flood exposure."""
    result = router.route(places["Eastleigh"], places["Kibera"], grid_60)
    safe, fast = result["safe"], result["fastest"]

    assert safe.exposure_km["CRITICAL"] == 0.0, "recommended route must avoid critical flooding"
    assert fast.worst_level > safe.worst_level or \
        sum(fast.exposure_km.values()) > sum(safe.exposure_km.values())
    assert safe.duration_min >= fast.duration_min, "avoiding flooding should not be faster"
    assert result["latency_sec"] < 2.0


@needs_model
def test_flooded_stretches_are_named(router, places, grid_60):
    """A warning must say which road and which neighbourhood."""
    fast = router.route(places["Eastleigh"], places["Kibera"], grid_60)["fastest"]
    assert fast.flooded_stretches, "the fastest route crosses flooding in this scenario"
    stretch = fast.flooded_stretches[0]
    assert stretch["road"] and stretch["place"]
    assert stretch["level"] in LEVEL_NAMES.values()
    assert stretch["km"] > 0


def test_point_far_from_any_road_is_detected(router):
    """TC-05: snapping distance must expose points the network cannot serve."""
    _node, metres = router.nearest_node(-0.15, 37.30)        # far outside Nairobi
    assert metres > 600, "a distant point must not silently snap to a Nairobi road"

    _node, close = router.nearest_node(-1.2864, 36.8172)     # city centre
    assert close < 200


@needs_model
def test_same_start_and_destination_is_refused(router, places):
    result = router.route(places["Kibera"], places["Kibera"], None)
    assert result["safe"] is None
    assert "same point" in (result["note"] or "").lower()


@needs_model
def test_route_exposure_changes_as_the_forecast_changes(router, places, predictor, grid_60):
    """Time-aware routing: the same road is clear in one hour and flooded in another."""
    dry = predictor.probability_for(0.0)
    safe = router.route(places["Eastleigh"], places["Kibera"], grid_60)["safe"]
    levels = router.exposure_over_time(safe.edge_ids, [dry, grid_60])
    assert levels[0] == 0
    assert levels[1] >= levels[0]


def test_flood_field_is_dilated_for_routing(grid_60):
    """Roads at the edge of a flood band should inherit that band's risk."""
    field = smooth_for_routing(grid_60)
    assert field.shape == grid_60.shape
    assert (level_of(field) > 0).sum() >= (level_of(grid_60) > 0).sum()


def test_places_are_named_inside_the_grid_only():
    assert place_name_at(-1.2864, 36.8172) not in ("", "outside the model area")
    assert place_name_at(-0.15, 37.30) == "outside the model area"
