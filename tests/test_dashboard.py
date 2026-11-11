"""
tests.test_dashboard
====================
The web application: page structure, health endpoint, map generation and the
place summary the side panel reads.

Traceability: TC-08, TC-10 (CHAPTER_5_GUIDE.md).
"""

from __future__ import annotations

import time

import pytest

from tests.conftest import needs_model

pytestmark = needs_model

REQUIRED_IDS = {
    "3d-map-frame", "mode-radio", "forecast-source", "hour-slider", "rain-slider",
    "route-origin", "route-destination", "outlook-card", "alert-log-body",
    "zone-risk-table", "metric-affected-pop", "btn-theme",
}


def _collect_ids(component, found=None):
    found = set() if found is None else found
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_ids(child, found)
    elif children is not None:
        _collect_ids(children, found)
    return found


def test_page_contains_every_control_the_callbacks_drive():
    from src.dashboard.layouts import build_dashboard_layout
    missing = REQUIRED_IDS - _collect_ids(build_dashboard_layout())
    assert not missing, f"layout is missing: {sorted(missing)}"


def test_health_endpoint_reports_status():
    """TC-10: the platform health check must answer while warm-up continues."""
    from src.dashboard.app import server
    response = server.test_client().get("/healthz")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "warmup_complete" in body
    assert body["model"].startswith("segmentation_model_v2")


def test_map_document_is_built_with_flood_and_route_layers(grid_60):
    """TC-08: a city-wide flood renders within the interactive budget."""
    from src.dashboard.components.map_3d import (
        ROUTE_LAYER_IDS, create_3d_digital_twin_deck, get_deck_html_with_embedded_legend,
    )
    start = time.perf_counter()
    deck = create_3d_digital_twin_deck(depth_grid=grid_60, value_is_probability=True, theme="light")
    html = get_deck_html_with_embedded_legend(deck)
    elapsed = time.perf_counter() - start

    layer_ids = {layer.id for layer in deck.layers}
    assert {"flood-main", "flood-halo"} <= layer_ids
    assert set(ROUTE_LAYER_IDS) <= layer_ids, "route layers must exist even when empty"
    assert deck._flood_features, "a 60 mm storm should produce flood polygons"
    assert "floodUpdate" in html and "pydeck-embedded-legend" in html
    assert elapsed < 10.0, f"map build took {elapsed:.1f}s"


def test_light_and_dark_use_different_basemaps(grid_60):
    from src.dashboard.components.map_3d import create_3d_digital_twin_deck
    light = create_3d_digital_twin_deck(depth_grid=grid_60, value_is_probability=True, theme="light")
    dark = create_3d_digital_twin_deck(depth_grid=grid_60, value_is_probability=True, theme="dark")
    assert light.map_style != dark.map_style


def test_flood_polygons_are_summarised_by_place(grid_60):
    """The side panel and the map must describe the same flooding."""
    from src.dashboard.components.map_3d import create_3d_digital_twin_deck, summarise_flooded_regions
    deck = create_3d_digital_twin_deck(depth_grid=grid_60, value_is_probability=True)
    regions = summarise_flooded_regions(deck._flood_features)

    assert regions, "no places reported for a 60 mm storm"
    name, first = next(iter(regions.items()))
    assert isinstance(name, str) and name
    assert first["risk_level"] in ("CRITICAL", "HIGH", "MODERATE", "LOW")
    assert 0 <= first["flooded_pct"] <= 100
    assert 0 <= first["peak_probability_pct"] <= 100


def test_no_flooding_produces_no_polygons(predictor):
    from src.dashboard.components.map_3d import create_3d_digital_twin_deck, summarise_flooded_regions
    dry = predictor.probability_for(0.0)
    deck = create_3d_digital_twin_deck(depth_grid=dry, value_is_probability=True)
    assert deck._flood_features == []
    assert summarise_flooded_regions(deck._flood_features) == {}
