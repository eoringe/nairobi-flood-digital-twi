"""
tests.test_map_labels
=====================
Map labels (sub-counties, rivers, facilities) and the map document's layer
settings.
"""

from __future__ import annotations

import json

from src.geo import layers


def test_every_sub_county_is_labelled_inside_itself():
    from shapely.geometry import Point
    from src.geo import boundaries
    rows = layers.subcounty_label_rows()
    subs = boundaries.load()["sub_county"]
    assert len(rows) == len(subs) == 19
    for r, a in zip(rows, subs):
        assert r["text"] == a["name"].upper() and a["geom"].contains(Point(*r["position"]))


def test_swahili_river_names_read_in_english():
    assert layers.river_name("Mto Nairobi") == "Nairobi River"
    assert layers.river_name("Ngong River") == "Ngong River"
    texts = {r["text"] for r in layers.river_label_rows()}
    assert "Nairobi River" in texts and not any(t.startswith("Mto ") for t in texts)
    assert all(-90 <= r["angle"] <= 90 for r in layers.river_label_rows()), "labels stay upright"


def test_facility_labels_are_thinned_with_hospitals_first():
    items = [{"name": "Big School", "category": "school", "lat": -1.2600, "lon": 36.8600},
             {"name": "Mathare Hospital", "category": "hospital", "lat": -1.26001, "lon": 36.86001},
             {"name": "Bridge on residential road", "category": "bridge", "lat": -1.27, "lon": 36.87},
             {"name": "Far Clinic", "category": "health_centre", "lat": -1.30, "lon": 36.80}]
    texts = [r["text"] for r in layers.facility_label_rows(items)]
    assert texts == ["Mathare Hospital", "Far Clinic"], "one label per square, generic bridge names skipped"


def test_no_layer_setting_reaches_the_browser_as_an_expression():
    """pydeck turns plain strings into JS expressions: 'meters' became an undefined variable."""
    from src.dashboard.components.map_3d import create_3d_digital_twin_deck
    deck = create_3d_digital_twin_deck()
    for layer in json.loads(deck.to_json())["layers"]:
        for key, value in layer.items():
            if key.endswith("Units") or key in ("characterSet", "fontFamily"):
                assert not str(value).startswith("@@="), f"{layer['id']}.{key} = {value}"


# ------------------------------------------------------------ performance --
def test_the_map_document_stays_small():
    """The fallback buildings (~1 MB) are fetched on demand, not embedded in every rebuild."""
    from src.dashboard.components.map_3d import create_3d_digital_twin_deck, get_deck_html_with_embedded_legend
    html = get_deck_html_with_embedded_legend(create_3d_digital_twin_deck())
    assert len(html) < 300_000, f"map document is {len(html) // 1024} KB"
    assert "/data/buildings-fallback.json" in html


def test_the_map_does_not_redraw_every_frame():
    from src.dashboard.components.map_3d import CITY_JS
    assert "_animate: !!water" not in CITY_JS, "continuous redraw even with no flooding on screen"
    assert "onAfterRender: null" not in CITY_JS, "deck.gl calls onAfterRender unguarded; null breaks rendering"
    assert 'type: "perf"' in CITY_JS or "msg.type !== \"perf\"" in CITY_JS


def test_zooming_over_flooding_stays_light():
    from src.dashboard.components.map_3d import CITY_JS
    # The water shader runs once per pixel on the main flood fill only, at 3 octaves.
    assert 'layer.id === "flood-main" || layer.id === "flood-halo"' not in CITY_JS
    assert "i < 5; i++" not in CITY_JS and "dFdx(h0)" in CITY_JS
    # Lighter keeps 3D buildings but drops the water shader.
    assert "extruded: !lite" not in CITY_JS   # flat footprints read as missing buildings
    assert "extensions: lite ? [] : [water]" in CITY_JS
    # Auto also watches real frame times while panning or zooming.
    assert "onInteractionStateChange" in CITY_JS and 'applyPerf(true, "moving")' in CITY_JS
    assert 'refinementStrategy: "no-overlap"' in CITY_JS
    # 3D buildings only from zoom 14 in both modes: zoomed out, the tilted view
    # pulled in dozens of building tiles to the horizon.
    assert "function gateCity" in CITY_JS and "perf.lite ? 15" not in CITY_JS
    assert "zoom >= 13.9 : zoom >= 14" in CITY_JS   # hysteresis, no flicker at 14
    assert "minZoom: 14," in CITY_JS
    # getViewports asserts before deck.gl's first frame; that once killed the perf block.
    assert "try { vp = d.viewManager && d.getViewports()[0]; }" in CITY_JS
    # No hover picking (a full redraw of every building) while the map moves.
    assert 'ctx.renderPass === "picking:hover"' in CITY_JS


def test_fallback_buildings_are_served_on_request():
    from src.dashboard.app import server
    r = server.test_client().get("/data/buildings-fallback.json?theme=light")
    assert r.status_code == 200 and isinstance(r.get_json(), list) and r.get_json()
