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
