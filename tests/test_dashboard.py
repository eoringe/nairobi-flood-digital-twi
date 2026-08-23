"""
tests.test_dashboard
====================
Unit tests for 3D Pydeck canvas rendering and Dash application layout.
"""

import numpy as np
import pydeck as pdk

from src.dashboard.components.map_3d import create_3d_digital_twin_deck
from src.dashboard.layouts import build_dashboard_layout


def test_pydeck_3d_creation():
    """Verify Pydeck 3D deck constructs with layers and html output."""
    dummy_grid = np.random.uniform(0, 3.0, (198, 252)).astype(np.float32)
    deck = create_3d_digital_twin_deck(depth_grid=dummy_grid)
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) >= 1
    html_str = deck.to_html(as_string=True)
    assert "deckgl" in html_str.lower() or "mapbox" in html_str.lower() or "<html" in html_str.lower()


def test_dashboard_layout_construction():
    """Verify Dash layout constructs container tree without error."""
    layout = build_dashboard_layout()
    assert layout is not None
