"""
src.grid_config
================
Single source of truth for the Nairobi flood-prediction grid.

Every stage of the pipeline (dataset assembly, training, inference,
dashboard rendering) must agree on exactly which patch of the earth
pixel (r, c) of the (GRID_H, GRID_W) array represents. Previously this
window was defined once in src.models.predict and silently assumed
elsewhere, while src.preprocessing.dataset_builder built its terrain
arrays from the *entire* SRTM mosaic (a ~2 deg x 2 deg region spanning
well beyond Nairobi) resized to (GRID_H, GRID_W) without ever cropping
to this window first. The two arrays covered different ground at the
same pixel indices. Importing the bounds from here instead of
redefining them keeps that from happening again.
"""

from __future__ import annotations

GRID_H: int = 198
GRID_W: int = 252

# Nairobi flood-prone core window (EPSG:4326). Deliberately tighter than
# the county-wide ingestion bbox (-1.45/-1.15, 36.65/37.10) used by the
# raw data fetchers — this is the sub-window the model actually predicts
# over, matching the hotspots in src.models.predict.NAIROBI_LOCATIONS.
LAT_NORTH: float = -1.23   # row 0
LAT_SOUTH: float = -1.35   # row GRID_H - 1
LON_WEST: float = 36.72    # col 0
LON_EAST: float = 36.90    # col GRID_W - 1
