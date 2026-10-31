"""
src.reports.satellite
=====================
Flooding observed from space: the Copernicus Global Flood Monitoring (GFM)
flood map from Sentinel-1 radar, for the map's optional satellite layer.

SOURCE
------
GFM maps flood extent from every Sentinel-1 pass at 20 m, usually within
8 hours of the pass, and publishes it through a free WMS-T service with no key
(https://geoserver.gfm.eodc.eu/geoserver/gfm/wms). Copernicus Emergency
Management Service, (c) European Union.

WHAT IT CAN AND CANNOT SHOW (measured for Nairobi, April-May 2024)
------------------------------------------------------------------
* Sentinel-1 passed over Nairobi about every six days (24 April, 1 May, 6 May),
  so the newest map can be days old: it is not live.
* Radar cannot map water between buildings (the signal bounces off walls), so
  GFM masks out built-up areas: 74-87% of the city on those passes. During the
  worst floods (1 May 2024) it detected no flooding in the visible part of the
  map area. The layer therefore shows open ground and riverbanks at best, and
  marks where the satellite cannot see.

A pass is recognised by the exclusion mask, which is only drawn for days with
an acquisition over the area.
"""

from __future__ import annotations

import io
import threading
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
from loguru import logger
from PIL import Image

from src.models.predict import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

GFM_WMS = "https://geoserver.gfm.eodc.eu/geoserver/gfm/wms"
# The plain layers: the group layers also draw the swath outline, which
# would read as a line of flooding along the edge of every pass.
FLOOD_LAYER = "observed_flood_extent"
MASK_LAYER = "exclusion_mask"
#: How far back to look for the newest pass.
SEARCH_DAYS = 14
#: Output size: about 15 m per pixel over the prediction grid.
WIDTH, HEIGHT = 1200, 800
#: Drawn colours: observed flooding, and where the radar cannot see.
WATER_RGBA = (56, 189, 248, 220)
UNSEEN_RGBA = (148, 163, 184, 46)

_cache: dict[date, dict | None] = {}
_lock = threading.Lock()


def _get_map(layer: str, day: date, width: int, height: int) -> np.ndarray:
    """One WMS GetMap over the prediction grid, as an RGBA array."""
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer, "styles": "",
        # WMS 1.3.0 with EPSG:4326 takes the box as lat, lon.
        "crs": "EPSG:4326", "bbox": f"{LAT_SOUTH},{LON_WEST},{LAT_NORTH},{LON_EAST}",
        "width": width, "height": height, "format": "image/png", "transparent": "true",
        "time": f"{day.isoformat()}T00:00:00Z",
    }
    req = urllib.request.Request(f"{GFM_WMS}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"})
    raw = urllib.request.urlopen(req, timeout=45).read()
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))


def _is_grey(rgba: np.ndarray) -> np.ndarray:
    r, g, b = (rgba[..., i].astype(int) for i in range(3))
    return (abs(r - g) < 12) & (abs(g - b) < 12)


def latest_pass(on_or_before: date, days: int = SEARCH_DAYS) -> date | None:
    """The newest day with a Sentinel-1 acquisition over Nairobi, or None."""
    for back in range(days + 1):
        day = on_or_before - timedelta(days=back)
        if (_get_map(MASK_LAYER, day, 120, 80)[..., 3] > 0).any():
            return day
    return None


def render(flood: np.ndarray, mask: np.ndarray) -> tuple[bytes, float, float]:
    """(PNG bytes, % of the area seen flooded, % the radar could see) from the two layers."""
    water = (flood[..., 3] > 30) & ~_is_grey(flood)
    unseen = (mask[..., 3] > 30) & _is_grey(mask) & ~water
    out = np.zeros(flood.shape[:2] + (4,), np.uint8)
    out[unseen] = UNSEEN_RGBA
    out[water] = WATER_RGBA
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue(), round(100.0 * float(water.mean()), 2), round(100.0 * float(1 - unseen.mean()), 1)


def observed(anchor: date) -> dict | None:
    """
    The newest satellite flood map on or before `anchor`: {"date", "png",
    "flooded_pct", "visible_pct"}, or None if no pass was found. Cached per anchor.
    """
    with _lock:
        if anchor in _cache:
            return _cache[anchor]
    try:
        day = latest_pass(anchor)
        result = None
        if day is not None:
            png, flooded, visible = render(_get_map(FLOOD_LAYER, day, WIDTH, HEIGHT),
                                           _get_map(MASK_LAYER, day, WIDTH, HEIGHT))
            result = {"date": day.isoformat(), "png": png, "flooded_pct": flooded, "visible_pct": visible}
            logger.info(f"Satellite flood map: Sentinel-1 pass {day}, {flooded}% flooded, "
                        f"{visible}% of the area visible to radar")
    except Exception as exc:                              # noqa: BLE001
        logger.warning(f"Satellite flood map unavailable: {exc}")
        return None                                      # not cached, so it is retried
    with _lock:
        _cache[anchor] = result
    return result
