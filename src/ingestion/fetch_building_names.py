"""
src.ingestion.fetch_building_names
==================================
Named buildings across Nairobi from OpenStreetMap, for the map's hover labels.

WHY
---
The 3D city is drawn from OpenFreeMap vector tiles, which carry each building's
height but not its name. Hovering a tower should say what it is ("Times Tower,
134 m"), so the named buildings are fetched once from OpenStreetMap and looked
up by position when the pointer is over a building.

CACHING
-------
Written to `data/processed/nairobi_building_names.json` and committed, so the
dashboard never depends on Overpass at runtime. Re-run with --refresh to update.
The file records the query box and the fetch date.

ATTRIBUTION
-----------
Building names are from OpenStreetMap, (c) OpenStreetMap contributors, available
under the Open Database Licence (ODbL).

USAGE
-----
    python -m src.ingestion.fetch_building_names
    python -m src.ingestion.fetch_building_names --refresh
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from loguru import logger

from src.ingestion.fetch_place_names import OVERPASS_MIRRORS

OUT_FILE = Path("data/processed/nairobi_building_names.json")

#: The prediction grid (-1.35..-1.23, 36.72..36.90).
BBOX = (-1.35, 36.72, -1.23, 36.90)          # south, west, north, east

QUERY = """
[out:json][timeout:120];
(
  way["building"]["name"]({s},{w},{n},{e});
  relation["building"]["name"]({s},{w},{n},{e});
);
out tags geom;
"""


def _height(tags: dict) -> float | None:
    """Tagged height in metres, else floors x 3.4 m, else None."""
    try:
        return round(float(str(tags["height"]).lower().replace("m", "").split()[0]), 1)
    except (KeyError, ValueError, IndexError):
        pass
    try:
        return round(float(tags["building:levels"]) * 3.4, 1)
    except (KeyError, ValueError):
        return None


def _outer_ring(el: dict) -> list[list[float]] | None:
    """The footprint as [[lon, lat], ...]: the way itself, or a relation's first outer way."""
    if el["type"] == "way":
        pts = el.get("geometry") or []
    else:
        outer = [m for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
        pts = outer[0]["geometry"] if outer else []
    if len(pts) < 3:
        return None
    return [[round(p["lon"], 6), round(p["lat"], 6)] for p in pts]


def fetch() -> list[dict]:
    s, w, n, e = BBOX
    q = QUERY.format(s=s, w=w, n=n, e=e)
    logger.info(f"Querying Overpass for named buildings in {BBOX}...")
    raw, last_error = None, None
    for attempt, url in enumerate(OVERPASS_MIRRORS, start=1):
        try:
            req = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"},
            )
            raw = json.loads(urllib.request.urlopen(req, timeout=150).read().decode())
            logger.info(f"  answered by mirror {attempt}: {url}")
            break
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            logger.warning(f"  mirror {attempt} failed ({type(exc).__name__}); trying next")
    if raw is None:
        raise RuntimeError(f"every Overpass mirror failed; last error: {last_error}. "
                           f"The cached file is committed, so this only matters when refreshing.")

    buildings = []
    for el in raw.get("elements", []):
        tags = el.get("tags", {})
        ring = _outer_ring(el)
        if not tags.get("name") or ring is None:
            continue
        b = {"name": tags["name"].strip(), "ring": ring}
        h = _height(tags)
        if h:
            b["height"] = h
        buildings.append(b)
    buildings.sort(key=lambda b: b["name"].lower())
    return buildings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch even when the cache already exists")
    args = ap.parse_args()
    if OUT_FILE.exists() and not args.refresh:
        existing = json.load(open(OUT_FILE, encoding="utf-8"))
        logger.info(f"{OUT_FILE} already holds {existing['count']} buildings "
                    f"(fetched {existing['fetched']}). Use --refresh to update.")
        return

    buildings = fetch()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "source": "OpenStreetMap via Overpass API",
        "licence": "ODbL - (c) OpenStreetMap contributors",
        "fetched": date.today().isoformat(),
        "bbox_south_west_north_east": list(BBOX),
        "count": len(buildings),
        "buildings": buildings,
    }, separators=(",", ":")), encoding="utf-8")
    logger.info(f"Wrote {len(buildings)} named buildings -> {OUT_FILE}")


if __name__ == "__main__":
    main()
