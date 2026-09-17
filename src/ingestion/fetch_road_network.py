"""
src.ingestion.fetch_road_network
================================
Download the Nairobi road network from OpenStreetMap for flood-aware routing.

WHY
---
The router needs a graph of roads it can drive: which points connect, how long
each stretch is, what class of road it is and whether it is one-way. OSM carries
all of that for Nairobi. Only the road classes a car can use are fetched;
footpaths, tracks and private driveways are excluded.

CACHING
-------
The result is written as gzip-compressed JSON to
`data/processed/nairobi_roads.json.gz` and committed, for the same reason as the
place gazetteer: the dashboard must never depend on Overpass being reachable
while it runs. Re-run with --refresh to update.

FORMAT
------
    {
      "nodes": {"<osm id>": [lat, lon], ...},        only nodes used by a way
      "ways":  [{"id", "highway", "name", "oneway", "nodes": [ids]}, ...]
    }

`oneway` is normalised to 1 (forward only), -1 (reverse only) or 0 (both ways).

ATTRIBUTION
-----------
Road data (c) OpenStreetMap contributors, Open Database Licence (ODbL).

USAGE
-----
    python -m src.ingestion.fetch_road_network
    python -m src.ingestion.fetch_road_network --refresh
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from loguru import logger

from src.ingestion.fetch_place_names import OVERPASS_MIRRORS

OUT_FILE = Path("data/processed/nairobi_roads.json.gz")

#: The prediction grid (-1.35..-1.23, 36.72..36.90) plus a small margin, so a
#: route between two places near the edge can still leave the grid briefly.
BBOX = (-1.36, 36.71, -1.22, 36.91)          # south, west, north, east

#: Drivable classes. `service` is kept because many Nairobi estates are reached
#: only by roads tagged that way, but parking aisles and driveways are dropped.
HIGHWAY_CLASSES = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service",
)

QUERY = """
[out:json][timeout:180];
way["highway"~"^({classes})$"]
   ["access"!~"^(private|no)$"]
   ["service"!~"^(parking_aisle|driveway|drive-through)$"]
   ({s},{w},{n},{e})->.roads;
.roads out body;
node(w.roads);
out skel qt;
"""


def _oneway(tags: dict) -> int:
    v = tags.get("oneway", "").lower()
    if v in ("yes", "1", "true"):
        return 1
    if v == "-1":
        return -1
    if v in ("no", "0", "false"):
        return 0
    # Roundabouts and motorways are one-way unless tagged otherwise.
    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") == "motorway":
        return 1
    return 0


def fetch() -> dict:
    s, w, n, e = BBOX
    q = QUERY.format(classes="|".join(HIGHWAY_CLASSES), s=s, w=w, n=n, e=e)
    logger.info(f"Querying Overpass for drivable roads in {BBOX}...")

    raw, last_error = None, None
    for attempt, url in enumerate(OVERPASS_MIRRORS, start=1):
        try:
            req = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"},
            )
            raw = json.loads(urllib.request.urlopen(req, timeout=200).read().decode())
            logger.info(f"  answered by mirror {attempt}: {url}")
            break
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            logger.warning(f"  mirror {attempt} failed ({type(exc).__name__}); trying next")
    if raw is None:
        raise RuntimeError(f"every Overpass mirror failed; last error: {last_error}")

    nodes, ways = {}, []
    for el in raw.get("elements", []):
        if el["type"] == "node":
            nodes[str(el["id"])] = [round(el["lat"], 6), round(el["lon"], 6)]
        elif el["type"] == "way" and len(el.get("nodes", [])) >= 2:
            tags = el.get("tags", {})
            ways.append({
                "id": el["id"],
                "highway": tags.get("highway", "unclassified"),
                "name": tags.get("name") or tags.get("ref") or "",
                "oneway": _oneway(tags),
                "nodes": [str(x) for x in el["nodes"]],
            })

    # Drop references to nodes the query did not return (ways clipped by the box).
    for wy in ways:
        wy["nodes"] = [x for x in wy["nodes"] if x in nodes]
    ways = [wy for wy in ways if len(wy["nodes"]) >= 2]
    used = {x for wy in ways for x in wy["nodes"]}
    nodes = {k: v for k, v in nodes.items() if k in used}
    return {"nodes": nodes, "ways": ways}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if OUT_FILE.exists() and not args.refresh:
        logger.info(f"{OUT_FILE} already exists. Use --refresh to update.")
        return

    net = fetch()
    by_class: dict[str, int] = {}
    for wy in net["ways"]:
        by_class[wy["highway"]] = by_class.get(wy["highway"], 0) + 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "OpenStreetMap via Overpass API",
        "licence": "ODbL - (c) OpenStreetMap contributors",
        "fetched": date.today().isoformat(),
        "bbox_south_west_north_east": list(BBOX),
        **net,
    }
    with gzip.open(OUT_FILE, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    logger.info(f"Wrote {len(net['ways']):,} ways / {len(net['nodes']):,} nodes -> "
                f"{OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
    for c, k in sorted(by_class.items(), key=lambda kv: -kv[1]):
        logger.info(f"   {c:<16} {k}")


if __name__ == "__main__":
    main()
