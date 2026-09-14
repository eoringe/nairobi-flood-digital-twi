"""
src.ingestion.fetch_search_index
================================
Build an offline search index of every named location in Nairobi, so the trip
planner can find "Sarit Centre", "Kenyatta National Hospital" or "Moi Avenue",
not only the 332 suburb and neighbourhood names.

WHAT GOES IN
------------
* Every OpenStreetMap feature with a name inside the routing area that is not a
  road: shops, malls, schools, hospitals, churches, offices, bus stages,
  estates, parks, named buildings.
* Every named road, taken from the cached road network
  (data/processed/nairobi_roads.json.gz), one entry per stretch of road.

Features sharing a name are merged when they are within MERGE_RADIUS_M of each
other, so a mall mapped as both a point and a building outline appears once,
while "Shell" petrol stations across the city remain separate entries.

WHY OFFLINE
-----------
Searching a local file is instant and works without internet during a
presentation. The dashboard falls back to the Photon geocoder only for queries
the index cannot answer (src/routing/geocoder.py).

OUTPUT
------
data/processed/nairobi_search_index.json.gz
    {"source", "licence", "fetched", "entries": [[name, lat, lon, kind], ...]}

(c) OpenStreetMap contributors, ODbL.

USAGE
-----
    python -m src.ingestion.fetch_search_index
    python -m src.ingestion.fetch_search_index --refresh
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from loguru import logger

from src.ingestion.fetch_place_names import OVERPASS_MIRRORS
from src.ingestion.fetch_road_network import BBOX, OUT_FILE as ROADS_FILE

OUT_FILE = Path("data/processed/nairobi_search_index.json.gz")
MERGE_RADIUS_M = 250.0
TILES = 3

QUERY = """
[out:json][timeout:170][maxsize:536870912];
nwr["name"]["highway"!~"."]({s},{w},{n},{e});
out center tags qt;
"""

#: First matching tag decides the label shown beside a result.
KIND_TAGS = (
    ("amenity", {"hospital": "hospital", "clinic": "clinic", "school": "school", "university": "university",
                 "college": "college", "place_of_worship": "place of worship", "marketplace": "market",
                 "bus_station": "bus station", "police": "police", "fuel": "fuel station",
                 "bank": "bank", "restaurant": "restaurant", "cafe": "cafe", "pharmacy": "pharmacy",
                 "kindergarten": "school", "townhall": "government office"}),
    ("shop", {"mall": "mall", "supermarket": "supermarket"}),
    ("public_transport", {"*": "bus stop"}),
    ("railway", {"station": "railway station", "halt": "railway station"}),
    ("aeroway", {"aerodrome": "airport", "terminal": "airport"}),
    ("tourism", {"hotel": "hotel", "museum": "museum", "attraction": "attraction", "*": "tourism"}),
    ("leisure", {"park": "park", "stadium": "stadium", "sports_centre": "sports centre", "*": "leisure"}),
    ("office", {"government": "government office", "*": "office"}),
    ("healthcare", {"*": "health facility"}),
    ("place", {"*": "area"}),
    ("landuse", {"residential": "estate", "*": "area"}),
    ("building", {"*": "building"}),
    ("natural", {"*": "natural feature"}),
    ("waterway", {"*": "river"}),
)


def _kind(tags: dict) -> str:
    for key, mapping in KIND_TAGS:
        if key in tags:
            # Unlisted values use the value itself ("fast_food" -> "fast food"),
            # not the key: labelling 3,500 places just "amenity" said nothing.
            return mapping.get(tags[key]) or mapping.get("*") or tags[key].replace("_", " ")
    for key in ("amenity", "shop", "craft", "club", "sport", "man_made", "historic"):
        if key in tags:
            return tags[key].replace("_", " ")
    return "place"


#: Raw tile responses, kept so an interrupted build resumes instead of
#: re-downloading tiles that already succeeded. data/raw is not tracked.
TILE_CACHE = Path("data/raw/overpass_search_tiles")


def _query_tile(s: float, w: float, n: float, e: float) -> list[dict]:
    """One tile, trying each mirror. An empty answer counts as a failure."""
    cache = TILE_CACHE / f"{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    q = QUERY.format(s=s, w=w, n=n, e=e)
    last = None
    for attempt in range(4):
        for i, url in enumerate(OVERPASS_MIRRORS, 1):
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                             headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"})
                els = json.loads(urllib.request.urlopen(req, timeout=200).read().decode()).get("elements", [])
                if els:
                    TILE_CACHE.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(els), encoding="utf-8")
                    return els
                last = "empty result"
            except Exception as exc:                        # noqa: BLE001
                last = exc
            logger.warning(f"    mirror {i} failed ({type(last).__name__ if not isinstance(last, str) else last})")
        time.sleep(30 * (attempt + 1))                      # public servers recover slowly
    raise RuntimeError(f"tile {s, w, n, e} failed on every mirror: {last}")


def _fetch_features() -> list[list]:
    """
    Named features, fetched as a TILES x TILES grid. One query over the whole
    area times out on the public servers (HTTP 504); nine smaller ones do not.
    """
    s0, w0, n0, e0 = BBOX
    dlat, dlon = (n0 - s0) / TILES, (e0 - w0) / TILES
    seen, out = set(), []
    for r in range(TILES):
        for c in range(TILES):
            s, w = s0 + r * dlat, w0 + c * dlon
            logger.info(f"  tile {r * TILES + c + 1}/{TILES * TILES}")
            for el in _query_tile(s, w, s + dlat, w + dlon):
                key = (el["type"], el["id"])
                if key in seen:
                    continue
                seen.add(key)
                tags = el.get("tags", {})
                name = (tags.get("name") or "").strip()
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lon = el.get("lon") or el.get("center", {}).get("lon")
                if not name or lat is None or lon is None:
                    continue
                out.append([name, round(float(lat), 6), round(float(lon), 6), _kind(tags)])
            time.sleep(2)                                   # be polite to a shared service
    return out


def _road_entries() -> list[list]:
    data = json.loads(gzip.open(ROADS_FILE, "rt", encoding="utf-8").read())
    nodes = data["nodes"]
    out = []
    for way in data["ways"]:
        if not way["name"]:
            continue
        mid = nodes[way["nodes"][len(way["nodes"]) // 2]]
        out.append([way["name"], mid[0], mid[1], "road"])
    return out


def _merge(entries: list[list]) -> list[list]:
    """Collapse same-name entries that lie within MERGE_RADIUS_M of a kept one."""
    m_lat, m_lon = 110_570.0, 111_320.0 * math.cos(math.radians(1.29))
    by_name: dict[str, list[list]] = {}
    for e in entries:
        by_name.setdefault(e[0].lower(), []).append(e)
    merged = []
    for group in by_name.values():
        # Prefer the more specific kind when a name is both, say, a mall and a building.
        group.sort(key=lambda e: (e[3] in ("building", "place", "area"), e[3] == "road"))
        kept: list[list] = []
        for e in group:
            if all(math.hypot((e[1] - k[1]) * m_lat, (e[2] - k[2]) * m_lon) > MERGE_RADIUS_M for k in kept):
                kept.append(e)
        merged.extend(kept)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if OUT_FILE.exists() and not args.refresh:
        logger.info(f"{OUT_FILE} already exists. Use --refresh to update.")
        return

    features = _fetch_features()
    roads = _road_entries()
    entries = _merge(features + roads)
    entries.sort(key=lambda e: e[0].lower())

    with gzip.open(OUT_FILE, "wt", encoding="utf-8") as fh:
        json.dump({"source": "OpenStreetMap via Overpass API", "licence": "ODbL - (c) OpenStreetMap contributors",
                   "fetched": date.today().isoformat(), "bbox_south_west_north_east": list(BBOX),
                   "entries": entries}, fh, separators=(",", ":"), ensure_ascii=False)

    kinds: dict[str, int] = {}
    for e in entries:
        kinds[e[3]] = kinds.get(e[3], 0) + 1
    logger.info(f"Wrote {len(entries):,} locations ({len(features):,} features + {len(roads):,} road stretches "
                f"before merging) -> {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
    for k, c in sorted(kinds.items(), key=lambda kv: -kv[1])[:15]:
        logger.info(f"   {k:<20} {c}")


if __name__ == "__main__":
    main()
