"""
src.ingestion.fetch_place_names
===============================
Build a gazetteer of named places across Nairobi from OpenStreetMap.

WHY
---
Region labelling previously used ten hand-picked monitoring points. That was a
limitation of the interface, not of the model: predictions are produced for all
49,896 grid cells, so any flooded patch anywhere in the county should be
nameable. With ten reference points most water fell between them and could only
be reported as "Unmonitored area".

OpenStreetMap carries several hundred named suburbs, neighbourhoods, quarters
and villages across Nairobi. Using them turns region naming from a curated
shortlist into coverage of the whole prediction grid.

CACHING
-------
The result is written to `data/processed/nairobi_places.json` and committed, so
the dashboard never depends on Overpass being reachable at runtime. Re-run this
script to refresh. The file records the query, the bounding box and the fetch
date, so what is in it can be audited rather than trusted.

ATTRIBUTION
-----------
Place names are from OpenStreetMap, (c) OpenStreetMap contributors, available
under the Open Database Licence (ODbL). Any map or figure using these names must
carry that attribution.

USAGE
-----
    python -m src.ingestion.fetch_place_names
    python -m src.ingestion.fetch_place_names --refresh
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from loguru import logger

OUT_FILE = Path("data/processed/nairobi_places.json")
#: The public Overpass instances are shared and rate-limited; 504s are routine.
#: Tried in order until one answers, so a busy mirror does not fail the build.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    # overpass.osm.ch is deliberately absent: it holds only Swiss data and
    # answers a Nairobi query with a successful, empty result.
    "https://overpass.openstreetmap.ru/api/interpreter",
)

#: Slightly wider than the prediction grid (-1.35..-1.23, 36.72..36.90) so a
#: polygon at the edge can still be named after a place just outside it.
BBOX = (-1.37, 36.70, -1.21, 36.92)          # south, west, north, east

#: Settlement-scale tags only. Excludes city/county-level entries, which would
#: label a specific flooded street "Nairobi" and tell the reader nothing.
PLACE_TYPES = ("suburb", "neighbourhood", "quarter", "village", "hamlet",
               "locality", "city_block")

QUERY = """
[out:json][timeout:90];
(
  node["place"~"^({types})$"]({s},{w},{n},{e});
  way["place"~"^({types})$"]({s},{w},{n},{e});
  relation["place"~"^({types})$"]({s},{w},{n},{e});
);
out center tags;
"""


def fetch() -> list[dict]:
    s, w, n, e = BBOX
    q = QUERY.format(types="|".join(PLACE_TYPES), s=s, w=w, n=n, e=e)
    logger.info(f"Querying Overpass for named places in {BBOX}...")

    raw = None
    last_error: Exception | None = None
    for attempt, url in enumerate(OVERPASS_MIRRORS, start=1):
        try:
            req = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"},
            )
            raw = json.loads(urllib.request.urlopen(req, timeout=110).read().decode())
            logger.info(f"  answered by mirror {attempt}: {url}")
            break
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            logger.warning(f"  mirror {attempt} failed ({type(exc).__name__}); trying next")
    if raw is None:
        raise RuntimeError(
            f"every Overpass mirror failed; last error: {last_error}. "
            f"The cached gazetteer is committed, so this only matters when refreshing."
        )

    places, seen = [], set()
    for el in raw.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        # OSM often carries the same settlement as both a node and a way.
        key = (name.lower(), round(lat, 3), round(lon, 3))
        if key in seen:
            continue
        seen.add(key)
        places.append({
            "name": name,
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "place_type": tags.get("place"),
        })

    places.sort(key=lambda p: p["name"])
    return places


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even when the cache already exists")
    args = ap.parse_args()

    if OUT_FILE.exists() and not args.refresh:
        existing = json.load(open(OUT_FILE, encoding="utf-8"))
        logger.info(
            f"{OUT_FILE} already holds {len(existing['places'])} places "
            f"(fetched {existing['fetched']}). Use --refresh to update."
        )
        return

    places = fetch()
    by_type: dict[str, int] = {}
    for p in places:
        by_type[p["place_type"]] = by_type.get(p["place_type"], 0) + 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "source": "OpenStreetMap via Overpass API",
        "licence": "ODbL - (c) OpenStreetMap contributors",
        "fetched": date.today().isoformat(),
        "bbox_south_west_north_east": list(BBOX),
        "place_types": list(PLACE_TYPES),
        "count": len(places),
        "places": places,
    }, indent=1), encoding="utf-8")

    logger.info(f"Wrote {len(places)} named places -> {OUT_FILE}")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        logger.info(f"   {t:<16} {c}")


if __name__ == "__main__":
    main()
