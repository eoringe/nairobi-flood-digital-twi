"""
src.ingestion.fetch_osm_layers
==============================
Rivers, water bodies and infrastructure for the operations map, from
OpenStreetMap (ODbL), clipped to the model grid.

    data/processed/nairobi_rivers.json          named rivers and streams, water bodies
    data/processed/nairobi_infrastructure.json  schools, health facilities, bridges,
                                                public buildings, emergency services

The infrastructure file is the interim source for specification §2.9 until the
Infrastructure Risk Mapping System is connected; the interface labels it as
OpenStreetMap data. OSM coverage of informal settlements is incomplete, so a
missing facility on the map is not evidence that none exists.

USAGE
-----
    python -m src.ingestion.fetch_osm_layers
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from loguru import logger

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST
from src.ingestion.fetch_place_names import OVERPASS_MIRRORS

RIVERS_FILE = Path("data/processed/nairobi_rivers.json")
INFRA_FILE = Path("data/processed/nairobi_infrastructure.json")
BBOX = (LAT_SOUTH, LON_WEST, LAT_NORTH, LON_EAST)          # south, west, north, east
ATTRIBUTION = "© OpenStreetMap contributors, ODbL"

RIVERS_QUERY = """
[out:json][timeout:120];
(
  way["waterway"~"^(river|stream|canal)$"]({bbox});
  way["natural"="water"]({bbox});
  relation["natural"="water"]({bbox});
);
out tags geom;
"""

INFRA_QUERY = """
[out:json][timeout:150];
(
  nwr["amenity"~"^(school|college|university|kindergarten)$"]({bbox});
  nwr["amenity"~"^(hospital|clinic|doctors|health_post)$"]({bbox});
  nwr["healthcare"~"^(hospital|clinic|centre|health_post)$"]({bbox});
  nwr["amenity"~"^(police|fire_station|townhall|courthouse|community_centre|social_facility)$"]({bbox});
  nwr["office"="government"]({bbox});
  way["bridge"="yes"]["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]({bbox});
);
out tags center;
"""

#: OSM tags -> the specification's infrastructure categories (§2.9).
def _category(tags: dict) -> str | None:
    a, h = tags.get("amenity", ""), tags.get("healthcare", "")
    if tags.get("bridge") == "yes" and tags.get("highway"):
        return "bridge"
    if a in ("school", "college", "university", "kindergarten"):
        return "school"
    if a == "hospital" or h == "hospital":
        return "hospital"
    if a in ("clinic", "doctors", "health_post") or h in ("clinic", "centre", "health_post"):
        return "health_centre"
    if a in ("police", "fire_station"):
        return "emergency_service"
    if a in ("townhall", "courthouse", "community_centre", "social_facility") or tags.get("office") == "government":
        return "public_building"
    return None


def _overpass(query: str) -> dict:
    bbox = ",".join(str(v) for v in BBOX)
    q = query.format(bbox=bbox)
    last: Exception | None = None
    for i, url in enumerate(OVERPASS_MIRRORS, start=1):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"})
            out = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
            logger.info(f"  answered by mirror {i}: {url}")
            return out
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            logger.warning(f"  mirror {i} failed ({type(exc).__name__}: {exc}); trying next")
    raise RuntimeError(f"every Overpass mirror failed; last error: {last}")


def _line(geom: list[dict]) -> list[list[float]]:
    return [[round(p["lon"], 5), round(p["lat"], 5)] for p in geom]


def build_rivers(raw: dict) -> dict:
    rivers, water = [], []
    for el in raw.get("elements", []):
        t = el.get("tags", {})
        if el["type"] == "way" and t.get("waterway") and el.get("geometry"):
            rivers.append({"name": t.get("name:en") or t.get("name") or "", "kind": t["waterway"],
                           "path": _line(el["geometry"])})
        elif t.get("natural") == "water":
            if el["type"] == "way" and el.get("geometry") and len(el["geometry"]) >= 4:
                water.append({"name": t.get("name") or "", "kind": t.get("water", "water"),
                              "polygon": _line(el["geometry"])})
            elif el["type"] == "relation":
                for m in el.get("members", []):
                    if m.get("role") == "outer" and len(m.get("geometry") or []) >= 4:
                        water.append({"name": t.get("name") or "", "kind": t.get("water", "water"),
                                      "polygon": _line(m["geometry"])})
    return {"source": ATTRIBUTION, "fetched": date.today().isoformat(), "rivers": rivers, "water_bodies": water}


def build_infrastructure(raw: dict) -> dict:
    assets, seen = [], set()
    for el in raw.get("elements", []):
        t = el.get("tags", {})
        cat = _category(t)
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if cat is None or lat is None:
            continue
        name = t.get("name") or ""
        if cat == "bridge":
            name = name or (f"Bridge on {t['highway']} road" if not t.get("ref") else f"Bridge, {t['ref']}")
        elif not name:
            continue                                   # an unnamed school point cannot be acted on
        key = (cat, name.lower(), round(lat, 3), round(lon, 3))
        if key in seen:
            continue
        seen.add(key)
        assets.append({"id": f"osm-{el['type'][0]}{el['id']}", "name": name, "category": cat,
                       "lat": round(float(lat), 6), "lon": round(float(lon), 6)})
    assets.sort(key=lambda a: (a["category"], a["name"]))
    return {"source": ATTRIBUTION, "fetched": date.today().isoformat(), "assets": assets}


def main() -> None:
    logger.info("Rivers and water bodies...")
    rivers = build_rivers(_overpass(RIVERS_QUERY))
    RIVERS_FILE.write_text(json.dumps(rivers, separators=(",", ":")), encoding="utf-8")
    logger.info(f"  {len(rivers['rivers'])} river/stream lines, {len(rivers['water_bodies'])} water bodies -> {RIVERS_FILE}")
    logger.info("Infrastructure...")
    infra = build_infrastructure(_overpass(INFRA_QUERY))
    INFRA_FILE.write_text(json.dumps(infra, separators=(",", ":")), encoding="utf-8")
    from collections import Counter
    logger.info(f"  {len(infra['assets'])} assets {dict(Counter(a['category'] for a in infra['assets']))} -> {INFRA_FILE}")


if __name__ == "__main__":
    main()
