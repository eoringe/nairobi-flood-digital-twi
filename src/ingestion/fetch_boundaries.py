"""
src.ingestion.fetch_boundaries
==============================
Counties, sub-counties (constituencies) and wards that overlap the model grid,
from geoBoundaries (specification §2.2, §2.12, FR-013).

    ADM1  counties        Public Domain (RCMRD)
    ADM2  sub-counties    CC BY 3.0 IGO (IEBC, OCHA ROSEA)
    ADM3  wards           CC BY 4.0 (Omare and Omare)

Each sub-county is linked to the county containing it, and each ward to the
sub-county containing it, by representative point. Output:
data/processed/nairobi_boundaries.json (committed).

USAGE
-----
    python -m src.ingestion.fetch_boundaries                 # downloads to data/raw/boundaries
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import shapely
from loguru import logger
from shapely.geometry import box, shape

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

RAW = Path("data/raw/boundaries")
OUT = Path("data/processed/nairobi_boundaries.json")
URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/KEN/{lvl}/"
       "geoBoundaries-KEN-{lvl}_simplified.geojson")
LEVELS = {"ADM1": "county", "ADM2": "sub_county", "ADM3": "ward"}
LICENSES = {"county": "Public Domain (RCMRD)", "sub_county": "CC BY 3.0 IGO (IEBC, OCHA ROSEA)",
            "ward": "CC BY 4.0 (B. A. Omare and G. Omare)"}


def _load(lvl: str) -> dict:
    path = RAW / f"KEN_{lvl}.geojson"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading geoBoundaries {lvl}...")
        req = urllib.request.Request(URL.format(lvl=lvl), headers={"User-Agent": "nairobi-flood-digital-twin/1.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    grid = box(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)
    levels: dict[str, list[dict]] = {}
    for lvl, level in LEVELS.items():
        rows = []
        for f in _load(lvl)["features"]:
            g = shape(f["geometry"])
            # Ignore slivers: a neighbouring unit that only grazes the grid edge.
            if not g.intersects(grid) or g.intersection(grid).area < 1e-6:
                continue
            g = shapely.set_precision(g, 1e-5)
            rows.append({"id": f["properties"]["shapeID"], "name": f["properties"]["shapeName"].replace(" Ii", " II"),
                         "level": level, "parent": None, "geom": g})
        levels[level] = rows
    for child, parent in (("sub_county", "county"), ("ward", "sub_county")):
        for c in levels[child]:
            p = c["geom"].representative_point()
            owner = next((x for x in levels[parent] if x["geom"].contains(p)), None)
            c["parent"] = owner["id"] if owner else None
    out = {"source": "geoBoundaries gbOpen KEN (release 9469f09), simplified", "licenses": LICENSES,
           **{lvl: [{**{k: v for k, v in r.items() if k != "geom"}, "wkt": r["geom"].wkt}
                    for r in sorted(rows, key=lambda r: r["name"])] for lvl, rows in levels.items()}}
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    logger.info(f"{len(levels['county'])} counties, {len(levels['sub_county'])} sub-counties, "
                f"{len(levels['ward'])} wards -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
