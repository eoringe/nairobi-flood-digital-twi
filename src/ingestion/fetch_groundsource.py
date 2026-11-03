"""
src.ingestion.fetch_groundsource
================================
Past Nairobi floods from Google's Groundsource dataset.

WHAT IT IS
----------
Groundsource (Google Research, 2026) is 2.6 million flood events that Gemini
extracted from news articles worldwide, 2000 to February 2026. Each event has a
polygon for the place reported flooded and the dates it was reported. It is the
dataset Google trained its Flood Hub flash-flood model on.

    https://research.google/blog/introducing-groundsource-turning-news-reports-into-data-with-gemini/
    https://doi.org/10.5281/zenodo.18647053   (CC BY 4.0)

Google's manual review found 60% of events right in both place and timing and
82% usable. News coverage is uneven, so a place with no event is not evidence
that it stayed dry.

WHAT THIS KEEPS
---------------
Events whose polygon touches the model grid (src.grid_config) and covers at most
MAX_AREA_KM2 -- larger polygons are "Nairobi" or "Kenya" and say nothing about
where in the city. Written to data/processed/nairobi_groundsource.json, which is
small enough to commit; the 667 MB parquet is not kept.

USAGE
-----
    pip install pyarrow
    python -m src.ingestion.fetch_groundsource [--parquet path/to/groundsource_2026.parquet]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from src.grid_config import LAT_NORTH, LAT_SOUTH, LON_EAST, LON_WEST

PARQUET_URL = "https://zenodo.org/api/records/18647054/files/groundsource_2026.parquet/content"
OUT = Path("data/processed/nairobi_groundsource.json")
MAX_AREA_KM2 = 200.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=Path("data/raw/groundsource_2026.parquet"))
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import shapely
    from shapely.geometry import box

    if not args.parquet.exists():
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        print(f"[GET] {PARQUET_URL} (667 MB)")
        urllib.request.urlretrieve(PARQUET_URL, args.parquet)

    table = pq.read_table(args.parquet, columns=["uuid", "area_km2", "geometry", "start_date", "end_date"])
    print(f"[LOAD] {table.num_rows:,} events worldwide")
    grid = box(LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    touches = shapely.intersects(geoms, grid)
    area = table.column("area_km2").to_numpy(zero_copy_only=False)
    keep = touches & (area <= MAX_AREA_KM2)
    print(f"[FILTER] {int(touches.sum())} touch the grid, {int(keep.sum())} of them <= {MAX_AREA_KM2:.0f} km2")

    cols = {n: table.column(n).to_pylist() for n in ("uuid", "start_date", "end_date")}
    events = []
    for i in keep.nonzero()[0]:
        g = shapely.set_precision(geoms[i], 1e-5)
        events.append({
            "uuid": cols["uuid"][i],
            "start": str(cols["start_date"][i])[:10],
            "end": str(cols["end_date"][i])[:10],
            "area_km2": round(float(area[i]), 3),
            "wkt": g.wkt,
        })
    events.sort(key=lambda e: (e["start"], e["uuid"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "Groundsource (Google Research, 2026), doi:10.5281/zenodo.18647053, CC BY 4.0",
        "max_area_km2": MAX_AREA_KM2,
        "events": events,
    }, separators=(",", ":")))
    print(f"[SAVE] {len(events)} events -> {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
