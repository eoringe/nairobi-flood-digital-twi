"""
src.routing.geocoder
====================
Find any location in Nairobi by name, for the trip planner.

SEARCH ORDER
------------
1. Coordinates typed directly ("-1.2921, 36.8219").
2. The offline index (src/ingestion/fetch_search_index.py): named OpenStreetMap
   features and roads across the routing area, plus the 332 suburb names. It
   is instant and needs no internet.
3. Photon (photon.komoot.io), an OpenStreetMap geocoder, only when the index
   finds fewer than ONLINE_MIN_LOCAL results. It is searched across the county,
   cached, and rate-limited to one request per second out of courtesy to a free
   public service.

Anything outside the routing area is listed but disabled, labelled "outside the
mapped area", so a miss is explained rather than looking like a broken search.

PRIVACY
-------
Step 3 sends the typed search text to a third-party service. Nothing else is
sent: no user identity and no coordinates.

Option values encode the point itself, "loc:<lat>,<lon>|<name>", so a selection
survives without any server-side state.
"""

from __future__ import annotations

import gzip
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from loguru import logger

from src.routing.flood_router import place_name_at

INDEX_FILE = Path("data/processed/nairobi_search_index.json.gz")
PLACES_FILE = Path("data/processed/nairobi_places.json")

#: The routing area (fetch_road_network.BBOX): south, west, north, east.
AREA = (-1.36, 36.71, -1.22, 36.91)
#: Photon is searched over the whole county, wider than the routing area, so a
#: place just outside it (Two Rivers Mall, JKIA) is reported as out of range
#: rather than silently returning nothing.
COUNTY = (-1.45, 36.65, -1.15, 37.10)
PHOTON_URL = "https://photon.komoot.io/api/?q={q}&bbox={w},{s},{e},{n}&limit=10&lang=en"
ONLINE_MIN_LOCAL = 5
MAX_RESULTS = 15

#: Areas first, then landmarks people navigate by, then roads, then the rest.
KIND_RANK = {
    "area": 0, "estate": 0, "suburb": 0, "neighbourhood": 0,
    "mall": 1, "hospital": 1, "university": 1, "college": 1, "school": 1, "market": 1,
    "bus station": 1, "railway station": 1, "airport": 1, "stadium": 1, "park": 1,
    "government office": 1, "hotel": 1, "museum": 1, "police": 1, "place of worship": 1,
    "road": 2, "building": 4,
}

#: Mapped but not destinations anyone navigates to; they only crowd results.
SKIP_KINDS = {"toilets", "drinking water", "waste basket", "waste disposal", "bench",
              "vending machine", "recycling", "shelter", "water point", "atm"}

_COORD_RE = re.compile(r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def _in_area(lat: float, lon: float) -> bool:
    s, w, n, e = AREA
    return s <= lat <= n and w <= lon <= e


def encode_value(lat: float, lon: float, name: str) -> str:
    return f"loc:{lat:.5f},{lon:.5f}|{name}"


def decode_value(value: str) -> tuple[float, float, str] | None:
    """'loc:<lat>,<lon>|<name>' -> (lat, lon, name)."""
    if not value or not value.startswith("loc:"):
        return None
    try:
        coords, _, name = value[4:].partition("|")
        lat, lon = (float(x) for x in coords.split(","))
        return lat, lon, name
    except ValueError:
        return None


class LocationSearch:
    def __init__(self, index_file: Path = INDEX_FILE, places_file: Path = PLACES_FILE) -> None:
        entries: list[list] = []
        if index_file.exists():
            entries = json.loads(gzip.open(index_file, "rt", encoding="utf-8").read())["entries"]
        else:
            logger.warning(f"{index_file} missing - search covers suburb names only. "
                           f"Run `python -m src.ingestion.fetch_search_index`.")
        try:
            for p in json.loads(places_file.read_text(encoding="utf-8"))["places"]:
                entries.append([p["name"], p["lat"], p["lon"], "area"])
        except Exception:                                  # noqa: BLE001
            pass

        entries = [e for e in entries if e[3] not in SKIP_KINDS]
        self.names = [e[0] for e in entries]
        self.norm = [_norm(e[0]) for e in entries]
        self.padded = [" " + n for n in self.norm]
        self.lat = np.array([e[1] for e in entries])
        self.lon = np.array([e[2] for e in entries])
        self.kind = [e[3] for e in entries]
        self._online_cache: dict[str, list[dict]] = {}
        self._online_lock = threading.Lock()
        self._last_online = 0.0
        logger.info(f"LocationSearch: {len(entries):,} searchable locations")

    # ------------------------------------------------------------ local --
    def _local(self, q: str) -> list[tuple]:
        tokens = q.split()
        hits = []
        for i, n in enumerate(self.norm):
            if q not in n:
                if len(tokens) < 2 or not all(t in n for t in tokens):
                    continue
                score = 4
            elif n == q:
                score = 0
            elif n.startswith(q):
                score = 1
            elif (" " + q) in self.padded[i]:
                score = 2
            else:
                score = 3
            hits.append((score, KIND_RANK.get(self.kind[i], 3), len(n), i))
        hits.sort()
        return hits

    def _option(self, lat: float, lon: float, name: str, kind: str, online: bool = False) -> dict:
        near = place_name_at(lat, lon)
        parts = [name, kind]
        if near and near.lower() != name.lower() and near != "outside the model area":
            parts.append(near)
        label = " · ".join(parts) + ("  (online)" if online else "")
        return {"label": label, "value": encode_value(lat, lon, name)}

    # ----------------------------------------------------------- online --
    def _photon(self, q: str) -> list[dict]:
        if q in self._online_cache:
            return self._online_cache[q]
        with self._online_lock:
            wait = 1.0 - (time.time() - self._last_online)
            if wait > 0:
                time.sleep(wait)
            s, w, n, e = COUNTY
            url = PHOTON_URL.format(q=urllib.parse.quote(q), s=s, w=w, n=n, e=e)
            out: list[dict] = []
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "nairobi-flood-digital-twin/1.0 (academic project)"})
                data = json.loads(urllib.request.urlopen(req, timeout=4).read().decode("utf-8"))
                for f in data.get("features", []):
                    lon, lat = f["geometry"]["coordinates"]
                    pr = f.get("properties", {})
                    name = pr.get("name") or " ".join(x for x in (pr.get("housenumber"), pr.get("street")) if x)
                    if not name:
                        continue
                    kind = (pr.get("osm_value") or pr.get("type") or "place").replace("_", " ")
                    if _in_area(lat, lon):
                        out.append(self._option(lat, lon, name, kind, online=True))
                    else:
                        out.append({"label": f"{name} · {kind} · outside the mapped area, cannot route",
                                    "value": f"out:{lat:.5f},{lon:.5f}", "disabled": True})
            except Exception as exc:                        # noqa: BLE001
                logger.warning(f"Photon search failed for {q!r}: {exc}")
            finally:
                self._last_online = time.time()
            self._online_cache[q] = out
            return out

    # ----------------------------------------------------------- public --
    def search(self, query: str, online: bool = True) -> list[dict]:
        """Up to MAX_RESULTS dropdown options for a query."""
        raw = (query or "").strip()
        m = _COORD_RE.match(raw)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if _in_area(lat, lon):
                return [self._option(lat, lon, f"{lat:.5f}, {lon:.5f}", "coordinates")]
            return [{"label": f"{lat:.5f}, {lon:.5f} · outside the mapped area, cannot route",
                     "value": f"out:{lat},{lon}", "disabled": True}]

        q = _norm(raw)
        if len(q) < 2:
            return []
        options, seen = [], set()
        for _score, _rank, _len, i in self._local(q):
            # The same name can be many petrol stations or bus stops; show each
            # distinct neighbourhood once so the list stays readable.
            key = (self.norm[i], self.kind[i], place_name_at(self.lat[i], self.lon[i]))
            if key in seen:
                continue
            seen.add(key)
            lat, lon = float(self.lat[i]), float(self.lon[i])
            if _in_area(lat, lon):
                options.append(self._option(lat, lon, self.names[i], self.kind[i]))
            else:
                # Features that straddle the edge (JKIA's outline) are indexed,
                # but their centre is beyond the road network.
                options.append({"label": f"{self.names[i]} · {self.kind[i]} · outside the mapped area, cannot route",
                                "value": f"out:{lat:.5f},{lon:.5f}", "disabled": True})
            if len(options) >= MAX_RESULTS:
                break

        if online and len(options) < ONLINE_MIN_LOCAL and len(q) >= 3:
            have = {o["label"].split(" · ")[0].lower() for o in options}
            online_opts = [o for o in self._photon(q) if o["label"].split(" · ")[0].lower() not in have]
            # Routable results first; out-of-area ones only explain a miss.
            options += sorted(online_opts, key=lambda o: bool(o.get("disabled")))
        return options[:MAX_RESULTS]
