"""
src.routing.flood_router
========================
Route between two points in Nairobi while avoiding predicted flooding.

HOW IT WORKS
------------
The OSM road network (src/ingestion/fetch_road_network.py) becomes a directed
graph whose edge weight is travel time. For a flood scenario, every edge is
scored with the flood probability under it, and its weight is multiplied by a
penalty for the band that probability falls in:

    below 25%      x1      no penalty
    MODERATE       x3      used if it saves a real detour
    HIGH           x25     used only when there is no reasonable alternative
    CRITICAL       closed  never used, unless nothing else connects the points

Two routes come back: the fastest route ignoring floods (what an ordinary
navigation app would suggest) and the flood-aware route. Showing both lets the
user see what is being avoided and what it costs in time.

WHAT IT DOES NOT CLAIM
----------------------
* Flood extent is validated at neighbourhood scale (~70 m cells), not street
  scale. A bridge or raised road inside a flooded cell is still avoided, and a
  low underpass in a dry cell is not. The route is lower-risk, not certified safe.
* Travel times use typical speeds per road class. There is no live traffic.
* Probability is sampled from the same smoothed field the map draws, dilated by
  one cell, so a road visibly running through a shaded band is treated as in it.

Shortest paths use scipy's compiled Dijkstra rather than networkx: the graph has
~167k nodes and a pure-Python search took over a second per route.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter, grey_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from src.grid_config import GRID_H, GRID_W, LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST

ROADS_FILE = Path("data/processed/nairobi_roads.json.gz")
PLACES_FILE = Path("data/processed/nairobi_places.json")

#: Typical urban driving speeds in Nairobi, km/h. Deliberately conservative:
#: posted limits are rarely achievable in town. Not live traffic.
SPEED_KMH = {
    "motorway": 60, "motorway_link": 40, "trunk": 45, "trunk_link": 30,
    "primary": 35, "primary_link": 25, "secondary": 30, "secondary_link": 22,
    "tertiary": 25, "tertiary_link": 20, "unclassified": 20, "residential": 18,
    "living_street": 10, "service": 12,
}

#: Probability bands, identical to the map's (map_3d.generate_flood_contour_geojson).
BANDS = ((0.82, 3, "CRITICAL"), (0.55, 2, "HIGH"), (0.25, 1, "MODERATE"))
LEVEL_NAMES = {0: "CLEAR", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}
PENALTY = {0: 1.0, 1: 3.0, 2: 25.0}
#: Used only in the fallback search when critical flooding cannot be avoided.
CRITICAL_FALLBACK_PENALTY = 200.0

#: Same smoothing the map applies before drawing bands.
MAP_SMOOTHING_SIGMA = 2.4
#: Sample spacing along an edge, metres. About half a grid cell.
SAMPLE_SPACING_M = 35.0

_M_PER_DEG_LAT = 110_570.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(1.29))


_PLACE_RASTER: tuple[np.ndarray, list[str]] | None = None


def place_raster(places_file: Path = PLACES_FILE) -> tuple[np.ndarray, list[str]]:
    """
    Nearest named place for every grid cell - a rasterised Voronoi partition,
    the same assignment the map uses to name flood polygons.

    Returns (flat index array of length GRID_H*GRID_W + 1, names). The final
    entry is a sentinel for points outside the grid and holds -1.
    """
    global _PLACE_RASTER
    if _PLACE_RASTER is None:
        places = json.loads(places_file.read_text(encoding="utf-8"))["places"]
        names = [p["name"] for p in places]
        tree = cKDTree(np.array([[p["lat"] * _M_PER_DEG_LAT, p["lon"] * _M_PER_DEG_LON] for p in places]))
        lats = np.linspace(LAT_NORTH, LAT_SOUTH, GRID_H)
        lons = np.linspace(LON_WEST, LON_EAST, GRID_W)
        gy, gx = np.meshgrid(lats, lons, indexing="ij")
        _d, idx = tree.query(np.column_stack([gy.ravel() * _M_PER_DEG_LAT, gx.ravel() * _M_PER_DEG_LON]))
        _PLACE_RASTER = (np.append(idx.astype(np.int64), -1), names)
    return _PLACE_RASTER


def place_name_at(lat: float, lon: float) -> str:
    """Name of the place a point falls in, or a plain description outside the grid."""
    idx, names = place_raster()
    row = int(round((LAT_NORTH - lat) / (LAT_NORTH - LAT_SOUTH) * (GRID_H - 1)))
    col = int(round((lon - LON_WEST) / (LON_EAST - LON_WEST) * (GRID_W - 1)))
    if 0 <= row < GRID_H and 0 <= col < GRID_W:
        return names[idx[row * GRID_W + col]]
    return "outside the model area"


def smooth_for_routing(probability: np.ndarray) -> np.ndarray:
    """The field the map draws, dilated by one cell to cover band edges."""
    smooth = gaussian_filter(probability.astype(np.float32), sigma=MAP_SMOOTHING_SIGMA)
    return grey_dilation(smooth, size=(3, 3))


def level_of(p: np.ndarray | float):
    """Band level 0..3 for a probability (vectorised)."""
    p = np.asarray(p)
    return np.select([p >= 0.82, p >= 0.55, p >= 0.25], [3, 2, 1], default=0)


@dataclass
class RouteResult:
    found: bool
    node_path: list[int]
    edge_ids: np.ndarray
    distance_km: float
    duration_min: float
    exposure_km: dict            # level name -> km driven in that band
    worst_level: int
    flooded_stretches: list[dict]   # [{road, place, level, km}]
    coordinates: list[list[float]]  # [[lon, lat], ...]
    used_critical: bool = False

    def summary(self) -> dict:
        return {
            "found": self.found,
            "distance_km": round(self.distance_km, 2),
            "duration_min": round(self.duration_min, 1),
            "exposure_km": {k: round(v, 2) for k, v in self.exposure_km.items()},
            "worst_level": LEVEL_NAMES[self.worst_level],
            "flooded_stretches": self.flooded_stretches,
            "used_critical": self.used_critical,
        }


class FloodAwareRouter:
    """Road graph plus the precomputed edge-to-grid sampling needed to score it."""

    def __init__(self, roads_file: Path = ROADS_FILE, places_file: Path = PLACES_FILE) -> None:
        t0 = time.perf_counter()
        data = json.loads(gzip.open(roads_file, "rt", encoding="utf-8").read())

        node_ids = list(data["nodes"].keys())
        index = {nid: i for i, nid in enumerate(node_ids)}
        coords = np.array([data["nodes"][nid] for nid in node_ids], dtype=np.float64)
        self.node_lat, self.node_lon = coords[:, 0], coords[:, 1]

        # Vectorised: every way's node sequence is concatenated into one array,
        # and a consecutive pair is an edge when both ends belong to the same way.
        # A per-segment Python loop took ~45 s for the 300k segments.
        self.road_names = []
        name_index: dict[str, int] = {}
        way_name, way_speed, way_oneway, seqs = [], [], [], []
        for way in data["ways"]:
            nm = way["name"] or ""
            if nm not in name_index:
                name_index[nm] = len(self.road_names)
                self.road_names.append(nm)
            way_name.append(name_index[nm])
            way_speed.append(SPEED_KMH.get(way["highway"], 15))
            way_oneway.append(way["oneway"])
            seqs.append([index[x] for x in way["nodes"]])

        lens = np.array([len(sq) for sq in seqs])
        flat = np.fromiter((i for sq in seqs for i in sq), dtype=np.int64, count=int(lens.sum()))
        way_of = np.repeat(np.arange(len(seqs)), lens)
        pair = (way_of[:-1] == way_of[1:]) & (flat[:-1] != flat[1:])
        a, b, w = flat[:-1][pair], flat[1:][pair], way_of[:-1][pair]
        d = np.hypot((coords[a, 0] - coords[b, 0]) * _M_PER_DEG_LAT,
                     (coords[a, 1] - coords[b, 1]) * _M_PER_DEG_LON)
        ow = np.array(way_oneway)[w]
        fwd, rev = ow >= 0, ow <= 0
        src = np.concatenate([a[fwd], b[rev]])
        dst = np.concatenate([b[fwd], a[rev]])
        length = np.concatenate([d[fwd], d[rev]])
        wid = np.concatenate([w[fwd], w[rev]])
        speed = np.array(way_speed, np.float64)[wid]
        name_idx = np.array(way_name, np.int32)[wid]

        # A CSR matrix sums duplicate (src, dst) entries, which would double the
        # cost of a street tagged as two overlapping ways. Keep the shortest.
        order = np.lexsort((length, dst, src))
        src, dst, length, speed, name_idx = (a[order] for a in (src, dst, length, speed, name_idx))
        keep = np.ones(len(src), bool)
        keep[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
        self.src, self.dst = src[keep], dst[keep]
        self.length_m = length[keep]
        self.time_s = self.length_m / (speed[keep] / 3.6)
        self.edge_name = name_idx[keep]
        self.n_nodes = len(node_ids)
        self.n_edges = len(self.src)

        # Only nodes in the largest strongly connected component are offered as
        # start or end points; anything else can be unreachable by construction
        # (a one-way fragment, a road clipped by the download box).
        base = csr_matrix((np.ones(self.n_edges), (self.src, self.dst)),
                          shape=(self.n_nodes, self.n_nodes))
        _n, labels = connected_components(base, directed=True, connection="strong")
        main = np.bincount(labels).argmax()
        self.routable = np.flatnonzero(labels == main)
        self._kdtree = cKDTree(np.column_stack([self.node_lat[self.routable] * _M_PER_DEG_LAT,
                                                self.node_lon[self.routable] * _M_PER_DEG_LON]))

        # Sample points along each edge -> flat grid-cell indices. Built once;
        # scoring a scenario is then a gather and a reduceat, not a geometry pass.
        n_samples = np.maximum(2, np.ceil(self.length_m / SAMPLE_SPACING_M).astype(np.int64) + 1)
        self.edge_ptr = np.concatenate([[0], np.cumsum(n_samples)])
        rep = np.repeat(np.arange(self.n_edges), n_samples)
        offset = np.arange(self.edge_ptr[-1]) - self.edge_ptr[:-1][rep]
        frac = offset / (n_samples[rep] - 1)
        s_lat = self.node_lat[self.src[rep]] + frac * (self.node_lat[self.dst[rep]] - self.node_lat[self.src[rep]])
        s_lon = self.node_lon[self.src[rep]] + frac * (self.node_lon[self.dst[rep]] - self.node_lon[self.src[rep]])
        row = np.rint((LAT_NORTH - s_lat) / (LAT_NORTH - LAT_SOUTH) * (GRID_H - 1)).astype(np.int64)
        col = np.rint((s_lon - LON_WEST) / (LON_EAST - LON_WEST) * (GRID_W - 1)).astype(np.int64)
        inside = (row >= 0) & (row < GRID_H) & (col >= 0) & (col < GRID_W)
        # Samples outside the prediction grid point at a sentinel cell holding 0:
        # no prediction there, so no penalty - disclosed in the route summary.
        self.sample_cell = np.where(inside, row * GRID_W + col, GRID_H * GRID_W)
        self.edge_in_grid = np.maximum.reduceat(inside.astype(np.int8), self.edge_ptr[:-1]).astype(bool)

        self.place_of_cell, self.place_names = place_raster(places_file)

        logger.info(
            f"FloodAwareRouter: {self.n_nodes:,} nodes, {self.n_edges:,} directed edges, "
            f"{len(self.routable):,} routable, built in {time.perf_counter() - t0:.1f}s"
        )

    # ----------------------------------------------------------- places --
    def place_at(self, lat: float, lon: float) -> str:
        return place_name_at(lat, lon)

    # ---------------------------------------------------------- scoring --
    def edge_probability(self, routing_field: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        """Worst flood probability along each edge (all edges, or a subset)."""
        flat = np.append(routing_field.ravel(), 0.0)
        if edges is None:
            return np.maximum.reduceat(flat[self.sample_cell], self.edge_ptr[:-1])
        out = np.empty(len(edges), dtype=np.float32)
        for k, e in enumerate(edges):
            out[k] = flat[self.sample_cell[self.edge_ptr[e]:self.edge_ptr[e + 1]]].max()
        return out

    def nearest_node(self, lat: float, lon: float) -> tuple[int, float]:
        d, i = self._kdtree.query([lat * _M_PER_DEG_LAT, lon * _M_PER_DEG_LON])
        return int(self.routable[i]), float(d)

    # ---------------------------------------------------------- routing --
    def _shortest(self, weights: np.ndarray, mask: np.ndarray, s: int, t: int) -> list[int] | None:
        g = csr_matrix((weights[mask], (self.src[mask], self.dst[mask])),
                       shape=(self.n_nodes, self.n_nodes))
        dist, pred = dijkstra(g, directed=True, indices=s, return_predecessors=True)
        if not np.isfinite(dist[t]):
            return None
        path = [t]
        while path[-1] != s:
            path.append(int(pred[path[-1]]))
        return path[::-1]

    def _edges_of(self, path: list[int]) -> np.ndarray:
        """Edge ids for consecutive node pairs (edges are sorted by src, dst)."""
        a = np.array(path[:-1], np.int64)
        b = np.array(path[1:], np.int64)
        key = self.src * self.n_nodes + self.dst
        return np.searchsorted(key, a * self.n_nodes + b)

    def _describe(self, path: list[int], edge_prob: np.ndarray, used_critical: bool) -> RouteResult:
        edges = self._edges_of(path)
        probs = edge_prob[edges]
        levels = level_of(probs)
        km = self.length_m[edges] / 1000.0
        exposure = {LEVEL_NAMES[l]: float(km[levels == l].sum()) for l in (1, 2, 3)}

        # Flooded edges grouped by (road, place, band) so the summary reads
        # "Racecourse Road, River Side, critical, 0.6 km" once rather than once per
        # OSM segment, and a stretch's length is only the part in that band.
        groups: dict[tuple[str, str, int], dict] = {}
        for k in np.flatnonzero(levels > 0):
            e = edges[k]
            lvl = int(levels[k])
            road = self.road_names[self.edge_name[e]] or "unnamed road"
            place = self.place_at(self.node_lat[self.src[e]], self.node_lon[self.src[e]])
            g = groups.setdefault((road, place, lvl), {"road": road, "place": place, "level": lvl, "km": 0.0})
            g["km"] += float(km[k])
        stretches = sorted(groups.values(), key=lambda s: (-s["level"], -s["km"]))
        for st in stretches:
            st["level"] = LEVEL_NAMES[st["level"]]
            st["km"] = round(st["km"], 2)

        return RouteResult(
            found=True,
            node_path=path,
            edge_ids=edges,
            distance_km=float(km.sum()),
            duration_min=float(self.time_s[edges].sum() / 60.0),
            exposure_km=exposure,
            worst_level=int(levels.max()) if len(levels) else 0,
            flooded_stretches=stretches[:6],
            coordinates=[[round(float(self.node_lon[n]), 6), round(float(self.node_lat[n]), 6)] for n in path],
            used_critical=used_critical,
        )

    def route(self, origin: tuple[float, float], destination: tuple[float, float],
              probability: np.ndarray | None) -> dict:
        """
        Fastest and flood-aware routes between two (lat, lon) points.

        `probability` is the model's raw per-cell probability grid, or None for a
        dry scenario. Returns a dict with `fastest`, `safe` (RouteResult or None)
        and diagnostics about snapping and fallbacks.
        """
        t0 = time.perf_counter()
        s, ds = self.nearest_node(*origin)
        t, dt = self.nearest_node(*destination)
        out = {"snap_m": (round(ds), round(dt)), "fastest": None, "safe": None,
               "note": None, "same_point": s == t}
        if s == t:
            out["note"] = "Start and destination are the same point on the road network."
            return out

        field = (smooth_for_routing(probability) if probability is not None
                 else np.zeros((GRID_H, GRID_W), np.float32))
        edge_prob = self.edge_probability(field)
        levels = level_of(edge_prob)
        all_edges = np.ones(self.n_edges, bool)

        fast_path = self._shortest(self.time_s, all_edges, s, t)
        if fast_path is None:
            out["note"] = "No road connection between these points."
            return out
        out["fastest"] = self._describe(fast_path, edge_prob, used_critical=False)

        penalty = np.select([levels == 0, levels == 1, levels == 2],
                            [PENALTY[0], PENALTY[1], PENALTY[2]], default=np.inf)
        weights = self.time_s * penalty
        safe_path = self._shortest(weights, np.isfinite(weights), s, t)
        used_critical = False
        if safe_path is None:
            # Nothing avoids critical flooding - typically the start or end is
            # inside it. Route through as little of it as possible and say so.
            weights = self.time_s * np.where(levels == 3, CRITICAL_FALLBACK_PENALTY, penalty)
            safe_path = self._shortest(weights, all_edges, s, t)
            used_critical = True
            out["note"] = ("No route avoids critical flooding - the start or destination "
                           "is likely inside a flooded area. This route crosses as little "
                           "of it as possible.")
        out["safe"] = self._describe(safe_path, edge_prob, used_critical=used_critical)
        out["outside_grid"] = bool((~self.edge_in_grid[out["safe"].edge_ids]).any())
        out["latency_sec"] = round(time.perf_counter() - t0, 3)
        return out

    def exposure_over_time(self, edge_ids: np.ndarray, probabilities: list[np.ndarray | None]) -> list[int]:
        """Worst band level along a fixed route for each of a series of grids."""
        levels = []
        for prob in probabilities:
            if prob is None:
                levels.append(0)
                continue
            p = self.edge_probability(smooth_for_routing(prob), edge_ids)
            levels.append(int(level_of(p).max()) if len(p) else 0)
        return levels
