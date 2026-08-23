"""
src.ingestion.fetch_sentinel_targets
======================================
GeoStream -- Sentinel-1 SAR Temporal Water Mask Downloader

PURPOSE
-------
Query the Google Earth Engine (GEE) 'COPERNICUS/S1_GRD' collection,
apply server-side speckle filtering and radar backscatter thresholding
to isolate open surface water, and stream the resulting annual binary
water-mask GeoTIFFs to:

    data/raw/vectors/nairobi_s1_temporal_targets/

These files are the temporal ground-truth Y labels for the LSTM
flood-prediction training loop.

GEE SERVER-SIDE PIPELINE (all computation stays on Google's servers)
---------------------------------------------------------------------
For each target storm window (year, season):

  1. Filter COPERNICUS/S1_GRD  by:
       - instrumentMode == 'IW'       (Interferometric Wide swath)
       - transmitterReceiverPolarisation contains 'VV'
       - orbitProperties_pass == 'DESCENDING'  (eliminates viewing-angle drift)
       - date window  (storm-season weeks per year)
       - spatial bounds (Nairobi bbox)

  2. Speckle filter: focal_median(radius=30 m, circle kernel) per image
  3. Water mask   : VV < -16 dB  →  1 = open water, 0 = dry
  4. Composite    : median reducer across all scenes in window
  5. Clip         : .clip(nairobi_geometry) — discards extra national data
                     BEFORE any bytes reach this machine

CLIENT-SIDE (only the clipped tile is downloaded)
--------------------------------------------------
  6. getDownloadURL(scale=30 m, format='GEO_TIFF') → HTTPS stream
  7. requests streaming → io.BytesIO buffer (< 5 MB per tile)
  8. rasterio MemoryFile → write deflate GeoTIFF to disk

MEMORY CONTRACT  (MEMORY_CONSTRAINTS.md)
-----------------------------------------
- Spatial domain is chunked: 30 m pixels over the 45 x 30 km Nairobi
  bbox yields ~1500 x 1000 px = 1.5 M pixels per scene (< 6 MB float32).
- Global variables never hold raw imagery; each tile is written to disk
  and the buffer is released before the next iteration.
- RSS is monitored after every download; aborts if > 14 GB.
- Startup aborts if free RAM < 1 GB.

PREREQUISITES
-------------
  pip install earthengine-api requests rasterio psutil
  earthengine authenticate          # one-time browser OAuth flow
  -- OR set GEE_SERVICE_ACCOUNT and GEE_KEY_FILE env vars for headless auth

USAGE
-----
  python -m src.ingestion.fetch_sentinel_targets
  python -m src.ingestion.fetch_sentinel_targets --project my-gee-project
  python -m src.ingestion.fetch_sentinel_targets --years 2022 2023 --scale 30
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Windows consoles often default to cp1252, which can't encode the arrows
# and dashes used in this script's progress output — reconfigure stdout to
# UTF-8 (available since Python 3.7) so a print() never crashes a download
# mid-run over a purely cosmetic character.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psutil

# ---------------------------------------------------------------------------
# Load .env automatically so GEE_PROJECT_ID etc. are available without
# the caller having to export them manually in the shell.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # does not override already-set shell vars
except ImportError:
    pass  # python-dotenv is optional; env vars can be set in the shell instead

# ---------------------------------------------------------------------------
# Dependency guards  — surface clear install instructions if missing
# ---------------------------------------------------------------------------
try:
    import ee
    _EE_OK = True
except ImportError:
    _EE_OK = False

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False


# ============================================================================
# CONFIGURATION
# ============================================================================

#: Nairobi County spatial domain (EPSG:4326)
MIN_LON: float = 36.65
MIN_LAT: float = -1.45
MAX_LON: float = 37.10
MAX_LAT: float = -1.15

#: GEE collection and filter parameters
S1_COLLECTION: str = "COPERNICUS/S1_GRD"
INSTRUMENT_MODE: str = "IW"
POLARISATION: str = "VV"
ORBIT_PASS: str = "DESCENDING"

#: Water mask threshold: VV backscatter < -16 dB == open water
WATER_THRESHOLD_DB: float = -16.0

#: Speckle filter: focal median with 30 m circular kernel
SPECKLE_RADIUS_M: float = 30.0

#: Download resolution in metres (30 m balances detail vs memory per §2)
DEFAULT_SCALE_M: int = 30

#: Target years — Sentinel-1 IW-mode coverage over Nairobi is usable back to
#: 2015 (verified scene counts: 2015 = 16+14 scenes/season, 2017+ = 20-21/season;
#: 2014 long_rains has 0 scenes, so 2015 is the practical floor). Matches the
#: CHIRPS backfill in src.ingestion.fetch_chirps_archive.
DEFAULT_YEARS: list[int] = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

#: Storm-season windows per year.
#: East Africa receives two rainy seasons:
#:   Long  rains: March – May  (MAM)
#:   Short rains: October – December  (OND)
STORM_WINDOWS: list[dict] = [
    {"label": "long_rains",  "month_start": 3,  "month_end": 5},
    {"label": "short_rains", "month_start": 10, "month_end": 12},
]

#: Local output directory
DEFAULT_OUT_DIR: Path = Path("data/raw/vectors/nairobi_s1_temporal_targets")

#: RAM floor for this ingestion script (< 50 MB peak RSS expected per tile)
MIN_FREE_RAM_GB: float = 0.5

#: HTTP chunk size for streaming downloads
CHUNK_SIZE: int = 256 * 1024  # 256 KB


# ============================================================================
# MEMORY HELPERS
# ============================================================================

def _check_startup_memory() -> None:
    """Abort if available RAM < MIN_FREE_RAM_GB (MEMORY_CONSTRAINTS.md §2)."""
    mem = psutil.virtual_memory()
    free_gb  = mem.available / 1024 ** 3
    total_gb = mem.total     / 1024 ** 3
    print(
        f"[MEMORY] total={total_gb:.1f} GB | "
        f"free={free_gb:.2f} GB | "
        f"floor={MIN_FREE_RAM_GB} GB"
    )
    if free_gb < MIN_FREE_RAM_GB:
        print(
            f"[FATAL ] Free RAM {free_gb:.2f} GB < {MIN_FREE_RAM_GB} GB floor. "
            "Aborting per MEMORY_CONSTRAINTS.md.",
            file=sys.stderr,
        )
        sys.exit(1)


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 3


def _check_rss(stage: str) -> None:
    rss = _rss_gb()
    if rss > 14.0:
        raise MemoryError(
            f"[{stage}] RSS {rss:.2f} GB exceeds 14 GB critical ceiling. "
            "Aborting per MEMORY_CONSTRAINTS.md."
        )
    if rss > 12.0:
        print(f"[WARN ] [{stage}] RSS {rss:.2f} GB > 12 GB WARNING threshold.")
    else:
        print(f"[MEM  ] [{stage}] RSS={rss:.2f} GB  OK")


# ============================================================================
# GEE INITIALISATION
# ============================================================================

def _init_gee(project: str | None = None) -> None:
    """
    Initialise Google Earth Engine.

    Priority order for authentication:
      1. GEE_SERVICE_ACCOUNT + GEE_KEY_FILE env vars  (headless / CI)
      2. ee.Initialize(project=...)  using cached user credentials
         (requires prior  `earthengine authenticate`  in terminal)

    Parameters
    ----------
    project : str or None
        GEE cloud project ID.  Can also be set via GEE_PROJECT env var.
        Required for the modern high-volume EE API.
    """
    if not _EE_OK:
        print(
            "[FATAL] earthengine-api not installed.\n"
            "        pip install earthengine-api",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve project from env vars — check both key names used across the repo
    project = (
        project
        or os.environ.get("GEE_PROJECT")
        or os.environ.get("GEE_PROJECT_ID")   # key defined in .env.example
    )

    sa  = os.environ.get("GEE_SERVICE_ACCOUNT")
    key = os.environ.get("GEE_KEY_FILE")

    if sa and key:
        print(f"[GEE  ] Authenticating via service account: {sa}")
        credentials = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials, project=project)
    else:
        print("[GEE  ] Authenticating via cached user credentials ...")
        init_kwargs: dict = {}
        if project:
            init_kwargs["project"] = project
        try:
            ee.Initialize(**init_kwargs)
        except ee.EEException:
            print(
                "[GEE  ] Credentials not found. Running ee.Authenticate() ...\n"
                "        Follow the browser prompt, then re-run this script.",
            )
            ee.Authenticate()
            ee.Initialize(**init_kwargs)

    print("[GEE  ] Initialised successfully.")


# ============================================================================
# SERVER-SIDE GEE PIPELINE
# ============================================================================

def _build_nairobi_geometry() -> "ee.Geometry":
    """Return the Nairobi County bounding box as a GEE Geometry."""
    return ee.Geometry.BBox(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)


def _apply_speckle_filter(image: "ee.Image") -> "ee.Image":
    """
    Apply focal median speckle filter to the VV band of a single S1 image.

    Uses a 30 m radius circular kernel which suppresses SAR speckle noise
    while preserving the spatial extent of inundation features.
    """
    vv_filtered = (
        image.select(POLARISATION)
        .focal_median(
            radius=SPECKLE_RADIUS_M,
            kernelType="circle",
            units="meters",
        )
    )
    return image.addBands(vv_filtered.rename("VV_filtered"), overwrite=False)


def _apply_water_mask(image: "ee.Image") -> "ee.Image":
    """
    Derive binary water mask from filtered VV backscatter.

    Pixels with VV < WATER_THRESHOLD_DB (-16 dB) are classified as
    open surface water (value 1). All other pixels are 0.

    The -16 dB threshold is a widely-validated empirical cutoff for
    C-band SAR open water detection at 10–30 m resolution
    (see Twele et al., 2016; Chini et al., 2017).
    """
    water = (
        image.select("VV_filtered")
        .lt(WATER_THRESHOLD_DB)
        .rename("water_mask")
        .toUint8()
    )
    return image.addBands(water)


def _query_and_composite(
    nairobi_geom: "ee.Geometry",
    year: int,
    window: dict,
) -> tuple["ee.Image | None", int, str]:
    """
    Build a server-side composite water mask for one storm window.

    Orbit-pass fallback strategy
    ----------------------------
    1. Try DESCENDING  (preferred — consistent viewing geometry)
    2. If 0 scenes found, try ASCENDING
    3. If still 0, drop the orbit filter entirely (ANY pass)
    4. If still 0, return (None, 0, 'NONE') — window is genuinely empty

    The same speckle filter (focal_median 30 m) and water-mask threshold
    (VV < -16 dB) are applied identically regardless of which pass succeeds.
    The composite is server-side clipped to the Nairobi bbox before any
    bytes leave Google's infrastructure.

    Parameters
    ----------
    nairobi_geom : ee.Geometry
    year         : int
    window       : dict with keys 'label', 'month_start', 'month_end'

    Returns
    -------
    composite      : ee.Image or None
    scene_count    : int
    orbit_pass_used: str  ('DESCENDING', 'ASCENDING', 'ANY', or 'NONE')
    """
    date_start = f"{year}-{window['month_start']:02d}-01"
    end_month  = window["month_end"]
    end_year   = year + 1 if end_month == 12 else year
    end_month_next = 1 if end_month == 12 else end_month + 1
    date_end   = f"{end_year}-{end_month_next:02d}-01"

    # Base collection: spatial + temporal + instrument + polarisation filters
    # (no orbit filter yet — added per-attempt below)
    base = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(nairobi_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", INSTRUMENT_MODE))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", POLARISATION))
    )

    # Fallback ladder: DESCENDING -> ASCENDING -> ANY (no orbit filter)
    attempts = [
        ("DESCENDING", base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))),
        ("ASCENDING",  base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))),
        ("ANY",        base),   # no orbit filter — all available passes
    ]

    collection     = None
    scene_count    = 0
    orbit_pass_used = "NONE"

    for pass_label, candidate in attempts:
        count = candidate.size().getInfo()   # lightweight client-side integer
        if count > 0:
            collection      = candidate
            scene_count     = count
            orbit_pass_used = pass_label
            break
        print(f"  [ORBIT] {pass_label}: 0 scenes — trying next pass ...")

    if collection is None or scene_count == 0:
        return None, 0, "NONE"

    # -- Server-side processing chain (identical for all orbit passes) ---------
    processed = (
        collection
        .map(_apply_speckle_filter)
        .map(_apply_water_mask)
    )

    composite = (
        processed
        .select("water_mask")
        .median()                           # pixel-wise median across scenes
        .gt(0)                              # binarise (median may be 0.0-1.0)
        .toUint8()
        .rename("water_mask")
        .clip(nairobi_geom)                 # server-side clip BEFORE download
    )

    return composite, scene_count, orbit_pass_used


# ============================================================================
# CLIENT-SIDE DOWNLOAD
# ============================================================================

def _download_image_to_tif(
    image: "ee.Image",
    nairobi_geom: "ee.Geometry",
    out_path: Path,
    scale_m: int,
) -> float:
    """
    Stream a GEE Image to a local deflate-compressed GeoTIFF.

    Uses getDownloadURL() to obtain a short-lived HTTPS link, then
    streams the bytes through requests → io.BytesIO → rasterio MemoryFile
    → disk write.  The in-memory buffer is explicitly deleted after write
    to free heap RAM immediately.

    Parameters
    ----------
    image       : GEE server-side Image (no client RAM until download)
    nairobi_geom: GEE server-side Geometry for region parameter
    out_path    : local destination .tif path
    scale_m     : pixel size in metres

    Returns
    -------
    float
        Downloaded tile size in MB.
    """
    if not _REQUESTS_OK:
        raise RuntimeError("'requests' is not installed. pip install requests")
    if not _RASTERIO_OK:
        raise RuntimeError("'rasterio' is not installed. pip install rasterio>=1.3")

    # -- Obtain short-lived signed download URL from GEE ----------------------
    url = image.getDownloadURL({
        "scale":   scale_m,
        "region":  nairobi_geom,
        "format":  "GEO_TIFF",
        "bands":   ["water_mask"],
    })

    # -- Stream into memory buffer ---------------------------------------------
    buf = io.BytesIO()
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            buf.write(chunk)
    downloaded_mb = buf.tell() / 1024 ** 2
    buf.seek(0)

    # -- Write deflate-compressed GeoTIFF via rasterio MemoryFile -------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with MemoryFile(buf) as mf:
        with mf.open() as src:
            profile = src.profile.copy()
            profile.update(
                compress="deflate",
                predictor=2,
                zlevel=6,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                interleave="band",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())

    # Release buffer immediately (MEMORY_CONSTRAINTS.md §2)
    del buf

    return downloaded_mb


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def fetch_sentinel_targets(
    years: list[int] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    scale_m: int = DEFAULT_SCALE_M,
    project: str | None = None,
) -> None:
    """
    Main orchestration loop.

    For each (year, storm_window) combination:
      - Queries and composites the S1 GRD collection server-side
      - Streams only the clipped Nairobi water mask to disk
      - Logs scene counts and memory metrics

    Parameters
    ----------
    years   : list of target years; defaults to DEFAULT_YEARS (2020-2026)
    out_dir : local directory for output .tif files
    scale_m : download pixel resolution in metres
    project : GEE Cloud project ID (or set GEE_PROJECT env var)
    """
    if years is None:
        years = DEFAULT_YEARS

    # -- Startup checks --------------------------------------------------------
    _check_startup_memory()
    _init_gee(project=project)

    out_dir.mkdir(parents=True, exist_ok=True)

    nairobi_geom = _build_nairobi_geometry()

    print()
    print("=" * 68)
    print("  GeoStream | Sentinel-1 SAR Temporal Water Mask Downloader")
    print("=" * 68)
    print(f"  Collection  : {S1_COLLECTION}")
    print(f"  Filters     : mode={INSTRUMENT_MODE} | pol={POLARISATION} | pass={ORBIT_PASS}")
    print(f"  Threshold   : VV < {WATER_THRESHOLD_DB} dB  =>  water=1")
    print(f"  Speckle     : focal_median radius={SPECKLE_RADIUS_M} m (circle)")
    print(f"  Resolution  : {scale_m} m")
    print(f"  Bbox        : lon [{MIN_LON}, {MAX_LON}]  lat [{MIN_LAT}, {MAX_LAT}]")
    print(f"  Years       : {years}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 68)

    total_tasks = len(years) * len(STORM_WINDOWS)
    completed   = 0
    skipped     = 0
    failed      = 0

    for year in years:
        print(f"\n{'-'*68}")
        print(f"  YEAR {year}")
        print(f"{'-'*68}")

        for window in STORM_WINDOWS:
            label       = window["label"]
            task_label  = f"{year}_{label}"
            out_filename = f"s1_water_mask_{task_label}.tif"
            out_path     = out_dir / out_filename

            print(f"\n  [{task_label}]  "
                  f"Months {window['month_start']:02d}-{window['month_end']:02d}")

            # Skip if already downloaded (idempotent re-runs)
            if out_path.exists():
                print(f"  [SKIP ] {out_filename} already exists. "
                      "Delete to re-download.")
                skipped += 1
                continue

            try:
                # -- Server-side query & composite ----------------------------
                print(f"  [QUERY] Querying GEE for S1 scenes ...")
                t0 = time.perf_counter()

                composite, scene_count, orbit_pass_used = _query_and_composite(
                    nairobi_geom, year, window
                )

                elapsed = time.perf_counter() - t0
                print(f"  [INFO ] Scenes found : {scene_count}  "
                      f"orbit={orbit_pass_used}  "
                      f"(query took {elapsed:.1f} s)")

                if composite is None or scene_count == 0:
                    print(f"  [SKIP ] No S1 scenes for {task_label} "
                          "on any orbit pass. Skipping download.")
                    skipped += 1
                    continue

                # -- Client-side stream download -------------------------------
                print(f"  [DL   ] Streaming clipped tile to {out_filename} ...")
                t1 = time.perf_counter()

                dl_mb = _download_image_to_tif(
                    composite, nairobi_geom, out_path, scale_m
                )

                dl_time = time.perf_counter() - t1
                disk_mb = out_path.stat().st_size / 1024 ** 2

                print(f"  [OK   ] Downloaded {dl_mb:.2f} MB → "
                      f"deflate {disk_mb:.3f} MB  "
                      f"({dl_time:.1f} s)")

                _check_rss(f"after_{task_label}")
                completed += 1

            except MemoryError as me:
                print(f"  [CRITICAL] {me}", file=sys.stderr)
                raise
            except Exception as exc:
                print(
                    f"  [ERROR] {task_label} failed: {exc}",
                    file=sys.stderr,
                )
                failed += 1
                # Continue to next window rather than aborting entire run
                continue

    # -- Final summary ---------------------------------------------------------
    print()
    print("=" * 68)
    print("  FETCH COMPLETE")
    print("=" * 68)
    print(f"  Tasks total    : {total_tasks}")
    print(f"  Downloaded     : {completed}")
    print(f"  Skipped        : {skipped}")
    print(f"  Failed         : {failed}")
    print(f"  Output dir     : {out_dir.resolve()}")

    if failed > 0:
        print(
            f"\n  [WARN ] {failed} task(s) failed. "
            "Check stderr above for details.",
            file=sys.stderr,
        )

    print()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingestion.fetch_sentinel_targets",
        description=(
            "Query Google Earth Engine for historical Sentinel-1 SAR "
            "water masks over Nairobi County (2020-2026) and stream "
            "compressed GeoTIFFs to data/raw/vectors/nairobi_s1_temporal_targets/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        metavar="YEAR",
        help="Target years to download (space-separated).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="PATH",
        help="Local output directory for water-mask GeoTIFFs.",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE_M,
        metavar="METRES",
        help=(
            "Download pixel resolution in metres. "
            "30 m = ~1.5 M pixels for Nairobi bbox (< 6 MB/tile). "
            "Use 10 m only if disk space allows (~13 M pixels/tile)."
        ),
    )
    p.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="GEE_PROJECT_ID",
        help=(
            "Google Earth Engine Cloud project ID. "
            "Also readable from the GEE_PROJECT environment variable."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    fetch_sentinel_targets(
        years=args.years,
        out_dir=args.out_dir,
        scale_m=args.scale,
        project=args.project,
    )
