"""
src.ingestion.download_flood_target
=====================================
GeoStream -- ICPAC WCS Flood Inundation Downloader

PURPOSE
-------
Stream the Kenya flood inundation raster from the ICPAC GeoServer WCS
endpoint, clip it to the Nairobi County spatial domain **in transit**,
and persist only the clipped window to disk.

WCS 2.0.1 subsetting strategy (DescribeCoverage findings)
----------------------------------------------------------
Coverage CRS : EPSG:4326  with axis order  Lat Long  (not lon/lat)
Grid size    : 9586 cols x 11152 rows  (~585 MB uncompressed)
Pixel size   : ~0.000833 deg  (~90 m)

We pass subset=Lat(min,max)&subset=Long(min,max) in the GetCoverage
request so GeoServer clips server-side before transmitting.  Only the
small Nairobi tile (~0.45 x 0.30 deg) crosses the wire.

MEMORY CONTRACT (MEMORY_CONSTRAINTS.md)
----------------------------------------
- No full-resolution national array is ever loaded into active memory.
- Peak RSS during this script is expected to be < 100 MB.
- RSS is checked at startup and after the window read.

OUTPUT
------
  data/raw/vectors/nairobi_flood_100yr.tif
    Single-band GeoTIFF, deflate-compressed.

USAGE
-----
  python -m src.ingestion.download_flood_target
  python -m src.ingestion.download_flood_target --out data/raw/vectors/nairobi_flood_100yr.tif
"""

from __future__ import annotations

import argparse
import copy
import io
import sys
from pathlib import Path
from urllib.parse import urlencode

import psutil

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    import requests
    _REQUESTS_OK = True
except ImportError:  # pragma: no cover
    _REQUESTS_OK = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds as window_from_bounds
    _RASTERIO_OK = True
except ImportError:  # pragma: no cover
    _RASTERIO_OK = False


# ============================================================================
# CONFIGURATION
# ============================================================================

#: ICPAC GeoServer WCS base endpoint
WCS_BASE: str = "https://geoportal.icpac.net/geoserver/ows"

#: Coverage identifier (confirmed via GetCapabilities)
COVERAGE_ID: str = "geonode__Kenya_flood_inundation"

#: Nairobi County spatial domain (EPSG:4326 geographic degrees)
#: Axis order in this coverage is Lat/Long (confirmed via DescribeCoverage)
MIN_LON: float = 36.65
MIN_LAT: float = -1.45
MAX_LON: float = 37.10
MAX_LAT: float = -1.15

#: Default output path (relative to repo root)
DEFAULT_OUT: Path = Path("data/raw/vectors/nairobi_flood_100yr.tif")

#: Ingestion-only RAM floor. Peak RSS here is < 100 MB.
#: The 4 GB floor in MEMORY_CONSTRAINTS.md targets training loops.
MIN_FREE_RAM_GB: float = 1.0

#: Streaming chunk size for HTTP download (bytes)
CHUNK_SIZE: int = 1024 * 256  # 256 KB


# ============================================================================
# HELPERS
# ============================================================================

def _check_startup_memory() -> None:
    """Abort if available RAM < MIN_FREE_RAM_GB (MEMORY_CONSTRAINTS.md §1)."""
    mem = psutil.virtual_memory()
    free_gb  = mem.available / 1024 ** 3
    total_gb = mem.total     / 1024 ** 3
    print(
        f"[MEMORY] total={total_gb:.1f} GB | "
        f"free={free_gb:.2f} GB | "
        f"threshold={MIN_FREE_RAM_GB} GB"
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
            f"[{stage}] RSS {rss:.2f} GB > 14 GB critical ceiling. "
            "Aborting per MEMORY_CONSTRAINTS.md."
        )
    if rss > 12.0:
        print(f"[WARN ] [{stage}] RSS {rss:.2f} GB > 12 GB WARNING threshold.")
    else:
        print(f"[MEM  ] [{stage}] RSS={rss:.2f} GB  OK")


def _build_wcs_url() -> str:
    """
    Build a WCS 2.0.1 GetCoverage URL with server-side Nairobi subset.

    GeoServer EPSG:4326 axis order is Lat/Long (confirmed via DescribeCoverage).
    Using subset=Lat(min_lat,max_lat)&subset=Long(min_lon,max_lon) clips the
    raster server-side so only the Nairobi tile is transmitted.
    """
    params = {
        "service":    "WCS",
        "version":    "2.0.1",
        "request":    "GetCoverage",
        "coverageId": COVERAGE_ID,
        "format":     "image/tiff",
        "subset":     [
            f"Lat({MIN_LAT},{MAX_LAT})",
            f"Long({MIN_LON},{MAX_LON})",
        ],
    }
    # urlencode handles the duplicate 'subset' key correctly with doseq=True
    return f"{WCS_BASE}?{urlencode(params, doseq=True)}"


# ============================================================================
# CORE DOWNLOAD LOGIC
# ============================================================================

def download_flood_target(out_path: Path = DEFAULT_OUT) -> None:
    """
    Server-side clip and stream the ICPAC Kenya flood inundation raster
    to the Nairobi domain, then write a deflate-compressed GeoTIFF.
    """
    if not _REQUESTS_OK:
        print("[FATAL] 'requests' is not installed. pip install requests", file=sys.stderr)
        sys.exit(1)
    if not _RASTERIO_OK:
        print("[FATAL] 'rasterio' is not installed. pip install rasterio>=1.3", file=sys.stderr)
        sys.exit(1)

    # ── 0. Startup memory guard ──────────────────────────────────────────────
    _check_startup_memory()

    wcs_url = _build_wcs_url()

    print()
    print("=" * 66)
    print("  GeoStream | ICPAC WCS Flood Inundation Downloader")
    print("=" * 66)
    print(f"  Coverage  : {COVERAGE_ID}")
    print(f"  Bbox (geo): lon [{MIN_LON}, {MAX_LON}]  lat [{MIN_LAT}, {MAX_LAT}]")
    print(f"  Subset URL: {wcs_url}")
    print(f"  Output    : {out_path}")
    print("=" * 66)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Stream HTTP response into memory buffer ───────────────────────
        print("\n[1/4] Streaming server-side clipped tile from ICPAC WCS ...")
        print(f"      Sending GET request ...")

        response = requests.get(wcs_url, stream=True, timeout=120)

        content_type = response.headers.get("Content-Type", "")
        content_len  = response.headers.get("Content-Length", "unknown")
        print(f"      HTTP status   : {response.status_code}")
        print(f"      Content-Type  : {content_type}")
        print(f"      Content-Length: {content_len} bytes")

        if response.status_code != 200:
            body_preview = response.text[:800]
            raise RuntimeError(
                f"Server returned HTTP {response.status_code}.\n"
                f"Response body:\n{body_preview}"
            )

        if "tiff" not in content_type.lower() and "image" not in content_type.lower():
            raise RuntimeError(
                f"Unexpected Content-Type '{content_type}'. "
                "Expected image/tiff. Server may have returned an error XML."
                f"\nResponse body:\n{response.text[:800]}"
            )

        # Accumulate into an in-memory buffer (Nairobi tile << 10 MB)
        buf = io.BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            buf.write(chunk)
            downloaded += len(chunk)

        buf.seek(0)
        downloaded_mb = downloaded / 1024 ** 2
        print(f"      Downloaded    : {downloaded_mb:.2f} MB")
        _check_rss("after_download")

        # ── 2. Open buffer with rasterio to inspect metadata ─────────────────
        print("\n[2/4] Inspecting clipped raster geometry ...")

        with MemoryFile(buf) as mf:
            with mf.open() as src:
                transform  = src.transform
                crs        = src.crs
                band_count = src.count
                dtype      = src.dtypes[0]
                h, w       = src.height, src.width

                print(f"      CRS           : {crs}")
                print(f"      Dimensions    : {w} W x {h} H px")
                print(f"      Bands         : {band_count}  |  dtype: {dtype}")
                print(f"      Transform     : {transform}")

                # ── 3. Read band 1 (multi-model flood agreement 0-6) ─────────
                print("\n[3/4] Reading band 1 (flood agreement 0-6) ...")
                clipped_array = src.read(1)

                print(f"      Array shape   : {clipped_array.shape}")
                import numpy as _np
                _vmin = _np.nanmin(clipped_array)
                _vmax = _np.nanmax(clipped_array)
                print(f"      Value range   : {_vmin} -- {_vmax}")
                _check_rss("after_read")

                # ── 4. Build output profile and write GeoTIFF ─────────────────
                print(f"\n[4/4] Writing deflate-compressed GeoTIFF to {out_path} ...")

                out_profile = copy.deepcopy(src.profile)
                out_profile.update(
                    driver="GTiff",
                    count=1,
                    height=h,
                    width=w,
                    transform=transform,
                    crs=crs,
                    dtype=dtype,
                    compress="deflate",
                    predictor=2,
                    zlevel=6,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    interleave="band",
                )

                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(clipped_array, 1)

        out_mb = out_path.stat().st_size / 1024 ** 2
        print(f"      File size     : {out_mb:.2f} MB (deflate-compressed)")
        _check_rss("after_write")

    except MemoryError:
        raise
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}", file=sys.stderr)
        raise RuntimeError(
            "download_flood_target: upstream failure -- see stderr above."
        ) from exc

    print()
    print("=" * 66)
    print("  SUCCESS -- Nairobi flood inundation raster acquired.")
    print(f"  Output  : {out_path.resolve()}")
    print(f"  Size    : {out_path.stat().st_size / 1024**2:.2f} MB")
    print("=" * 66)
    print()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingestion.download_flood_target",
        description=(
            "Server-side clip and stream the ICPAC Kenya flood inundation "
            "WCS layer to the Nairobi County bbox and write a "
            "deflate-compressed GeoTIFF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        metavar="PATH",
        help="Destination path for the clipped GeoTIFF.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    download_flood_target(out_path=args.out)
