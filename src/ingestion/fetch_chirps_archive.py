"""
src.ingestion.fetch_chirps_archive
====================================
Nairobi Urban Flood Digital Twin — CHIRPS Global Daily Archive Downloader

PURPOSE
-------
src.ingestion.fetch_chirps only *processes* whatever .nc files already sit
in data/raw/climate/ — it never fetches them. Those files were previously
placed there manually and only covered 2020-2026, which capped
src.preprocessing.dataset_builder's real Sentinel-1 training events to
that same window even though usable Sentinel-1 SAR coverage over Nairobi
goes back to 2015 (verified directly against Google Earth Engine scene
counts). This script pulls the missing years directly from the public
CHIRPS v2.0 global daily archive (UCSB Climate Hazards Center) so
fetch_chirps.py has real rainfall data to pair with the older Sentinel-1
composites.

USAGE
-----
    python -m src.ingestion.fetch_chirps_archive
    python -m src.ingestion.fetch_chirps_archive --years 2015 2016 2017 2018 2019
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import psutil
from loguru import logger

CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25"
DEFAULT_OUT_DIR = Path("data/raw/climate")
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019]
CHUNK_SIZE = 1024 * 1024  # 1 MB
MIN_FREE_RAM_GB = 1.0


def _check_free_disk_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(path).free / 1024 ** 3


def fetch_chirps_archive(years: list[int] = DEFAULT_YEARS, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    free_gb = psutil.virtual_memory().available / 1024 ** 3
    if free_gb < MIN_FREE_RAM_GB:
        logger.error(f"Free RAM {free_gb:.2f} GB < {MIN_FREE_RAM_GB} GB floor. Aborting.")
        sys.exit(1)

    disk_free_gb = _check_free_disk_gb(out_dir)
    logger.info(f"Disk free at {out_dir}: {disk_free_gb:.1f} GB")

    for year in years:
        out_path = out_dir / f"chirps-v2.0.{year}.days_p25.nc"
        if out_path.exists() and out_path.stat().st_size > 1_000_000:
            logger.info(f"[SKIP] {out_path.name} already present ({out_path.stat().st_size / 1024**2:.1f} MB).")
            continue

        url = f"{CHIRPS_BASE_URL}/chirps-v2.0.{year}.days_p25.nc"
        logger.info(f"Downloading {url} ...")

        try:
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total_bytes = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                tmp_path = out_path.with_suffix(".nc.part")
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                tmp_path.rename(out_path)
            logger.info(
                f"[OK] {out_path.name} — {downloaded / 1024**2:.1f} MB "
                f"(expected {total_bytes / 1024**2:.1f} MB)"
            )
        except Exception as e:
            logger.error(f"[FAIL] {year}: {e}")
            continue

    logger.info("CHIRPS archive backfill complete.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingestion.fetch_chirps_archive",
        description="Download missing years of CHIRPS global daily precipitation NetCDF from the public UCSB CHC archive.",
    )
    p.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, metavar="YEAR")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    fetch_chirps_archive(years=args.years, out_dir=args.out_dir)
