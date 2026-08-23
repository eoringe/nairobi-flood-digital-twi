"""
src.ingestion.fetch_chirps
===========================
Nairobi Urban Flood Digital Twin — CHIRPS Rainfall Processor

PURPOSE
-------
1. Read CHIRPS daily precipitation NetCDF datasets (.nc) from data/raw/climate/
2. Subset spatial window to Nairobi County domain (-1.45 <= Lat <= -1.15, 36.65 <= Lon <= 37.10)
3. Extract daily mean rainfall time-series (mm/day) and gridded precipitation matrices
4. Save standardized numpy array (T, Lat, Lon) and date metadata to data/processed/arrays/

MEMORY-FIRST DESIGN
-------------------
Reads NetCDF files slice-by-slice / year-by-year to keep RAM < 500 MB.

USAGE
-----
    python -m src.ingestion.fetch_chirps
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger

try:
    import xarray as xr
    _XARRAY_OK = True
except ImportError:
    _XARRAY_OK = False

try:
    import netCDF4
    _NETCDF_OK = True
except ImportError:
    _NETCDF_OK = False

DEFAULT_CLIMATE_DIR = Path("data/raw/climate")
DEFAULT_OUT_DIR = Path("data/processed/arrays")

NAIROBI_LAT_MIN, NAIROBI_LAT_MAX = -1.45, -1.15
NAIROBI_LON_MIN, NAIROBI_LON_MAX = 36.65, 37.10


def process_chirps_netcdf(
    climate_dir: Path = DEFAULT_CLIMATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(list(climate_dir.glob("*.nc")))

    if not nc_files:
        logger.warning(f"No NetCDF files found in {climate_dir}")
        return

    logger.info(f"Processing {len(nc_files)} CHIRPS NetCDF files in {climate_dir}")

    rainfall_list = []
    dates_list = []

    for nc_path in nc_files:
        logger.info(f"Processing: {nc_path.name}")
        if _XARRAY_OK:
            try:
                ds = xr.open_dataset(nc_path)
                # CHIRPS variables: precip, latitude/lat, longitude/lon
                lat_var = "latitude" if "latitude" in ds.coords else "lat"
                lon_var = "longitude" if "longitude" in ds.coords else "lon"
                time_var = "time"

                sub = ds.sel(
                    {
                        lat_var: slice(NAIROBI_LAT_MIN, NAIROBI_LAT_MAX),
                        lon_var: slice(NAIROBI_LON_MIN, NAIROBI_LON_MAX),
                    }
                )

                precip = sub["precip"].values  # shape: (time, lat, lon)
                # Handle missing value (-9999)
                precip = np.where(precip < 0, 0.0, precip).astype(np.float32)

                times = [str(t)[:10] for t in sub[time_var].values]

                rainfall_list.append(precip)
                dates_list.extend(times)
                ds.close()
                continue
            except Exception as e:
                logger.warning(f"xarray failed for {nc_path.name}: {e}. Falling back to netCDF4.")

        if _NETCDF_OK:
            try:
                ds = netCDF4.Dataset(nc_path, "r")
                lat_name = "latitude" if "latitude" in ds.variables else "lat"
                lon_name = "longitude" if "longitude" in ds.variables else "lon"
                
                lats = ds.variables[lat_name][:]
                lons = ds.variables[lon_name][:]

                lat_idx = np.where((lats >= NAIROBI_LAT_MIN) & (lats <= NAIROBI_LAT_MAX))[0]
                lon_idx = np.where((lons >= NAIROBI_LON_MIN) & (lons <= NAIROBI_LON_MAX))[0]

                if len(lat_idx) == 0 or len(lon_idx) == 0:
                    # Depending on coordinate orientation
                    lat_idx = np.where((lats <= NAIROBI_LAT_MAX) & (lats >= NAIROBI_LAT_MIN))[0]

                precip = ds.variables["precip"][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                precip = np.where(precip < 0, 0.0, precip).astype(np.float32)

                # Time units
                time_var = ds.variables["time"]
                times = [f"{nc_path.name.split('.')[2]}-day-{i+1:03d}" for i in range(precip.shape[0])]

                rainfall_list.append(precip)
                dates_list.extend(times)
                ds.close()
            except Exception as e:
                logger.error(f"netCDF4 failed for {nc_path.name}: {e}")

    if rainfall_list:
        combined_rainfall = np.concatenate(rainfall_list, axis=0)
        np.save(out_dir / "rainfall_chirps.npy", combined_rainfall)
        with open(out_dir / "rainfall_dates.json", "w") as f:
            json.dump(dates_list, f, indent=2)

        # Calculate daily mean series across Nairobi domain
        daily_mean = np.mean(combined_rainfall, axis=(1, 2))
        np.save(out_dir / "rainfall_daily_mean.npy", daily_mean)

        logger.info(
            f"Successfully processed CHIRPS data! Total days: {combined_rainfall.shape[0]}, "
            f"Grid shape: {combined_rainfall.shape[1:]}, Saved to {out_dir}/rainfall_chirps.npy"
        )
    else:
        logger.error("No CHIRPS rainfall data could be processed.")


if __name__ == "__main__":
    process_chirps_netcdf()
