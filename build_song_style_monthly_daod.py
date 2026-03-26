#!/usr/bin/env python3
"""Build Song-style monthly DAOD/TAOD NetCDF files from daily L3 products."""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


FILE_PATTERNS = {
    ("MYD08_D3", "xgboost"): re.compile(r"MYD08_D3_DustAOD_XGB_(\d{4})-(\d{2})-(\d{2})\.nc$"),
    ("MYD08_D3", "li_ginoux"): re.compile(r"MYD08_D3_DustAOD_LiGinoux_(\d{4})-(\d{2})-(\d{2})\.nc$"),
    ("MOD08_D3", "xgboost"): re.compile(r"MOD08_D3_DustAOD_XGB_(\d{4})-(\d{2})-(\d{2})\.nc$"),
    ("MOD08_D3", "li_ginoux"): re.compile(r"MOD08_D3_DustAOD_LiGinoux_(\d{4})-(\d{2})-(\d{2})\.nc$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate daily MODIS L3 dust AOD outputs into a Song-style monthly "
            "NetCDF with dimensions (year, month, lat, lon)."
        )
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        required=True,
        help="Directory containing daily L3 dust AOD NetCDF files.",
    )
    parser.add_argument(
        "--platform",
        choices=["MYD08_D3", "MOD08_D3"],
        required=True,
        help="Platform/product prefix used in the daily filenames.",
    )
    parser.add_argument(
        "--method",
        choices=["xgboost", "li_ginoux"],
        required=True,
        help="Dust AOD method represented by the daily files.",
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output monthly NetCDF path.",
    )
    return parser.parse_args()


def index_daily_files(
    daily_dir: Path, platform: str, method: str, start_year: int, end_year: int
) -> dict[tuple[int, int, int], Path]:
    pattern = FILE_PATTERNS[(platform, method)]
    indexed: dict[tuple[int, int, int], Path] = {}
    for path in sorted(daily_dir.glob("*.nc")):
        match = pattern.match(path.name)
        if not match:
            continue
        year, month, day = map(int, match.groups())
        if start_year <= year <= end_year:
            indexed[(year, month, day)] = path
    if not indexed:
        raise FileNotFoundError(f"No daily files found in {daily_dir} for {platform} {method}")
    return indexed


def read_grid(example_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(example_file) as ds:
        lat2d = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon2d = np.asarray(ds.variables["lon"][:], dtype=np.float32)
    lat = lat2d[:, 0].copy()
    lon = lon2d[0, :].copy()
    return lat, lon


def monthly_means(
    files: list[Path], ny: int, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    taod_sum = np.zeros((ny, nx), dtype=np.float64)
    daod_sum = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int16)

    for path in files:
        with Dataset(path) as ds:
            taod = np.asarray(ds.variables["db_aod_550"][:], dtype=np.float64)
            daod = np.asarray(ds.variables["dust_aod"][:], dtype=np.float64)

        valid_total = np.isfinite(taod)
        if not valid_total.any():
            continue

        # Follow the requested rule exactly:
        # if total AOD exists but dust AOD is missing, count the day and use zero.
        daod_effective = np.where(valid_total, np.where(np.isfinite(daod), daod, 0.0), np.nan)

        taod_sum[valid_total] += taod[valid_total]
        daod_sum[valid_total] += daod_effective[valid_total]
        count[valid_total] += 1

    taod_month = np.full((ny, nx), np.nan, dtype=np.float32)
    daod_month = np.full((ny, nx), np.nan, dtype=np.float32)
    valid = count > 0
    taod_month[valid] = (taod_sum[valid] / count[valid]).astype(np.float32)
    daod_month[valid] = (daod_sum[valid] / count[valid]).astype(np.float32)
    return daod_month, taod_month, count


def write_output(
    output_path: Path,
    years: np.ndarray,
    months: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    daod: np.ndarray,
    taod: np.ndarray,
    sample_count: np.ndarray,
    platform: str,
    method: str,
    daily_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_path, "w") as ds:
        ds.createDimension("year", len(years))
        ds.createDimension("month", len(months))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))

        year_var = ds.createVariable("year", "i4", ("year",))
        month_var = ds.createVariable("month", "i4", ("month",))
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        daod_var = ds.createVariable("daod", "f4", ("year", "month", "lat", "lon"), zlib=True, complevel=4, fill_value=np.nan)
        taod_var = ds.createVariable("taod", "f4", ("year", "month", "lat", "lon"), zlib=True, complevel=4, fill_value=np.nan)
        count_var = ds.createVariable("n_days", "i2", ("year", "month", "lat", "lon"), zlib=True, complevel=4)

        year_var[:] = years
        month_var[:] = months
        lat_var[:] = lat
        lon_var[:] = lon
        daod_var[:] = daod
        taod_var[:] = taod
        count_var[:] = sample_count

        year_var.units = "year"
        year_var.long_name = "year"
        month_var.units = "month of a year"
        month_var.long_name = "month"
        lat_var.units = "degrees_north"
        lat_var.long_name = "center latitude"
        lon_var.units = "degrees_east"
        lon_var.long_name = "center longitude"
        daod_var.units = "unitless"
        daod_var.long_name = "monthly mean dust aerosol optical depth at 550 nm"
        taod_var.units = "unitless"
        taod_var.long_name = "monthly mean total aerosol optical depth at 550 nm"
        count_var.long_name = "number of daily total-AOD retrievals contributing to monthly mean"

        ds.title = "Monthly MODIS Deep Blue dust AOD and total AOD"
        ds.platform = platform
        ds.method = method
        ds.source_daily_directory = str(daily_dir)
        ds.averaging_rule = (
            "Days with finite total AOD contribute to both means; if dust AOD is missing "
            "on such a day it is treated as zero for the dust-AOD monthly mean."
        )
        ds.history = "Created by build_song_style_monthly_daod.py"


def main() -> None:
    args = parse_args()
    indexed = index_daily_files(
        args.daily_dir, args.platform, args.method, args.start_year, args.end_year
    )
    example_file = indexed[min(indexed)]
    lat, lon = read_grid(example_file)

    years = np.arange(args.start_year, args.end_year + 1, dtype=np.int32)
    months = np.arange(1, 13, dtype=np.int32)
    daod = np.full((len(years), 12, len(lat), len(lon)), np.nan, dtype=np.float32)
    taod = np.full_like(daod, np.nan)
    sample_count = np.zeros((len(years), 12, len(lat), len(lon)), dtype=np.int16)

    for yi, year in enumerate(years):
        for mi, month in enumerate(months, start=0):
            files = []
            for day in range(1, calendar.monthrange(int(year), int(month))[1] + 1):
                path = indexed.get((int(year), int(month), day))
                if path is not None:
                    files.append(path)
            if not files:
                continue
            daod_month, taod_month, count = monthly_means(files, len(lat), len(lon))
            daod[yi, mi] = daod_month
            taod[yi, mi] = taod_month
            sample_count[yi, mi] = count

    write_output(
        args.output,
        years,
        months,
        lat,
        lon,
        daod,
        taod,
        sample_count,
        args.platform,
        args.method,
        args.daily_dir,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
