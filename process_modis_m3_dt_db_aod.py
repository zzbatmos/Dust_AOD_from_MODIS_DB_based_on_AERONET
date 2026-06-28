#!/usr/bin/env python3
"""Extract monthly MOD08_M3/MYD08_M3 AOD fields for quick DT/DB trend analysis."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import earthaccess
import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "modis_m3_dt_db_quicklook"
PRODUCTS = ("MOD08_M3", "MYD08_M3")

FIELDS = {
    "dt_land_ocean_aod550": {
        "sds": "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean",
        "index": None,
        "long_name": "Dark Target land and ocean AOD at 550 nm, monthly mean of daily means",
    },
    "dt_land_aod550": {
        "sds": "Aerosol_Optical_Depth_Land_Mean_Mean",
        "index": 1,
        "long_name": "Dark Target land AOD at 550 nm, monthly mean of daily means",
    },
    "dt_land_qa_aod550": {
        "sds": "Aerosol_Optical_Depth_Land_QA_Mean_Mean",
        "index": 1,
        "long_name": "Dark Target land AOD at 550 nm, monthly mean of QA-weighted daily means",
    },
    "db_land_aod550": {
        "sds": "Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean_Mean",
        "index": None,
        "long_name": "Deep Blue land AOD at 550 nm, monthly mean of daily means",
    },
    "dt_db_combined_aod550": {
        "sds": "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean",
        "index": None,
        "long_name": "Combined Dark Target and Deep Blue AOD at 550 nm, monthly mean of daily means",
    },
}

COUNT_FIELDS = {
    "dt_land_ocean_pixel_counts": "Aerosol_Optical_Depth_Land_Ocean_Pixel_Counts",
    "db_land_pixels_used_550": "Deep_Blue_Number_Pixels_Used_550_Land_Mean_Mean",
}

REGIONS = {
    "Global": (-90.0, 90.0, -180.0, 180.0),
    "East China": (20.0, 45.0, 100.0, 125.0),
    "East Asia": (20.0, 50.0, 95.0, 135.0),
    "India": (5.0, 35.0, 65.0, 90.0),
    "North Africa/Sahel": (0.0, 35.0, -20.0, 35.0),
    "Middle East": (10.0, 35.0, 35.0, 65.0),
    "Tropical S. America": (-20.0, 10.0, -80.0, -45.0),
    "Europe": (35.0, 60.0, -10.0, 40.0),
    "CONUS": (25.0, 50.0, -125.0, -65.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-hdf", action="store_true", help="Keep downloaded HDF files under output-dir/hdf.")
    return parser.parse_args()


def month_start_end(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    if month == 12:
        end = dt.date(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return start, end


def parse_modis_date_from_name(name: str) -> dt.date | None:
    marker = ".A"
    if marker not in name:
        return None
    token = name.split(marker, 1)[1][:7]
    if len(token) != 7 or not token.isdigit():
        return None
    year = int(token[:4])
    jday = int(token[4:])
    return dt.date(year, 1, 1) + dt.timedelta(days=jday - 1)


def item_name(item) -> str:
    details = getattr(item, "details", {})
    name = details.get("name")
    if name:
        return Path(name).name
    links = item.data_links()
    return Path(links[0]).name if links else ""


def download_item_https(item, output_dir: Path, session, retries: int = 3) -> Path | None:
    links = item.data_links()
    if not links:
        return None
    url = links[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / Path(url).name
    if output.exists() and output.stat().st_size > 0:
        return output

    part = output.with_suffix(output.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with part.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            part.replace(output)
            return output
        except Exception as exc:
            try:
                part.unlink()
            except OSError:
                pass
            if attempt == retries:
                print(f"[ERROR download] {Path(url).name}: {exc}", flush=True)
                return None
            sleep_time = 5 * attempt
            print(f"[RETRY download] {Path(url).name}: attempt {attempt} failed; sleeping {sleep_time}s", flush=True)
            time.sleep(sleep_time)
    return None


def find_month_item(short_name: str, year: int, month: int):
    start, end = month_start_end(year, month)
    items = earthaccess.search_data(
        short_name=short_name,
        version="6.1",
        temporal=(start.isoformat(), end.isoformat()),
        count=8,
    )
    target = start
    for item in items:
        if parse_modis_date_from_name(item_name(item)) == target:
            return item
    return None


def read_scaled_hdf_field(hdf: SD, sds_name: str, index: int | None = None) -> np.ndarray:
    sds = hdf.select(sds_name)
    attrs = sds.attributes()
    data = sds.get()
    arr = np.asarray(data, dtype=np.float64)
    if index is not None:
        arr = arr[index, :, :]
    fill = attrs.get("_FillValue")
    valid_range = attrs.get("valid_range")
    if fill is not None:
        arr[arr == float(fill)] = np.nan
    if valid_range is not None and len(valid_range) == 2:
        lo, hi = float(valid_range[0]), float(valid_range[1])
        arr[(arr < lo) | (arr > hi)] = np.nan
    scale = float(attrs.get("scale_factor", 1.0))
    offset = float(attrs.get("add_offset", 0.0))
    return arr * scale + offset


def read_hdf_fields(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hdf = SD(str(path), SDC.READ)
    fields = {
        out_name: read_scaled_hdf_field(hdf, spec["sds"], spec["index"])
        for out_name, spec in FIELDS.items()
    }
    counts = {}
    for out_name, sds_name in COUNT_FIELDS.items():
        if sds_name in hdf.datasets():
            counts[out_name] = read_scaled_hdf_field(hdf, sds_name, None)
    hdf.end()
    return fields, counts


def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # MOD08 1-degree grid is stored north-to-south with longitude from west to east.
    lat = np.arange(89.5, -90.0, -1.0, dtype=np.float64)
    lon = np.arange(-179.5, 180.0, 1.0, dtype=np.float64)
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (lat.size, lon.size))
    return lat, lon, weights


def region_mask(lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = bounds
    return (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max) & (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)


def weighted_mean(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(field) & mask
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(field[valid] * weights[valid]) / np.nansum(weights[valid]))


def write_product_netcdf(
    output: Path,
    product: str,
    years: list[int],
    months: list[int],
    lat: np.ndarray,
    lon: np.ndarray,
    fields: dict[str, list[np.ndarray]],
    counts: dict[str, list[np.ndarray]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    nt = len(years)
    with Dataset(output, "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        t = ds.createVariable("time", "i4", ("time",))
        y = ds.createVariable("year", "i4", ("time",))
        m = ds.createVariable("month", "i4", ("time",))
        latv = ds.createVariable("lat", "f4", ("lat",))
        lonv = ds.createVariable("lon", "f4", ("lon",))
        t[:] = np.arange(nt)
        y[:] = years
        m[:] = months
        latv[:] = lat.astype(np.float32)
        lonv[:] = lon.astype(np.float32)
        latv.units = "degrees_north"
        lonv.units = "degrees_east"

        for name, arrays in fields.items():
            var = ds.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=np.nan)
            var[:] = np.asarray(arrays, dtype=np.float32)
            var.units = "1"
            var.long_name = FIELDS[name]["long_name"]
            var.source_sds = FIELDS[name]["sds"]

        for name, arrays in counts.items():
            var = ds.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=np.nan)
            var[:] = np.asarray(arrays, dtype=np.float32)
            var.units = "1"
            var.long_name = name.replace("_", " ")
            var.source_sds = COUNT_FIELDS[name]

        ds.title = f"{product} monthly Dark Target and Deep Blue AOD quick-look extraction"
        ds.product = product
        ds.collection = "MODIS Collection 6.1"
        ds.grid_note = "1 degree MOD08 grid; latitude centers 89.5 to -89.5, longitude centers -179.5 to 179.5"
        ds.history = "Created by process_modis_m3_dt_db_aod.py using earthaccess downloads."


def write_regional_csv(output: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["product", "year", "month", "region", *FIELDS.keys()]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf_dir = args.output_dir / "hdf"
    if args.keep_hdf:
        hdf_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login(strategy="netrc")
    https_session = earthaccess.get_requests_https_session()
    lat, lon, weights = grid()
    masks = {name: region_mask(lat, lon, bounds) for name, bounds in REGIONS.items()}
    all_rows: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="modis_m3_", dir="/tmp") as tmp:
        tmp_dir = Path(tmp)
        for product in args.products:
            product_years: list[int] = []
            product_months: list[int] = []
            product_fields = {name: [] for name in FIELDS}
            product_counts = {name: [] for name in COUNT_FIELDS}

            for year in range(args.start_year, args.end_year + 1):
                for month in range(1, 13):
                    out_nc = args.output_dir / f"{product}_AOD_monthly_{args.start_year}_{args.end_year}.nc"
                    print(f"[SEARCH] {product} {year}-{month:02d}", flush=True)
                    item = find_month_item(product, year, month)
                    if item is None:
                        print(f"[MISS] {product} {year}-{month:02d}", flush=True)
                        continue

                    download_dir = hdf_dir if args.keep_hdf else tmp_dir
                    local_hdf = download_item_https(item, download_dir, https_session)
                    if local_hdf is None:
                        print(f"[MISS download] {product} {year}-{month:02d}", flush=True)
                        continue
                    print(f"[READ] {local_hdf.name}", flush=True)
                    fields, counts = read_hdf_fields(local_hdf)

                    product_years.append(year)
                    product_months.append(month)
                    for name, arr in fields.items():
                        product_fields[name].append(arr)
                    for name in product_counts:
                        product_counts[name].append(counts.get(name, np.full((lat.size, lon.size), np.nan)))

                    for region, mask in masks.items():
                        row: dict[str, object] = {"product": product, "year": year, "month": month, "region": region}
                        for field_name, arr in fields.items():
                            row[field_name] = weighted_mean(arr, weights, mask)
                        all_rows.append(row)

                    if not args.keep_hdf:
                        try:
                            local_hdf.unlink()
                        except OSError:
                            pass

            if product_years:
                output_nc = args.output_dir / f"{product}_AOD_monthly_{args.start_year}_{args.end_year}.nc"
                write_product_netcdf(
                    output_nc,
                    product,
                    product_years,
                    product_months,
                    lat,
                    lon,
                    product_fields,
                    product_counts,
                )
                print(f"[WROTE] {output_nc}", flush=True)

    csv_path = args.output_dir / f"MODIS_M3_AOD_monthly_region_means_{args.start_year}_{args.end_year}.csv"
    write_regional_csv(csv_path, all_rows)
    print(f"[WROTE] {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
