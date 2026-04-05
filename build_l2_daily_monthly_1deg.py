#!/usr/bin/env python3
"""Aggregate saved L2 per-granule no-plot outputs into daily and monthly 1x1 degree products."""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


LAT_CENTERS = np.arange(-89.5, 90.0, 1.0, dtype=np.float32)
LON_CENTERS = np.arange(-179.5, 180.0, 1.0, dtype=np.float32)
NY = LAT_CENTERS.size
NX = LON_CENTERS.size

TWO_STAGE_THRESHOLDS = {
    1: 0.4,
    2: 0.6,
    3: 0.7,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=here / "aqua_l2_2017_three_methods_noplot_by_day" / "MYD04_L2_2017-01-01_2017-12-31",
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=here / "aqua_l2_2017_1deg_daily" / "MYD04_L2_three_methods_daily_1deg_2017.nc",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=here / "aqua_l2_2017_1deg_monthly" / "MYD04_L2_three_methods_monthly_1deg_2017.nc",
    )
    return parser.parse_args()


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def indices_from_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_wrapped = wrap_lon(lon)
    valid = np.isfinite(lat) & np.isfinite(lon_wrapped) & (lat >= -90.0) & (lat <= 90.0)
    lat_idx = np.floor(lat + 90.0).astype(np.int16)
    lon_idx = np.floor(lon_wrapped + 180.0).astype(np.int16)
    valid &= (lat_idx >= 0) & (lat_idx < NY) & (lon_idx >= 0) & (lon_idx < NX)
    return lat_idx, lon_idx, valid


def nanmean_time(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, valid_count, out=out, where=valid_count > 0)
    return out, valid_count.astype(np.int16)


def aggregate_day(day_dir: Path) -> dict[str, np.ndarray]:
    granules = sorted((day_dir / "granules").glob("*.npz"))
    if not granules:
        return {
            "total_aod": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_li_ginoux": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_two_stage_qa1": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_two_stage_qa2": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_two_stage_qa3": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_dbtype_qa1": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_dbtype_qa2": np.full((NY, NX), np.nan, dtype=np.float32),
            "dust_aod_dbtype_qa3": np.full((NY, NX), np.nan, dtype=np.float32),
            "n_pixels": np.zeros((NY, NX), dtype=np.int32),
        }

    sum_total = np.zeros((NY, NX), dtype=np.float64)
    sum_li = np.zeros((NY, NX), dtype=np.float64)
    sum_ts = {k: np.zeros((NY, NX), dtype=np.float64) for k in (1, 2, 3)}
    sum_db = {k: np.zeros((NY, NX), dtype=np.float64) for k in (1, 2, 3)}
    count = np.zeros((NY, NX), dtype=np.int32)

    for granule in granules:
        data = np.load(granule)
        lat = np.asarray(data["lat"], dtype=np.float64)
        lon = np.asarray(data["lon"], dtype=np.float64)
        aod = np.asarray(data["aod550"], dtype=np.float64)
        li = np.asarray(data["li_dust_aod"], dtype=np.float64)
        two_prob = np.asarray(data["two_stage_probability"], dtype=np.float64)
        two_frac = np.asarray(data["two_stage_fraction"], dtype=np.float64)
        db_frac = np.asarray(data["db_type_fraction"], dtype=np.float64)
        db_qa = np.asarray(data["db_type_qa"], dtype=np.float64)

        lat_idx, lon_idx, valid_geo = indices_from_lat_lon(lat, lon)
        valid = valid_geo & np.isfinite(aod)
        if not np.any(valid):
            continue

        flat_idx = (lat_idx[valid].astype(np.int32) * NX + lon_idx[valid].astype(np.int32)).ravel()
        aod_valid = aod[valid].ravel()
        li_valid = np.nan_to_num(li[valid].ravel(), nan=0.0)

        np.add.at(sum_total.ravel(), flat_idx, aod_valid)
        np.add.at(sum_li.ravel(), flat_idx, li_valid)
        np.add.at(count.ravel(), flat_idx, 1)

        for qa_level, threshold in TWO_STAGE_THRESHOLDS.items():
            pass_mask = (
                valid
                & np.isfinite(two_prob)
                & np.isfinite(two_frac)
                & (two_prob >= threshold)
            )
            ts_vals = np.zeros_like(aod_valid, dtype=np.float64)
            if np.any(pass_mask):
                ts_vals[np.flatnonzero(pass_mask[valid])] = (
                    np.clip(two_frac[pass_mask], 0.0, 1.0) * aod[pass_mask]
                ).ravel()
            np.add.at(sum_ts[qa_level].ravel(), flat_idx, ts_vals)

        for qa_level in (1, 2, 3):
            pass_mask = (
                valid
                & np.isfinite(db_qa)
                & np.isfinite(db_frac)
                & (db_qa >= float(qa_level))
            )
            db_vals = np.zeros_like(aod_valid, dtype=np.float64)
            if np.any(pass_mask):
                db_vals[np.flatnonzero(pass_mask[valid])] = (
                    np.clip(db_frac[pass_mask], 0.0, 1.0) * aod[pass_mask]
                ).ravel()
            np.add.at(sum_db[qa_level].ravel(), flat_idx, db_vals)

    out: dict[str, np.ndarray] = {
        "n_pixels": count,
    }
    denom = count.astype(np.float64)
    total_mean = np.full((NY, NX), np.nan, dtype=np.float32)
    li_mean = np.full((NY, NX), np.nan, dtype=np.float32)
    np.divide(sum_total, denom, out=total_mean, where=count > 0)
    np.divide(sum_li, denom, out=li_mean, where=count > 0)
    out["total_aod"] = total_mean
    out["dust_aod_li_ginoux"] = li_mean
    for qa_level in (1, 2, 3):
        ts_mean = np.full((NY, NX), np.nan, dtype=np.float32)
        db_mean = np.full((NY, NX), np.nan, dtype=np.float32)
        np.divide(sum_ts[qa_level], denom, out=ts_mean, where=count > 0)
        np.divide(sum_db[qa_level], denom, out=db_mean, where=count > 0)
        out[f"dust_aod_two_stage_qa{qa_level}"] = ts_mean
        out[f"dust_aod_dbtype_qa{qa_level}"] = db_mean
    return out


def write_daily_output(output_path: Path, dates: list[datetime], variables: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_path, "w") as ds:
        ds.createDimension("time", len(dates))
        ds.createDimension("lat", NY)
        ds.createDimension("lon", NX)

        time_var = ds.createVariable("time", "i4", ("time",))
        year_var = ds.createVariable("year", "i4", ("time",))
        month_var = ds.createVariable("month", "i4", ("time",))
        day_var = ds.createVariable("day", "i4", ("time",))
        lat_var = ds.createVariable("lat", "f4", ("lat",), zlib=True, complevel=4)
        lon_var = ds.createVariable("lon", "f4", ("lon",), zlib=True, complevel=4)

        base = datetime(dates[0].year, 1, 1)
        time_var[:] = np.array([(d - base).days for d in dates], dtype=np.int32)
        time_var.units = f"days since {base:%Y-%m-%d} 00:00:00"
        year_var[:] = np.array([d.year for d in dates], dtype=np.int32)
        month_var[:] = np.array([d.month for d in dates], dtype=np.int32)
        day_var[:] = np.array([d.day for d in dates], dtype=np.int32)
        lat_var[:] = LAT_CENTERS
        lon_var[:] = LON_CENTERS

        for name, data in variables.items():
            dtype = "i4" if name == "n_pixels" else "f4"
            fill = -1 if name == "n_pixels" else np.float32(np.nan)
            var = ds.createVariable(name, dtype, ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=fill)
            var[:] = data
            if name == "n_pixels":
                var.long_name = "number of valid total-AOD pixels contributing to daily 1x1 degree mean"
            else:
                var.units = "unitless"

        ds.title = "Daily 1x1 degree daytime mean MODIS DB total and dust AOD from L2 granules"
        ds.averaging_rule = (
            "Daily grid means use all valid total-AOD pixels in each 1x1 degree box. "
            "Pixels with invalid total AOD are discarded. "
            "For QA-thresholded methods, pixels with valid total AOD but QA below threshold are assigned dust AOD=0."
        )
        ds.history = "Created by build_l2_daily_monthly_1deg.py"


def write_monthly_output(output_path: Path, year: int, daily_vars: dict[str, np.ndarray], dates: list[datetime]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    months = np.arange(1, 13, dtype=np.int32)
    monthly_vars: dict[str, np.ndarray] = {}
    for name, data in daily_vars.items():
        out_shape = (12, NY, NX)
        if name == "n_pixels":
            # Daily pixel counts are not averaged; monthly file stores number of valid daily means.
            continue
        out = np.full(out_shape, np.nan, dtype=np.float32)
        valid_days = np.zeros(out_shape, dtype=np.int16)
        for month in months:
            sel = [i for i, d in enumerate(dates) if d.month == int(month)]
            if not sel:
                continue
            mean_field, count_field = nanmean_time(data[np.array(sel)])
            out[month - 1] = mean_field
            valid_days[month - 1] = count_field
        monthly_vars[name] = out
        monthly_vars[f"{name}_valid_days"] = valid_days

    with Dataset(output_path, "w") as ds:
        ds.createDimension("month", 12)
        ds.createDimension("lat", NY)
        ds.createDimension("lon", NX)
        year_var = ds.createVariable("year", "i4")
        month_var = ds.createVariable("month", "i4", ("month",))
        lat_var = ds.createVariable("lat", "f4", ("lat",), zlib=True, complevel=4)
        lon_var = ds.createVariable("lon", "f4", ("lon",), zlib=True, complevel=4)
        year_var.assignValue(year)
        month_var[:] = months
        lat_var[:] = LAT_CENTERS
        lon_var[:] = LON_CENTERS

        for name, data in monthly_vars.items():
            is_count = name.endswith("_valid_days")
            dtype = "i2" if is_count else "f4"
            fill = np.int16(-1) if is_count else np.float32(np.nan)
            var = ds.createVariable(name, dtype, ("month", "lat", "lon"), zlib=True, complevel=4, fill_value=fill)
            var[:] = data
            if is_count:
                var.long_name = "number of daily means contributing to monthly mean"
            else:
                var.units = "unitless"

        ds.title = "Monthly 1x1 degree daytime mean MODIS DB total and dust AOD from daily L2 means"
        ds.averaging_rule = (
            "Monthly means are simple averages of the daily 1x1 degree means for each month. "
            "Days with no valid daily mean at a grid cell are ignored."
        )
        ds.history = "Created by build_l2_daily_monthly_1deg.py"


def main() -> None:
    args = parse_args()
    day_dirs = []
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(args.year, month)[1] + 1):
            day_str = f"{args.year:04d}-{month:02d}-{day:02d}"
            day_dir = args.input_root / day_str
            if not day_dir.exists():
                raise FileNotFoundError(f"Missing day directory: {day_dir}")
            day_dirs.append((datetime.strptime(day_str, "%Y-%m-%d"), day_dir))

    daily_stack: dict[str, list[np.ndarray]] = {
        "total_aod": [],
        "dust_aod_li_ginoux": [],
        "dust_aod_two_stage_qa1": [],
        "dust_aod_two_stage_qa2": [],
        "dust_aod_two_stage_qa3": [],
        "dust_aod_dbtype_qa1": [],
        "dust_aod_dbtype_qa2": [],
        "dust_aod_dbtype_qa3": [],
        "n_pixels": [],
    }

    dates: list[datetime] = []
    for dt, day_dir in day_dirs:
        print(f"[DAY] aggregating {dt:%Y-%m-%d}", flush=True)
        agg = aggregate_day(day_dir)
        dates.append(dt)
        for name in daily_stack:
            daily_stack[name].append(agg[name])

    daily_vars = {name: np.stack(arrs).astype(np.int32 if name == "n_pixels" else np.float32) for name, arrs in daily_stack.items()}
    write_daily_output(args.daily_output, dates, daily_vars)
    write_monthly_output(args.monthly_output, args.year, daily_vars, dates)
    print(f"Wrote {args.daily_output}")
    print(f"Wrote {args.monthly_output}")


if __name__ == "__main__":
    main()
