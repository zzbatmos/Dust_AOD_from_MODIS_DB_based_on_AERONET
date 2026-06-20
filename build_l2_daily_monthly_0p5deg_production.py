#!/usr/bin/env python3
"""Aggregate production-freeze L2 NPZ outputs to daily/monthly 0.5 degree products."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


GRID_STEP = 0.5
LAT_CENTERS = np.arange(-89.75, 90.0, GRID_STEP, dtype=np.float32)
LON_CENTERS = np.arange(-179.75, 180.0, GRID_STEP, dtype=np.float32)
NY = LAT_CENTERS.size
NX = LON_CENTERS.size
TWO_STAGE_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
PRODUCT_VERSION = "production-freeze-l2-daod-v1"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=here / "modis_l2_production_freeze_v1" / "MYD04_L2_2023-01-01_2023-12-31",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=here / "modis_l3_production_freeze_v1" / "MYD04_L2_daily_0p5deg_2023.nc",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=here / "modis_l3_production_freeze_v1" / "MYD04_L2_monthly_0p5deg_2023.nc",
    )
    return parser.parse_args()


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def indices_from_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_wrapped = wrap_lon(lon)
    valid = np.isfinite(lat) & np.isfinite(lon_wrapped) & (lat >= -90.0) & (lat <= 90.0)
    lat_idx = np.floor((lat + 90.0) / GRID_STEP).astype(np.int32)
    lon_idx = np.floor((lon_wrapped + 180.0) / GRID_STEP).astype(np.int32)
    valid &= (lat_idx >= 0) & (lat_idx < NY) & (lon_idx >= 0) & (lon_idx < NX)
    return lat_idx, lon_idx, valid


def require_key(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} is missing required production-freeze key {key!r}")
    return data[key]


def empty_day() -> dict[str, np.ndarray]:
    return {
        "total_aod": np.full((NY, NX), np.nan, dtype=np.float32),
        "dust_aod_li_ginoux": np.full((NY, NX), np.nan, dtype=np.float32),
        "dust_aod_li_ginoux_raw": np.full((NY, NX), np.nan, dtype=np.float32),
        "dust_aod_two_stage_qa1": np.full((NY, NX), np.nan, dtype=np.float32),
        "dust_aod_two_stage_qa2": np.full((NY, NX), np.nan, dtype=np.float32),
        "dust_aod_two_stage_qa3": np.full((NY, NX), np.nan, dtype=np.float32),
        "n_pixels": np.zeros((NY, NX), dtype=np.int32),
    }


def aggregate_day(day_dir: Path) -> dict[str, np.ndarray]:
    granules = sorted((day_dir / "granules").glob("*.npz"))
    if not granules:
        return empty_day()

    sum_total = np.zeros((NY, NX), dtype=np.float64)
    sum_li = np.zeros((NY, NX), dtype=np.float64)
    sum_li_raw = np.zeros((NY, NX), dtype=np.float64)
    sum_ts = {qa: np.zeros((NY, NX), dtype=np.float64) for qa in TWO_STAGE_THRESHOLDS}
    count = np.zeros((NY, NX), dtype=np.int32)

    for granule in granules:
        data = np.load(granule)
        lat = np.asarray(require_key(data, "lat", granule), dtype=np.float64)
        lon = np.asarray(require_key(data, "lon", granule), dtype=np.float64)
        aod = np.asarray(require_key(data, "aod550", granule), dtype=np.float64)
        li_raw = np.asarray(require_key(data, "li_dust_aod", granule), dtype=np.float64)
        li_screen = np.asarray(require_key(data, "li_ginoux_ae_screen_dust_aod", granule), dtype=np.float64)
        prob = np.asarray(require_key(data, "two_stage_probability", granule), dtype=np.float64)
        frac_raw = np.asarray(require_key(data, "two_stage_fraction_raw", granule), dtype=np.float64)

        lat_idx, lon_idx, valid_geo = indices_from_lat_lon(lat, lon)
        valid = valid_geo & np.isfinite(aod)
        if not np.any(valid):
            continue

        flat_idx = (lat_idx[valid] * NX + lon_idx[valid]).ravel()
        aod_valid = aod[valid].ravel()
        np.add.at(sum_total.ravel(), flat_idx, aod_valid)
        np.add.at(sum_li.ravel(), flat_idx, np.nan_to_num(li_screen[valid].ravel(), nan=0.0))
        np.add.at(sum_li_raw.ravel(), flat_idx, np.nan_to_num(li_raw[valid].ravel(), nan=0.0))
        np.add.at(count.ravel(), flat_idx, 1)

        for qa, threshold in TWO_STAGE_THRESHOLDS.items():
            pass_mask = valid & np.isfinite(prob) & np.isfinite(frac_raw) & (prob >= threshold)
            values = np.zeros_like(aod_valid, dtype=np.float64)
            if np.any(pass_mask):
                valid_positions = np.flatnonzero(pass_mask[valid])
                values[valid_positions] = (np.clip(frac_raw[pass_mask], 0.0, 1.0) * aod[pass_mask]).ravel()
            np.add.at(sum_ts[qa].ravel(), flat_idx, values)

    denom = count.astype(np.float64)
    out: dict[str, np.ndarray] = {"n_pixels": count}
    for name, total in [
        ("total_aod", sum_total),
        ("dust_aod_li_ginoux", sum_li),
        ("dust_aod_li_ginoux_raw", sum_li_raw),
    ]:
        field = np.full((NY, NX), np.nan, dtype=np.float32)
        np.divide(total, denom, out=field, where=count > 0)
        out[name] = field
    for qa, total in sum_ts.items():
        field = np.full((NY, NX), np.nan, dtype=np.float32)
        np.divide(total, denom, out=field, where=count > 0)
        out[f"dust_aod_two_stage_qa{qa}"] = field
    return out


def nanmean_time(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, valid_count, out=out, where=valid_count > 0)
    return out, valid_count.astype(np.int16)


def set_var_attrs(name: str, var) -> None:
    var.units = "unitless"
    if name == "n_pixels":
        var.long_name = "number of valid total-AOD pixels contributing to daily 0.5x0.5 degree mean"
    elif name == "dust_aod_li_ginoux":
        var.long_name = "Li-Ginoux dust AOD with SSA412<SSA470 and FMF<=0.7 AE-equivalent screen"
    elif name == "dust_aod_li_ginoux_raw":
        var.long_name = "Li-Ginoux dust AOD with SSA412<SSA470 before FMF<=0.7 AE-equivalent screen"
    elif name.startswith("dust_aod_two_stage_qa"):
        qa = int(name[-1])
        var.long_name = (
            f"two-stage XGBoost dust AOD at QA{qa}; valid total-AOD pixels below "
            f"P(dust)>={TWO_STAGE_THRESHOLDS[qa]:.1f} are assigned zero dust AOD"
        )


def write_common_attrs(ds, *, is_monthly: bool) -> None:
    ds.product_version = PRODUCT_VERSION
    ds.title = (
        "Monthly 0.5x0.5 degree daytime mean MODIS DB total and dust AOD from L2 granules"
        if is_monthly
        else "Daily 0.5x0.5 degree daytime mean MODIS DB total and dust AOD from L2 granules"
    )
    ds.two_stage_qa_definition = (
        "QA1=P(dust)>=0.4, QA2=P(dust)>=0.6, QA3=P(dust)>=0.7. "
        "QA2 is the recommended balanced product. QA-thresholded valid total-AOD "
        "pixels below threshold are assigned dust AOD=0."
    )
    ds.li_ginoux_definition = (
        "Li-Ginoux raw dust AOD uses FMF=0.085*AE^2+0.336*AE+0.051, "
        "dust AOD=(1-FMF)*AOD550, and SSA412<SSA470. The recommended "
        "dust_aod_li_ginoux field additionally applies FMF<=0.7, approximately AE<=1.42."
    )
    ds.excluded_experimental_products = (
        "DB-type-aware, suspect-veto, and high-AOD-rescue branches are excluded from this production freeze."
    )
    ds.averaging_rule = (
        "Daily means use all valid total-AOD L2 pixels in each grid cell. Monthly means are simple "
        "averages of daily grid-cell means with missing days ignored."
        if is_monthly
        else "Daily means use all valid total-AOD L2 pixels in each grid cell. Invalid total AOD is discarded."
    )
    ds.history = "Created by build_l2_daily_monthly_0p5deg_production.py"


def write_daily_output(output_path: Path, dates: list[datetime], variables: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_path, "w") as ds:
        ds.createDimension("time", len(dates))
        ds.createDimension("lat", NY)
        ds.createDimension("lon", NX)

        base = datetime(dates[0].year, 1, 1)
        time_var = ds.createVariable("time", "i4", ("time",))
        time_var[:] = np.array([(d - base).days for d in dates], dtype=np.int32)
        time_var.units = f"days since {base:%Y-%m-%d} 00:00:00"
        ds.createVariable("year", "i4", ("time",))[:] = np.array([d.year for d in dates], dtype=np.int32)
        ds.createVariable("month", "i4", ("time",))[:] = np.array([d.month for d in dates], dtype=np.int32)
        ds.createVariable("day", "i4", ("time",))[:] = np.array([d.day for d in dates], dtype=np.int32)
        ds.createVariable("lat", "f4", ("lat",), zlib=True, complevel=4)[:] = LAT_CENTERS
        ds.createVariable("lon", "f4", ("lon",), zlib=True, complevel=4)[:] = LON_CENTERS

        for name, data in variables.items():
            dtype = "i4" if name == "n_pixels" else "f4"
            fill = -1 if name == "n_pixels" else np.float32(np.nan)
            var = ds.createVariable(name, dtype, ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=fill)
            var[:] = data
            set_var_attrs(name, var)
        write_common_attrs(ds, is_monthly=False)


def write_monthly_output(output_path: Path, year: int, daily_vars: dict[str, np.ndarray], dates: list[datetime]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    months = np.arange(1, 13, dtype=np.int32)
    monthly_vars: dict[str, np.ndarray] = {}
    for name, data in daily_vars.items():
        if name == "n_pixels":
            continue
        field = np.full((12, NY, NX), np.nan, dtype=np.float32)
        valid_days = np.zeros((12, NY, NX), dtype=np.int16)
        for month in months:
            sel = [i for i, d in enumerate(dates) if d.month == int(month)]
            if not sel:
                continue
            field[month - 1], valid_days[month - 1] = nanmean_time(data[np.array(sel)])
        monthly_vars[name] = field
        monthly_vars[f"{name}_valid_days"] = valid_days

    with Dataset(output_path, "w") as ds:
        ds.createDimension("month", 12)
        ds.createDimension("lat", NY)
        ds.createDimension("lon", NX)
        ds.createVariable("year", "i4").assignValue(year)
        ds.createVariable("month", "i4", ("month",))[:] = months
        ds.createVariable("lat", "f4", ("lat",), zlib=True, complevel=4)[:] = LAT_CENTERS
        ds.createVariable("lon", "f4", ("lon",), zlib=True, complevel=4)[:] = LON_CENTERS

        for name, data in monthly_vars.items():
            is_count = name.endswith("_valid_days")
            dtype = "i2" if is_count else "f4"
            fill = np.int16(-1) if is_count else np.float32(np.nan)
            var = ds.createVariable(name, dtype, ("month", "lat", "lon"), zlib=True, complevel=4, fill_value=fill)
            var[:] = data
            if is_count:
                var.long_name = "number of daily means contributing to monthly mean"
            else:
                set_var_attrs(name, var)
        write_common_attrs(ds, is_monthly=True)


def main() -> int:
    args = parse_args()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime(args.year, 1, 1)
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime(args.year, 12, 31)
    if end_dt < start_dt:
        raise ValueError("--end-date must be on or after --start-date")

    day_dirs: list[tuple[datetime, Path]] = []
    current = start_dt
    while current <= end_dt:
        day_dir = args.input_root / current.strftime("%Y-%m-%d")
        if not day_dir.exists():
            raise FileNotFoundError(f"Missing day directory: {day_dir}")
        day_dirs.append((current, day_dir))
        current += timedelta(days=1)

    daily_stack: dict[str, list[np.ndarray]] = {
        "total_aod": [],
        "dust_aod_li_ginoux": [],
        "dust_aod_li_ginoux_raw": [],
        "dust_aod_two_stage_qa1": [],
        "dust_aod_two_stage_qa2": [],
        "dust_aod_two_stage_qa3": [],
        "n_pixels": [],
    }
    dates: list[datetime] = []
    for dt, day_dir in day_dirs:
        print(f"[DAY] aggregating {dt:%Y-%m-%d}", flush=True)
        agg = aggregate_day(day_dir)
        dates.append(dt)
        for name in daily_stack:
            daily_stack[name].append(agg[name])

    daily_vars = {
        name: np.stack(arrs).astype(np.int32 if name == "n_pixels" else np.float32)
        for name, arrs in daily_stack.items()
    }
    write_daily_output(args.daily_output, dates, daily_vars)
    write_monthly_output(args.monthly_output, args.year, daily_vars, dates)
    print(f"Wrote {args.daily_output}")
    print(f"Wrote {args.monthly_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
