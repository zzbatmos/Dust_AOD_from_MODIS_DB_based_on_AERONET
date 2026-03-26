#!/usr/bin/env python3

from __future__ import annotations

import argparse
import calendar
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three L3 dust-AOD methods over a full year."
    )
    parser.add_argument("--li-dir", type=Path, required=True)
    parser.add_argument("--xgb-dir", type=Path, required=True)
    parser.add_argument("--two-stage-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--short-name", default="MOD08_D3")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_daily_stack(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = []
    lat = lon = None
    for path in files:
        ds = xr.open_dataset(path)
        if lat is None:
            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
        stack.append(np.asarray(ds["dust_aod"].values, dtype=float))
        ds.close()
    return np.stack(stack, axis=0), lat, lon


def nanmean_no_warning(stack: np.ndarray, axis: int) -> np.ndarray:
    valid_count = np.sum(np.isfinite(stack), axis=axis)
    total = np.nansum(stack, axis=axis)
    out = np.full_like(total, np.nan, dtype=float)
    np.divide(total, valid_count, out=out, where=valid_count > 0)
    return out


def files_for_year(directory: Path, prefix: str, year: int) -> list[Path]:
    return sorted(directory.glob(f"{prefix}_{year:04d}-*.nc"))


def monthly_global_means(stack: np.ndarray, files: list[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    by_month: dict[int, list[np.ndarray]] = {month: [] for month in range(1, 13)}
    for arr, path in zip(stack, files):
        month = int(path.stem.split("_")[-1].split("-")[1])
        by_month[month].append(arr)
    for month in range(1, 13):
        if not by_month[month]:
            rows.append({"month": month, "mean": np.nan})
            continue
        month_stack = np.stack(by_month[month], axis=0)
        rows.append({"month": month, "mean": float(np.nanmean(month_stack))})
    return rows


def save_annual_netcdf(path: Path, data: np.ndarray, lat: np.ndarray, lon: np.ndarray, attrs: dict) -> None:
    xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), data.astype("float32")),
            lat=(("y", "x"), lat.astype("float32")),
            lon=(("y", "x"), lon.astype("float32")),
        ),
        attrs=attrs,
    ).to_netcdf(path)


def add_geo(ax) -> None:
    ax.set_global()
    ax.coastlines(linewidth=0.7)
    ax.gridlines(
        draw_labels=False,
        xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-90, 91, 30),
        linewidth=0.4,
        color="0.5",
        alpha=0.5,
        linestyle="--",
    )


def plot_map(ax, lon: np.ndarray, lat: np.ndarray, data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    mesh = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    add_geo(ax)
    ax.set_title(title)
    return mesh


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    li_prefix = f"{args.short_name}_DustAOD_LiGinoux"
    xgb_prefix = f"{args.short_name}_DustAOD_XGB"
    two_stage_prefix = f"{args.short_name}_DustAOD_TwoStage"

    li_files = files_for_year(args.li_dir, li_prefix, args.year)
    xgb_files = files_for_year(args.xgb_dir, xgb_prefix, args.year)
    two_stage_files = files_for_year(args.two_stage_dir, two_stage_prefix, args.year)
    if not li_files:
        raise FileNotFoundError(f"No Li-Ginoux daily files found for {args.year}")
    if not xgb_files:
        raise FileNotFoundError(f"No XGBoost daily files found for {args.year}")
    if not two_stage_files:
        raise FileNotFoundError(f"No two-stage daily files found for {args.year}")

    li_stack, lat, lon = load_daily_stack(li_files)
    xgb_stack, _, _ = load_daily_stack(xgb_files)
    two_stage_stack, _, _ = load_daily_stack(two_stage_files)

    li_annual = nanmean_no_warning(li_stack, axis=0)
    xgb_annual = nanmean_no_warning(xgb_stack, axis=0)
    two_stage_annual = nanmean_no_warning(two_stage_stack, axis=0)

    save_annual_netcdf(
        args.output_dir / f"{args.short_name}_DustAOD_LiGinoux_annual_mean_{args.year}.nc",
        li_annual,
        lat,
        lon,
        {"year": args.year, "method": "Li-Ginoux", "daily_files": len(li_files)},
    )
    save_annual_netcdf(
        args.output_dir / f"{args.short_name}_DustAOD_XGB_annual_mean_{args.year}.nc",
        xgb_annual,
        lat,
        lon,
        {"year": args.year, "method": "XGBoost", "daily_files": len(xgb_files)},
    )
    save_annual_netcdf(
        args.output_dir / f"{args.short_name}_DustAOD_TwoStage_annual_mean_{args.year}.nc",
        two_stage_annual,
        lat,
        lon,
        {"year": args.year, "method": "TwoStage", "daily_files": len(two_stage_files)},
    )

    combined = np.concatenate(
        [
            li_annual[np.isfinite(li_annual)],
            xgb_annual[np.isfinite(xgb_annual)],
            two_stage_annual[np.isfinite(two_stage_annual)],
        ]
    )
    vmax = float(np.nanpercentile(combined, 99))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    mesh = plot_map(axes[0], lon, lat, li_annual, f"Li-Ginoux {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[1], lon, lat, xgb_annual, f"Direct XGBoost {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[2], lon, lat, two_stage_annual, f"Two-stage XGBoost {args.year}", "YlOrBr", 0.0, vmax)
    cbar = fig.colorbar(mesh, ax=axes, location="right", pad=0.02, shrink=0.72, fraction=0.03)
    cbar.set_label("Annual mean dust AOD at 550 nm")
    fig.suptitle(f"{args.short_name} annual mean dust AOD comparison for {args.year}", y=0.98)
    fig.savefig(args.output_dir / f"{args.short_name}_DustAOD_annual_mean_compare_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    diff1 = two_stage_annual - li_annual
    diff2 = two_stage_annual - xgb_annual
    diff_abs = np.concatenate([np.abs(diff1[np.isfinite(diff1)]), np.abs(diff2[np.isfinite(diff2)])])
    dv = float(np.nanpercentile(diff_abs, 99))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    mesh = plot_map(axes[0], lon, lat, diff1, f"Two-stage minus Li-Ginoux {args.year}", "RdBu_r", -dv, dv)
    plot_map(axes[1], lon, lat, diff2, f"Two-stage minus direct XGBoost {args.year}", "RdBu_r", -dv, dv)
    cbar = fig.colorbar(mesh, ax=axes, location="right", pad=0.02, shrink=0.72, fraction=0.04)
    cbar.set_label("Annual mean DAOD difference")
    fig.savefig(args.output_dir / f"{args.short_name}_DustAOD_annual_mean_difference_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    li_monthly = pd.DataFrame(monthly_global_means(li_stack, li_files)).rename(columns={"mean": "li_ginoux"})
    xgb_monthly = pd.DataFrame(monthly_global_means(xgb_stack, xgb_files)).rename(columns={"mean": "xgboost"})
    two_stage_monthly = pd.DataFrame(monthly_global_means(two_stage_stack, two_stage_files)).rename(columns={"mean": "two_stage"})
    monthly = li_monthly.merge(xgb_monthly, on="month").merge(two_stage_monthly, on="month")
    monthly["month_name"] = [calendar.month_abbr[m] for m in monthly["month"]]
    monthly.to_csv(args.output_dir / f"{args.short_name}_DustAOD_monthly_global_mean_{args.year}.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.plot(monthly["month"], monthly["li_ginoux"], marker="o", linewidth=2.0, label="Li-Ginoux")
    ax.plot(monthly["month"], monthly["xgboost"], marker="o", linewidth=2.0, label="Direct XGBoost")
    ax.plot(monthly["month"], monthly["two_stage"], marker="o", linewidth=2.0, label="Two-stage XGBoost")
    ax.set_xticks(monthly["month"], monthly["month_name"])
    ax.set_ylabel("Global mean dust AOD at 550 nm")
    ax.set_title(f"{args.short_name} monthly global mean dust AOD in {args.year}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / f"{args.short_name}_DustAOD_monthly_global_mean_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {"method": "Li-Ginoux", "n_daily_files": len(li_files), "annual_global_mean": float(np.nanmean(li_annual))},
            {"method": "Direct XGBoost", "n_daily_files": len(xgb_files), "annual_global_mean": float(np.nanmean(xgb_annual))},
            {"method": "Two-stage XGBoost", "n_daily_files": len(two_stage_files), "annual_global_mean": float(np.nanmean(two_stage_annual))},
        ]
    )
    summary.to_csv(args.output_dir / f"{args.short_name}_DustAOD_annual_summary_{args.year}.csv", index=False)


if __name__ == "__main__":
    main()
