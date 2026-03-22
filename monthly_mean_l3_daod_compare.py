#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute monthly mean DAOD and plot side-by-side maps.")
    parser.add_argument("--li-dir", type=Path, required=True)
    parser.add_argument("--xgb-dir", type=Path, required=True)
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format.")
    parser.add_argument("--short-name", default="MOD08_D3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vmax", type=float, default=None)
    return parser.parse_args()


def mean_from_daily(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_stack = []
    lat = lon = None
    for path in files:
        ds = xr.open_dataset(path)
        if lat is None:
            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
        data_stack.append(np.asarray(ds["dust_aod"].values, dtype=float))
        ds.close()
    stacked = np.stack(data_stack, axis=0)
    mean = np.nanmean(stacked, axis=0)
    return mean, lat, lon


def plot_panel(lat: np.ndarray, lon: np.ndarray, data: np.ndarray, title: str, ax, vmin: float, vmax: float) -> None:
    ax.set_global()
    mesh = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="YlOrBr",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines(linewidth=0.7)
    ax.set_title(title)
    return mesh


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    month_prefix = args.month + "-"
    li_files = sorted(args.li_dir.glob(f"{args.short_name}_DustAOD_LiGinoux_{month_prefix}*.nc"))
    xgb_files = sorted(args.xgb_dir.glob(f"{args.short_name}_DustAOD_XGB_{month_prefix}*.nc"))
    if not li_files:
        raise FileNotFoundError(f"No Li-Ginoux daily files found for {args.month} in {args.li_dir}")
    if not xgb_files:
        raise FileNotFoundError(f"No XGBoost daily files found for {args.month} in {args.xgb_dir}")

    li_mean, lat, lon = mean_from_daily(li_files)
    xgb_mean, _, _ = mean_from_daily(xgb_files)

    out_li = args.output_dir / f"{args.short_name}_DustAOD_LiGinoux_monthly_mean_{args.month}.nc"
    out_xgb = args.output_dir / f"{args.short_name}_DustAOD_XGB_monthly_mean_{args.month}.nc"
    xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), li_mean.astype("float32")),
            lat=(("y", "x"), lat.astype("float32")),
            lon=(("y", "x"), lon.astype("float32")),
        ),
        attrs={"month": args.month, "method": "Li-Ginoux", "daily_files": len(li_files)},
    ).to_netcdf(out_li)
    xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), xgb_mean.astype("float32")),
            lat=(("y", "x"), lat.astype("float32")),
            lon=(("y", "x"), lon.astype("float32")),
        ),
        attrs={"month": args.month, "method": "XGBoost", "daily_files": len(xgb_files)},
    ).to_netcdf(out_xgb)

    combined = np.concatenate([li_mean[np.isfinite(li_mean)], xgb_mean[np.isfinite(xgb_mean)]])
    vmin = 0.0
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(combined, 99))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.6),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    mesh = plot_panel(lat, lon, li_mean, f"Li-Ginoux {args.month}", axes[0], vmin, vmax)
    plot_panel(lat, lon, xgb_mean, f"XGBoost {args.month}", axes[1], vmin, vmax)
    cbar = fig.colorbar(mesh, ax=axes, location="right", pad=0.02, shrink=0.68, fraction=0.035)
    cbar.set_label("Monthly mean dust AOD at 550 nm")
    fig.suptitle(f"{args.short_name} monthly mean dust AOD comparison for {args.month}", y=0.98)
    fig.savefig(args.output_dir / f"{args.short_name}_DustAOD_monthly_mean_compare_{args.month}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
