#!/usr/bin/env python3
"""Compare Song et al. monthly DAOD with our Aqua monthly products."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--song",
        type=Path,
        default=Path("Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc"),
    )
    parser.add_argument(
        "--xgb",
        type=Path,
        default=Path("Song2021_data/MYD08_D3_DustAOD_XGB_2003_2019_Monthly_1degX1deg.nc"),
    )
    parser.add_argument(
        "--li",
        type=Path,
        default=Path("Song2021_data/MYD08_D3_DustAOD_LiGinoux_2003_2019_Monthly_1degX1deg.nc"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("song2021_comparison"),
    )
    return parser.parse_args()


def load_monthly(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        out = {
            "year": np.asarray(ds.variables["year"][:], dtype=int),
            "month": np.asarray(ds.variables["month"][:], dtype=int),
            "lat": np.asarray(ds.variables["lat"][:], dtype=np.float32),
            "lon": np.asarray(ds.variables["lon"][:], dtype=np.float32),
            "daod": np.asarray(ds.variables["daod"][:], dtype=np.float32),
            "taod": np.asarray(ds.variables["taod"][:], dtype=np.float32),
        }
        if "n_days" in ds.variables:
            out["n_days"] = np.asarray(ds.variables["n_days"][:], dtype=np.int16)
    if not np.all(np.diff(out["lat"]) > 0):
        order = np.argsort(out["lat"])
        out["lat"] = out["lat"][order]
        out["daod"] = out["daod"][:, :, order, :]
        out["taod"] = out["taod"][:, :, order, :]
        if "n_days" in out:
            out["n_days"] = out["n_days"][:, :, order, :]
    return out


def weighted_spatial_mean(field: np.ndarray, weights_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    arr = np.where(mask_2d[None, None, :, :], field, np.nan)
    valid = np.isfinite(arr)
    w = np.where(valid, weights_2d[None, None, :, :], 0.0)
    numerator = np.nansum(arr * w, axis=(2, 3))
    denominator = np.sum(w, axis=(2, 3))
    return numerator / denominator


def monthly_climatology(field: np.ndarray) -> np.ndarray:
    return np.nanmean(field, axis=0)


def mean_map(field: np.ndarray) -> np.ndarray:
    return np.nanmean(field, axis=(0, 1))


def save_time_series(dates: pd.DatetimeIndex, data: dict[str, np.ndarray], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    for label, series in data.items():
        ax.plot(dates, series, label=label, linewidth=1.6)
    ax.set_ylabel("Global Land Mean Dust AOD")
    ax.set_title("Monthly Mean Dust AOD Over Land (2003-2019)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_monthly_cycle(months: np.ndarray, data: dict[str, np.ndarray], output: Path) -> None:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for label, series in data.items():
        ax.plot(months, series, marker="o", label=label, linewidth=1.8)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Climatological Mean Dust AOD")
    ax.set_title("Seasonal Cycle of Global Land Mean Dust AOD (2003-2019)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_map_panel(lat: np.ndarray, lon: np.ndarray, maps: list[np.ndarray], titles: list[str], cmap: str, output: Path) -> None:
    lon2d, lat2d = np.meshgrid(lon, lat)
    vmax = float(np.nanpercentile(np.stack([m for m in maps]), 99))
    fig, axes = plt.subplots(
        1,
        len(maps),
        figsize=(6.1 * len(maps), 4.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    if len(maps) == 1:
        axes = [axes]
    mappable = None
    for ax, field, title in zip(axes, maps, titles):
        ax.set_global()
        mappable = ax.pcolormesh(
            lon2d,
            lat2d,
            field,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.add_feature(cfeature.LAND, facecolor="0.96", zorder=0)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.35")
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.45,
            color="0.45",
            alpha=0.45,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.x_inline = False
        gl.y_inline = False
        ax.set_title(title)
    cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.72, pad=0.02, fraction=0.032)
    cbar.set_label("Dust AOD")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_difference_panel(lat: np.ndarray, lon: np.ndarray, maps: list[np.ndarray], titles: list[str], output: Path) -> None:
    lon2d, lat2d = np.meshgrid(lon, lat)
    vmax = float(np.nanpercentile(np.abs(np.stack([m for m in maps])), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        1,
        len(maps),
        figsize=(6.1 * len(maps), 4.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    if len(maps) == 1:
        axes = [axes]
    mappable = None
    for ax, field, title in zip(axes, maps, titles):
        ax.set_global()
        mappable = ax.pcolormesh(
            lon2d,
            lat2d,
            field,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            cmap="RdBu_r",
            norm=norm,
        )
        ax.add_feature(cfeature.LAND, facecolor="0.96", zorder=0)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.35")
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.45,
            color="0.45",
            alpha=0.45,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.x_inline = False
        gl.y_inline = False
        ax.set_title(title)
    cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.72, pad=0.02, fraction=0.032)
    cbar.set_label("Dust AOD Difference")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    song = load_monthly(args.song)
    xgb = load_monthly(args.xgb)
    li = load_monthly(args.li)

    land_mask = np.isfinite(xgb["taod"]).any(axis=(0, 1)) | np.isfinite(li["taod"]).any(axis=(0, 1))
    weights = np.cos(np.deg2rad(song["lat"]))[:, None] * np.ones((len(song["lat"]), len(song["lon"])), dtype=np.float32)

    song_daod_land = np.where(land_mask[None, None, :, :], song["daod"], np.nan)
    xgb_daod_land = np.where(land_mask[None, None, :, :], xgb["daod"], np.nan)
    li_daod_land = np.where(land_mask[None, None, :, :], li["daod"], np.nan)
    song_taod_land = np.where(land_mask[None, None, :, :], song["taod"], np.nan)
    xgb_taod_land = np.where(land_mask[None, None, :, :], xgb["taod"], np.nan)
    li_taod_land = np.where(land_mask[None, None, :, :], li["taod"], np.nan)

    dates = pd.to_datetime(
        [f"{year:04d}-{month:02d}-01" for year in song["year"] for month in song["month"]]
    )

    monthly_means = {
        "Song2021": weighted_spatial_mean(song_daod_land, weights, land_mask).reshape(-1),
        "XGBoost": weighted_spatial_mean(xgb_daod_land, weights, land_mask).reshape(-1),
        "Li-Ginoux": weighted_spatial_mean(li_daod_land, weights, land_mask).reshape(-1),
    }
    save_time_series(
        dates,
        monthly_means,
        args.output_dir / "aqua_2003_2019_global_land_monthly_mean_daod_timeseries.png",
    )

    monthly_cycle = {
        "Song2021": np.nanmean(weighted_spatial_mean(song_daod_land, weights, land_mask), axis=0),
        "XGBoost": np.nanmean(weighted_spatial_mean(xgb_daod_land, weights, land_mask), axis=0),
        "Li-Ginoux": np.nanmean(weighted_spatial_mean(li_daod_land, weights, land_mask), axis=0),
    }
    save_monthly_cycle(
        song["month"],
        monthly_cycle,
        args.output_dir / "aqua_2003_2019_global_land_monthly_cycle_daod.png",
    )

    climatology_maps = [
        mean_map(song_daod_land),
        mean_map(xgb_daod_land),
        mean_map(li_daod_land),
    ]
    save_map_panel(
        song["lat"],
        song["lon"],
        climatology_maps,
        ["Song2021 DAOD", "XGBoost DAOD", "Li-Ginoux DAOD"],
        "YlOrBr",
        args.output_dir / "aqua_2003_2019_daod_climatology_maps.png",
    )

    difference_maps = [
        mean_map(xgb_daod_land - song_daod_land),
        mean_map(li_daod_land - song_daod_land),
        mean_map(xgb_daod_land - li_daod_land),
    ]
    save_difference_panel(
        song["lat"],
        song["lon"],
        difference_maps,
        ["XGBoost - Song2021", "Li-Ginoux - Song2021", "XGBoost - Li-Ginoux"],
        args.output_dir / "aqua_2003_2019_daod_difference_maps.png",
    )

    rows = []
    for idx, date in enumerate(dates):
        yi = idx // 12
        mi = idx % 12
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "year": int(song["year"][yi]),
                "month": int(song["month"][mi]),
                "song_daod_land_mean": float(monthly_means["Song2021"][idx]),
                "xgb_daod_land_mean": float(monthly_means["XGBoost"][idx]),
                "li_ginoux_daod_land_mean": float(monthly_means["Li-Ginoux"][idx]),
                "song_taod_land_mean": float(weighted_spatial_mean(song_taod_land, weights, land_mask).reshape(-1)[idx]),
                "xgb_taod_land_mean": float(weighted_spatial_mean(xgb_taod_land, weights, land_mask).reshape(-1)[idx]),
                "li_ginoux_taod_land_mean": float(weighted_spatial_mean(li_taod_land, weights, land_mask).reshape(-1)[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "aqua_2003_2019_global_land_monthly_means.csv", index=False)

    climatology_df = pd.DataFrame(
        {
            "month": song["month"],
            "song_daod_land_climatology": monthly_cycle["Song2021"],
            "xgb_daod_land_climatology": monthly_cycle["XGBoost"],
            "li_ginoux_daod_land_climatology": monthly_cycle["Li-Ginoux"],
        }
    )
    climatology_df.to_csv(args.output_dir / "aqua_2003_2019_global_land_monthly_cycle.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "product": "Song2021",
                "global_land_mean_daod": float(np.nanmean(song_daod_land)),
                "global_land_mean_taod": float(np.nanmean(song_taod_land)),
            },
            {
                "product": "XGBoost",
                "global_land_mean_daod": float(np.nanmean(xgb_daod_land)),
                "global_land_mean_taod": float(np.nanmean(xgb_taod_land)),
            },
            {
                "product": "Li-Ginoux",
                "global_land_mean_daod": float(np.nanmean(li_daod_land)),
                "global_land_mean_taod": float(np.nanmean(li_taod_land)),
            },
        ]
    )
    summary.to_csv(args.output_dir / "aqua_2003_2019_land_summary.csv", index=False)

    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
