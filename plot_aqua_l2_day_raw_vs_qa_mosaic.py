#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot Aqua L2 day global mosaics for raw and QA-gated dust AOD methods."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=here / "aqua_l2_2017-04-17_three_methods_noplot" / "granules",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "aqua_l2_2017-04-17_three_methods_noplot",
    )
    parser.add_argument("--date-label", default="2017-04-17")
    return parser.parse_args()


def enforce_semantics(dust_aod: np.ndarray, total_aod: np.ndarray) -> np.ndarray:
    total_aod = np.asarray(total_aod, dtype=float)
    dust_aod = np.asarray(dust_aod, dtype=float).copy()
    total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
    dust_aod[~np.isfinite(total_aod)] = np.nan
    dust_aod[total_valid & ~np.isfinite(dust_aod)] = 0.0
    return dust_aod


def collect_points(input_dir: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path) as d:
            lat = np.asarray(d["lat"], dtype=float)
            lon = np.asarray(d["lon"], dtype=float)
            aod550 = np.asarray(d["aod550"], dtype=float)
            if variable == "db_type_raw_dust_aod":
                val = enforce_semantics(np.asarray(d["db_type_fraction"], dtype=float) * aod550, aod550)
            else:
                val = np.asarray(d[variable], dtype=float)
        lon = ((lon + 180.0) % 360.0) - 180.0
        mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
        if not np.any(mask):
            continue
        lats.append(lat[mask].ravel())
        lons.append(lon[mask].ravel())
        vals.append(val[mask].ravel())

    if not vals:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(lats), np.concatenate(lons), np.concatenate(vals)


def plot_panel(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    val: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
):
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle=":")
    if val.size == 0:
        ax.set_title(f"{title} (no valid data)")
        return None
    sc = ax.scatter(
        lon,
        lat,
        c=val,
        s=0.8,
        cmap="YlOrBr",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.set_title(title)
    return sc


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        ("li_dust_aod", "Li-Ginoux"),
        ("two_stage_dust_aod", "Two-stage"),
        ("db_type_raw_dust_aod", "DB-type raw"),
        ("db_type_qa2_dust_aod", "DB-type QA>=2"),
    ]
    datasets = []
    all_vals: list[np.ndarray] = []
    for var, label in variables:
        lat, lon, val = collect_points(args.input_dir, var)
        datasets.append((label, lat, lon, val))
        if val.size:
            all_vals.append(val)

    if not all_vals:
        raise RuntimeError("No valid data found in the input NPZ files.")

    merged = np.concatenate(all_vals)
    vmax = float(np.nanpercentile(merged, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(merged))
    vmin = 0.0

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8.5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    mappable = None
    for ax, (label, lat, lon, val) in zip(axes.ravel(), datasets):
        maybe = plot_panel(ax, lon, lat, val, f"{label} DAOD", vmin, vmax)
        if mappable is None and maybe is not None:
            mappable = maybe

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.82, pad=0.03)
        cbar.set_label("Dust AOD at 550 nm")
    fig.suptitle(f"Aqua MYD04_L2 raw and QA-gated dust AOD mosaics, {args.date_label}", y=1.02)
    out = args.output_dir / f"MYD04_L2_{args.date_label}_raw_vs_qa_mosaic.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
