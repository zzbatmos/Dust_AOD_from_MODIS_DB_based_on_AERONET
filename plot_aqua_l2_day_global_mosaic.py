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
        description="Plot global mosaics from saved Aqua L2 day no-plot NPZ outputs."
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
    parser.add_argument(
        "--date-label",
        default="2017-04-17",
    )
    return parser.parse_args()


def collect_points(input_dir: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path) as data:
            lat = np.asarray(data["lat"], dtype=float)
            lon = np.asarray(data["lon"], dtype=float)
            val = np.asarray(data[variable], dtype=float)
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
) -> None:
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle=":")
    if val.size == 0:
        ax.set_title(f"{title} (no valid data)")
        return
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
        ("db_type_qa2_dust_aod", "DB-type QA>=2"),
    ]
    collected: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    all_vals: list[np.ndarray] = []
    for var, label in variables:
        lat, lon, val = collect_points(args.input_dir, var)
        collected.append((var, label, lat, lon, val))
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
        1,
        3,
        figsize=(18, 5.8),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    mappable = None
    for ax, (_, label, lat, lon, val) in zip(axes, collected):
        mappable = plot_panel(ax, lon, lat, val, f"{label} DAOD", vmin, vmax)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.82, pad=0.03)
        cbar.set_label("Dust AOD at 550 nm")
    fig.suptitle(f"Aqua MYD04_L2 dust AOD global mosaic, {args.date_label}", y=1.02)
    out = args.output_dir / f"MYD04_L2_{args.date_label}_global_mosaic_three_methods.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
