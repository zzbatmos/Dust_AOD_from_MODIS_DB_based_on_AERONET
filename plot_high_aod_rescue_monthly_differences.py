#!/usr/bin/env python3
"""Plot monthly high-AOD rescue differences against Li-Ginoux and two-stage QA2."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


DEFAULT_MONTHLY_FILES = [
    Path(
        "two_stage_high_aod_dust_rescue_test/production_aug2017_l3/"
        "MYD04_L2_high_aod_rescue_monthly_0p5deg_2017-08.nc"
    ),
    Path(
        "two_stage_high_aod_dust_rescue_test/production_dec2017_l3/"
        "MYD04_L2_high_aod_rescue_monthly_0p5deg_2017-12.nc"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-files",
        type=Path,
        nargs="*",
        default=DEFAULT_MONTHLY_FILES,
        help="Monthly 0.5-degree NetCDF files with high-AOD rescue output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("two_stage_high_aod_dust_rescue_test/monthly_difference_figures"),
    )
    return parser.parse_args()


def as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def infer_month_index(path: Path) -> int:
    match = re.search(r"2017-(\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer month from file name: {path.name}")
    return int(match.group(1)) - 1


def finite_percentile(arrays: list[np.ndarray], percentile: float, minimum: float) -> float:
    vals = []
    for arr in arrays:
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return minimum
    merged = np.concatenate(vals)
    return float(max(np.nanpercentile(merged, percentile), minimum))


def weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, None]
    valid = np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(field * weights) / np.nansum(weights * valid))


def add_map(ax, lon2d: np.ndarray, lat2d: np.ndarray, data: np.ndarray, *, title: str, cmap: str, vmin: float, vmax: float):
    if HAS_CARTOPY:
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        mesh = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title)
    return mesh


def plot_month(path: Path, output_dir: Path) -> dict[str, object]:
    month_index = infer_month_index(path)
    month = month_index + 1
    year = 2017

    with Dataset(path) as ds:
        lat = as_float_array(ds.variables["lat"][:])
        lon = as_float_array(ds.variables["lon"][:])
        li = as_float_array(ds.variables["dust_aod_li_ginoux"][month_index][:])
        two = as_float_array(ds.variables["dust_aod_two_stage_qa2"][month_index][:])
        rescue = as_float_array(ds.variables["dust_aod_two_stage_high_aod_rescue_qa2"][month_index][:])
        rescued_fraction = as_float_array(ds.variables["high_aod_rescued_pixel_fraction"][month_index][:])

    rescue_minus_li = rescue - li
    rescue_minus_two = rescue - two
    lon2d, lat2d = np.meshgrid(lon, lat)

    daod_vmax = finite_percentile([li, two, rescue], 98.5, 0.2)
    diff_abs = finite_percentile([np.abs(rescue_minus_li), np.abs(rescue_minus_two)], 99.0, 0.05)
    frac_vmax = finite_percentile([rescued_fraction], 99.5, 0.01)

    panels = [
        ("Li-Ginoux DAOD", li, "plasma", 0.0, daod_vmax),
        ("Two-stage QA2 DAOD", two, "plasma", 0.0, daod_vmax),
        ("QA2 + high-AOD rescue", rescue, "plasma", 0.0, daod_vmax),
        ("Rescue - Li-Ginoux", rescue_minus_li, "RdBu_r", -diff_abs, diff_abs),
        ("Rescue - two-stage QA2", rescue_minus_two, "magma", 0.0, diff_abs),
        ("Rescued-pixel fraction", rescued_fraction, "viridis", 0.0, frac_vmax),
    ]

    subplot_kw = {"projection": ccrs.Robinson()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), subplot_kw=subplot_kw)
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        mesh = add_map(ax, lon2d, lat2d, data, title=title, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)

    fig.suptitle(f"MYD04_L2 {year}-{month:02d}: high-AOD rescue monthly global comparison", fontsize=15)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"MYD04_L2_{year}-{month:02d}_high_aod_rescue_global_differences.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "year": year,
        "month": month,
        "input_file": str(path),
        "figure": str(output_path),
        "global_mean_li_ginoux": weighted_mean(li, lat),
        "global_mean_two_stage_qa2": weighted_mean(two, lat),
        "global_mean_rescue_qa2": weighted_mean(rescue, lat),
        "global_mean_rescue_minus_li": weighted_mean(rescue_minus_li, lat),
        "global_mean_rescue_minus_two_stage": weighted_mean(rescue_minus_two, lat),
        "global_mean_rescued_pixel_fraction": weighted_mean(rescued_fraction, lat),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.monthly_files:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(plot_month(path, args.output_dir))

    summary_path = args.output_dir / "high_aod_rescue_monthly_difference_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['year']}-{row['month']:02d}: "
            f"rescue-li={row['global_mean_rescue_minus_li']:.5f}, "
            f"rescue-two={row['global_mean_rescue_minus_two_stage']:.5f}, "
            f"figure={row['figure']}"
        )
    print(summary_path)


if __name__ == "__main__":
    main()
