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
        description="Plot Aqua L2 day mosaic of DB-type-aware QA>=2 minus raw dust AOD."
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


def collect_points(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path) as d:
            lat = np.asarray(d["lat"], dtype=float)
            lon = np.asarray(d["lon"], dtype=float)
            aod550 = np.asarray(d["aod550"], dtype=float)
            raw = enforce_semantics(np.asarray(d["db_type_fraction"], dtype=float) * aod550, aod550)
            qa2 = np.asarray(d["db_type_qa2_dust_aod"], dtype=float)
            diff = qa2 - raw
        lon = ((lon + 180.0) % 360.0) - 180.0
        mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(diff)
        if not np.any(mask):
            continue
        lats.append(lat[mask].ravel())
        lons.append(lon[mask].ravel())
        vals.append(diff[mask].ravel())

    if not vals:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(lats), np.concatenate(lons), np.concatenate(vals)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon, diff = collect_points(args.input_dir)
    if diff.size == 0:
        raise RuntimeError("No valid QA2-minus-raw difference data found.")

    dv = float(np.nanpercentile(np.abs(diff), 98))
    if not np.isfinite(dv) or dv <= 0:
        dv = float(np.nanmax(np.abs(diff)))

    proj = ccrs.Robinson()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10.5, 5.8),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle=":")
    sc = ax.scatter(
        lon,
        lat,
        c=diff,
        s=0.8,
        cmap="RdBu_r",
        vmin=-dv,
        vmax=dv,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.set_title(f"DB-type-aware QA>=2 minus raw DAOD, {args.date_label}")
    cbar = fig.colorbar(sc, ax=ax, location="right", shrink=0.82, pad=0.03)
    cbar.set_label("Dust AOD difference at 550 nm")
    out = args.output_dir / f"MYD04_L2_{args.date_label}_dbtype_qa2_minus_raw.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
