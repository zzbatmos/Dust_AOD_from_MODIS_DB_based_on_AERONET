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
    parser = argparse.ArgumentParser(description="Plot a global DAOD map from a NetCDF file.")
    parser.add_argument("--input", type=Path, required=True, help="Input NetCDF with dust_aod, lat, lon.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="inferno")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = xr.open_dataset(args.input)
    dust = np.asarray(ds["dust_aod"].values, dtype=float)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)

    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = ax.pcolormesh(
        lon,
        lat,
        dust,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    ax.coastlines(linewidth=0.6)
    ax.set_global()
    ax.set_title(args.title)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.88)
    cbar.set_label("Dust AOD at 550 nm")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    ds.close()


if __name__ == "__main__":
    main()
