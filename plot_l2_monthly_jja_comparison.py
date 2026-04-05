from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


HERE = Path(__file__).resolve().parent
MONTHLY_FILE = (
    HERE
    / "aqua_l2_2017_1deg_monthly"
    / "MYD04_L2_three_methods_monthly_1deg_2017.nc"
)
OUTPUT_FILE = (
    HERE
    / "aqua_l2_2017_1deg_monthly"
    / "MYD04_L2_three_methods_monthly_1deg_2017_JJA_comparison.png"
)


def main() -> None:
    ds = xr.open_dataset(MONTHLY_FILE)

    methods = [
        ("dust_aod_li_ginoux", "Li-Ginoux"),
        ("dust_aod_two_stage_qa2", "Two-stage QA>=2"),
        ("dust_aod_dbtype_qa2", "DB-type-aware QA>=2"),
    ]
    months = [(6, "June"), (7, "July"), (8, "August")]

    arrays = [
        ds[var].sel(month=month).values
        for var, _ in methods
        for month, _ in months
    ]
    vmax = np.nanpercentile(np.concatenate([a[np.isfinite(a)] for a in arrays]), 99)
    vmax = float(max(vmax, 0.05))

    lon = ds["lon"].values
    lat = ds["lat"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(17, 11),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    mesh = None
    for row, (var, method_label) in enumerate(methods):
        for col, (month, month_label) in enumerate(months):
            ax = axes[row, col]
            data = ds[var].sel(month=month).values
            mesh = ax.pcolormesh(
                lon2d,
                lat2d,
                data,
                transform=data_crs,
                cmap="YlOrBr",
                vmin=0.0,
                vmax=vmax,
                shading="auto",
            )
            ax.set_global()
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.25)
            ax.gridlines(
                crs=data_crs,
                linewidth=0.25,
                color="gray",
                alpha=0.4,
                linestyle="--",
                draw_labels=False,
            )
            ax.set_title(f"{method_label} | {month_label}", fontsize=11)

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="vertical",
        shrink=0.78,
        pad=0.03,
    )
    cbar.set_label("Dust AOD at 550 nm")

    fig.suptitle("Aqua MYD04_L2 2017 Monthly Mean Dust AOD Comparison (JJA)", fontsize=15)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
