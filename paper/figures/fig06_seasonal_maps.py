"""Native-grid seasonal maps on the frozen common CALIOP comparison support."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


REGIONS = ["Sahara", "Sahel", "Middle East", "Central Asia", "East Asian deserts",
           "SW North America", "India", "East China", "Amazonia_Cerrado", "southern_Africa",
           "boreal_North_America", "Siberia", "mainland_SE_Asia"]


def draw_map(ax, frame, value, difference=False):
    lat, lon = np.sort(frame.lat.unique()), np.sort(frame.lon.unique())
    rows = frame.sort_values(["lat_index", "lon_index"])
    shape = (len(lat), len(lon))
    if len(rows) != np.prod(shape) or rows.duplicated(["lat_index", "lon_index"]).any():
        raise ValueError("Incomplete or duplicated native grid")
    x = np.r_[rows.lon_lower.unique(), rows.lon_upper.max()]
    y = np.r_[rows.lat_lower.unique(), rows.lat_upper.max()]
    z = rows[value].to_numpy().reshape(shape)
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    levels = np.linspace(-.15, .15, 11) if difference else np.linspace(0, .5, 11)
    cmap = style.CMAP_DIFF if difference else style.CMAP_DAOD
    if difference:
        missing = (rows.in_domain & rows.has_land & ~rows.supported).to_numpy().reshape(shape)
        ax.pcolor(x, y, np.ma.masked_where(~missing, np.ones(shape)),
                  cmap=ListedColormap(["#E0E0E0"]), hatch="////", edgecolor="#BDBDBD",
                  linewidth=0, transform=ccrs.PlateCarree(), rasterized=True)
    else:
        ax.pcolormesh(x, y, np.ma.masked_where(~(z == 0), np.zeros(shape)),
                      cmap=ListedColormap([style.ZERO_GREY]), transform=ccrs.PlateCarree(),
                      rasterized=True)
        z = np.where(z == 0, np.nan, z)
    mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(z), cmap=cmap,
                         norm=BoundaryNorm(levels, cmap.N), transform=ccrs.PlateCarree(),
                         rasterized=True)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", edgecolor="none", zorder=3)
    ax.coastlines(resolution="110m", linewidth=.4, color="#4D4D4D", zorder=4)
    grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 61, 30),
                       linewidth=.2, color="#BDBDBD", zorder=5)
    grid.top_labels = grid.right_labels = grid.geo_labels = grid.left_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 6.5}
    grid.rotate_labels = False
    ymin, ymax = ax.get_ylim()
    for latitude, label in ((-30, "30\N{DEGREE SIGN}S"), (0, "0\N{DEGREE SIGN}"), (30, "30\N{DEGREE SIGN}N")):
        _, projected_y = ax.projection.transform_point(0, latitude, ccrs.PlateCarree())
        ax.text(-.025, (projected_y - ymin) / (ymax - ymin), label, transform=ax.transAxes,
                ha="right", va="center", fontsize=6.5)
    return mesh


def make(registry_path, output):
    registry = Registry.read(registry_path)
    data = registry.table("derived/seasonal_maps")
    boxes = registry.table("derived/seasonal_region_boxes")
    boxes = boxes.loc[boxes.region.isin(REGIONS)]
    if set(boxes.region) != set(REGIONS):
        raise ValueError("Missing prescribed regional boundaries")
    fig = plt.figure(figsize=(style.DOUBLE, 170 * style.MM))
    for row, season in enumerate(("DJF", "MAM", "JJA", "SON")):
        q = data.loc[data.period.eq(season)]
        for col, value in enumerate(("ev", "difference")):
            ax = fig.add_axes([.06 + col * .50, .76 - row * .21, .40, .18],
                              projection=ccrs.Robinson())
            mesh = draw_map(ax, q, value, difference=col == 1)
            if row == 0:
                ax.set_title("Expected-value DAOD" if col == 0 else "Expected value minus CALIOP", pad=11)
            ax.text(.02, .95, season, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.5), zorder=10)
            style.panel_label(ax, "abcdefgh"[row * 2 + col], x=-.12, y=1.02)
            if row == 2 and col == 0:
                for _, box in boxes.iterrows():
                    ax.add_patch(Rectangle((box.lon_lower, box.lat_lower),
                                          box.lon_upper - box.lon_lower, box.lat_upper - box.lat_lower,
                                          fill=False, edgecolor="black", linewidth=.4,
                                          transform=ccrs.PlateCarree(), zorder=6))
            if row == 3:
                cax = fig.add_axes([.15 + col * .50, .065, .22, 2.5 / 170])
                bar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                                   extend="both" if col else "max")
                bar.set_ticks([-.15, 0, .15] if col else [0, .25, .5])
                bar.set_label("DAOD difference (550 nm)" if col else "DAOD (550 nm)")
    evidence = [{"table": name, "source_file": registry.payload["tables"][name]["source_file"],
                 "source_sha256": registry.payload["tables"][name]["source_sha256"],
                 "derivation": registry.payload["tables"][name]["derivation"]}
                for name in ("derived/seasonal_maps", "derived/seasonal_region_boxes", "derived/seasonal_map_metrics")]
    style.export(fig, output, "Fig06_seasonal_maps",
                 pd.concat([data.assign(record="grid"), boxes.assign(record="region_boundary")], ignore_index=True),
                 registry_path, evidence, __file__, raster=True)
    (Path(output) / "Fig06_caption.md").write_text(
        "Seasonal expected-value DAOD and differences from CALIOP on common native-grid land support. "
        "DJF includes December 2016. Zero DAOD is grey; missing is white, with missing land hatched "
        "in difference panels. JJA outlines: Sahara, Sahel, Middle East, Central Asia, East Asian deserts, "
        "SW North America, India, East China, Amazonia/Cerrado, southern Africa, boreal North America, "
        "Siberia, and mainland SE Asia. All three months must be valid; seasons are equal-month means.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
