"""Development-case diagnostics from registered frozen pixels and GIBS context."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from PIL import Image

from paper.figures import figstyle as style
from paper.figures import difference_encoding as signs
from paper.numbers import Registry, digest


CASES = ["Kansas_20250314", "Amazonia_20240922", "NorthAmerica_20170829"]


def native_arrays(rows, geometry):
    shapes = geometry[["swath_rows", "swath_cols"]].drop_duplicates()
    if len(shapes) != 1:
        raise ValueError("Ambiguous swath shape")
    shape = tuple(shapes.iloc[0].astype(int))
    geo = geometry.sort_values("pixel_index")
    if not np.array_equal(geo.pixel_index.to_numpy(), np.arange(np.prod(shape))):
        raise ValueError("Incomplete native geolocation")
    lon, lat = (geo[name].to_numpy().reshape(shape) for name in ("lon", "lat"))
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Cannot fabricate missing swath coordinates")
    if rows.pixel_index.duplicated().any():
        raise ValueError("Duplicate plotted pixel")
    expected = geo.set_index("pixel_index").loc[rows.pixel_index]
    if not (np.array_equal(rows.lat.to_numpy(), expected.lat.to_numpy())
            and np.array_equal(rows.lon.to_numpy(), expected.lon.to_numpy())):
        raise ValueError("Display pixels differ from registered full-swath geolocation")

    def values(name):
        output = np.full(np.prod(shape), np.nan)
        output[rows.pixel_index.to_numpy(int)] = rows[name].to_numpy(float)
        return output.reshape(shape)
    return lon, lat, values


def decorate(ax, bounds, first):
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=.4, color="#4D4D4D", zorder=5)
    xticks = np.arange(np.ceil(bounds[0] / 5) * 5, bounds[1] + .01, 5)
    yticks = np.arange(np.ceil(bounds[2] / 5) * 5, bounds[3] + .01, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{abs(int(x))}\N{DEGREE SIGN}{'W' if x < 0 else 'E'}" if x % 10 == 0 and x < bounds[1] - 2.5 else ""
                        for x in xticks])
    ax.set_yticklabels([f"{abs(int(y))}\N{DEGREE SIGN}{'S' if y < 0 else 'N'}" if first else ""
                        for y in yticks])
    ax.tick_params(length=2, pad=1)


def make(registry_path, image_dir, output):
    registry = Registry.read(registry_path)
    pixels = registry.table("derived/case_display")
    geometry = registry.table("derived/case_geolocation")
    fig = plt.figure(figsize=(style.DOUBLE, 120 * style.MM))
    titles = ["True color", "Dust probability", "QA2 DAOD", "Expected-value\nDAOD", "LG25 DAOD", "Expected value\nminus LG25"]
    fields = [None, "dust_probability", "daod550_qa2", "daod550_ev", "lg25_daod", "ev_minus_lg25"]
    evidence = []
    colorbars = {}
    for r, case in enumerate(CASES):
        q = pixels.loc[pixels.case.eq(case)]
        geo = geometry.loc[geometry.case.eq(case)]
        row = q.iloc[0]
        bounds = [row.zoom_west, row.zoom_east, row.zoom_south, row.zoom_north]
        lon, lat, values = native_arrays(q, geo)
        image_path = image_dir / f"{case}_GIBS_truecolor.png"
        image_meta = json.loads(image_path.with_suffix(".json").read_text())
        if (image_meta["sha256"] != digest(image_path) or image_meta["platform"] != row.platform
                or image_meta["date"] != row.date or image_meta["extent_lonlat"] != bounds):
            raise ValueError("Context image provenance does not match case")
        evidence.append(image_meta)
        for c, field in enumerate(fields):
            ax = fig.add_axes([.05 + .157 * c, [.67, .35, .11][r], .145, .23], projection=ccrs.PlateCarree())
            if r == 0:
                fig.text(.1225 + .157 * c, .95, titles[c], ha="center", va="bottom", fontsize=6.5)
            if c == 0:
                ax.imshow(Image.open(image_path), extent=bounds, origin="upper", transform=ccrs.PlateCarree())
                time = row.granule_start_utc[11:16]
                ax.text(.025, .975, f"{row.date}\n{row.platform} {time} UTC", transform=ax.transAxes,
                        va="top", fontsize=6.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=.5), zorder=6)
            else:
                z = values(field)
                if c == 1:
                    cmap, levels = style.CMAP_P, np.linspace(0, 1, 11)
                elif c == 5:
                    cmap, levels = style.CMAP_DIFF, np.linspace(-.5, .5, 11)
                    missing = values("difference_missing")
                    ax.pcolor(lon, lat, np.ma.masked_where(~(missing == 1), missing),
                              cmap=ListedColormap(["#E0E0E0"]), hatch="////", edgecolor="#BDBDBD",
                              linewidth=0, shading="auto", transform=ccrs.PlateCarree(), rasterized=True)
                else:
                    cmap, levels = style.CMAP_DAOD, np.linspace(0, 1, 11)
                    ax.pcolormesh(lon, lat, np.ma.masked_where(~(z == 0), np.zeros_like(z)),
                                  cmap=ListedColormap([style.ZERO_GREY]), shading="auto",
                                  transform=ccrs.PlateCarree(), rasterized=True)
                    z = np.where(z == 0, np.nan, z)
                mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(z), cmap=cmap,
                                     norm=BoundaryNorm(levels, cmap.N), shading="auto",
                                     transform=ccrs.PlateCarree(), rasterized=True)
                if c == 5:
                    signs.add_positive_stipple(ax, mesh, z)
                ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", edgecolor="none", zorder=3)
                if c == 1:
                    full8 = values("full8_branch")
                    if np.nanmin(full8) < .5 < np.nanmax(full8):
                        contour = ax.contour(lon, lat, full8, levels=[.5], colors="black", linewidths=.4,
                                             transform=ccrs.PlateCarree(), zorder=4)
                        contour.set_path_effects([pe.Stroke(linewidth=1.1, foreground="white"), pe.Normal()])
                colorbars[c] = mesh
            decorate(ax, bounds, c == 0)
            style.panel_label(ax, "abcdefghijklmnopqr"[r * 6 + c], x=0, y=1.015)
    for c, left, width, label, extend in ((1, .235, .085, "Probability", "neither"),
                                         (3, .425, .25, "DAOD (550 nm)", "max"),
                                         (5, .86, .085, "Difference", "both")):
        cax = fig.add_axes([left, .08, width, 2.5 / 120])
        bar = fig.colorbar(colorbars[c], cax=cax, orientation="horizontal", extend=extend)
        bar.set_ticks([-.5, 0, .5] if c == 5 else [0, .5, 1])
        bar.set_label(label)
    fig.legend(handles=signs.sign_handles(), loc="lower right", bbox_to_anchor=(.98, .114),
               ncol=2, fontsize=6.5, handlelength=1.5)
    evidence.append(signs.provenance())
    for name in ("derived/case_display", "derived/case_geolocation"):
        item = registry.payload["tables"][name]
        evidence.append({"table": name, "source_file": item["source_file"],
                         "source_sha256": item["source_sha256"], "derivation": item["derivation"]})
    style.export(fig, output, "Fig08_cases",
                 pd.concat([pixels.assign(record="science_pixel"), geometry.assign(record="geolocation")], ignore_index=True),
                 registry_path, evidence, __file__, raster=True)
    (Path(output) / "Fig08_caption.md").write_text(
        "Development cases: Kansas dust, Amazonia smoke, North American transported smoke (top-bottom). "
        "Columns show daily GIBS true color, probability, QA2 DAOD, expected-value DAOD, unscreened LG25, "
        "and EV-minus-LG25. White-haloed black contours delimit full8 availability. Zero DAOD is grey; "
        "missing is white, diagonally hatched within swaths in difference panels. Dots mark strictly "
        "positive differences, not significance; zero/negative differences are undotted. Daily RGB "
        "mosaics need not match labelled L2 start times. Diagnostic cases, not independent validation.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.image_dir, args.output_dir)
