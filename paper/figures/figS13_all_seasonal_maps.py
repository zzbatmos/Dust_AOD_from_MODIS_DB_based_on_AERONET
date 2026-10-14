"""Registry-only S13a absolute maps and S13b differences on explicit supports.

No direct science-cache reads, interpolation, or online Cartopy resolvers.
Each page has its own FigS13a/FigS13b data, provenance and caption stem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.text import Text
import cartopy.crs as ccrs
from cartopy.io import shapereader
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper.numbers import Registry, digest
from paper.figures import figstyle as style
from paper.figures import difference_encoding as signs

MAP_TABLE = "derived/all_seasonal_maps"
METRIC_TABLE = "derived/all_seasonal_map_metrics"
METHODS = ("E1", "Eselected", "E5", "E7", "E8", "Song", "CALIOP", "E6")
SEASONS = ("DJF", "MAM", "JJA", "SON")
PERIOD_MONTHS = {"DJF": "2016-12|2017-01|2017-02", "MAM": "2017-03|2017-04|2017-05",
                 "JJA": "2017-06|2017-07|2017-08", "SON": "2017-09|2017-10|2017-11"}
HEIGHT_MM = 230
DEFAULT_GEOGRAPHY = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"
LABELS = {"E1": "QA2\n(E1)", "Eselected": "Expected value\n(Eselected)",
          "E5": "DB aerosol type\n(E5)", "E7": "LG25\n(E7)", "E8": "AE refit\n(E8)",
          "Song": "Song et al.\n(2021)", "CALIOP": "CALIOP\nV2", "E6": "LG25 screened*\n(E6)"}


def support_for(estimator):
    return "full_common_E6" if estimator == "E6" else "core_common"


def reconstruct(frame):
    """Require a complete native grid and preserve exact registered science values."""
    dy, dx = CONFIG["grids"]["2x5"]
    lat = np.arange(CONFIG["inputs"]["latitude_bounds"][0] + dy / 2,
                    CONFIG["inputs"]["latitude_bounds"][1], dy)
    lon = np.arange(CONFIG["inputs"]["longitude_bounds"][0] + dx / 2,
                    CONFIG["inputs"]["longitude_bounds"][1], dx)
    shape = (len(lat), len(lon))
    rows = frame.sort_values(["lat_index", "lon_index"])
    iy, ix = np.indices(shape)
    if len(rows) != iy.size or rows.duplicated(["lat_index", "lon_index"]).any():
        raise ValueError("Incomplete or duplicated native grid")
    for column, expected in (("lat_index", iy), ("lon_index", ix), ("lat", lat[iy]), ("lon", lon[ix]),
                              ("lat_lower", lat[iy] - dy / 2), ("lat_upper", lat[iy] + dy / 2),
                              ("lon_lower", lon[ix] - dx / 2), ("lon_upper", lon[ix] + dx / 2)):
        if not np.array_equal(rows[column].to_numpy().reshape(shape), expected):
            raise ValueError("Wrong native grid coordinate/index: " + column)
    for column in ("supported", "in_domain", "has_land"):
        if not rows[column].isin([True, False]).all():
            raise ValueError("Invalid support flag: " + column)
    supported = rows.supported.to_numpy(bool)
    domain = (rows.lat >= -60) & (rows.lat < 60)
    if not np.array_equal(rows.in_domain, domain):
        raise ValueError("Unexpected 60S-60N domain")
    for column in ("prior_joint_months", "modis_months", "mean_only_months"):
        if not rows[column].isin(range(4)).all():
            raise ValueError("Invalid three-month support diagnostic")
    if not np.array_equal(supported, rows.in_domain & rows.prior_joint_months.eq(3)):
        raise ValueError("Support is not the complete prior_joint intersection")
    if ((rows.prior_joint_months > rows.mean_only_months) | (rows.mean_only_months > rows.modis_months)).any():
        raise ValueError("Inconsistent support diagnostic ordering")
    fields = rows[["value", "caliop", "difference"]].to_numpy(float)
    if not np.isfinite(fields[supported]).all() or not np.isnan(fields[~supported]).all():
        raise ValueError("Science missingness differs from saved support")
    if (fields[supported, :2] < 0).any():
        raise ValueError("Negative supported absolute DAOD")
    if not np.allclose(fields[supported, 0] - fields[supported, 1], fields[supported, 2], atol=1e-12, rtol=0):
        raise ValueError("Difference does not use the same-support CALIOP field")
    return dict(x=np.r_[lon - dx / 2, lon[-1] + dx / 2],
                y=np.r_[lat - dy / 2, lat[-1] + dy / 2],
                value=fields[:, 0].reshape(shape), difference=fields[:, 2].reshape(shape),
                missing_land=(rows.in_domain & rows.has_land & ~rows.supported).to_numpy().reshape(shape),
                supported=supported.reshape(shape))


def select(registry):
    data = registry.table(MAP_TABLE)
    metrics = registry.table(METRIC_TABLE)
    for frame in (data, metrics):
        for column, value in dict(reference=CONFIG["reference"]["name"], variant="hierarchy",
                                  reference_mask="prior_joint", region="Land 60S-60N").items():
            if not frame[column].eq(value).all():
                raise ValueError("Unexpected map/metric selection: " + column)
    keys = ["estimator", "support", "period"]
    expected = {(e, support_for(e), p) for e in METHODS for p in SEASONS}
    if set(data[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing/extra method-support-season map combination")
    if metrics.duplicated(keys).any():
        raise ValueError("Ambiguous saved seasonal metrics")
    displayed = metrics.loc[metrics.display_map.eq(True)]
    if set(displayed[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Displayed metrics do not match registered maps")
    grids = {}
    for (estimator, support, period), frame in data.groupby(keys, sort=False):
        if not frame.months.eq(PERIOD_MONTHS[period]).all():
            raise ValueError("Wrong seasonal months, including DJF December")
        grid = reconstruct(frame)
        row = metrics.loc[metrics.estimator.eq(estimator) & metrics.support.eq(support) & metrics.period.eq(period)].iloc[0]
        if row.n != grid["supported"].sum():
            raise ValueError("Map support count differs from checked seasonal metric")
        grids[(estimator, period)] = grid
    # Core rows must share the exact seasonal bitmap, not just its cell count.
    for period in SEASONS:
        core = grids[("CALIOP", period)]["supported"]
        for estimator in METHODS:
            supported = grids[(estimator, period)]["supported"]
            if estimator == "E6":
                if (supported & ~core).any():
                    raise ValueError("E6 support is not a subset of core")
            elif not np.array_equal(supported, core):
                raise ValueError("Core maps do not share the same support")
    evidence = [dict(table=name, source_file=registry.payload["tables"][name]["source_file"],
                     source_sha256=registry.payload["tables"][name]["source_sha256"],
                     derivation=registry.payload["tables"][name]["derivation"])
                for name in (MAP_TABLE, METRIC_TABLE)]
    return data, grids, evidence


def local_geography(directory=DEFAULT_GEOGRAPHY):
    """Use existing local files directly, never Cartopy's downloading resolver."""
    result, sources = {}, {}
    for name in ("coastline", "ocean"):
        path = Path(directory) / f"ne_110m_{name}.shp"
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            component = path.with_suffix(suffix).resolve(strict=True)
            sources[str(component)] = digest(component)
        reader = shapereader.Reader(path)
        try:
            result[name] = tuple(reader.geometries())
        finally:
            reader.close()
    return result, sources


def draw_map(ax, grid, geography, difference=False, left_labels=False, bottom_labels=False):
    x, y = grid["x"], grid["y"]
    z = grid["difference" if difference else "value"]
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    if difference:
        ax.pcolor(x, y, np.ma.masked_where(~grid["missing_land"], np.ones_like(z)),
                  cmap=ListedColormap(["#E0E0E0"]), hatch="////", edgecolor="#BDBDBD", lw=0,
                  transform=ccrs.PlateCarree(), rasterized=True)
    ax.pcolormesh(x, y, np.ma.masked_where(z != 0, np.zeros_like(z)),
                  cmap=ListedColormap([style.ZERO_GREY]), transform=ccrs.PlateCarree(), rasterized=True)
    cmap = style.CMAP_DIFF if difference else style.CMAP_DAOD
    levels = np.linspace(-.15, .15, 11) if difference else np.linspace(0, .5, 11)
    mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(np.where(z == 0, np.nan, z)),
                         cmap=cmap, norm=BoundaryNorm(levels, cmap.N),
                         transform=ccrs.PlateCarree(), rasterized=True)
    if difference:
        signs.add_positive_stipple(ax, mesh, z)
    ax.add_geometries(geography["ocean"], ccrs.PlateCarree(), facecolor="white", edgecolor="none", zorder=3)
    ax.add_geometries(geography["coastline"], ccrs.PlateCarree(), facecolor="none", edgecolor="#4D4D4D", lw=.4, zorder=4)
    ax.gridlines(crs=ccrs.PlateCarree(), xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 61, 30),
                 linewidth=.2, color="#BDBDBD", draw_labels=False, zorder=5)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if left_labels:
        for latitude, label in ((-30, "30S"), (0, "0"), (30, "30N")):
            _, py = ax.projection.transform_point(0, latitude, ccrs.PlateCarree())
            ax.text(-.035, (py - ymin) / (ymax - ymin), label, transform=ax.transAxes,
                    ha="right", va="center", fontsize=6.5)
    if bottom_labels:
        for longitude, label in ((-120, "120W"), (0, "0"), (120, "120E")):
            px, _ = ax.projection.transform_point(longitude, 0, ccrs.PlateCarree())
            ax.text((px - xmin) / (xmax - xmin), -.09, label, transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5)
    return mesh


def panel_code(index):
    return chr(ord("a") + index) if index < 26 else "a" + chr(ord("a") + index - 26)


def build_figure(grids, geography, difference=False):
    methods = tuple(e for e in METHODS if not difference or e != "CALIOP")
    fig = plt.figure(figsize=(style.DOUBLE, HEIGHT_MM * style.MM))
    fig.text(.02, .989, "Primary reference | hierarchy | " + ("Estimator minus CALIOP" if difference else "Seasonal DAOD"),
             fontsize=7, va="top")
    for row, (estimator, bottom) in enumerate(zip(methods, np.linspace(.87, .135, len(methods)))):
        fig.text(.012, bottom + .039, LABELS[estimator], fontsize=7, va="center", linespacing=1.3)
        for col, period in enumerate(SEASONS):
            ax = fig.add_axes([.185 + col * .202, bottom, .19, .078], projection=ccrs.Robinson())
            mesh = draw_map(ax, grids[(estimator, period)], geography, difference,
                            left_labels=col == 0, bottom_labels=row == len(methods) - 1)
            style.panel_label(ax, panel_code(row * 4 + col), x=.01, y=1.03)
            if row == 0:
                ax.set_title(period, fontsize=7, pad=8)
    cax = fig.add_axes([.43, .068, .30, 2.5 / HEIGHT_MM])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both" if difference else "max")
    bar.set_ticks([-.15, 0, .15] if difference else [0, .25, .5])
    bar.set_label("DAOD difference (550 nm)" if difference else "DAOD (550 nm)")
    handles = [Patch(facecolor=style.ZERO_GREY, edgecolor="none", label="Exact zero"),
               Patch(facecolor="#E0E0E0" if difference else "white", edgecolor="#BDBDBD",
                     hatch="////" if difference else None, label="Missing land" if difference else "Missing")]
    if difference:
        handles.append(signs.sign_handles()[0])
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .036), fontsize=6.5)
    fig.text(.02, .012, "* E6: separate full_common_E6 support; all other rows: core_common. Not a paired E6/core comparison.",
             fontsize=6.5, va="bottom")
    return fig


def check_layout(fig):
    fig.canvas.draw()
    renderer, boxes = fig.canvas.get_renderer(), []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if text.get_fontsize() < 6.5:
            raise ValueError("Text below 6.5 pt: " + text.get_text())
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError("Off-canvas text: " + text.get_text())
        for previous, other in boxes:
            if min(box.x1, other.x1) - max(box.x0, other.x0) > .5 and min(box.y1, other.y1) - max(box.y0, other.y0) > .5:
                raise ValueError(f"Overlapping text: {previous!r} / {text.get_text()!r}")
        boxes.append((text.get_text(), box))


def caption(difference):
    return (("Seasonal estimator-minus-CALIOP DAOD550 differences" if difference else "Seasonal DAOD550 for eight methods") +
            " for the primary experimental1020 hierarchy. Columns are DJF, MAM, JJA and SON; "
            "DJF includes December 2016. Native 2x5-degree land cells between 60S and 60N retain "
            "the saved prior_joint intersection of all three months, with equal-month means. "
            "E1, Eselected, E5, E7, E8, Song and CALIOP use core_common. E6 (LG25 screened) "
            "uses separate full_common_E6 support; its reference is CALIOP on that support, "
            "not the displayed core CALIOP mean. E6 versus core-method differences are not paired "
            "comparisons. Song is the saved 550-nm field; CALIOP is transferred from 532 to 550 nm "
            "using the frozen dust Angstrom exponent before averaging. " +
            ("Differences span -0.15 to +0.15; missing land is light grey with diagonal hatching. "
             "Dots mark strictly positive differences, not significance; zero and negative differences are undotted. " if difference else
             "Absolute values span 0 to 0.5; missing is white. ") +
            "Colorbar extensions indicate out-of-range values without altering data. Exact zeros "
            "are grey; ocean is white. Gridlines are 30 degrees with sparse outer-edge geographic "
            "labels to remain readable. No interpolation or new masks are applied to the scientific "
            "arrays. CALIOP is a comparison, not dust truth.\n")


def make(registry_path, output, geography_directory=DEFAULT_GEOGRAPHY, page="both"):
    if page not in ("both", "absolute", "difference"):
        raise ValueError("Unknown S13 page")
    before = digest(Path(registry_path))
    data, grids, evidence = select(Registry.read(registry_path))
    geography, geography_sources = local_geography(geography_directory)
    evidence.append(dict(geography_sources=geography_sources, rendering="native pcolormesh, no interpolation",
                         map_shape=[90, 72], absolute_panels=[8, 4], difference_panels=[7, 4]))
    for difference, stem in ((False, "FigS13a_all_seasonal_maps"), (True, "FigS13b_all_seasonal_differences")):
        if page != "both" and difference != (page == "difference"):
            continue
        fig = build_figure(grids, geography, difference)
        try:
            check_layout(fig)
            if digest(Path(registry_path)) != before:
                raise RuntimeError("Registry changed during rendering")
            exported = data.loc[data.estimator.ne("CALIOP")] if difference else data
            page_evidence = evidence + ([signs.provenance()] if difference else [])
            style.export(fig, output, stem, exported, registry_path, page_evidence, __file__, raster=True)
            (Path(output) / (stem.split("_")[0] + "_caption.md")).write_text(caption(difference))
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geography-directory", type=Path, default=DEFAULT_GEOGRAPHY)
    parser.add_argument("--page", choices=("both", "absolute", "difference"), default="both")
    args = parser.parse_args(argv)
    make(args.registry, args.output_dir, args.geography_directory, args.page)


if __name__ == "__main__":
    main()
