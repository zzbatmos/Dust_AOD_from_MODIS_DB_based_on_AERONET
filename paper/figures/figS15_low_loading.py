"""Registry-only S15: available-month EV contributions and separate floor diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper.numbers import Registry, digest
from paper.figures import figstyle as style
from paper.figures.figS13_all_seasonal_maps import local_geography, check_layout

TABLES = dict(maps="low_loading_maps", summary="low_loading_summary", samples="low_loading_floor_samples",
              floor="low_loading_floor_summary", audit="low_loading_floor_audit")
METHODS = ("E1", "Eselected", "E5", "E7", "E8")
SUBSETS = ("AOD550_lt_0p2", "P_lt_0p3")
REGIMES = ("Amazon_DJF", "dual_reference_low_dust")
LABELS = {"E1": "QA2", "Eselected": "Expected value", "E5": "DB type", "E7": "LG25", "E8": "AE refit"}


def support_label(cell_months, spatial_cells):
    return (f"{int(cell_months)} cell-month{'s' if cell_months != 1 else ''}; "
            f"{int(spatial_cells)} cell{'s' if spatial_cells != 1 else ''}")


def reconstruct(frame):
    dy, dx = CONFIG["grids"]["2x5"]
    lat = np.arange(CONFIG["inputs"]["latitude_bounds"][0] + dy / 2, CONFIG["inputs"]["latitude_bounds"][1], dy)
    lon = np.arange(CONFIG["inputs"]["longitude_bounds"][0] + dx / 2, CONFIG["inputs"]["longitude_bounds"][1], dx)
    shape = len(lat), len(lon)
    iy, ix = np.indices(shape)
    rows = frame.sort_values(["lat_index", "lon_index"])
    if len(rows) != iy.size or rows.duplicated(["lat_index", "lon_index"]).any():
        raise ValueError("Incomplete/duplicated contribution grid")
    for name, expected in (("lat_index", iy), ("lon_index", ix), ("lat", lat[iy]), ("lon", lon[ix]),
                           ("lat_lower", lat[iy] - dy / 2), ("lat_upper", lat[iy] + dy / 2),
                           ("lon_lower", lon[ix] - dx / 2), ("lon_upper", lon[ix] + dx / 2)):
        if not np.array_equal(rows[name].to_numpy().reshape(shape), expected):
            raise ValueError("Wrong native contribution coordinate/index")
    if not rows.month_count.isin(range(13)).all() or not rows.supported.isin([True, False]).all():
        raise ValueError("Invalid available-month counts/support")
    if not np.array_equal(rows.in_domain, (rows.lat >= -60) & (rows.lat < 60)):
        raise ValueError("Wrong primary land-domain flag")
    valid = rows.month_count.gt(0)
    if not np.array_equal(rows.supported, valid) or (valid & ~rows.in_domain).any():
        raise ValueError("Map support differs from available-month count/domain")
    values = rows.value.to_numpy(float)
    if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all() or (values[valid] < 0).any():
        raise ValueError("Missing/zero/nonnegative contribution contract violated")
    return dict(x=np.r_[lon - dx / 2, lon[-1] + dx / 2], y=np.r_[lat - dy / 2, lat[-1] + dy / 2],
                value=values.reshape(shape), month_count=rows.month_count.to_numpy().reshape(shape))


def select(registry):
    frames = {name: registry.table("derived/" + table) for name, table in TABLES.items()}
    for frame in frames.values():
        for key, value in dict(reference=CONFIG["reference"]["name"], variant="hierarchy", support="core_common").items():
            if not frame[key].eq(value).all():
                raise ValueError("Wrong primary diagnostic support")
    maps, summary, samples, floor, audit = (frames[k] for k in TABLES)
    if set(maps.subset) != set(SUBSETS) or len(audit) != 1:
        raise ValueError("Missing contribution subsets or ambiguous floor audit")
    for key, value in dict(estimator="Eselected", period="2017", reference_mask="modis_only", denominator="full_paired_count").items():
        column = maps[key].astype(str) if key == "period" else maps[key]
        if not column.eq(value).all():
            raise ValueError("Wrong map semantics: " + key)
    counts = None
    for subset, frame in maps.groupby("subset"):
        grid = reconstruct(frame)
        expected = CONFIG["diagnostics"]["low_aod_threshold" if subset == SUBSETS[0] else "low_probability_threshold"]
        if not frame.threshold.eq(expected).all():
            raise ValueError("Map subset label/threshold mismatch")
        if counts is not None and not np.array_equal(counts, grid["month_count"]):
            raise ValueError("Contribution maps do not share MODIS support")
        counts = grid["month_count"]
        row = summary.loc[summary.estimator.eq("Eselected") & summary.region.eq("Land 60S-60N") & summary.subset.eq(subset)]
        if len(row) != 1 or row.iloc[0].cell_months != counts.sum():
            raise ValueError("Map month counts differ from global cell-month summary")
    if set(floor[["estimator", "regime"]].itertuples(index=False, name=None)) != {(e, r) for e in METHODS for r in REGIMES} or len(floor) != 10:
        raise ValueError("Missing/duplicate saved floor rows")
    if samples.duplicated(["regime", "estimator", "month", "lat_index", "lon_index"]).any():
        raise ValueError("Duplicate diagnostic cell-month")
    if not samples.estimator.isin(METHODS).all() or not samples.regime.isin(REGIMES).all():
        raise ValueError("Unexpected diagnostic method/regime")
    for regime in REGIMES:
        regime_samples = samples.loc[samples.regime.eq(regime)]
        keys = None
        for e in METHODS:
            part = regime_samples.loc[regime_samples.estimator.eq(e)]
            row = floor.loc[floor.estimator.eq(e) & floor.regime.eq(regime)].iloc[0]
            if len(part) != row.cell_months or not np.isfinite(part.value).all() or (part.value < 0).any():
                raise ValueError("Floor selected count/values disagree with saved summary")
            these = set(part[["month", "lat_index", "lon_index"]].itertuples(index=False, name=None))
            if keys is not None and these != keys:
                raise ValueError("Floor estimators are not on paired support")
            keys = these
        if regime == "Amazon_DJF":
            if not regime_samples.month.isin(["2016-12", "2017-01", "2017-02"]).all() or not regime_samples.reference_mask.eq("modis_only").all():
                raise ValueError("Amazon DJF support/months changed")
            audit_n = audit.iloc[0].Amazon_DJF_cell_months
        else:
            bound = audit.iloc[0].low_dust_threshold
            if (not regime_samples.month.str.startswith("2017-").all()
                    or not regime_samples.reference_mask.eq("reference_mean_only_low_dust").all()
                    or not (regime_samples[["caliop", "song"]].ge(0) & regime_samples[["caliop", "song"]].lt(bound)).all().all()):
                raise ValueError("Low-dust reference-mean-only selection changed")
            audit_n = audit.iloc[0].paired_low_dust_cell_months
        if len(keys) != audit_n:
            raise ValueError("Floor audit does not match selected cell-months")
    evidence = [dict(table="derived/" + name, source_file=registry.payload["tables"]["derived/" + name]["source_file"],
                     source_sha256=registry.payload["tables"]["derived/" + name]["source_sha256"],
                     derivation=registry.payload["tables"]["derived/" + name]["derivation"]) for name in TABLES.values()]
    return frames, evidence


def draw_map(ax, grid, geography, vmax):
    z = grid["value"]
    ax.set_extent([-180, 180, -60, 60], ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.pcolormesh(grid["x"], grid["y"], np.ma.masked_where(z != 0, np.zeros_like(z)),
                  cmap=ListedColormap([style.ZERO_GREY]), transform=ccrs.PlateCarree(), rasterized=True)
    mesh = ax.pcolormesh(grid["x"], grid["y"], np.ma.masked_invalid(np.where(z == 0, np.nan, z)),
                         cmap=style.CMAP_DAOD, norm=BoundaryNorm(np.linspace(0, vmax, 11), style.CMAP_DAOD.N),
                         transform=ccrs.PlateCarree(), rasterized=True)
    ax.add_geometries(geography["ocean"], ccrs.PlateCarree(), facecolor="white", edgecolor="none", zorder=3)
    ax.add_geometries(geography["coastline"], ccrs.PlateCarree(), facecolor="none", edgecolor="#4D4D4D", lw=.4, zorder=4)
    ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 61, 30), linewidth=.2, color="#BDBDBD")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for latitude, label in ((-30, "30S"), (0, "0"), (30, "30N")):
        _, py = ax.projection.transform_point(0, latitude, ccrs.PlateCarree())
        ax.text(-.02, (py - ymin) / (ymax - ymin), label, transform=ax.transAxes,
                ha="right", va="center", fontsize=6.5)
    for longitude, label in ((-120, "120W"), (0, "0"), (120, "120E")):
        px, _ = ax.projection.transform_point(longitude, 0, ccrs.PlateCarree())
        ax.text((px - xmin) / (xmax - xmin), -.06, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5)
    return mesh


def build_figure(frames, geography):
    fig = plt.figure(figsize=(style.DOUBLE, 200 * style.MM))
    fig.text(.02, .985, "Primary reference | hierarchy | core_common", fontsize=7, va="top")
    for col, (subset, title, vmax) in enumerate(zip(SUBSETS, ("Low AOD contribution", "Low probability contribution"), (.01, .05))):
        frame = frames["maps"].loc[frames["maps"].subset.eq(subset)]
        grid = reconstruct(frame)
        left = .07 + col * .49
        ax = fig.add_axes([left, .77, .39, .15], projection=ccrs.Robinson())
        mesh = draw_map(ax, grid, geography, vmax)
        ax.set_title(title + f" ({'AOD550' if col == 0 else 'P'} < {frame.threshold.iloc[0]:g})", fontsize=7, pad=8)
        style.panel_label(ax, chr(ord("a") + col), x=-.07, y=1.04)
        bar = fig.colorbar(mesh, cax=fig.add_axes([left + .025, .72, .34, .012]), orientation="horizontal", extend="max")
        bar.set_ticks([0, vmax / 2, vmax])
        bar.set_label("EV contribution to DAOD550")
        row = frames["summary"].loc[frames["summary"].estimator.eq("Eselected") & frames["summary"].region.eq("Land 60S-60N") & frames["summary"].subset.eq(subset)].iloc[0]
        fig.text(left + .195, .64, f"Weighted cell-month mean = {row.area_weighted_mean:.5f}\n"
                 f"N = {int(row.cell_months):,} cell-months (not a mean of this map)", ha="center", va="top", fontsize=6.5)
    available = frames["maps"].loc[frames["maps"].supported, "month_count"]
    span = f"{int(available.min())}-{int(available.max())}" if len(available) else "none"
    fig.text(.5, .585, f"Maps: available-month arithmetic means, calendar 2017; {span} months/cell. Zero: grey; missing: white.",
             ha="center", fontsize=6.5)
    audit = frames["audit"].iloc[0]
    ax = fig.add_axes([.09, .22, .36, .27])
    for e in METHODS:
        values = np.sort(frames["samples"].loc[frames["samples"].regime.eq("Amazon_DJF") & frames["samples"].estimator.eq(e), "value"].to_numpy())
        _, color, marker, linestyle = style.EST[e]
        if len(values):
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, marker=marker,
                    markevery=max(1, len(values) // 5), ms=3, linestyle=linestyle, drawstyle="steps-post", label=LABELS[e])
    ax.axvline(audit.descriptive_floor_threshold, color="#4D4D4D", linestyle=":", lw=.6)
    ax.set(xlabel="Estimated DAOD550", ylabel="Empirical cumulative probability", ylim=(0, 1.05))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_title("Amazonia DJF: MODIS-only\n" + support_label(audit.Amazon_DJF_cell_months, audit.Amazon_DJF_spatial_cells), fontsize=7, pad=9)
    style.panel_label(ax, "c", x=-.16, y=1.10)
    ax.legend(loc="upper left", bbox_to_anchor=(-.06, -.30), ncol=2, fontsize=6.5)
    ax = fig.add_axes([.64, .22, .32, .27])
    selected = frames["floor"].loc[frames["floor"].regime.eq("dual_reference_low_dust")].set_index("estimator")
    for index, e in enumerate(METHODS):
        row = selected.loc[e]
        _, color, marker, _ = style.EST[e]
        if row.cell_months:
            ax.plot([row.q05, row.q95], [index, index], color=color, lw=1)
            ax.plot(row["median"], index, marker=marker, color=color, ms=4, linestyle="none")
    ax.axvline(audit.descriptive_floor_threshold, color="#4D4D4D", linestyle=":", lw=.6)
    ax.set(yticks=range(len(METHODS)), yticklabels=[LABELS[e] for e in METHODS], xlabel="Estimated DAOD550", ylim=(-.5, len(METHODS) - .5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.invert_yaxis()
    ax.set_title("Dual-reference low dust: 2017\n" + support_label(audit.paired_low_dust_cell_months, audit.paired_low_dust_spatial_cells), fontsize=7, pad=9)
    style.panel_label(ax, "d", x=-.20, y=1.10)
    note = "Only one eligible cell-month;\nnot a population-level floor test." if audit.paired_low_dust_cell_months == 1 else "Median and 5th-95th percentiles;\ndistribution range, not a CI."
    if audit.paired_low_dust_cell_months == 0:
        note = "No eligible cell-months;\nfloor test unavailable."
    fig.text(.80, .12, note, ha="center", va="top", fontsize=6.5)
    fig.text(.02, .025, f"Different supports in (c) and (d); neither is selected by QA2=0. Dotted lines: descriptive {audit.descriptive_floor_threshold:g} flag.", fontsize=6.5)
    return fig


def caption(frames):
    audit = frames["audit"].iloc[0]
    return ("Primary experimental1020 hierarchy on core_common. (a,b) Native 2x5-degree EV "
            "low-AOD and low-P contributions on verified saved MODIS-only support, Land 60S-60N. "
            "Each cell averages its available calendar-2017 months equally; month counts accompany "
            "the arrays. There is no fixed-year intersection. Contributions use the full paired "
            "retrieval-count denominator, not subset means; P<0.3 excludes missing P in the frozen "
            "recipe. The two subsets can overlap and must not be added. Annotated global means "
            "instead area-weight all eligible cell-months with original float64 land geometry; "
            "they are not spatial averages of the displayed map. Color scales are 0-0.01 and "
            "0-0.05 respectively; extensions denote values beyond the scale without changing data. "
            "Zero is grey; missing and ocean are white. No interpolation or new masks. "
            "(c) Unweighted empirical distributions of five saved estimators in Amazonia_Cerrado "
            "on MODIS-only DJF support, including December 2016, January and February 2017. "
            "(d) Saved medians and 5th-95th percentile ranges, not confidence intervals, on "
            "calendar-2017 reference-mean-only support where both transferred CALIOP550 and "
            f"Song550 are below {audit.low_dust_threshold:g}. This is not prior_joint support. "
            "The two floor selections are different and are not selected by QA2=0. "
            f"There are {int(audit.paired_low_dust_cell_months)} dual-reference eligible cell-months; "
            "a singleton provides no population-level floor test. Source selection-step counts "
            "and all saved regional loading/floor summaries accompany the data. Dotted lines "
            f"mark the saved descriptive {audit.descriptive_floor_threshold:g} flag, not a new cutoff. "
            "CALIOP and Song are comparisons, not dust truth. No fits or new bootstrap.\n")


def make(registry_path, output):
    before = digest(Path(registry_path))
    frames, evidence = select(Registry.read(registry_path))
    geography, sources = local_geography()
    evidence.append(dict(geography_sources=sources, rendering="native pcolormesh, no interpolation"))
    fig = build_figure(frames, geography)
    try:
        check_layout(fig)
        if digest(Path(registry_path)) != before:
            raise RuntimeError("Registry changed during render")
        data = pd.concat([frame.assign(table_kind=key) for key, frame in frames.items()], ignore_index=True)
        style.export(fig, output, "FigS15_low_loading", data, registry_path, evidence, __file__, raster=True)
        (Path(output) / "FigS15_caption.md").write_text(caption(frames))
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make(args.registry, args.output_dir)


if __name__ == "__main__":
    main()
