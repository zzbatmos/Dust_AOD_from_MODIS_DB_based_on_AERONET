"""Author-approved qualified Figure 3; saved data only, no new uncertainty fits."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from paper.numbers import Registry, digest
from paper.figures import figstyle as style
from paper.figures.figS13_all_seasonal_maps import local_geography, check_layout

BINS = ("lt0.2", "0.2-lt0.4", "0.4-lt0.8", "ge0.8")
MODELS = ("full8", "fullminusAE7", "compact4", "singleSSA3", "noSSA5", "minimal2")
BASELINES = ("typeconditional", "LGscreened_coarseproxy", "zero")
CAPTION = (
    "Deep Blue information and availability. (a) Dated Aqua coverage pilot, not annual; "
    "Sulfate has no observations. (b) Annual count-weighted full8 availability. "
    "(c) Common-row QA2 ablation; vertical bars are saved 95% site-bootstrap intervals; "
    "AUC intervals are unavailable. Grey baselines show RMSE only. "
    "(d) Fixed noSSA5 E4 minus E8 RMSE, with saved intervals and practical margin; "
    "not nested Eselected or an equivalence test. Panels (a,b) have different temporal support.\n"
)


def pilot_counts(frame, inventory):
    if (len(inventory) != 1 or str(inventory.iloc[0].day) != "2017-08-29"
            or not isinstance(inventory.iloc[0].complete, (bool, np.bool_))
            or not inventory.iloc[0].complete or inventory.iloc[0].failed != 0
            or inventory.iloc[0].completed != inventory.iloc[0].expected):
        raise ValueError("Expected the completed 29 August 2017 pilot inventory")
    q = frame.loc[frame["sample"].eq("raw_Aqua2017") & frame.region.eq("All swath")].copy()
    if set(q.month) != {"2017-08"}:
        raise ValueError("Pilot coverage is not the registered August sample")
    for column in ("db_type", "algorithm"):
        q[column] = q[column].astype(str)
    total = q.loc[q.db_type.eq("ALL") & q.algorithm.eq("ALL") & q.aod_bin.eq("ALL")]
    joint = q.loc[q.db_type.isin(["0", "1", "2", "3"]) & ~q.algorithm.eq("ALL") & q.aod_bin.isin(BINS)]
    columns = ["n_valid_aod", "n_ssa_observed", "n_ssa_missing"]
    if len(total) != 1 or joint.empty or joint.duplicated(["month", "db_type", "algorithm", "aod_bin"]).any():
        raise ValueError("Ambiguous pilot marginal/joint rows")
    if (not np.isfinite(joint[columns]).all().all() or (joint[columns] < 0).any().any()
            or not joint.n_ssa_observed.eq(joint.n_valid_aod).all()
            or (joint.n_ssa_missing > joint.n_valid_aod).any()):
        raise ValueError("Invalid audited SSA denominator/counts")
    if not np.array_equal(joint[columns].sum().to_numpy(), total[columns].iloc[0].to_numpy()):
        raise ValueError("Pilot joint counts do not close to the overall marginal")
    counts = joint.groupby(["db_type", "aod_bin"])[columns].sum().reindex(
        pd.MultiIndex.from_product([["0", "1", "2", "3"], BINS], names=["db_type", "aod_bin"]), fill_value=0).reset_index()
    counts["available"] = counts.n_ssa_observed - counts.n_ssa_missing
    counts["fraction"] = counts.available / counts.n_valid_aod.where(counts.n_valid_aod.gt(0))
    counts["date"] = "2017-08-29"
    return counts


def annual_grid(frame):
    q = frame.sort_values(["lat", "lon"])
    lat, lon = np.arange(-89.5, 90), np.arange(-179.5, 180)
    if len(q) != len(lat) * len(lon) or q.duplicated(["lat", "lon"]).any():
        raise ValueError("Incomplete annual 1x1 grid")
    if (not np.array_equal(q.lat, np.repeat(lat, len(lon)))
            or not np.array_equal(q.lon, np.tile(lon, len(lat)))
            or not q.year.eq(2017).all() or not q.reference.eq("experimental1020").all()
            or not q.support.eq("calendar_year_valid_AOD").all()):
        raise ValueError("Incorrect annual coordinates or population")
    counts = q[["n_full8", "n_noSSA5", "n_ml", "n_valid_aod"]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Invalid annual counts")
    if (not q.n_full8.add(q.n_noSSA5).eq(q.n_ml).all() or (q.n_ml > q.n_valid_aod).any()):
        raise ValueError("Annual branch partition fails")
    expected = q.n_full8 / q.n_valid_aod.where(q.n_valid_aod.gt(0))
    if not np.allclose(q.full8_fraction_of_valid_aod, expected, atol=1e-14, rtol=0, equal_nan=True):
        raise ValueError("Annual fraction is not pooled-count weighted")
    return np.arange(-180, 181), np.arange(-90, 91), expected.to_numpy().reshape(180, 360)


def select(registry, inventory):
    counts = pilot_counts(registry.table("coverage/T1_coverage"), inventory)
    annual = registry.table("derived/annual_full8_coverage_2017")
    annual_grid(annual)
    evidence, rows = [], []
    selectors = dict(reference="experimental1020", population="common", scheme="site_cv", QA=.6, scope="overall", group="all")
    for model in (*MODELS, *BASELINES):
        numbers = {metric: registry.number("grouped_validation/metrics", metric, model=model, **selectors)
                   for metric in ("AUC", "DAOD_RMSE")}
        rmse, auc = numbers["DAOD_RMSE"], numbers["AUC"]
        if rmse["N"]["unique_rows"] != 3673 or rmse["N"]["sites"] != 231:
            raise ValueError("Ablation does not use the identical saved population")
        if rmse["interval"] is None or (model in MODELS and auc["interval"] is not None):
            raise ValueError("Interval availability changed; review Figure 3 candidate")
        rows.append(dict(model=model, auc=auc["value"], rmse=rmse["value"],
                         ci_low=rmse["interval"][0], ci_high=rmse["interval"][1],
                         unique_rows=3673, sites=231, auc_interval_available=False))
        evidence.extend(numbers.values())
    paired_filters = dict(stage="b", reference="experimental1020", branch="noSSA5_own", scheme="site_cv",
                          hypothesis="E8_recalibration_contrast", metric="rmse", candidate="E4", comparator="E8")
    paired = registry.table("expected_value_test/stage_b/paired_differences", **paired_filters)
    if len(paired) != 2 or set(paired.level) != {"collocation", "site_month"}:
        raise ValueError("Ambiguous E4 versus E8 contrast")
    if not paired.margin.eq(.005).all() or not np.isfinite(paired[["difference", "ci_low", "ci_high"]]).all().all():
        raise ValueError("Missing saved paired interval or changed margin")
    for name in ("coverage/T1_coverage", "derived/annual_full8_coverage_2017", "expected_value_test/stage_b/paired_differences"):
        table = registry.payload["tables"][name]
        evidence.append(dict(table=name, source_file=table["source_file"], source_sha256=table["source_sha256"],
                             derivation=table.get("derivation"), filters=paired_filters if name.endswith("paired_differences") else None))
    return dict(pilot=counts, annual=annual, ablation=pd.DataFrame(rows), paired=paired), evidence


def build(frames, geography):
    fig = plt.figure(figsize=(style.DOUBLE, 120 * style.MM))
    a = fig.add_axes([.08, .65, .36, .24])
    for code, label, color, marker in [("1", "Dust", "#D55E00", "o"), ("0", "Mixed", "#009E73", "s"), ("2", "Smoke", "#0072B2", "^")]:
        q = frames["pilot"].loc[frames["pilot"].db_type.eq(code)].set_index("aod_bin").loc[list(BINS)]
        a.plot(range(4), q.fraction, label=label, color=color, marker=marker, ms=3)
    a.set(xticks=range(4), xticklabels=["<0.2", "0.2-0.4", "0.4-0.8", ">=0.8"], ylim=(0, 1.05),
          yticks=[0, .5, 1], xlabel="DB AOD550", ylabel="All-SSA availability", title="Aqua pilot: 29 August 2017")
    a.legend(loc="lower left", bbox_to_anchor=(-.03, 1.19), ncol=3, handlelength=1.2, columnspacing=.7)
    a.text(.5, -.43, "Sulfate: no observations", transform=a.transAxes, ha="center", fontsize=6.5)
    style.panel_label(a, "a", x=-.17, y=1.03)
    b = fig.add_axes([.57, .65, .39, .26], projection=ccrs.Robinson())
    x, y, z = annual_grid(frames["annual"])
    b.set_global()
    mesh = b.pcolormesh(x, y, np.ma.masked_invalid(z), transform=ccrs.PlateCarree(), cmap=style.CMAP_P,
                       norm=BoundaryNorm(np.linspace(0, 1, 11), style.CMAP_P.N), rasterized=True)
    b.add_geometries(geography["ocean"], ccrs.PlateCarree(), facecolor="white", edgecolor="none", zorder=3)
    b.add_geometries(geography["coastline"], ccrs.PlateCarree(), facecolor="none", edgecolor="#555555", lw=.35, zorder=4)
    b.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 61, 30),
                linewidth=.2, color="#BDBDBD")
    xmin, xmax = b.get_xlim()
    ymin, ymax = b.get_ylim()
    for latitude, label in ((-60, "60S"), (0, "0"), (60, "60N")):
        _, py = b.projection.transform_point(0, latitude, ccrs.PlateCarree())
        b.text(-.025, (py-ymin)/(ymax-ymin), label, transform=b.transAxes,
               ha="right", va="center", fontsize=6.5)
    for longitude, label in ((-120, "120W"), (0, "0"), (120, "120E")):
        px, _ = b.projection.transform_point(longitude, 0, ccrs.PlateCarree())
        b.text((px-xmin)/(xmax-xmin), -.04, label, transform=b.transAxes,
               ha="center", va="top", fontsize=6.5)
    b.set_title("Annual Aqua: January-December 2017", pad=10)
    style.panel_label(b, "b", x=-.09, y=1.06)
    bar = fig.colorbar(mesh, cax=fig.add_axes([.63, .565, .27, .015]), orientation="horizontal", ticks=[0, .5, 1])
    bar.set_label("full8 / valid-AOD counts")
    c = fig.add_axes([.205, .17, .235, .25])
    base = fig.add_axes([.08, .17, .09, .25], sharey=c)
    markers, colors = ["o", "s", "^", "D", "v", "p"], ["#332288", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#555555"]
    handles = []
    q = frames["ablation"].set_index("model")
    for i, (model, marker, color) in enumerate(zip(MODELS, markers, colors)):
        row = q.loc[model]
        c.vlines(row.auc, row.ci_low, row.ci_high, color=color, lw=.75)
        c.plot(row.auc, row.rmse, marker=marker, color=color, ms=4, linestyle="none")
        handles.append(Line2D([], [], color=color, marker=marker, ls="none", label=model, ms=3))
    start, end = q.loc["full8"], q.loc["noSSA5"]
    c.annotate("", xy=(end.auc, end.rmse), xytext=(start.auc, start.rmse), arrowprops=dict(arrowstyle="->", lw=.7, color="black"))
    for i, model in enumerate(BASELINES):
        row = q.loc[model]
        base.vlines(i, row.ci_low, row.ci_high, color="#777777", lw=.7)
        base.plot(i, row.rmse, "o", color="#777777", ms=3)
    base.set(xticks=range(3), xticklabels=["DB", "LG", "0"], xlim=(-.5, 2.5), ylabel="QA2 DAOD RMSE", xlabel="Baselines")
    c.set(xlabel="AUC", xticks=[.74, .76, .78], xlim=(.73, .80))
    c.tick_params(labelleft=False)
    base.set_title("Common rows; saved RMSE intervals", loc="left", pad=11)
    style.panel_label(base, "c", x=-.65, y=1.04)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.055, .005), ncol=3,
               handlelength=.6, columnspacing=.6, labelspacing=.3, fontsize=6.5)
    d = fig.add_axes([.65, .17, .31, .25])
    d.axvspan(-.005, .005, color="#E5E5E5", zorder=0)
    d.axvline(0, color="#777777", lw=.7)
    for i, level in enumerate(("collocation", "site_month")):
        row = frames["paired"].loc[frames["paired"].level.eq(level)].iloc[0]
        d.hlines(i, row.ci_low, row.ci_high, color="#333333", lw=1)
        d.plot(row.difference, i, "o", color="#D55E00", ms=4)
    d.set(yticks=[0, 1], yticklabels=["Collocation", "Site-month"], ylim=(1.6, -.6),
          xlim=(-.009, .007), xticks=[-.005, 0, .005], xlabel="RMSE(E4) - RMSE(E8)", title="noSSA5: fixed direct form E4")
    style.panel_label(d, "d", x=-.26, y=1.04)
    fig.text(.65, .035, "Shading: +/-0.005 practical margin\nNo equivalence claim; AUC intervals unavailable in (c).", fontsize=6.5, ha="center")
    return fig


def make(registry_path, output, pilot_inventory):
    registry_path, pilot_inventory = Path(registry_path), Path(pilot_inventory)
    before, inventory_sha = digest(registry_path), digest(pilot_inventory)
    frames, evidence = select(Registry.read(registry_path), pd.read_csv(pilot_inventory))
    geography, geography_sources = local_geography()
    evidence.append(dict(pilot_inventory=str(pilot_inventory.resolve()), sha256=inventory_sha,
                         geography_sources=geography_sources, status="Qualified presentation approved by author on 2026-10-14"))
    fig = build(frames, geography)
    try:
        check_layout(fig)
        if digest(registry_path) != before or digest(pilot_inventory) != inventory_sha:
            raise RuntimeError("Figure 3 sources changed")
        style.export(fig, output, "Fig03_information_content", pd.concat(
            [frame.assign(panel_source=key) for key, frame in frames.items()], ignore_index=True),
            registry_path, evidence, __file__, raster=True)
        (Path(output) / "Fig03_caption.md").write_text(CAPTION)
        (Path(output) / "Fig03_review_notes.md").write_text(
            "# Figure 3 presentation notes\n\nThe author approved the qualified presentation on 2026-10-14: the pilot/annual temporal "
            "distinction, omitted horizontal AUC intervals, and fixed-E4 comparison. No new bootstrap was run. "
            "Panel c uses 3,673 unique collocations at 231 sites, "
            "with pooled held-out prediction events across saved repeats. Baselines DB/LG/0 are the fitted DB-type "
            "conditional estimator, screened LG25 and zero, not three probability models.\n\nPanel d is the fixed "
            "E4 form evaluated out of fold, not nested Eselected. Saved contrasts use the noSSA5 own population; "
            "collocation and site-month supports differ. The shaded margin is not an equivalence confidence region. "
            "Panel a requires all three SSA bands; panel b additionally requires other full8 features. Pilot missing "
            "type/bin support is not zero availability. Annual missing cells are white; zero availability remains zero.\n")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-inventory", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir, args.pilot_inventory)
