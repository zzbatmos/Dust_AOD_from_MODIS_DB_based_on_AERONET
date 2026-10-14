"""Regional monthly cycles on each month's frozen joint comparison support."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from paper.derive_monthly_cycles import METHODS
from paper.figures import figstyle as style
from paper.numbers import Registry

TABLE = "derived/regional_monthly_cycles"
DISPLAY_NAMES = {"Land 60S-60N": "Global land (60S-60N)", "Amazonia_Cerrado": "Amazonia/Cerrado",
    "southern_Africa": "Southern Africa", "boreal_North_America": "Boreal N. America (50-60N)",
    "Siberia": "Siberia (50-60N)", "mainland_SE_Asia": "Mainland SE Asia"}


def select(registry):
    data = registry.table(TABLE, reference="experimental1020", variant="hierarchy",
                          support="core_common", reference_mask="prior_joint")
    if data.empty or data.duplicated(["region", "date", "estimator"]).any():
        raise ValueError("Empty or ambiguous monthly cycles")
    expected_dates = {"2016-12", *[f"2017-{m:02d}" for m in range(1, 12)]}
    for region, q in data.groupby("region"):
        if set(q.date) != expected_dates or set(q.estimator) != set(METHODS) or len(q) != 12 * len(METHODS):
            raise ValueError("Incomplete monthly cycle: " + region)
        for _, month in q.groupby("date"):
            if month.n.nunique() != 1 or month.land_area_weight_sum.nunique() != 1:
                raise ValueError("Unpaired estimator support in month")
            valid = month.n.gt(0)
            if not (month.supported == valid).all() or not np.isfinite(month.loc[valid, "mean"]).all():
                raise ValueError("Invalid supported monthly values")
            if month.loc[~valid, "mean"].notna().any():
                raise ValueError("Missing month must not be zero filled")
    primary = data.loc[~data.sensitivity]
    if primary.region.nunique() != 15:
        raise ValueError("Expected 15 frozen primary regions including global")
    return data


def make(registry_path, output):
    registry = Registry.read(registry_path)
    data = select(registry)
    primary = data.loc[~data.sensitivity]
    regions = primary.region.drop_duplicates().tolist()
    methods = list(METHODS[:7])
    fig, axes = plt.subplots(5, 3, figsize=(style.DOUBLE, 215 * style.MM), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, top=.95, bottom=.15, hspace=.62, wspace=.36)
    for i, (region, ax) in enumerate(zip(regions, axes.flat)):
        q = primary.loc[primary.region.eq(region)]
        low = q.loc[q.estimator.eq("CALIOP_low")].sort_values("row_order")
        high = q.loc[q.estimator.eq("CALIOP_high")].sort_values("row_order")
        ax.fill_between(low.row_order, low["mean"], high["mean"], color="#BDBDBD", alpha=.42, lw=0)
        for estimator in methods:
            s = q.loc[q.estimator.eq(estimator)].sort_values("row_order")
            _, color, marker, line = style.EST[estimator]
            ax.plot(s.row_order, s["mean"], color=color, marker=marker, ls=line,
                    markersize=2.4, lw=.8, zorder=4 if estimator == "CALIOP" else 3)
        count = low.n
        ax.set_title(DISPLAY_NAMES.get(region, region), pad=8, fontsize=7)
        ax.text(.03, .96, f"Cells: {count.min():.0f}-{count.max():.0f}", transform=ax.transAxes,
                va="top", fontsize=6.5)
        maximum = q["mean"].max()
        ax.set(xlim=(-.1, 11.1), ylim=(0, max(.02, float(maximum) * 1.25)),
               xticks=[0, 3, 6, 9, 11], xticklabels=["Dec", "Mar", "Jun", "Sep", "Nov"])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(axis="y", lw=.35, color="#E6E6E6")
        if i % 3 == 0:
            ax.set_ylabel("DAOD (550 nm)")
        if i >= 12:
            ax.set_xlabel("Dec 2016-Nov 2017")
        style.panel_label(ax, "abcdefghijklmno"[i], x=-.17, y=1.06)
    handles = [Line2D([], [], color=style.EST[e][1], marker=style.EST[e][2], ls=style.EST[e][3],
                      markersize=3.5, label=style.EST[e][0]) for e in methods]
    handles.append(Patch(facecolor="#BDBDBD", alpha=.42, label="CALIOP low/high scenarios"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .05), ncol=3,
               columnspacing=1., handlelength=1.8)
    fig.text(.5, .018, "Monthly joint support varies with sampling; individual y-axis ranges. Shading is not a confidence interval.",
             ha="center", fontsize=6.5)
    metadata = registry.payload["tables"][TABLE]
    style.export(fig, output, "FigS14_monthly_cycles", data.assign(plotted=~data.sensitivity), registry_path,
                 [dict(table=TABLE, source_file=metadata["source_file"], source_sha256=metadata["source_sha256"],
                       derivation=metadata.get("derivation"))], __file__)
    (Path(output) / "FigS14_caption.md").write_text(
        "Monthly regional DAOD at 550 nm from December 2016 through November 2017, using the primary reference "
        "and hierarchy. All curves within a region-month use identical saved core joint support and land-area "
        "weights; that support can change between months. Counts give the range of supported native 2x5-degree "
        "cells, not independent observations. CALIOP low/high shading represents retrieval scenarios, not a "
        "confidence interval; all CALIOP fields are spectrally transferred to 550 nm. Regional y-axis ranges differ. "
        "Missing months remain gaps. Screened LG25 is excluded because its support differs. The two 50-75N boreal "
        "sensitivity regions are retained separately in companion data, not substituted for primary 50-60N boxes. "
        "Monthly curves should not be averaged to reproduce seasonal statistics, which require the complete "
        "three-month spatial intersection.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
