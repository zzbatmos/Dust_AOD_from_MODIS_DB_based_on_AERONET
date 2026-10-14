"""Author-approved mixed-scale Figure 5, exclusively from saved registry records."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from paper.derive_configuration_tables import guard_output
from paper.figures import figstyle as style
from paper.figures.figS11_site_month_sensitivity import check_layout
from paper.numbers import Registry, clean, digest


STEM = "Fig05_site_month"
ROOT = "expected_value_test/stage_b/"
POINTS = "derived/site_months_hierarchy"
ESTIMATORS = ("E1", "E5", "Eselected")
BASE = dict(stage="b", reference="experimental1020", branch="hierarchy_own", scheme="site_cv")
MONTHLY = dict(**BASE, pairing="core", minimum=3, repeat_summary="mean_over_repeats")
STRATA = ("all", "DB_mislabel_regimes", "dust_source_regimes", "Sahara", "Sahel", "Middle East",
          "Central Asia", "East Asian deserts", "SW North America", "India", "East China")
COMPONENTS = {
    "missed_dust_FN": ("Missed dust", "#0072B2", "///"),
    "false_dust_FP": ("False dust", "#E69F00", "xxx"),
    "true_positive_TP": ("TP error", "#009E73", "..."),
    "subthreshold_reference_TN": ("Subthreshold", "#CC79A7", "\\\\"),
}
COUNTS = ["prediction_events", "sites", "repeats", "unique_rows", "unique_site_months"]
STATUS = "Explicit mixed-scale presentation approved by author on 2026-10-14."
CAPTION = (
    "Experimental1020 hierarchy, held-out sites. "
    "(a-c) Site-month DAOD: minimum three collocations; both repeats plotted; annotations use saved "
    "repeat-specific metric means. (d) COLLOCATION-scale QA2 bias: four additive components, black total ticks; "
    "not the site-month bias decomposition. (e) Site-month bias and saved 95% site-bootstrap intervals on "
    "fixed, potentially overlapping strata. Open markers or asterisks denote fewer than five sites "
    "(exploratory); parentheses give site counts. No new uncertainty calculation."
)


def _counts(frame):
    values = frame[COUNTS].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any() or (values != np.floor(values)).any():
        raise ValueError("Invalid saved support counts")
    if not frame.repeats.eq(2).all():
        raise ValueError("Expected two saved site-CV repeats")


def _intervals(frame, point, low, high):
    values = frame[[point, low, high]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 1] > values[:, 2]).any():
        raise ValueError("Missing or reversed saved interval; do not fabricate uncertainty")


def select(registry):
    """No resampling, monthly grouping, model calls, or recalculation of skill."""
    data, evidence = [], []

    def get(name, filters, role):
        frame = registry.table(name, **filters)
        if frame.empty:
            raise ValueError(f"Required saved source missing: {name}/{filters}")
        frame["source_record_index"] = frame.index
        frame["source_table"] = name
        frame["plot_role"] = role
        metadata = registry.payload["tables"][name]
        evidence.append(dict(table=name, filters=filters,
                             **copy.deepcopy({k: v for k, v in metadata.items() if k != "records"})))
        return frame

    points = get(POINTS, dict(reference="experimental1020", scheme="site_cv", minimum=3, pairing="core"), "scatter")
    keys = ["site_key", "year", "month"]
    if points[keys + ["repeat"]].isna().any().any() or points.duplicated(keys + ["repeat"]).any():
        raise ValueError("Missing or duplicate saved site-month-repeat key")
    if set(points["repeat"]) != {0, 1} or not points.groupby(keys).size().eq(2).all():
        raise ValueError("Both saved repeats must be retained per site-month")
    if not points.groupby(keys)[["truth_daod", "unique_rows"]].nunique(dropna=False).eq(1).all().all():
        raise ValueError("Reference/support changes between repeats")
    values = points[["truth_daod", *[e + "_daod" for e in ESTIMATORS]]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid saved scatter coordinate")
    n = points.unique_rows.to_numpy(float)
    if not np.isfinite(n).all() or (n < 3).any() or (n != np.floor(n)).any():
        raise ValueError("Saved cells violate minimum-three support")
    points["panel"] = "abc"
    for e in ESTIMATORS:
        points[e + "_outside_axes"] = (points.truth_daod > .8) | (points[e + "_daod"] > .8)
    data.append(points)

    scores = get(ROOT + "site_month_metrics", dict(**MONTHLY, stratum="all"), "annotation")
    scores = scores.loc[scores.estimator.isin(ESTIMATORS)].copy()
    if scores.estimator.duplicated().any() or set(scores.estimator) != set(ESTIMATORS):
        raise ValueError("Missing or ambiguous saved annotation")
    _counts(scores)
    if not np.isfinite(scores[["bias", "rmse", "r"]].to_numpy(float)).all():
        raise ValueError("Missing saved annotation value")
    count_values = dict(prediction_events=len(points), unique_site_months=len(points[keys].drop_duplicates()),
                        sites=points.site_key.nunique(), repeats=2)
    for column, count in count_values.items():
        if not scores[column].eq(count).all():
            raise ValueError(f"Scatter/annotation count mismatch: {column}")
    for _, repeat in points.groupby("repeat"):
        if not scores.unique_rows.eq(repeat.unique_rows.sum()).all():
            raise ValueError("Underlying collocation count differs from saved score support")
    scores["panel"] = scores.estimator.map(dict(zip(ESTIMATORS, "abc")))
    data.append(scores)

    decomposition = get(ROOT + "bias_decomposition_E1", BASE, "source_only")
    if not decomposition.stratum.map(lambda s: s == "all" or s.startswith("region=")).all():
        raise ValueError("Unexpected decomposition stratum")
    if decomposition.duplicated(["stratum", "component"]).any() or "all" not in set(decomposition.stratum):
        raise ValueError("Ambiguous decomposition or missing All")
    _counts(decomposition)
    if not decomposition.prediction_events.eq(2 * decomposition.unique_rows).all():
        raise ValueError("Decomposition must retain collocation-repeat support")
    decomposition["panel"] = "d"
    decomposition["exploratory"] = decomposition.sites.lt(5)
    decomposition.loc[decomposition.component.isin(COMPONENTS), "plot_role"] = "stack"
    for stratum, group in decomposition.groupby("stratum", sort=False):
        if not set(COMPONENTS).issubset(set(group.component)) or group[COUNTS].nunique().gt(1).any():
            raise ValueError("Incomplete four-term decomposition or inconsistent component support")
        stack = group.loc[group.component.isin(COMPONENTS)].set_index("component")
        _intervals(stack, "contribution", "ci_low", "ci_high")
        positive = negative = 0.
        for component in COMPONENTS:
            value = float(stack.loc[component, "contribution"])
            left = positive if value >= 0 else negative
            mask = decomposition.stratum.eq(stratum) & decomposition.component.eq(component)
            decomposition.loc[mask, "stack_left"] = left
            decomposition.loc[mask, "stack_right"] = left + value
            if value >= 0:
                positive += value
            else:
                negative += value
        # These are display positions for the four saved additive terms, not a new statistic or CI.
        decomposition.loc[group.index, "stack_total"] = positive + negative
    data.append(decomposition)

    strata = get(ROOT + "stratum_bias", MONTHLY, "source_only")
    strata = strata.loc[strata.estimator.isin(ESTIMATORS)].copy()
    if strata.duplicated(["stratum", "estimator"]).any():
        raise ValueError("Ambiguous saved stratum/estimator")
    strata["panel"] = "e"
    strata["exploratory"] = strata.sites.lt(5)
    strata.loc[strata.stratum.isin(STRATA), "plot_role"] = "stratum_bias"
    for name in STRATA:
        group = strata.loc[strata.stratum.eq(name)]
        if set(group.estimator) != set(ESTIMATORS):
            raise ValueError(f"Missing fixed saved stratum: {name}; never backfill")
        _counts(group)
        _intervals(group, "bias", "bias_lo", "bias_hi")
        if group[COUNTS].nunique().gt(1).any() or not group.prediction_events.eq(2 * group.unique_site_months).all():
            raise ValueError("Unpaired stratum support")
    data.append(strata)
    evidence.append(dict(status=STATUS, estimators=list(ESTIMATORS), stratum_order=list(STRATA),
                         stack_components=list(COMPONENTS), annotation_metrics="Verbatim mean_over_repeats; not recomputed",
                         scales=dict(abc="site-month", d="collocation", e="site-month"),
                         exploratory_below_sites=5, display_arithmetic="Separate signed cumulative component positions and four-term total only",
                         intervals="Saved endpoints only in e; no component endpoint sums or total CI in d",
                         unused_rows="TP subcomponents and unselected seasonal strata retained as source_only"))
    result = pd.concat(data, ignore_index=True, sort=False)
    result["plotted"] = result.plot_role.ne("source_only")
    for source in evidence[:-1]:
        source["selected_record_indices"] = result.loc[result.source_table.eq(source["table"]), "source_record_index"].tolist()
    return result, clean(evidence)


def region_label(value):
    return {"all": "All", "DB_mislabel_regimes": "Suspected mislabel", "dust_source_regimes": "Dust-source regimes",
            "boreal_North_America": "Boreal N. America", "Amazonia_Cerrado": "Amazonia/Cerrado",
            "mainland_SE_Asia": "Mainland SE Asia", "SW North America": "SW N. America"}.get(
                value.removeprefix("region="), value.removeprefix("region=").replace("_", " "))


def decomposition_order(data):
    q = data.loc[data.plot_role.eq("stack")].drop_duplicates("stratum")
    return ["all", *q.loc[q.stratum.ne("all")].sort_values(["stack_total", "stratum"]).stratum.tolist()]


def limits(values):
    lo, hi = min(0., float(np.min(values))), max(0., float(np.max(values)))
    pad = .08 * (hi - lo or .1)
    return lo - pad, hi + pad


def bounded_ticks(ax):
    lo, hi = ax.get_xlim()
    ticks = MaxNLocator(4).tick_values(lo, hi)
    ax.set_xticks(ticks[(ticks >= lo) & (ticks <= hi)])


def build_figure(data):
    fig = plt.figure(figsize=(style.DOUBLE, 150 * style.MM))
    points = data.loc[data.plot_role.eq("scatter")]
    scores = data.loc[data.plot_role.eq("annotation")].set_index("estimator")
    for i, e in enumerate(ESTIMATORS):
        ax = fig.add_axes([.085 + .315 * i, .68, .25, .275])
        label, color, marker, _ = style.EST[e]
        ax.scatter(points.truth_daod, points[e + "_daod"], s=9, alpha=.5, color=color,
                   marker=marker, linewidths=0, rasterized=True)
        ax.plot([0, .8], [0, .8], color="#7F7F7F", lw=.6, ls="--")
        ax.set(xlim=(0, .8), ylim=(0, .8), xticks=[0, .4, .8], yticks=[0, .4, .8], title=label)
        if i == 0:
            ax.set_ylabel("Site-month DAOD (550 nm)")
        row = scores.loc[e]
        ax.text(.025, .97, f"N = {int(row.unique_site_months)}; sites = {int(row.sites)}\n"
                f"bias = {row.bias:+.3f}\nRMSE = {row.rmse:.3f}\nr = {row.r:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
        style.panel_label(ax, "abc"[i], x=-.15, y=1.035)
    fig.text(.53, .622, "AERONET reference site-month DAOD (550 nm)", ha="center", fontsize=7)
    handles = [Line2D([], [], marker=style.EST[e][2], color=style.EST[e][1], ls="none",
                      markersize=3.5, label=style.EST[e][0]) for e in ESTIMATORS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.54, .581), ncol=3,
               columnspacing=1.2, handletextpad=.4, fontsize=6.5)

    d = fig.add_axes([.215, .135, .245, .37])
    order = decomposition_order(data)
    stack = data.loc[data.plot_role.eq("stack")]
    labels = []
    for y, stratum in enumerate(order):
        group = stack.loc[stack.stratum.eq(stratum)].set_index("component")
        labels.append(region_label(stratum) + ("*" if group.exploratory.iloc[0] else ""))
        for component, (_, color, hatch) in COMPONENTS.items():
            row = group.loc[component]
            d.barh(y, row.contribution, left=row.stack_left, height=.7, color=color, hatch=hatch,
                   edgecolor="#444444", linewidth=.25)
        d.plot(group.stack_total.iloc[0], y, "|", color="black", ms=5, mew=1.)
    d.axvline(0, color="#7F7F7F", lw=.6, ls="--")
    d.set(yticks=range(len(order)), yticklabels=labels, ylim=(len(order) - .4, -.6),
          xlim=limits(stack[["stack_left", "stack_right", "stack_total"]].to_numpy(float)),
          xlabel="QA2 bias contribution (550 nm)")
    bounded_ticks(d)
    d.tick_params(axis="y", length=0, pad=3)
    d.set_title("COLLOCATION-scale decomposition", loc="right", pad=13)
    style.panel_label(d, "d", x=-.78, y=1.04)
    patches = [Patch(facecolor=color, hatch=hatch, edgecolor="#444444", linewidth=.25, label=label)
               for label, color, hatch in COMPONENTS.values()]
    fig.legend(handles=patches, loc="lower left", bbox_to_anchor=(.025, .005), ncol=2,
               columnspacing=.8, handlelength=1.4, labelspacing=.3, fontsize=6.5)

    e_ax = fig.add_axes([.735, .135, .245, .37])
    bias = data.loc[data.plot_role.eq("stratum_bias")]
    labels = []
    for y, stratum in enumerate(STRATA):
        group = bias.loc[bias.stratum.eq(stratum)].set_index("estimator")
        sites = int(group.sites.iloc[0])
        labels.append(f"{region_label(stratum)} ({sites})" + ("*" if sites < 5 else ""))
        for offset, e in zip([-.23, 0., .23], ESTIMATORS):
            row = group.loc[e]
            _, color, marker, _ = style.EST[e]
            e_ax.hlines(y + offset, row.bias_lo, row.bias_hi, color=color, lw=.8)
            e_ax.plot([row.bias_lo, row.bias_hi], [y + offset] * 2, "|", color=color, ms=4, mew=.8)
            e_ax.plot(row.bias, y + offset, marker=marker, color=color, ls="none", ms=3.5,
                      markerfacecolor="white" if row.exploratory else color, mew=.8)
    e_ax.set(yticks=range(len(STRATA)), yticklabels=labels, ylim=(len(STRATA) - .4, -.6),
             xlim=limits(bias[["bias", "bias_lo", "bias_hi"]].to_numpy(float)),
             xlabel="Site-month DAOD bias (550 nm)")
    bounded_ticks(e_ax)
    e_ax.tick_params(axis="y", length=0, pad=3)
    e_ax.axvline(0, color="#7F7F7F", lw=.6, ls="--")
    e_ax.grid(axis="x", color="#E6E6E6", lw=.4)
    e_ax.set_axisbelow(True)
    e_ax.set_title("Site-month strata", loc="right", pad=13)
    style.panel_label(e_ax, "e", x=-.89, y=1.04)
    fig.text(.52, .028, "Open markers / *: <5 sites, exploratory\nBlack ticks in (d): four-term total", fontsize=6.5, linespacing=1.5)
    return fig


def notes(data):
    points = data.loc[data.plot_role.eq("scatter")]
    scores = data.loc[data.plot_role.eq("annotation")].iloc[0]
    all_d = data.loc[data.plot_role.eq("stack") & data.stratum.eq("all")].iloc[0]
    outside = {e: int(points[e + "_outside_axes"].sum()) for e in ESTIMATORS}
    return (f"# Figure 5 Candidate\n\n{STATUS}\n\n"
            f"Site-month support: {int(scores.unique_site_months)} distinct cells, {int(scores.sites)} sites, "
            f"{int(scores.prediction_events)} cell-repeat points, {int(scores.unique_rows)} underlying collocations. "
            "Both CV repeats are plotted separately; annotations are copied from mean_over_repeats, not "
            "recomputed from pooled points or averaged predictions. Fractions are not multiplied again. "
            f"Outside the 0-0.8 axes (retained in CSV and saved statistics): {json.dumps(outside)}.\n\n"
            f"Panel d: {int(all_d.unique_rows)} distinct collocations, {int(all_d.sites)} sites, "
            f"{int(all_d.prediction_events)} prediction events; no minimum-three monthly filter. "
            "Its regional definitions are CV regions, not the overlapping geographic strata in panel e. "
            "Only the four named components are stacked; TP fraction/total-AOD subcomponents remain "
            "source_only. Signed cumulative positions and the additive total are display arithmetic only. "
            "No CI is inferred for the summed total; saved component CIs remain in the CSV, not on stacks.\n\n"
            "Panel e: saved fixed All/pooled/region strata; source masks precede monthly grouping. "
            "Potentially overlapping strata are not summed. Other saved seasonal rows remain source_only. "
            "Open markers and asterisks flag fewer than five sites. Single-site intervals can be degenerate "
            "and do not establish robust regional uncertainty. Saved CI endpoints are drawn directly, "
            "even if they do not enclose a point estimate. The monthly score and stratum tables retain "
            "their respective original core intersections; no new cross-table pairing.\n\n"
            "The author approved this explicitly labeled mixed-scale presentation on 2026-10-14. "
            "No new aggregation, fits, predictions, confidence intervals or bootstrap were performed.\n")


def make(registry_path, output):
    registry_path, output = Path(registry_path), Path(output)
    guard_output(output)
    before = digest(registry_path)
    data, evidence = select(Registry.read(registry_path))
    fig = build_figure(data)
    try:
        check_layout(fig)
        if digest(registry_path) != before:
            raise RuntimeError("Registry changed during rendering")
        style.export(fig, output, STEM, data, registry_path, evidence, __file__)
        (output / "Fig05_caption.md").write_text(CAPTION + "\n")
        (output / "Fig05_notes.md").write_text(notes(data))
        if digest(registry_path) != before:
            raise RuntimeError("Registry changed during export")
    finally:
        plt.close(fig)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make(args.registry, args.output_dir)


if __name__ == "__main__":
    main()
