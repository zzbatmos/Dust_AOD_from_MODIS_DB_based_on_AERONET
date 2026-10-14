"""Plot saved old-reference site-month sensitivity, never recomputing metrics.

Only hierarchy_own/core is selected. Keep all rows/columns from both selected
source tables in FigS11_data.csv; plot flags identify the deliberately smaller
display. Repeat details exist only at minimum 3 in the frozen Stage B tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry, clean, digest

ROOT = "expected_value_test/stage_b/"
TABLES = ("site_month_metrics", "stratum_bias")
FILTERS = dict(stage="b", reference="old675LR56", branch="hierarchy_own", pairing="core")
KEYS = ["stage", "reference", "branch", "scheme", "pairing", "stratum",
        "estimator", "repeat_summary", "minimum"]
COUNTS = ["prediction_events", "sites", "repeats", "unique_rows", "unique_site_months"]
MINIMA = (1, 3, 5)
REPEATS = ("repeat_0", "repeat_1")
SCHEMES = ("site_cv", "region_out")
ESTIMATORS = ("E1", "Eselected", "E5", "E7")
METRICS = ("bias", "rmse", "r")
# Figure-spec Fig. 5(e) defines strata with fewer than five sites as exploratory.
EXPLORATORY_BELOW_SITES = 5
STEM = "FigS11_site_month_sensitivity"


def _validate(frame, name):
    required = set(KEYS + COUNTS + list(METRICS))
    if not required.issubset(frame) or frame.empty:
        raise ValueError(f"Missing {name} rows or required columns")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Ambiguous {name} metric keys")
    for column in COUNTS:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid saved counts: {column}")
    if ((frame.sites > frame.unique_site_months) | (frame.unique_site_months > frame.unique_rows)
            | (frame.prediction_events < frame.unique_site_months)
            | (frame.prediction_events > frame.unique_site_months * frame.repeats)).any():
        raise ValueError("Inconsistent site-month/collocation/repeat counts")
    if frame.loc[frame.repeat_summary.isin(REPEATS), "repeats"].ne(1).any():
        raise ValueError("Per-repeat row reports multiple repeats")
    grouped = frame.groupby([key for key in KEYS if key != "estimator"], dropna=False)
    if grouped[COUNTS].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Estimator counts disagree within a source-table paired support")
    for metric in METRICS:
        bounds = [f"{metric}_lo", f"{metric}_hi"]
        if sum(column in frame for column in bounds) == 1:
            raise ValueError(f"Incomplete interval columns: {metric}")
        for column in bounds:
            if column not in frame:
                frame[column] = np.nan
        values = frame[[metric] + bounds].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite saved {metric}")
        finite = np.isfinite(values)
        available = finite[:, 1:].all(axis=1)
        if (finite[:, 1] != finite[:, 2]).any() or (values[available, 1] > values[available, 2]).any():
            raise ValueError(f"Malformed saved {metric} interval")
        if (available & ~finite[:, 0]).any():
            raise ValueError(f"Interval without a saved {metric} estimate")
        if metric == "rmse" and (values[finite] < 0).any():
            raise ValueError("Negative RMSE")
        if metric == "r" and (np.abs(values[finite]) > 1).any():
            raise ValueError("Correlation outside [-1, 1]")
        valid_column = f"{metric}_valid_bootstraps"
        if valid_column in frame:
            valid = frame[valid_column].to_numpy(float)
            if not np.isfinite(valid).all() or (valid < 0).any() or (valid != np.floor(valid)).any():
                raise ValueError("Invalid saved bootstrap count")
            if (available & (valid == 0)).any():
                raise ValueError("Saved interval has no valid bootstraps")
        frame[f"{metric}_interval_available"] = available


def select(registry):
    """Select exact supports; preserve absent strata/methods and all saved columns."""
    frames, evidence = {}, []
    for name in TABLES:
        frame = registry.table(ROOT + name, **FILTERS)
        frame = frame.loc[frame.scheme.isin(SCHEMES) & frame.minimum.isin(MINIMA)].copy()
        frame["source_record_index"] = frame.index
        _validate(frame, name)
        if set(frame.scheme) != set(SCHEMES):
            raise ValueError(f"Missing validation scheme in {name}")
        frame["source_table"] = ROOT + name
        frame["exploratory"] = frame.sites.lt(EXPLORATORY_BELOW_SITES)
        frame["plot_role"] = "source_only"
        frames[name] = frame
        metadata = registry.payload["tables"][ROOT + name]
        evidence.append(dict(table=ROOT + name, source_file=metadata["source_file"],
                             source_sha256=metadata["source_sha256"], semantics=metadata["semantics"],
                             filters=FILTERS, source_record_indices=frame.source_record_index.tolist()))

    monthly = frames["site_month_metrics"]
    # Fail rather than using region-out, another support, or another minimum as a substitute.
    main = monthly.scheme.eq("site_cv") & monthly.stratum.eq("all") & monthly.estimator.isin(ESTIMATORS)
    for minimum, summary in [(m, "mean_over_repeats") for m in MINIMA] + [(3, r) for r in REPEATS]:
        q = monthly.loc[main & monthly.minimum.eq(minimum) & monthly.repeat_summary.eq(summary)]
        if set(q.estimator) != set(ESTIMATORS):
            raise ValueError(f"Missing site-CV minimum/repeat rows: {minimum}/{summary}")
    monthly.loc[main & monthly.repeat_summary.eq("mean_over_repeats"), "plot_role"] = "minimum_sensitivity"
    monthly.loc[main & monthly.minimum.eq(3) & monthly.repeat_summary.isin(REPEATS), "plot_role"] = "repeat_detail"

    strata = frames["stratum_bias"]
    stratum_selection = (strata.minimum.eq(3) & strata.repeat_summary.eq("mean_over_repeats")
                         & strata.stratum.ne("all"))
    small = strata.loc[stratum_selection & strata.exploratory, "stratum"].unique()
    if not len(small):
        raise ValueError("No saved exploratory strata for this reference/support")
    # Show both schemes for the union of small-site strata, not a score-selected subset.
    strata.loc[stratum_selection & strata.stratum.isin(small) & strata.estimator.isin(ESTIMATORS),
               "plot_role"] = "exploratory_stratum"
    absence = []
    for (scheme, stratum), q in strata.loc[stratum_selection].groupby(["scheme", "stratum"]):
        for estimator in ESTIMATORS:
            if estimator not in set(q.estimator):
                absence.append(dict(scheme=scheme, stratum=stratum, estimator=estimator, status="not_saved"))
    evidence.append(dict(
        plot_selection="Site-CV minima 1/3/5 mean_over_repeats; site-CV repeat_0/repeat_1 at minimum 3; "
        "minimum-3 mean_over_repeats bias for all strata with <5 sites in either scheme. "
        "Other selected rows remain in CSV as source_only.",
        absent_estimator_strata=absence, exploratory_below_sites=EXPLORATORY_BELOW_SITES,
        exploratory_rule_source="FIGURE_SPEC_AND_STYLE_GUIDE.md Fig. 5(e)",
        monthly_core_estimators=["E0", "E1_QA1", "E1", "E1_QA3", "E2", "E3", "E5", "E7", "E3c", "E4", "E8", "Eselected"],
        stratum_core_estimators=["E1", "E3", "E3c", "E4", "Eselected", "E5"],
        support_semantics="Pair before monthly minimum, independently within each source table's core "
        "estimator intersection. Strata can overlap; no sums across strata or cross-table pooling.",
        count_semantics="prediction_events counts site-month-repeat records, unique_site_months counts "
        "distinct site/year/month cells, unique_rows counts underlying collocations; none are interchangeable.",
        repeat_semantics="mean_over_repeats is the saved arithmetic mean of repeat-specific metrics, "
        "not pooled events, averaged predictions, or an average calculated by this renderer."))
    data = pd.concat(frames.values(), ignore_index=True, sort=False)
    data["plotted"] = data.plot_role.ne("source_only")
    data["plot_metrics"] = np.where(data.plot_role.eq("exploratory_stratum"), "bias",
                                    np.where(data.plotted, "bias,rmse,r", ""))
    return data, clean(evidence)


def _limits(data, metric):
    values = data[[metric, f"{metric}_lo", f"{metric}_hi"]].to_numpy(float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"No saved finite {metric} values in plot selection")
    lo, hi = values.min(), values.max()
    if metric in ("bias", "rmse"):
        lo, hi = min(0, lo), max(0, hi)
    padding = .08 * (hi - lo or 1)
    ticks = MaxNLocator(nbins=4).tick_values(lo, hi)
    return (lo - padding, hi + padding), ticks[(ticks >= lo) & (ticks <= hi)]


def _point(ax, row, metric, position, horizontal=False, open_marker=False):
    _, color, marker, _ = style.EST[row.estimator]
    value = row[metric]
    if pd.isna(value):
        return  # A missing saved value is never converted to zero.
    if row[f"{metric}_interval_available"]:
        lo, hi = row[f"{metric}_lo"], row[f"{metric}_hi"]
        if horizontal:
            ax.hlines(position, lo, hi, color=color, lw=.8)
            ax.plot([lo, hi], [position] * 2, "|", color=color, markersize=4, mew=.8)
        else:
            ax.vlines(position, lo, hi, color=color, lw=.8)
            ax.plot([position] * 2, [lo, hi], "_", color=color, markersize=4, mew=.8)
    x, y = (value, position) if horizontal else (position, value)
    ax.plot(x, y, marker=marker, linestyle="none", color=color, markersize=3.5,
            markerfacecolor="white" if open_marker else color, markeredgewidth=.8)


def build_figure(data):
    fig = plt.figure(figsize=(style.DOUBLE, 225 * style.MM))
    fig.text(.08, .98, "Old 675-nm / 56-sr reference | hierarchy own | core paired supports", fontsize=7, va="top")
    handles = [Line2D([], [], color=style.EST[e][1], marker=style.EST[e][2], linestyle=style.EST[e][3],
                      markersize=4, label=style.EST[e][0]) for e in ESTIMATORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .959), ncol=4, fontsize=7,
               columnspacing=1.2, handlelength=1.8)
    top = data.loc[data.plot_role.isin(["minimum_sensitivity", "repeat_detail"])]
    for row, role in enumerate(("minimum_sensitivity", "repeat_detail")):
        selection = top.loc[top.plot_role.eq(role)]
        fig.text(.08, .9 if row == 0 else .645,
                 "Site CV: mean of repeat-specific metrics" if row == 0 else "Site CV: separate repeats, minimum 3 collocations",
                 fontsize=7)
        labels = MINIMA if row == 0 else REPEATS
        column = "minimum" if row == 0 else "repeat_summary"
        for col, metric in enumerate(METRICS):
            ax = fig.add_axes([.10 + .305 * col, .72 if row == 0 else .465, .235, .15])
            for offset, estimator in zip(np.linspace(-.21, .21, len(ESTIMATORS)), ESTIMATORS):
                q = selection.loc[selection.estimator.eq(estimator)].set_index(column)
                _, color, _, linestyle = style.EST[estimator]
                values = [q.loc[label, metric] if label in q.index else np.nan for label in labels]
                ax.plot(np.arange(len(labels)) + offset, values, color=color, ls=linestyle, lw=1.)
                for x, label in enumerate(labels):
                    if label in q.index:
                        _point(ax, q.loc[label], metric, x + offset)
            bounds, ticks = _limits(top, metric)
            ax.set(ylim=bounds, yticks=ticks, xlim=(-.5, len(labels) - .5), xticks=range(len(labels)),
                   xticklabels=[str(v) for v in MINIMA] if row == 0 else ["Repeat 0", "Repeat 1"],
                   xlabel="Minimum collocations / site-month" if row == 0 else "Minimum = 3",
                   ylabel={"bias": "DAOD bias (550 nm)", "rmse": "DAOD RMSE (550 nm)", "r": "DAOD correlation (r)"}[metric])
            if metric == "bias":
                ax.axhline(0, color="#7F7F7F", lw=.6, ls="--")
            ax.grid(axis="y", color="#E6E6E6", lw=.4)
            ax.set_axisbelow(True)
            style.panel_label(ax, "abcdef"[row * 3 + col], x=0, y=1.04)

    subset = data.loc[data.plot_role.eq("exploratory_stratum")]
    groups = sorted(subset.stratum.unique(), key=str.casefold)
    shown = [e for e in ESTIMATORS if subset.estimator.eq(e).any()]
    fig.text(.08, .389, "Exploratory strata (<5 sites): minimum 3; mean of repeat-specific metrics", fontsize=7)
    bounds, ticks = _limits(subset, "bias")
    for col, scheme in enumerate(SCHEMES):
        ax = fig.add_axes([.34 + .35 * col, .09, .275, .26])
        q = subset.loc[subset.scheme.eq(scheme)]
        labels = []
        for y, group in enumerate(groups):
            stratum = q.loc[q.stratum.eq(group)]
            if stratum.empty:
                labels.append(group.replace("_", " ") + "\nnot saved")
            else:
                record = stratum.iloc[0]
                name = group.replace("_", " ")
                cells, sites = int(record.unique_site_months), int(record.sites)
                labels.append(f"{name}\n{cells} site-month{'s' if cells != 1 else ''}; "
                              f"{sites} site{'s' if sites != 1 else ''}")
            for offset, estimator in zip(np.linspace(-.25, .25, len(shown)), shown):
                cell = stratum.loc[stratum.estimator.eq(estimator)]
                if not cell.empty:
                    record = cell.iloc[0]
                    _point(ax, record, "bias", y + offset, horizontal=True, open_marker=record.exploratory)
        ax.set(xlim=bounds, xticks=ticks, ylim=(len(groups) - .4, -.6), yticks=range(len(groups)),
               yticklabels=labels if col == 0 else [], xlabel="DAOD bias (550 nm)",
               title="Site CV" if scheme == "site_cv" else "Region-out")
        # Counts on the shared left labels must apply to both schemes, or be shown separately.
        if col == 1:
            left = subset.loc[subset.scheme.eq(SCHEMES[0])].groupby("stratum")[["sites", "unique_site_months"]].first()
            right = q.groupby("stratum")[["sites", "unique_site_months"]].first()
            if not left.equals(right):
                raise ValueError("Scheme counts differ: shared exploratory count labels would mislead")
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color="#7F7F7F", lw=.6, ls="--")
        ax.grid(axis="x", color="#E6E6E6", lw=.4)
        ax.set_axisbelow(True)
        style.panel_label(ax, "gh"[col], x=0, y=1.04)
    fig.text(.08, .016, "Bars: saved 95% site-bootstrap intervals. Open markers: exploratory.\n"
             "Single-site intervals can have zero width. Missing values/estimator-strata are not zero.",
             fontsize=6.5, va="bottom", linespacing=1.5)
    return fig


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if text.get_fontsize() < 6.5:
            raise ValueError(f"Text below 6.5 pt: {text.get_text()}")
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError(f"Text outside figure: {text.get_text()}")
        for previous, other in boxes:
            if min(box.x1, other.x1) - max(box.x0, other.x0) > .5 and min(box.y1, other.y1) - max(box.y0, other.y0) > .5:
                raise ValueError(f"Overlapping text: {previous!r} / {text.get_text()!r}")
        boxes.append((text.get_text(), box))


def caption(data):
    strata = data.loc[data.plot_role.eq("exploratory_stratum")]
    absent = [e for e in ESTIMATORS if not strata.estimator.eq(e).any()]
    return ("Old675LR56 sensitivity on saved hierarchy-own, core-paired site-month supports. "
            "(a-c) Site-CV bias, RMSE and correlation for minimum 1, 3 or 5 collocations per site-month. "
            "Each minimum changes the eligible population. (d-f) Separate site-CV repeat 0 and repeat 1 "
            "at minimum 3; no repeat-detail rows are saved for minima 1 or 5. Mean-over-repeats rows "
            "average repeat-specific metrics, not predictions or pooled events. (g-h) Minimum-3 bias "
            "for every saved stratum with fewer than five sites in either scheme, with open markers "
            "where exploratory; site-CV and region-out remain separate. Labels give distinct "
            "site-months and sites, not prediction-event counts. Strata may overlap and are not additive. "
            "Bars are saved 95% site-cluster-bootstrap intervals, not repeat ranges or new bootstrap. "
            "Single-site intervals can collapse to zero width and do not establish robust uncertainty. "
            "Bias is prediction minus reference; valid zeros and undefined values remain distinct. "
            "E1 is QA2, Eselected the saved expected-value selector, E5 the type-conditional baseline, "
            "and E7 unscreened LG25. " + (f"{', '.join(absent)} has no saved plotted stratum rows and is not backfilled. " if absent else "") +
            "The site-month and stratum tables retain their respective core estimator intersections "
            "(12 and 6 estimators); no cross-table pooling is performed. The accompanying CSV retains "
            "all old-reference hierarchy/core rows from both tables, including unplotted estimators, "
            "strata and region-out sensitivity, source counts, and plotting flags. No fits or new inference.\n")


def make(registry_path, output):
    before = digest(Path(registry_path))
    data, evidence = select(Registry.read(registry_path))
    fig = build_figure(data)
    try:
        check_layout(fig)
        if digest(Path(registry_path)) != before:
            raise RuntimeError("Registry changed during rendering")
        style.export(fig, output, STEM, data, registry_path, evidence, __file__)
        (Path(output) / "FigS11_caption.md").write_text(caption(data))
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
