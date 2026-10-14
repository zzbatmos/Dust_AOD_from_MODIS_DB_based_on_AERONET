"""Held-out pixel diagnostics, exclusively from registered arrays and statistics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output):
    registry = Registry.read(registry_path)
    data, evidence = [], []

    def table(name, panel, **filters):
        selected = registry.table(name, **filters)
        if selected.empty:
            raise ValueError(f"No plot data: {name}/{filters}")
        source = registry.payload["tables"][name]
        evidence.append({"table": name, "source_file": source["source_file"],
                         "source_sha256": source["source_sha256"], "filter": filters,
                         "derivation": source.get("derivation")})
        data.append(selected.assign(panel=panel, source_table=name))
        return selected

    fig = plt.figure(figsize=(style.DOUBLE, 150 * style.MM))
    a = fig.add_axes([.08, .65, .35, .27])
    histogram = fig.add_axes([.08, .56, .35, .055], sharex=a)
    b = fig.add_axes([.64, .58, .34, .34])
    c = fig.add_axes([.08, .15, .35, .31])
    d = fig.add_axes([.75, .13, .23, .34])
    branches = [("full8_common", "full8", "#332288", "o"),
                ("noSSA5_own", "noSSA5", "#88CCEE", "s")]
    for offset, (branch, label, color, marker) in zip((-.23, .23), branches):
        bins = table("expected_value_test/stage_b/calibration_bins", "a",
                     reference="experimental1020", scheme="site_cv", branch=branch)
        for estimator, suffix, linestyle, face in (("probability", "raw", "-", color),
                                                  ("probability_calibrated", "calibrated", "--", "white")):
            q = bins[bins.estimator.eq(estimator)].sort_values("bin_low")
            a.plot(q.predicted, q.observed, color=color, marker=marker, markerfacecolor=face,
                   markeredgecolor=color, ls=linestyle, markersize=3.5, label=f"{label} {suffix}")
            if estimator == "probability":
                widths = q.bin_high - q.bin_low
                histogram.bar((q.bin_low + q.bin_high) / 2 + widths * offset,
                              q.prediction_events, width=widths * .43,
                              color=color, edgecolor="#4D4D4D", linewidth=.3)
    a.plot([0, 1], [0, 1], ls="--", color="#7F7F7F", lw=.6)
    a.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed dust frequency")
    a.tick_params(labelbottom=False)
    a.legend(loc="upper left", fontsize=6.5, handlelength=1.8, labelspacing=.15)
    histogram.set(xlabel="Dust probability", ylabel="Events")
    histogram.ticklabel_format(axis="y", style="plain")
    histogram.yaxis.set_major_locator(plt.MaxNLocator(2))
    style.panel_label(a, "a")

    styles = [("full8", "#332288", "-"), ("noSSA5", "#88CCEE", "--"),
              ("hierarchy_prespecified", "#D55E00", "-.")]
    names = {"full8": "full8", "noSSA5": "noSSA5", "hierarchy_prespecified": "Hierarchy"}
    prevalence_drawn = []
    for model, color, linestyle in styles:
        curve = table("derived/pixel_pr_curves", "b", model=model).sort_values("point_index")
        n = int(curve.unique_rows.iloc[0])
        b.plot(curve.recall, curve.precision, color=color, ls=linestyle, lw=1,
               label=f"{names[model]} (N={n:,})")
        points = table("derived/pixel_qa_operating_points", "b", model=model)
        for tier, marker in ((1, "o"), (2, "s"), (3, "^")):
            q = points[points.qa_tier.eq(tier)]
            b.plot(q.recall, q.precision, marker=marker, markersize=4, color=color,
                   markeredgecolor="#4D4D4D", markeredgewidth=.4, linestyle="none")
        prevalence = table("derived/pixel_prevalences", "b", model=model, scope="classifier").prevalence.item()
        if not any(np.isclose(prevalence, previous, atol=0, rtol=0) for previous in prevalence_drawn):
            b.axhline(prevalence, color=color, ls=":", lw=.8)
            prevalence_drawn.append(prevalence)
    baseline = table("derived/pixel_baseline_points", "b", model="hierarchy_prespecified")
    for estimator, label, color, marker in (
        ("DB_dust_flag", "DB dust flag", "#7F7F7F", "D"),
        ("AEfixedthreshold", "AE < 0.6", "black", "+"),
        ("LGscreened_coarseproxy", "LG25 screened*", "#CC79A7", "v"),
    ):
        row = baseline[baseline.estimator.eq(estimator)]
        b.plot(row.recall, row.precision, marker=marker, color=color, ls="none",
               markersize=4, label=label)
    b.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    b.legend(loc="upper right", fontsize=6.5, handlelength=1.8, labelspacing=.15)
    b.text(.025, .05, "QA1: circle; QA2: square; QA3: triangle\nDotted: sample prevalence",
           transform=b.transAxes, fontsize=6.5, va="bottom")
    style.panel_label(b, "b", x=-.15)

    scatter = table("derived/pixel_ev_scatter", "c", reference="experimental1020", scheme="site_cv")
    density = c.hexbin(scatter.truth_daod, scatter.Eselected_daod, gridsize=45,
                       extent=(0, 1.5, 0, 1.5), mincnt=1, norm=LogNorm(),
                       cmap=style.CMAP_DENSITY, linewidths=0, rasterized=True)
    x = np.array([0, 1.5])
    for y in (x, x + .05 + .2 * x, x - .05 - .2 * x):
        c.plot(x, y, color="#7F7F7F", lw=.6, ls="--")
    c.set(xlim=(0, 1.5), ylim=(0, 1.5), xlabel="AERONET reference DAOD (550 nm)",
          ylabel="Expected-value DAOD (550 nm)")
    summary_filter = dict(stage="b", reference="experimental1020", branch="hierarchy_own",
                          scheme="site_cv", pairing="core", stratum="all", estimator="Eselected",
                          repeat_summary="mean_over_repeats", minimum=1)
    scores = {}
    for metric in ("unique_rows", "bias", "rmse", "r"):
        result = registry.number("expected_value_test/stage_b/collocation_metrics", metric, **summary_filter)
        scores[metric] = result["value"]
        evidence.append(result)
    outside = registry.number("derived/plot_range_counts", "n_outside", figure="Fig04", panel="c",
                              reference="experimental1020", scheme="site_cv", estimator="Eselected")
    evidence.append(outside)
    c.text(.035, .97, f"N = {int(scores['unique_rows']):,}\nbias = {scores['bias']:+.3f}; RMSE = {scores['rmse']:.3f}\n"
           f"r = {scores['r']:.2f}; outside axes: {int(outside['value'])} events",
           transform=c.transAxes, fontsize=6.5, va="top")
    colorbar = fig.colorbar(density, cax=fig.add_axes([.145, .055, .22, .012]), orientation="horizontal")
    colorbar.set_label("Prediction-event count", fontsize=7)
    style.panel_label(c, "c")

    overrides = table("derived/pixel_overrides", "d")
    overrides = overrides[overrides.eligible > 0].copy()
    overrides["order"] = overrides.group.map(lambda x: "" if x == "all" else x)
    overrides = overrides.sort_values("order")
    for index, row in enumerate(overrides.itertuples(index=False)):
        hatch = "////" if row.eligible_sites < 5 else None
        d.barh(index, row.supported_rejection_fraction, color="#0072B2", height=.72,
               edgecolor="#4D4D4D", linewidth=.3, hatch=hatch)
        d.barh(index, row.unsupported_rejection_fraction, left=row.supported_rejection_fraction,
               color="#E69F00", height=.72, edgecolor="#4D4D4D", linewidth=.3, hatch=hatch)
    labels = [("All" if r.group == "all" else r.group.replace("_", " ")) + f" ({int(r.eligible)})"
              for r in overrides.itertuples(index=False)]
    d.set_yticks(range(len(labels)), labels)
    d.set_ylim(len(labels) - .4, -.6)
    d.set_xlim(0, 1)
    d.set_xticks([0, .5, 1])
    d.tick_params(axis="y", length=0)
    d.set_xlabel("Fraction of DB-dust events")
    d.set_title("Hierarchy QA2 rejections", pad=9)
    style.panel_label(d, "d", x=-.95)
    fig.legend(handles=[Line2D([], [], color="#0072B2", lw=4, label="Reference-supported"),
                        Line2D([], [], color="#E69F00", lw=4, label="Reference-unsupported")],
               loc="lower right", bbox_to_anchor=(.98, .01), fontsize=6.5)
    style.export(fig, output, "Fig04_pixel_skill", pd.concat(data, ignore_index=True),
                 registry_path, evidence, __file__)
    caption = (
        "Held-out site diagnostics. (a) Raw and calibrated reliability with raw-score event counts. "
        "(b) Precision-recall curves on branch-specific samples; dotted lines show prevalence. "
        "Screened LG25 (*) has smaller support; DB dust flag is not the type-conditional baseline. "
        "(c) Expected-value DAOD; N counts distinct collocations, both repeats are plotted, and scores "
        "average repeat-specific metrics. Dashed lines indicate equality and the expected-error envelope. "
        "(d) Rejections divided by eligible DB-dust events (parentheses); hatching marks fewer than five eligible sites."
    )
    (Path(output) / "Fig04_caption.md").write_text(caption + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
