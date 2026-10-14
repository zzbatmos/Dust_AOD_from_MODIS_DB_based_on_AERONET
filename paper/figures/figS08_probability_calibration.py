"""Saved reliability curves and exact Brier accounting, without new fits."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper.derive_probability_diagnostics import BRANCHES, ESTIMATORS, REFERENCES
from paper.figures import figstyle as style
from paper.numbers import Registry

NAMES = {"full8_common": "full8: comparison", "noSSA5_own": "noSSA5: own",
         "minimal2_own": "minimal2: exploratory", "hierarchy_own": "Hierarchy: own"}
REF_NAMES = {"experimental1020": "Primary 1020-nm reference", "old675LR56": "675-nm / 56-sr sensitivity"}
CURVES = {"probability": ("Raw P", "#0072B2", "o", "-"),
          "probability_calibrated": ("Nested-calibrated P", "#D55E00", "s", "--")}


def select(registry):
    bins = registry.table("expected_value_test/stage_b/calibration_bins", stage="b")
    bins = bins.loc[bins.estimator.isin(ESTIMATORS) & bins.reference.isin(REFERENCES)
                    & bins.branch.isin(BRANCHES)].copy()
    scores = registry.table("derived/probability_brier")
    if bins.duplicated(["reference", "branch", "scheme", "estimator", "bin_low", "bin_high"]).any():
        raise ValueError("Ambiguous reliability curves")
    if scores.duplicated(["reference", "branch", "scheme", "estimator", "repeat_summary"]).any():
        raise ValueError("Ambiguous Brier scores")
    for reference in REFERENCES:
        for branch in BRANCHES:
            for estimator in ESTIMATORS:
                for scheme in ("site_cv", "region_out"):
                    q = scores.loc[scores.reference.eq(reference) & scores.branch.eq(branch)
                        & scores.estimator.eq(estimator) & scores.scheme.eq(scheme)
                        & scores.repeat_summary.eq("mean_over_repeats")]
                    b = bins.loc[bins.reference.eq(reference) & bins.branch.eq(branch)
                        & bins.estimator.eq(estimator) & bins.scheme.eq(scheme)]
                    if len(q) != 1 or b.empty:
                        raise ValueError("Missing requested reference/branch/scheme")
                    r = q.iloc[0]
                    if b.prediction_events.sum() != r.prediction_events:
                        raise ValueError("Brier and reliability support counts differ")
                    if not np.isclose(r.brier, r.reliability - r.resolution + r.uncertainty
                                      + r.within_bin_correction, atol=1e-12, rtol=0):
                        raise ValueError("Brier accounting does not conserve score")
    return bins, scores


def make(registry_path, output):
    registry = Registry.read(registry_path)
    bins, scores = select(registry)
    fig, axes = plt.subplots(4, 4, figsize=(style.DOUBLE, 215 * style.MM))
    fig.subplots_adjust(left=.085, right=.985, bottom=.135, top=.95, wspace=.34, hspace=.77)
    for row, reference in enumerate(REFERENCES):
        fig.text(.52, [.982, .753][row], REF_NAMES[reference], ha="center", fontsize=8)
        fig.text(.52, [.527, .299][row], REF_NAMES[reference] + ": Brier accounting", ha="center", fontsize=8)
        for col, branch in enumerate(BRANCHES):
            ax, score_ax = axes[row, col], axes[row + 2, col]
            for k, estimator in enumerate(ESTIMATORS):
                label, color, marker, line = CURVES[estimator]
                b = bins.loc[bins.reference.eq(reference) & bins.branch.eq(branch)
                    & bins.scheme.eq("site_cv") & bins.estimator.eq(estimator)].sort_values("bin_low")
                ax.vlines(b.predicted, b.observed_lo, b.observed_hi, color=color, lw=.65, alpha=.65)
                ax.plot(b.predicted, b.observed, color=color, ls=line, lw=.8)
                for sparse in (False, True):
                    points = b.loc[b.sites.lt(5).eq(sparse)]
                    ax.plot(points.predicted, points.observed, color=color, marker=marker, ls="none",
                            markersize=3, markerfacecolor="white" if sparse else color)
                s = scores.loc[scores.reference.eq(reference) & scores.branch.eq(branch)
                    & scores.scheme.eq("site_cv") & scores.estimator.eq(estimator)
                    & scores.repeat_summary.eq("mean_over_repeats")].iloc[0]
                terms = [s.reliability, -s.resolution, s.uncertainty, s.within_bin_correction]
                if min(terms + [s.brier]) < -.1 or max(terms + [s.brier]) > .4:
                    raise ValueError("Brier display limits would hide values")
                score_ax.bar(np.arange(4) + (k - .5) * .32, terms, width=.3, color=color,
                             edgecolor=color, hatch="//" if k else None, alpha=.75)
                score_ax.plot(4 + (k - .5) * .32, s.brier, marker=marker, color=color, ls="none", markersize=4)
                score_ax.text(.03, .96 - .13 * k, f"{label}: {s.brier:.3f}", color=color,
                              transform=score_ax.transAxes, va="top", fontsize=6.5)
            ax.plot([0, 1], [0, 1], color="#7F7F7F", ls=":", lw=.65)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1],
                   xlabel="Predicted P(dust)", title=NAMES[branch])
            if col == 0:
                ax.set_ylabel("Reference dust frequency")
                score_ax.set_ylabel("Score / signed contribution")
            score_ax.axhline(0, color="#7F7F7F", lw=.65)
            score_ax.set(xticks=range(5), xticklabels=["REL", "-RES", "UNC", "C", "BS"],
                         ylim=(-.1, .4), title=NAMES[branch])
            style.panel_label(ax, "abcdefgh"[row * 4 + col], x=-.16, y=1.04)
            style.panel_label(score_ax, "ijklmnop"[row * 4 + col], x=-.16, y=1.04)
    handles = [Line2D([], [], color=c, marker=m, ls=l, label=name) for name, c, m, l in CURVES.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .066), ncol=2)
    fig.text(.5, .044, "BS = REL - RES + UNC + C; C retains within-bin probability variation.", ha="center", fontsize=6.5)
    fig.text(.5, .025, "Held-out sites; equal repeat weights. Brier summaries are descriptive, without new confidence intervals.",
             ha="center", fontsize=6.5)
    evidence, frames = [], []
    for name, d in [("expected_value_test/stage_b/calibration_bins", bins), ("derived/probability_brier", scores)]:
        table = registry.payload["tables"][name]
        evidence.append(dict(table=name, source_file=table["source_file"], source_sha256=table["source_sha256"],
                             derivation=table.get("derivation"), display_scheme="site_cv"))
        frames.append(d.assign(source_table=name, plotted=d.scheme.eq("site_cv") & (
            d.repeat_summary.eq("mean_over_repeats") if "repeat_summary" in d else True)))
    style.export(fig, output, "FigS08_probability_calibration", pd.concat(frames, ignore_index=True),
                 registry_path, evidence, __file__)
    (Path(output) / "FigS08_caption.md").write_text(
        "Probability reliability (a-h) and Brier-score accounting (i-p) on held-out sites, for the primary "
        "1020-nm reference and the 675-nm/56-sr sensitivity. Dust labels require reference fraction >=0.2. "
        "Raw and nested-calibrated curves use saved bin means and 95% site-bootstrap intervals; open markers "
        "indicate fewer than five sites. Branch populations differ; noSSA5-own is not fallback-only and minimal2 "
        "is exploratory, not deployed. Bars show reliability (REL), negative resolution (-RES), uncertainty (UNC), "
        "and the signed within-bin correction (C); markers show actual Brier score (BS). The correction preserves "
        "the exact score for continuous probabilities. Metrics are computed within each repeat then equally averaged, "
        "not from averaged predictions. Brier terms have no new uncertainty estimates. Held-out-region results and "
        "per-repeat accounting are retained in the companion data.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
