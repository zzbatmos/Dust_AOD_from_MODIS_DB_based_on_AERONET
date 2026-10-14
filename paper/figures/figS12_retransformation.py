"""Display final Stage B smearing and AE-fit diagnostics, with no new fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import Registry


ROOT = "expected_value_test/stage_b_support/"
TABLES = ("f_pos_retransformation_stage_b", "E8_coefficients")
KEYS = ["reference", "population", "branch", "split_id", "scheme", "repeat"]
REFERENCES = ("experimental1020", "old675LR56")
BRANCHES = ("full8", "noSSA5")
COLORS = {"full8": "#0072B2", "noSSA5": "#D55E00"}
MARKERS = {"full8": "o", "noSSA5": "s"}


def select(registry):
    frames, evidence = {}, []
    for name in TABLES:
        frame = registry.table(ROOT + name, population="own")
        frame = frame.loc[frame.reference.isin(REFERENCES) & frame.branch.isin(BRANCHES)].copy()
        if frame.empty or frame.duplicated(KEYS).any():
            raise ValueError("Missing or ambiguous fold diagnostics")
        if set(zip(frame.reference, frame.branch)) != {(r, b) for r in REFERENCES for b in BRANCHES}:
            raise ValueError("Missing reference/branch diagnostics")
        if set(frame.scheme) != {"site_cv", "region_out"}:
            raise ValueError("Missing validation scheme")
        frames[name] = frame
        table = registry.payload["tables"][ROOT + name]
        evidence.append(dict(table=ROOT + name, source_file=table["source_file"],
                             source_sha256=table["source_sha256"], filters=dict(
                                 population="own", reference=list(REFERENCES), branch=list(BRANCHES))))
    diagnostic, coefficients = (frames[k] for k in TABLES)
    left = diagnostic.set_index(KEYS).sort_index()
    right = coefficients.set_index(KEYS).sort_index()
    if not left.index.equals(right.index) or not left.training_rows.equals(right.training_rows):
        raise ValueError("Smearing/coefficient fold populations disagree")
    if not coefficients.target.eq("dust fraction (NOT FMF)").all():
        raise ValueError("AE fit is not the saved dust-fraction target")
    if not coefficients.coefficient_order.eq("a + b*DB_AE + c*DB_AE^2").all():
        raise ValueError("Unknown coefficient ordering")
    if not np.isfinite(coefficients[["a", "b", "c"]].to_numpy()).all():
        raise ValueError("Nonfinite saved AE coefficient")
    if not diagnostic.selection_criterion.eq("site_month_RMSE").all():
        raise ValueError("Not the final Stage B selection criterion")
    means = diagnostic[["positive_observed_mean", "f_pos_mean", "f_pos_star_mean"]]
    present = diagnostic.evaluation_positive_rows.gt(0)
    if (diagnostic.evaluation_positive_rows < 0).any() or not means.loc[~present].isna().all().all():
        raise ValueError("Zero-positive diagnostics must remain missing")
    valid = means.loc[present].to_numpy(float)
    if not np.isfinite(valid).all() or ((valid < 0) | (valid > 1)).any():
        raise ValueError("Invalid saved positive-fraction mean")
    return frames, pd.concat([d.assign(diagnostic=k) for k, d in frames.items()],
                            ignore_index=True, sort=False), evidence


def make(registry_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper.figures import figstyle as style

    registry = Registry.read(registry_path)
    frames, exported, evidence = select(registry)
    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE, 155 * style.MM))
    fig.subplots_adjust(left=.12, right=.97, bottom=.17, top=.93, wspace=.30, hspace=.58)
    coefficient_limit = float(np.ceil(np.abs(frames[TABLES[1]][["a", "b", "c"]].to_numpy()).max() * 10) / 10)
    for col, reference in enumerate(REFERENCES):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=.7)
        for branch in BRANCHES:
            q = frames[TABLES[0]]
            q = q.loc[q.reference.eq(reference) & q.branch.eq(branch) & q.evaluation_positive_rows.gt(0)]
            for field, face in (("f_pos_mean", "none"), ("f_pos_star_mean", COLORS[branch])):
                ax.scatter(q.positive_observed_mean, q[field], marker=MARKERS[branch], s=15, alpha=.65,
                           facecolors=face, edgecolors=COLORS[branch], linewidths=.6)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Held-out positive-reference mean fraction",
               ylabel="Predicted positive mean fraction", title=("Primary 1020-nm reference" if col == 0
                                                                else "Old 675-nm / 56-sr reference"))
        style.panel_label(ax, "ab"[col], y=1.07)
        ax = axes[1, col]
        ax.axhline(0, color="#888888", lw=.7)
        for bi, branch in enumerate(BRANCHES):
            q = frames[TABLES[1]]
            q = q.loc[q.reference.eq(reference) & q.branch.eq(branch)].sort_values(["scheme", "split_id"])
            # Deterministic horizontal displacement separates fold points; no values are averaged.
            jitter = np.linspace(-.055, .055, len(q))
            for i, coefficient in enumerate(("a", "b", "c")):
                ax.scatter(i + (bi - .5) * .20 + jitter, q[coefficient], c=COLORS[branch],
                           marker=MARKERS[branch], s=10, alpha=.65)
        ax.set(xticks=range(3), xticklabels=["a (constant)", "b (linear)", "c (quadratic)"],
               ylim=(-coefficient_limit, coefficient_limit),
               ylabel="Saved AE-fit coefficient", title="Dust-fraction fit: a + b AE + c AE squared")
        style.panel_label(ax, "cd"[col], y=1.07)
    branch_handles = [Line2D([], [], color=COLORS[b], marker=MARKERS[b],
                              markerfacecolor="none", ls="none",
                              label=f"{b}: own population") for b in BRANCHES]
    smear_handles = [Line2D([], [], color="black", marker=m, markerfacecolor=f, ls="none", label=l)
                     for m, f, l in [("o", "none", "Unsmeared positive mean"),
                                      ("o", "black", "Residual-smeared positive mean")]]
    fig.legend(handles=branch_handles + smear_handles, loc="lower center", bbox_to_anchor=(.54, .02), ncol=2)
    for record in evidence:
        record["semantics"] = ("Final Stage B own-population fold artifacts; all site-CV repeats and "
                               "held-out regions retained. No refits, pooling of diagnostics or new bootstrap. "
                               "Zero-positive folds remain in CSV with missing means. Scatter spread is not a CI.")
    style.export(fig, output, "FigS12_retransformation", exported, registry_path, evidence, __file__)
    missing = int(frames[TABLES[0]].evaluation_positive_rows.eq(0).sum())
    (Path(output) / "FigS12_caption.md").write_text(
        "Final Stage B retransformation and AE-fit diagnostics under both AERONET references. "
        "Top: each marker is one held-out fold's positive-reference mean versus its unsmeared "
        "(open markers) or empirical inner-OOF residual-smeared (filled markers) positive prediction mean. "
        "Circles denote full8 and squares noSSA5 in both rows; top-row fill denotes smearing, not branch. "
        f"The {missing} folds without positive evaluation records retain missing means in the CSV and are "
        "not plotted. Bottom: saved coefficients of a + b AE + c AE squared fitted to reference dust "
        "fraction, not fine-mode fraction. Each point is one fold; horizontal offsets only separate points. "
        "Full8 and noSSA5 are evaluated on their own populations, not identical supports. Both site-CV "
        "repeats and region-held-out folds are shown; overlapping training sets mean the point spread "
        "is not a confidence interval. These are the final nested Stage B artifacts, not earlier Stage A fits.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
