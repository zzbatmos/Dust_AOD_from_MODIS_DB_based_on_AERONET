"""Saved expected-fraction calibration; not classifier probability reliability."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output):
    registry = Registry.read(registry_path)
    name = "expected_value_test/stage_b/calibration_bins"
    data = registry.table(name, stage="b", reference="experimental1020")
    branches = ["full8_common", "noSSA5_own", "hierarchy_own"]
    methods = ["Eselected", "E1", "E5"]
    schemes = ["site_cv", "region_out"]
    selected = data.loc[data.branch.isin(branches) & data.estimator.isin(methods) & data.scheme.isin(schemes)]
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE, 140 * style.MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.90, wspace=.20, hspace=.35)
    for r, scheme in enumerate(schemes):
        for c, branch in enumerate(branches):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color="#7F7F7F", lw=.6, ls="--")
            for method in methods:
                q = selected.loc[selected.branch.eq(branch) & selected.scheme.eq(scheme)
                                 & selected.estimator.eq(method)].sort_values("bin_low")
                if q.empty or q.duplicated(["bin_low", "bin_high"]).any():
                    raise ValueError("Missing or ambiguous saved calibration bins")
                _, color, marker, line = style.EST[method]
                ax.vlines(q.predicted, q.observed_lo, q.observed_hi, color=color, lw=.8, alpha=.7)
                ax.plot(q.predicted, q.observed, color=color, ls=line, lw=.8)
                for sparse in (False, True):
                    points = q.loc[q.sites.lt(5).eq(sparse)]
                    ax.plot(points.predicted, points.observed, marker=marker, color=color, ls="none",
                            markerfacecolor="white" if sparse else color, markersize=3.5)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.set_title({"full8_common": "full8: comparison population", "noSSA5_own": "noSSA5: own population",
                          "hierarchy_own": "Hierarchy: own population"}[branch], pad=7)
            if c == 0:
                ax.set_ylabel(("Held-out sites" if r == 0 else "Held-out regions") + "\nReference dust fraction")
            if r == 1:
                ax.set_xlabel("Predicted dust fraction")
            style.panel_label(ax, "abcdef"[r * 3 + c], x=-.10, y=1.04)
    handles = [Line2D([], [], color=style.EST[e][1], marker=style.EST[e][2], ls=style.EST[e][3],
                      label=style.EST[e][0]) for e in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045), ncol=3)
    table = registry.payload["tables"][name]
    evidence = [{"table": name, "source_file": table["source_file"], "source_sha256": table["source_sha256"],
                 "filters": {"reference": "experimental1020", "branch": branches, "scheme": schemes, "estimator": methods},
                 "semantics": "Saved within-bin reference/predicted fraction means; repeat-specific means averaged. "
                 "Vertical intervals are saved site-bootstrap intervals. No rebinning or new resampling."}]
    style.export(fig, output, "FigS10_fraction_reliability", selected, registry_path, evidence, __file__)
    (Path(output) / "FigS10_caption.md").write_text(
        "Binned dust-extinction-fraction reliability for held-out sites (top) and regions (bottom). "
        "These are reference-versus-predicted fractions, not dust occurrence probabilities. "
        "Points are saved within-bin means, averaged over the available cross-validation repeats; "
        "vertical lines are saved 95% site-bootstrap intervals. Dashed grey lines indicate equality. "
        "Populations differ between columns; noSSA5's own population is not limited to fallback pixels. "
        "Open markers flag bins with fewer than five sites. "
        "The DB aerosol-type curve represents its fitted conditional-mean baseline.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
