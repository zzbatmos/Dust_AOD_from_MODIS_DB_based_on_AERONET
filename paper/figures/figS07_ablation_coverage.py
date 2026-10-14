"""Own-population ablation skill and collocation coverage, not a paired ranking."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output):
    registry = Registry.read(registry_path)
    configs = ["full8", "fullminusAE7", "compact4", "singleSSA3", "noSSA5", "minimal2"]
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 85 * style.MM), sharey=True)
    fig.subplots_adjust(left=.16, right=.98, bottom=.23, top=.88, wspace=.4)
    evidence, data = [], []
    for y, model in enumerate(configs):
        common = dict(reference="experimental1020", model=model, population="own", scheme="site_cv",
                      scope="overall", group="all", QA=.6)
        cover = {metric: registry.number("grouped_validation/ablation_collocation_coverage", metric,
                                        reference="experimental1020", features=model)
                 for metric in ("collocation_coverage", "eligible_collocations", "gained_over_full8")}
        evidence.extend(cover.values())
        color = "#332288" if model == "full8" else ("#88CCEE" if model == "noSSA5" else "#BDBDBD")
        axes[0].barh(y, cover["collocation_coverage"]["value"], color=color, height=.55)
        n, gain = int(cover["eligible_collocations"]["value"]), int(cover["gained_over_full8"]["value"])
        axes[0].text(.04, y, f"N = {n:,}; +{gain:,}", va="center", fontsize=6.5,
                     color="white" if model == "full8" else "black")
        for key, number in cover.items():
            data.append(dict(model=model, metric=key, value=number["value"], registry_key=number["key"], panel="a"))
        for col, metric in enumerate(("DAOD_RMSE", "DAOD_bias"), start=1):
            number = registry.number("grouped_validation/metrics", metric, **common)
            if number["N"]["unique_rows"] != n:
                raise ValueError("Coverage and own-population skill counts differ")
            evidence.append(number)
            lo, hi = number["interval"]
            axes[col].hlines(y, lo, hi, color="black", lw=.8)
            axes[col].plot(number["value"], y, "o", color="black", markersize=4)
            data.append(dict(model=model, metric=metric, value=number["value"], ci_low=lo, ci_high=hi,
                             registry_key=number["key"], panel="abc"[col]))
    axes[0].set(xlim=(0, 1.05), xticks=[0, .5, 1], yticks=range(len(configs)), yticklabels=configs,
                xlabel="Collocation coverage", ylim=(len(configs) - .4, -.6))
    axes[1].set_xlabel("QA2 DAOD RMSE")
    axes[1].set_xlim(left=0)
    axes[2].set_xlabel("QA2 DAOD bias")
    axes[2].axvline(0, color="#7F7F7F", lw=.6, ls="--")
    for index, ax in enumerate(axes):
        ax.tick_params(axis="y", length=0)
        style.panel_label(ax, "abc"[index], x=-.05)
    style.export(fig, output, "FigS07_ablation_coverage", pd.DataFrame(data), registry_path, evidence, __file__)
    (Path(output) / "FigS07_caption.md").write_text(
        "Ablation on each configuration's own eligible collocations. (a) Coverage relative to the "
        "reference-and-DB-AOD denominator; labels give unique collocations and gains over full8. "
        "This is collocation coverage, not annual satellite-pixel availability. (b, c) QA2 DAOD "
        "errors under held-out-site validation, with saved 95% site-bootstrap intervals. "
        "Different populations prevent interpreting these differences as a strictly paired "
        "feature-importance ranking. Minimal2 is an experimental comparator, not a production branch.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
