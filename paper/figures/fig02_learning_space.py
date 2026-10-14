"""Controlled learning-space comparison, read only from the numbers registry."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output, reference="experimental1020"):
    registry = Registry.read(registry_path)
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 70 * style.MM),
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    fig.subplots_adjust(left=.16, right=.985, bottom=.25, top=.86, wspace=.7)
    models = ["AER_size500proxy", "AER_nonsphericity", "DB_size500proxy", "DB_nonsphericity",
              "full8_nonsphericity_comparator"]
    positions = [4.5, 3.5, 2, 1, -.5]
    labels = ["AERONET: size", "AERONET: PLDR", "DB: size", "DB: PLDR", "DB: full8"]
    data, evidence = [], []
    for panel, metric in enumerate(("DAOD_RMSE", "DAOD_MAE")):
        ax = axes[panel]
        for model, y in zip(models, positions):
            n = registry.number("grouped_validation/decomposition_2x2", metric,
                                reference=reference, model=model, scheme="site_cv", QA=.6)
            evidence.append(n)
            lo, hi = n["interval"]
            is_size = "size" in model
            color = "#0072B2" if is_size else "#D55E00"
            if model.startswith("full8"):
                color = "#7F7F7F"
            marker = "s" if is_size else "o"
            ax.errorbar(n["value"], y, xerr=[[n["value"] - lo], [hi - n["value"]]],
                        fmt=marker, color=color, markersize=4, elinewidth=.8, capsize=2)
            data.append({"panel": "ab"[panel], "model": model, "metric": metric,
                         "value": n["value"], "ci_low": lo, "ci_high": hi,
                         "y_position": y, "registry_key": n["key"]})
        ax.set_yticks(positions, labels if panel == 0 else [""] * len(labels))
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-1.2, 5.2)
        ax.set_xlim(left=0)
        ax.set_xlabel("DAOD RMSE" if panel == 0 else "DAOD MAE")
        ax.grid(axis="x", color="#E6E6E6", lw=.4)
        style.panel_label(ax, "ab"[panel], x=-.05)
    contrasts = [("DB_size500proxy", "AER_size500proxy", "DB - AER\n(size)"),
                 ("DB_nonsphericity", "AER_nonsphericity", "DB - AER\n(PLDR)"),
                 ("DB_nonsphericity", "DB_size500proxy", "PLDR - size\n(DB)"),
                 ("AER_nonsphericity", "AER_size500proxy", "PLDR - size\n(AER)")]
    ax = axes[2]
    ax.axvspan(-.005, .005, color="#E6E6E6", zorder=0)
    ax.axvline(0, color="#7F7F7F", ls="--", lw=.6)
    for y, (a, b, label) in enumerate(reversed(contrasts)):
        n = registry.number("grouped_validation/decomposition_paired_differences",
                            "delta_RMSE_A_minus_B", reference=reference, A=a, B=b,
                            scheme="site_cv", QA=.6)
        evidence.append(n)
        lo, hi = n["interval"]
        ax.errorbar(n["value"], y, xerr=[[n["value"] - lo], [hi - n["value"]]],
                    fmt="o", color="black", elinewidth=.8, capsize=2)
        data.append({"panel": "c", "model": a + " minus " + b, "metric": "delta_RMSE",
                     "value": n["value"], "ci_low": lo, "ci_high": hi,
                     "y_position": y, "registry_key": n["key"]})
    ax.set_yticks(range(4), [x[2] for x in reversed(contrasts)])
    ax.set_ylim(-.7, 3.7)
    ax.set_xlim(-.08, .04)
    ax.set_xticks([-.08, -.04, 0, .04])
    contrasts_data = [d for d in data if d["panel"] == "c"]
    if any(d["ci_low"] < -.08 or d["ci_high"] > .04 for d in contrasts_data):
        limit = math.ceil(max(max(abs(d["ci_low"]), abs(d["ci_high"]))
                              for d in contrasts_data) / .04) * .04
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-limit, 0, limit])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("RMSE difference\nNegative favors first term")
    style.panel_label(ax, "c", x=-.05)
    stem = "Fig02_learning_space" if reference == "experimental1020" else "FigS05_learning_space_sensitivity"
    style.export(fig, output, stem, pd.DataFrame(data), registry_path, evidence, __file__)
    counts = evidence[0]["N"]
    caption = (
        f"Controlled learning-space and target comparison at held-out sites "
        f"({int(counts['unique_rows']):,} collocations; {int(counts['sites'])} sites). "
        "(a, b) DAOD errors for size-proxy (blue squares) and depolarization "
        "(orange circles) targets; grey denotes the separate full8 comparator. "
        "(c) Paired RMSE differences; negative values favor the first term. "
        "Intervals are 95% site-bootstrap intervals. Shading marks the +/-0.005 "
        "practical margin. All methods use identical collocations; repeated "
        "cross-validation predictions are pooled. AER denotes AERONET; DB denotes Deep Blue."
    )
    if reference == "old675LR56":
        caption = "Sensitivity using the old 675-nm, 56-sr reference. " + caption
    (Path(output) / f"{stem.split('_')[0]}_caption.md").write_text(caption + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", choices=("experimental1020", "old675LR56"), default="experimental1020")
    args = parser.parse_args()
    make(args.registry, args.output_dir, args.reference)
