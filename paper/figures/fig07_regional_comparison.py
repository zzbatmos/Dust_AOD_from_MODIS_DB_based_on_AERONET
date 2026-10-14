"""Regime-specific seasonal comparisons, using registry values only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output):
    registry = Registry.read(registry_path)
    ratios = registry.table("derived/regional_ratio_summary")
    smoke = registry.table("derived/regional_smoke_summary")
    methods = list(ratios.estimator.drop_duplicates())
    regions = list(ratios.region.drop_duplicates())
    fig = plt.figure(figsize=(style.DOUBLE, 110 * style.MM))
    left = fig.add_axes([.15, .19, .29, .72])
    right = fig.add_axes([.73, .19, .255, .72])
    offsets = np.linspace(-.35, .35, len(methods))
    for index, method in enumerate(methods):
        q = ratios[ratios.estimator.eq(method)].set_index("region").loc[regions]
        _, color, marker, _ = style.EST[method]
        left.errorbar(q.ratio_mean, np.arange(len(regions)) + offsets[index],
                      xerr=np.stack((q.ratio_mean - q.ratio_min, q.ratio_max - q.ratio_mean)),
                      fmt=marker, color=color, markersize=3.5, capsize=0, elinewidth=.8)
    left.set_xscale("log", base=2)
    left.set_xlim(1 / 8, 8)
    left.set_xticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    left.axvline(1, color="#7F7F7F", ls="--", lw=.6)
    left.set_yticks(range(len(regions)), regions)
    left.set_ylim(len(regions) - .45, -.55)
    left.set_xlabel("Ratio to CALIOP")
    left.grid(axis="x", color="#E6E6E6", lw=.4)
    left.tick_params(axis="y", length=0)
    left.set_title("Dust-source and mixed regions", pad=9)
    style.panel_label(left, "a", x=-.43, y=1.03)

    all_methods = methods + ["CALIOP"]
    offsets = np.linspace(-.35, .35, len(all_methods))
    for index, method in enumerate(all_methods):
        q = smoke[smoke.estimator.eq(method)].sort_values("row_order")
        _, color, marker, _ = style.EST[method]
        right.plot(q.candidate_mean, q.row_order + offsets[index], linestyle="none",
                   marker=marker, color=color, markersize=4 if method == "CALIOP" else 3.5)
    labels = []
    display = {"Amazonia_Cerrado": "Amazonia/Cerrado", "southern_Africa": "Southern Africa",
               "boreal_North_America": "Boreal N. America", "mainland_SE_Asia": "Mainland SE Asia"}
    for _, row in smoke[smoke.estimator.eq("CALIOP")].sort_values("row_order").iterrows():
        labels.append(f"{display.get(row.region, row.region)} {row.period}\n(n = {int(row.n)})")
    right.set_yticks(range(len(labels)), labels)
    right.set_ylim(len(labels) - .45, -.55)
    right.set_xlim(0, max(.12, float(smoke.candidate_mean.max()) * 1.06))
    right.set_xticks([0, .04, .08, .12])
    right.set_xlabel("DAOD (550 nm)")
    right.tick_params(axis="y", length=0)
    right.grid(axis="x", color="#E6E6E6", lw=.4)
    right.set_title("Smoke-season and remote regions", pad=9)
    style.panel_label(right, "b", x=-.72, y=1.03)
    handles = [Line2D([], [], color=style.EST[m][1], marker=style.EST[m][2], ls="none",
                      markersize=4, label=style.EST[m][0]) for m in all_methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .005), ncol=4,
               columnspacing=.8, handletextpad=.35)
    evidence = [{"table": name, "source_file": registry.payload["tables"][name]["source_file"],
                 "source_sha256": registry.payload["tables"][name]["source_sha256"],
                 "derivation": registry.payload["tables"][name]["derivation"]}
                for name in ("derived/regional_ratio_summary", "derived/regional_smoke_summary")]
    style.export(fig, output, "Fig07_regional_comparison",
                 pd.concat([ratios.assign(panel="a"), smoke.assign(panel="b")], ignore_index=True),
                 registry_path, evidence, __file__)
    (Path(output) / "Fig07_caption.md").write_text(
        "Regional comparison with CALIOP on paired land support. (a) Mean of four seasonal "
        "DAOD ratios; whiskers show the seasonal range, not confidence intervals. (b) Absolute "
        "seasonal DAOD; n counts paired spatial cells. DJF spans December 2016-February 2017. "
        "The screened LG25 estimate is omitted because its support differs. Regional means "
        "are area weighted. Sparse boreal spring sampling remains visible rather than excluded.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
