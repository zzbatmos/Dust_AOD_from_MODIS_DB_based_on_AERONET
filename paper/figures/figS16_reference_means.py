"""Reference sensitivity of saved seasonal means, not an uncertainty estimate."""
import argparse
from pathlib import Path

import numpy as np

from paper.numbers import Registry


TABLE = "expected_value_test/stage_c/seasonal_metrics"
FILTERS = dict(variant="hierarchy", support="core_common", reference_mask="prior_joint")
REFS = ("experimental1020", "old675LR56")
METHODS = ("Eselected", "E1")
SEASONS = ("DJF", "MAM", "JJA", "SON")
GLOBAL = "Land 60S-60N"


def select(registry):
    data = registry.table(TABLE, **FILTERS)
    data = data.loc[data.reference.isin(REFS) & data.estimator.isin((*METHODS, "CALIOP"))
                    & ~data.region.str.endswith("_sensitivity")].copy()
    keys = ["reference", "estimator", "region", "period"]
    if data.empty or data.duplicated(keys).any():
        raise ValueError("Missing or ambiguous reference-sensitivity data")
    regions = list(data.region.unique())
    if GLOBAL not in regions:
        raise ValueError("Missing global comparison")
    for (reference, method, region), q in data.groupby(keys[:-1]):
        if set(q.period) != set(SEASONS) or len(q) != 4:
            raise ValueError("Incomplete four-season comparison")
    if len(data) != len(REFS) * (len(METHODS) + 1) * len(regions) * len(SEASONS):
        raise ValueError("Incomplete reference/method/region grid")
    for _, q in data.groupby(["region", "period"]):
        if q.n.nunique() != 1 or q.reference_mean.nunique(dropna=False) != 1:
            raise ValueError("Reference scenarios have different saved support")
        if q.candidate_mean.isna().all() and q.reference_mean.isna().all():
            if (q.n < 0).any():
                raise ValueError("Invalid saved seasonal count")
            continue
        if not np.isfinite(q.candidate_mean).all() or (q.candidate_mean < 0).any() or (q.n <= 0).any():
            raise ValueError("Invalid saved seasonal mean")
        cal = q.loc[q.estimator.eq("CALIOP")]
        if not cal.candidate_mean.eq(cal.reference_mean).all():
            raise ValueError("CALIOP mean identity mismatch")
    return data


def make(registry_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper.figures import figstyle as style

    registry = Registry.read(registry_path)
    data = select(registry)
    fig, axes = plt.subplots(3, 2, figsize=(style.DOUBLE, 220 * style.MM))
    fig.subplots_adjust(left=.12, right=.97, bottom=.09, top=.95, hspace=.65, wspace=.35)
    global_rows = data.loc[data.region.eq(GLOBAL)]
    for column, method in enumerate(METHODS):
        ax = axes[0, column]
        _, color, marker, _ = style.EST[method]
        for reference, line in zip(REFS, ("-", "--")):
            q = global_rows.loc[global_rows.estimator.eq(method) & global_rows.reference.eq(reference)]
            q = q.set_index("period").loc[list(SEASONS)]
            ax.plot(range(4), q.candidate_mean, marker=marker, color=color, ls=line)
        cal = global_rows.loc[global_rows.estimator.eq("CALIOP") & global_rows.reference.eq(REFS[0])]
        ax.plot(range(4), cal.set_index("period").loc[list(SEASONS)].candidate_mean,
                color="black", marker="*", ms=5)
        ax.set(xticks=range(4), xticklabels=SEASONS, ylim=(0, .15),
               ylabel="Land 60S-60N mean DAOD550", title=style.EST[method][0])
        style.panel_label(ax, "ab"[column], y=1.08)
    handles = [Line2D([], [], color="black", ls=line, label=label) for line, label in
               [("-", "Primary 1020-nm reference"), ("--", "Old 675-nm / 56-sr reference")]]
    handles.append(Line2D([], [], color="black", marker="*", label="CALIOP comparison"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .706), ncol=2)
    regional = data.loc[~data.region.eq(GLOBAL) & data.estimator.isin(METHODS)]
    limit = float(np.ceil(regional.candidate_mean.max() * 10) / 10)
    for index, season in enumerate(SEASONS):
        ax = axes[1 + index // 2, index % 2]
        ax.plot([0, limit], [0, limit], color="#888888", ls="--", lw=.7)
        valid_regions = None
        for method in METHODS:
            q = regional.loc[regional.period.eq(season) & regional.estimator.eq(method)]
            wide = q.pivot(index="region", columns="reference", values="candidate_mean")
            present = wide.notna().all(axis=1)
            if valid_regions is not None and not present.equals(valid_regions):
                raise ValueError("Methods differ in defined regional-mean support")
            valid_regions = present
            _, color, marker, _ = style.EST[method]
            ax.scatter(wide[REFS[1]], wide[REFS[0]], c=color, marker=marker, s=17, alpha=.8,
                       label=style.EST[method][0])
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Old-reference regional mean DAOD550",
               ylabel="Primary-reference regional mean DAOD550",
               title=f"{season}: {int(valid_regions.sum())}/{len(valid_regions)} regions defined")
        style.panel_label(ax, "cdef"[index], y=1.08)
    fig.legend(handles=[Line2D([], [], color=style.EST[e][1], marker=style.EST[e][2], ls="none",
                               label=style.EST[e][0]) for e in METHODS],
               loc="lower center", bbox_to_anchor=(.53, .01), ncol=2)
    table = registry.payload["tables"][TABLE]
    evidence = [dict(table=TABLE, source_file=table["source_file"], source_sha256=table["source_sha256"],
                     filters=dict(FILTERS, reference=list(REFS), estimator=[*METHODS, "CALIOP"],
                                  excluded_regions="*_sensitivity"),
                     semantics="Saved area-weighted seasonal means only. Same saved N and CALIOP mean "
                               "verified across references/methods. No annual aggregation, refit or new mask.")]
    style.export(fig, output, "FigS16_reference_means", data, registry_path, evidence, __file__)
    nregions = regional.region.nunique()
    (Path(output) / "FigS16_caption.md").write_text(
        "Sensitivity to the complete AERONET reference definition. Top panels show seasonal land means "
        "between 60S and 60N for the expected-value and QA2 products; solid/dashed colored lines use "
        "the primary 1020-nm and old 675-nm/56-sr references, respectively. Black stars show CALIOP. "
        f"Lower panels compare old versus primary means in {nregions} regions, one point per region "
        "and method; dashed diagonals indicate equality. All plotted means are saved area-weighted "
        "statistics on core_common/prior_joint seasonal support. DJF includes December 2016; the "
        "remaining seasons are in 2017. This is a multi-component reference sensitivity, not an "
        "isolated lidar-ratio effect, calibrated uncertainty interval or annual-mean comparison. "
        "Undefined saved means are omitted from scatter panels, never replaced by zero; panel titles "
        "state the number of regions with defined means. "
        "Region identities, cell counts and unrounded means are provided in the accompanying CSV.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
