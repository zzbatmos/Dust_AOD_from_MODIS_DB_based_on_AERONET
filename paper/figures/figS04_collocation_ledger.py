"""Saved-row eligibility ledger and year/platform sampling, not pixel coverage."""
import argparse
from pathlib import Path

import numpy as np

from paper.numbers import Registry


NAMES = ("collocation_ledger", "collocation_year_platform", "collocation_cohort_summary")
SAMPLES = ("all_collocations", "hierarchy_eligible")


def select(registry):
    frames = {k: registry.table("derived/" + k) for k in NAMES}
    ledger, annual, summary = (frames[k] for k in NAMES)
    if len(ledger) != 7 or ledger["sample"].duplicated().any() or set(ledger.row_order) != set(range(7)):
        raise ValueError("Ambiguous collocation ledger")
    ledger = ledger.sort_values("row_order")
    if (np.diff(ledger.iloc[:5].rows) > 0).any() or ledger.iloc[5:].rows.sum() != ledger.iloc[4].rows:
        raise ValueError("Ledger is not nested or branch partition fails")
    if summary["sample"].duplicated().any() or set(summary["sample"]) != set(SAMPLES):
        raise ValueError("Missing cohort summary")
    if annual.duplicated(["sample", "year", "platform"]).any():
        raise ValueError("Ambiguous year/platform counts")
    for sample in SAMPLES:
        q = annual.loc[annual["sample"].eq(sample)]
        saved = summary.loc[summary["sample"].eq(sample)].iloc[0]
        if q.empty or set(q.platform) != {"Terra", "Aqua"} or not q.groupby("year").size().eq(2).all():
            raise ValueError("Incomplete year/platform grid")
        if q.rows.sum() != saved.rows or q.loc[q.year.eq(2017), "rows"].sum() != saved.rows_2017:
            raise ValueError("Year totals disagree with cohort summary")
        if not np.isclose(saved.fraction_2017, saved.rows_2017 / saved.rows, atol=1e-15, rtol=0):
            raise ValueError("2017 share disagrees with its denominator")
        if ledger.loc[ledger["sample"].eq(sample), "rows"].iloc[0] != saved.rows:
            raise ValueError("Ledger and annual populations differ")
    return frames


def make(registry_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    import pandas as pd
    from paper.figures import figstyle as style

    registry = Registry.read(registry_path)
    frames = select(registry)
    ledger, annual, summary = (frames[k] for k in NAMES)
    ledger = ledger.sort_values("row_order")
    fig = plt.figure(figsize=(style.DOUBLE, 150 * style.MM))
    ax = fig.add_axes([.28, .58, .67, .35])
    q = ledger.iloc[:5]
    ax.barh(range(5), q.rows, color=["#AAAAAA"] * 4 + ["#0072B2"], height=.55)
    maximum = float(q.rows.max())
    for y, row in enumerate(q.itertuples()):
        ax.text(row.rows + maximum * .015, y, f"{row.rows:,}", va="center", fontsize=7)
    ax.set(yticks=range(5), yticklabels=["All collocations", "Non-calibration sites",
           "Finite primary dust reference", "Paired total AOD550", "Hierarchy features"],
           ylim=(4.7, -.7), xlim=(0, maximum * 1.18), xlabel="Retained collocations",
           title="Primary-reference eligibility ledger")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    style.panel_label(ax, "a", x=-.08, y=1.06)
    partitions = ledger.set_index("sample")
    fig.text(.28, .49, f"Hierarchy partition: full8 = {int(partitions.loc['full8_subset', 'rows']):,}; "
             f"noSSA5 fallback = {int(partitions.loc['noSSA5_fallback_subset', 'rows']):,}", fontsize=7)
    colors = {"Terra": "#E69F00", "Aqua": "#0072B2"}
    for col, sample in enumerate(SAMPLES):
        ax = fig.add_axes([.10 + col * .47, .15, .38, .25])
        part = annual.loc[annual["sample"].eq(sample)]
        wide = part.pivot(index="year", columns="platform", values="rows").sort_index()
        ax.axvspan(2016.55, 2017.45, color="black", alpha=.08, lw=0, zorder=0)
        base = np.zeros(len(wide))
        for platform in ("Terra", "Aqua"):
            ax.bar(wide.index, wide[platform], bottom=base, color=colors[platform], width=.8, linewidth=0)
            base += wide[platform].to_numpy()
        saved = summary.loc[summary["sample"].eq(sample)].iloc[0]
        ax.set_title(("All saved collocations" if col == 0 else "ML hierarchy eligible")
                     + f"\n2017: {int(saved.rows_2017):,}/{int(saved.rows):,} ({saved.fraction_2017:.1%})", pad=8)
        ax.set(xlabel="Collocation year", ylabel="Collocations", ylim=(0, float(base.max()) * 1.12),
               xlim=(wide.index.min() - 1, wide.index.max() + 1),
               xticks=[y for y in wide.index if y % 5 == 0])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"))
        style.panel_label(ax, "bc"[col], x=-.12, y=1.24)
    fig.legend(handles=[Patch(color=colors[p], label=p) for p in ("Terra", "Aqua")],
               loc="lower center", bbox_to_anchor=(.52, .015), ncol=2)
    evidence = [dict(table="derived/" + name, **{k: registry.payload["tables"]["derived/" + name][k]
                   for k in ("source_file", "source_sha256", "derivation")}) for name in NAMES]
    exported = pd.concat([d.assign(record_type=k) for k, d in frames.items()], ignore_index=True, sort=False)
    style.export(fig, output, "FigS04_collocation_ledger", exported, registry_path, evidence, __file__)
    (Path(output) / "FigS04_caption.md").write_text(
        "Collocation eligibility and temporal/platform sampling. The primary-reference ledger counts "
        "unique saved collocations, not MODIS pixels or repeated cross-validation events. Calibration-reserved "
        "sites are removed before reference and feature eligibility; full8 and noSSA5 fallback partition "
        "the final hierarchy. Lower panels retain separate denominators for all archived collocations and "
        "the ML-eligible subset. Stacked bars show Terra and Aqua; the pale band marks 2017. Percentages "
        "are the 2017 share of each complete cohort, not satellite coverage or dust occurrence. Years with "
        "zero collocations do not imply zero aerosol. The saved row identities and both held-out repeats "
        "were checked against the eligibility reconstruction; no fitting or new screening was performed.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
