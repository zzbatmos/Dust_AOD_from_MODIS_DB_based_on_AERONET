"""Plot saved temporal diagnostics without recomputing correlations or intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import Registry


ROOT = "expected_value_test/stage_b/temporal_supplement_"
METHODS = ("Eselected", "E1", "E5", "E7", "E8")
SCHEMES = ("site_cv", "region_out")
FILTERS = dict(stage="b", reference="experimental1020", branch="hierarchy_own",
               pairing="core", repeat_summary="mean_over_repeats", minimum_rows=3,
               seasonal_minimum_calendar_months=6, anomaly_minimum_years=3)


def select_tables(registry):
    """Fail on ambiguous summaries or incompatible support rather than pooling."""
    frames, evidence = {}, []
    for suffix in ("seasonal_summary", "anomalies", "support"):
        name = ROOT + suffix
        frame = registry.table(name, **FILTERS)
        frame = frame.loc[frame.scheme.isin(SCHEMES)].copy()
        keys = ["scheme"]
        if suffix != "support":
            frame = frame.loc[frame.estimator.isin(METHODS)].copy()
            keys += ["estimator"]
        expected = len(SCHEMES) * (len(METHODS) if suffix != "support" else 1)
        if len(frame) != expected or frame.duplicated(keys).any():
            raise ValueError(f"Missing or ambiguous temporal summaries: {suffix}")
        for scheme in SCHEMES:
            repeats = frame.loc[frame.scheme.eq(scheme), "expected_repeats"]
            if not repeats.eq(2 if scheme == "site_cv" else 1).all():
                raise ValueError("Unexpected repeat semantics")
        frames[suffix] = frame
        table = registry.payload["tables"][name]
        evidence.append(dict(table=name, source_file=table["source_file"],
                             source_sha256=table["source_sha256"],
                             filters=dict(FILTERS, scheme=list(SCHEMES),
                                          **({"estimator": list(METHODS)} if suffix != "support" else {}))))
    for suffix, prefix in (("seasonal_summary", "seasonal"), ("anomalies", "anomaly")):
        for _, row in frames[suffix].iterrows():
            support = frames["support"].loc[frames["support"].scheme.eq(row.scheme)].iloc[0]
            for field in ("sites", "prediction_events", "unique_rows", "unique_site_months", "repeats"):
                if row[field] != support[f"{prefix}_{field}"]:
                    raise ValueError(f"Temporal support mismatch: {suffix}/{field}")
            if suffix == "seasonal_summary":
                if not row.summary_defined or not 0 < row.finite_r_sites <= row.sites:
                    raise ValueError("Undefined seasonal summary")
                bounds = [row.q05_site_r, row.median_site_r, row.q95_site_r]
                if "not CIs" not in row.statistic_semantics:
                    raise ValueError("Seasonal spread must not be treated as a confidence interval")
            else:
                if not row.r_defined:
                    raise ValueError("Undefined anomaly correlation")
                bounds = [row.r_lo, row.r_hi]
                if not np.isfinite(row.r) or abs(row.r) > 1:
                    raise ValueError("Invalid anomaly correlation")
            if not np.isfinite(bounds).all() or min(bounds) < -1 or max(bounds) > 1:
                raise ValueError("Invalid saved correlation range")
            if any(a > b for a, b in zip(bounds, bounds[1:])):
                raise ValueError("Reversed saved correlation range")
    exported = pd.concat([frame.assign(diagnostic=kind) for kind, frame in frames.items()],
                         ignore_index=True, sort=False)
    return frames, exported, evidence


def make(registry_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper.figures import figstyle as style

    registry = Registry.read(registry_path)
    frames, exported, evidence = select_tables(registry)
    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE, 135 * style.MM), sharex=True)
    fig.subplots_adjust(left=.24, right=.97, bottom=.13, top=.90, wspace=.22, hspace=.55)
    labels = [style.EST[e][0] for e in METHODS]
    for row_index, scheme in enumerate(SCHEMES):
        support = frames["support"].loc[frames["support"].scheme.eq(scheme)].iloc[0]
        for column, suffix in enumerate(("seasonal_summary", "anomalies")):
            ax = axes[row_index, column]
            selected = frames[suffix].loc[frames[suffix].scheme.eq(scheme)].set_index("estimator")
            ax.axvline(0, color="#999999", lw=.7)
            for y, method in enumerate(METHODS):
                record = selected.loc[method]
                _, color, marker, _ = style.EST[method]
                if column == 0:
                    point, low, high = record.median_site_r, record.q05_site_r, record.q95_site_r
                else:
                    point, low, high = record.r, record.r_lo, record.r_hi
                ax.hlines(y, low, high, color=color, linewidth=1.4)
                ax.plot(point, y, marker=marker, color=color, linestyle="none", markersize=5)
                if column == 0:
                    ax.text(-.97, y, f"{int(record.finite_r_sites)}/{int(record.sites)}", va="center",
                            ha="left", fontsize=6.5, bbox=dict(facecolor="white", edgecolor="none", pad=.3))
            ax.set(xlim=(-1, 1), ylim=(len(METHODS) - .5, -.65), xticks=[-1, -.5, 0, .5, 1],
                   yticks=range(len(METHODS)), yticklabels=labels if column == 0 else [])
            ax.tick_params(axis="y", length=0)
            scheme_label = "Held-out sites" if scheme == "site_cv" else "Held-out regions"
            if column == 0:
                title = f"{scheme_label}: seasonal cycle\n{int(support.seasonal_sites)} eligible sites; finite / eligible shown"
                xlabel = "Median site r; 5th-95th percentile spread"
            else:
                title = (f"{scheme_label}: anomalies\n{int(support.anomaly_sites)} sites; "
                         f"{int(support.anomaly_unique_site_months)} distinct site-months")
                xlabel = "Pooled anomaly r; 95% site-bootstrap interval"
            ax.set_title(title, pad=10)
            ax.set_xlabel(xlabel)
            style.panel_label(ax, "abcd"[2 * row_index + column], x=-.07, y=1.23)
    for record in evidence:
        record["semantics"] = ("Saved summaries only. Seasonal ranges describe between-site spread, not confidence. "
                               "Anomaly confidence intervals are the saved site bootstrap. Repeats are averaged "
                               "at statistic level, not at prediction level. No new fitting/resampling.")
    style.export(fig, output, "FigS18_temporal", exported, registry_path, evidence, __file__)
    support_counts = frames["support"][["seasonal_sites", "anomaly_sites"]].drop_duplicates()
    support_note = "Support sizes are stated separately for each scheme in the panels."
    if len(support_counts) == 1:
        support_note = (f"Seasonal support is only {int(support_counts.iloc[0].seasonal_sites)} sites "
                        f"and is exploratory, unlike the {int(support_counts.iloc[0].anomaly_sites)}-site "
                        "anomaly diagnostic.")
    (Path(output) / "FigS18_caption.md").write_text(
        "Temporal diagnostics for the primary-reference hierarchy on core paired support. "
        "Site-month means require at least three collocations. Seasonal cycles require six represented calendar "
        "months per site: symbols show median site-specific correlations and lines their 5th-95th percentile "
        "spread, not confidence intervals. Counts distinguish finite correlations from eligible sites; constant "
        "predictions can make correlation undefined. Anomalies subtract each site's calendar-month mean and "
        "require at least three represented years for that month. Right-hand symbols show pooled site-month "
        "anomaly correlations with saved 95% site-bootstrap intervals. Held-out-site statistics are averaged "
        "equally across the two repeats; predictions are not averaged. " + support_note + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
