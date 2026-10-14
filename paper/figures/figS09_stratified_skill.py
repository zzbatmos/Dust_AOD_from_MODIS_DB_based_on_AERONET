"""Render saved QA2 stratified skill, without inference or new uncertainty estimates.

Select experimental1020 hierarchy_prespecified/own when present. Only when that
model is entirely absent, use full8/common and noSSA5/own as distinct series.
The known grouped-validation schema contains QA-gated DAOD, not expected-value
DAOD. Never relabel these scores as EV or import incompatible Stage-B strata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper.figures import figstyle as style
from paper.numbers import Registry, clean, digest

TABLE = "grouped_validation/metrics"
INTERVALS = "grouped_validation/site_bootstrap_95CI"
KEYS = ["reference", "population", "model", "scheme", "QA", "scope", "group"]
SCHEMES = ("site_cv", "region_out")
SCOPES = ("AOD_bin", "region")
METRICS = ("DAOD_RMSE", "DAOD_bias", "recall")
HIERARCHY = ("hierarchy_prespecified", "own")
FALLBACK = (("full8", "common"), ("noSSA5", "own"))
STEM = "FigS09_stratified_skill"


def select_data(registry):
    """Return full selected rows, exact-key saved intervals, and source evidence.

    Missing interval tables/rows/bounds are allowed, not synthesized. Partial
    bounds, duplicate keys, incomplete scheme/stratum coverage, or malformed
    counts fail closed. No pooling across models, populations, or schemes.
    """
    frame = registry.table(TABLE, reference=CONFIG["reference"]["name"])
    required = set(KEYS + list(METRICS) + ["n", "unique_rows", "sites", "TP", "FP", "FN", "TN"])
    if not required.issubset(frame):
        raise ValueError(f"Missing metrics columns: {sorted(required - set(frame))}")
    models = (HIERARCHY,) if frame.model.eq(HIERARCHY[0]).any() else FALLBACK
    qa = CONFIG["qa"]["thresholds"][CONFIG["qa"]["product_gate_index"]]
    chosen = frame.QA.eq(qa) & frame.scheme.isin(SCHEMES) & frame.scope.isin(SCOPES)
    support = pd.Series(False, index=frame.index)
    for model, population in models:
        support |= frame.model.eq(model) & frame.population.eq(population)
    data = frame.loc[chosen & support].copy()
    data["source_record_index"] = data.index
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate selected metric keys")
    for model, population in models:
        subset = data.loc[data.model.eq(model) & data.population.eq(population)]
        for scope in SCOPES:
            groups = [set(subset.loc[subset.scheme.eq(scheme) & subset.scope.eq(scope), "group"])
                      for scheme in SCHEMES]
            if not groups[0] or groups[0] != groups[1]:
                raise ValueError(f"Incomplete scheme/stratum coverage: {model}/{population}/{scope}")
    for column in ("n", "unique_rows", "sites", "TP", "FP", "FN", "TN"):
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid saved counts: {column}")
    if ((data.sites > data.unique_rows) | (data.unique_rows > data.n)).any():
        raise ValueError("Inconsistent event/collocation/site counts")
    if not data[["TP", "FP", "FN", "TN"]].sum(axis=1).eq(data.n).all():
        raise ValueError("Confusion counts do not sum to saved prediction events")

    evidence = [{"table": TABLE, **{k: registry.payload["tables"][TABLE][k]
                                  for k in ("source_file", "source_sha256", "semantics")},
                 "selection": {"reference": CONFIG["reference"]["name"], "QA": qa,
                               "model_population_pairs": models, "schemes": SCHEMES, "scopes": SCOPES},
                 "source_record_indices": data.source_record_index.tolist()}]
    ci_columns = []
    if INTERVALS in registry.payload["tables"]:
        ci = registry.table(INTERVALS)
        if not set(KEYS).issubset(ci):
            raise ValueError("Missing interval selector columns")
        ci = ci.assign(interval_source_record_index=ci.index)
        ci = ci.merge(data[KEYS], on=KEYS, how="inner", validate="many_to_one")
        if ci.duplicated(KEYS).any():
            raise ValueError("Duplicate selected interval keys")
        ci_columns = [c for c in ci if c not in KEYS]
        if set(ci_columns) & set(data):
            raise ValueError("Metric/interval column collision")
        data = data.merge(ci, on=KEYS, how="left", validate="one_to_one")
        evidence.append({"table": INTERVALS, **{k: registry.payload["tables"][INTERVALS][k]
                                               for k in ("source_file", "source_sha256")},
                         "source_record_indices": ci.interval_source_record_index.tolist()})
    for metric in METRICS:
        bounds = [f"{metric}_{side}" for side in ("lo", "hi")]
        if sum(c in ci_columns for c in bounds) == 1:
            raise ValueError(f"Incomplete interval columns: {metric}")
        for column in bounds:
            if column not in data:
                data[column] = np.nan
        values = data[[metric] + bounds].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite saved values: {metric}")
        finite = np.isfinite(values)
        available = finite[:, 1:].all(axis=1)
        if (finite[:, 1] != finite[:, 2]).any() or (values[available, 1] > values[available, 2]).any():
            raise ValueError(f"Invalid interval bounds: {metric}")
        if (available & (~finite[:, 0] | (data.sites.to_numpy() < 2))).any():
            raise ValueError(f"Unsupported saved interval: {metric}")
        if metric == "DAOD_RMSE" and (values[np.isfinite(values)] < 0).any():
            raise ValueError("Negative RMSE")
        if metric == "recall":
            if ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
                raise ValueError("Recall outside [0, 1]")
            if not np.array_equal(finite[:, 0], (data.TP + data.FN > 0).to_numpy()):
                raise ValueError("Undefined recall does not match saved positive counts")
        elif (~finite[:, 0] & data.n.gt(0).to_numpy()).any():
            raise ValueError(f"Missing {metric} despite saved evaluation events")
        if available.any():
            if "bootstrap_unit" not in data or not data.loc[available, "bootstrap_unit"].eq("site").all():
                raise ValueError("Intervals are not saved site-bootstrap intervals")
            if "replicates" not in data or not data.loc[available, "replicates"].gt(0).all():
                raise ValueError("Missing saved bootstrap replicate count")
        data[f"{metric}_interval_available"] = available
    data["estimator"] = "QA2"
    data["aod_bin_variable"] = "AER_AOD550"
    data["aod_bin_closure"] = "left_closed_right_open"
    data["ev_available"] = False
    data = data.sort_values(["model", "population", "scope", "group", "scheme"]).reset_index(drop=True)
    evidence.append({"ev_status": "No EV scores in the saved grouped-validation schema; QA2 only",
                     "aod_bin_semantics": "AERONET AOD550, left-closed/right-open saved bins; not DB AOD",
                     "support_semantics": "hierarchy own: full8 where eligible, noSSA5 otherwise; "
                     "fallback full8 common: intersection of both targets and full8 inputs; "
                     "fallback noSSA5 own: its target/feature eligible population. Never pooled.",
                     "configuration_file": str(Path(__file__).resolve().parents[2] / "configs/dbdust_v2.0.0.yaml"),
                     "configuration_sha256": digest(Path(__file__).resolve().parents[2] / "configs/dbdust_v2.0.0.yaml")})
    return data, clean(evidence)


def _groups(data, scope):
    groups = data.loc[data.scope.eq(scope), "group"].unique().tolist()
    if scope == "AOD_bin":
        def lower_bound(label):
            if label.startswith("<"):
                return -np.inf
            try:
                return float(label.removeprefix(">=").split("-")[0])
            except ValueError as exc:
                raise ValueError(f"Unknown saved AOD-bin label: {label}") from exc
        return sorted(groups, key=lower_bound)
    return sorted(groups, key=str.casefold)


def _series(data):
    supports = data[["model", "population"]].drop_duplicates().itertuples(index=False, name=None)
    output = []
    for index, (model, population) in enumerate(supports):
        for scheme in SCHEMES:
            color = "#0072B2" if scheme == "site_cv" else "#222222"
            marker = (("s", "D"), ("^", "o"))[index][SCHEMES.index(scheme)]
            label = "Site CV" if scheme == "site_cv" else "Region-out"
            if model != HIERARCHY[0]:
                label = f"{model}/{population}: {label}"
            output.append((model, population, scheme, color, marker, label))
    return output


def build_figure(data):
    """Create six panels; saved percentile endpoints need not contain the point."""
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE, 210 * style.MM),
                             gridspec_kw={"height_ratios": [1, 3.3]})
    fig.subplots_adjust(left=.25, right=.975, bottom=.095, top=.88, hspace=.39, wspace=.24)
    series = _series(data)
    offsets = np.linspace(-.23, .23, len(series))
    for row, scope in enumerate(SCOPES):
        groups = _groups(data, scope)
        subset = data.loc[data.scope.eq(scope)]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for offset, (model, population, scheme, color, marker, _) in zip(offsets, series):
                selection = subset.loc[subset.model.eq(model) & subset.population.eq(population)
                                       & subset.scheme.eq(scheme)].set_index("group")
                for y, group in enumerate(groups):
                    # A fallback support may not contain a region present in another support.
                    if group not in selection.index:
                        ax.text(.5, y + offset, "no support", transform=ax.get_yaxis_transform(),
                                fontsize=6.5, ha="center", va="center", color=color)
                        continue
                    record = selection.loc[group]
                    value = record[metric]
                    if pd.isna(value):
                        ax.text(.5, y + offset, "NA", transform=ax.get_yaxis_transform(),
                                fontsize=6.5, ha="center", va="center", color=color)
                        continue
                    if record[f"{metric}_interval_available"]:
                        lo, hi = record[f"{metric}_lo"], record[f"{metric}_hi"]
                        ax.hlines(y + offset, lo, hi, color=color, lw=.8)
                        ax.plot([lo, hi], [y + offset] * 2, "|", color=color, markersize=4, mew=.8)
                    ax.plot(value, y + offset, marker=marker, color=color, linestyle="none",
                            markersize=3.5, markerfacecolor=color if scheme == "site_cv" else "white",
                            markeredgewidth=.8)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([g.replace("_", " ") for g in groups] if col == 0 else [])
            ax.set_ylim(len(groups) - .4, -.6)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color="#E6E6E6", lw=.4)
            ax.set_axisbelow(True)
            style.panel_label(ax, "abcdef"[row * 3 + col], x=0, y=1.04)
            values = subset[[metric, f"{metric}_lo", f"{metric}_hi"]].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                raise ValueError(f"No finite values for {scope}/{metric}")
            low, high = min(0, values.min()), max(0, values.max())
            if metric == "recall":
                low, high = 0, 1
            pad = .06 * (high - low or 1)
            ax.set_xlim(low - pad, high + pad)
            ticks = MaxNLocator(nbins=4).tick_values(low, high)
            ax.set_xticks(ticks[(ticks >= low) & (ticks <= high)])
            ax.set_xlabel({"DAOD_RMSE": "QA2 DAOD RMSE (550 nm)",
                           "DAOD_bias": "QA2 DAOD bias (550 nm)", "recall": "QA2 dust recall"}[metric])
            if metric == "DAOD_bias":
                ax.axvline(0, color="#7F7F7F", lw=.6, ls="--")
        axes[row, 0].text(-.02, 1.16 if row == 0 else 1.10,
                          "AERONET AOD (550 nm) bins" if row == 0 else "Geographic regions",
                          transform=axes[row, 0].transAxes, fontsize=7, va="bottom")
    title = ("Prespecified hierarchy / own support" if data.model.eq(HIERARCHY[0]).all()
             else "Full8 / common and noSSA5 / own: distinct supports")
    fig.text(.25, .975, f"{CONFIG['reference']['name']} | {title}", fontsize=7, va="top")
    handles = [Line2D([], [], marker=marker, linestyle="none", color=color, markersize=4,
                      markerfacecolor=color if scheme == "site_cv" else "white", label=label)
               for _, _, scheme, color, marker, label in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.24, .955), ncol=2, fontsize=7)
    fig.text(.25, .025, "Bars: saved 95% site-bootstrap intervals; absent bars: unavailable.\n"
             "NA: undefined recall (no reference-positive events). No EV scores in this table.",
             fontsize=6.5, va="bottom", linespacing=1.5)
    return fig


def check_layout(fig):
    """Refuse undersized, off-canvas, or overlapping visible text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        if text.get_fontsize() < 6.5:
            raise ValueError(f"Text below 6.5 pt: {text.get_text()}")
        box = text.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError(f"Text outside figure: {text.get_text()}")
        for previous, other in boxes:
            overlap_x = min(box.x1, other.x1) - max(box.x0, other.x0)
            overlap_y = min(box.y1, other.y1) - max(box.y0, other.y0)
            if overlap_x > .5 and overlap_y > .5:
                raise ValueError(f"Overlapping text: {previous!r} / {text.get_text()!r}")
        boxes.append((text.get_text(), box))


def caption(data):
    support = ("The saved prespecified full8/noSSA5 hierarchy is evaluated on its own eligible "
               "support (full8 where eligible, noSSA5 otherwise)." if data.model.eq(HIERARCHY[0]).all()
               else "Full8 uses the common intersection of both reference targets and full8 inputs; "
               "noSSA5 uses its own target/feature-eligible population. These are distinct supports, "
               "not a paired feature-effect comparison.")
    qa = CONFIG["qa"]["thresholds"][CONFIG["qa"]["product_gate_index"]]
    label = CONFIG["reference"]["dust_label_threshold"]
    return (f"Saved experimental1020 QA2 skill by AERONET AOD550 bin (a-c) and geographic region "
            f"(d-f). {support} Filled squares denote site CV; open diamonds denote region-out "
            "validation" + ("; triangles/circles distinguish noSSA5 in the fallback display" if
                            not data.model.eq(HIERARCHY[0]).all() else "") + ". "
            "Site-CV regional summaries hold out sites, not entire regions; region-out holds out "
            "each displayed region. Scores retain the source's pooled held-out prediction events "
            "(two repeats for site CV, one region holdout per collocation for region-out), not "
            "metrics of averaged predictions. n counts events; unique_rows and sites are separate "
            "in the full selected-data CSV. DAOD bias is prediction minus reference. "
            f"QA2 gates at probability >= {qa:g}; recall uses reference dust fraction >= {label:g}. "
            "DAOD reference is reference fraction times AERONET AOD550, not DB AOD. Bins are "
            "left-closed/right-open, as saved; no unrepresented low-AOD stratum is invented. "
            "Bars are existing 95% site-bootstrap intervals retaining repeats within sites, "
            "not model-refit or calibration uncertainty. Missing intervals (including single-site "
            "strata) are omitted; NA means no reference-positive events, not zero recall. "
            "All selected source metrics, counts, and available interval columns are exported. "
            "EV is absent from this source and is not inferred from QA2. No new fits or bootstrap.\n")


def make(registry_path, output):
    before = digest(Path(registry_path))
    registry = Registry.read(registry_path)
    data, evidence = select_data(registry)
    fig = build_figure(data)
    try:
        check_layout(fig)
        if digest(Path(registry_path)) != before:
            raise RuntimeError("Registry changed during rendering")
        style.export(fig, output, STEM, data, registry_path, evidence, __file__)
        (Path(output) / "FigS09_caption.md").write_text(caption(data))
    finally:
        plt.close(fig)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make(args.registry, args.output_dir)


if __name__ == "__main__":
    main()
