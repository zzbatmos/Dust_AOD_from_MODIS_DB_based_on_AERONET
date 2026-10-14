"""Lossless candidate SI Tables S2-S5 from an immutable private registry only.

CSV files preserve all source scopes and precision. S2/S5 LaTeX uses explicit
reader-oriented subsets; S3/S4 retains source-row-linked panels. No source
CSVs, models, or experiments are opened.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from paper.numbers import Registry, clean, digest


METRICS = "grouped_validation/metrics"
INTERVALS = "grouped_validation/site_bootstrap_95CI"
DECOMPOSITION = "grouped_validation/decomposition_2x2"
CONTRASTS = "grouped_validation/decomposition_paired_differences"
COMPONENTS = "expected_value_test/stage_b/paired_differences"
DECISIONS = "expected_value_test/stage_b/decision_table"
SEASONAL = "expected_value_test/stage_c/seasonal_metrics"
KEYS = ["reference", "population", "model", "scheme", "QA", "scope", "group"]
CI_ROW = "interval_source_record_index"
S2_PANELS = (
    ("Classification", ["precision", "recall", "FPR", "F1", "AUC", "Brier"]),
    ("DAOD550", ["DAOD_bias", "DAOD_MAE", "DAOD_RMSE", "DAOD_R2", "EE_descriptive"]),
)
S5_VALUES = ["n", "candidate_mean", "reference_mean", "bias", "rmse", "mae", "r"]
VIEW_NOTES = {
    2: "Reader view only: scope=overall, group=all, population=own or common; all saved "
       "models, QA tiers, schemes and both references. TableS02_metrics.csv and "
       "TableS02_intervals.csv retain FULL source scope, all columns and unrounded values: "
       "overall, region, AOD_bin, platform, fold and decomposition populations where saved. "
       "Decomposition is also reported in S3. No other scope is typeset here. "
       "Saved 95% site-bootstrap intervals are joined on reference/population/model/scheme/QA/"
       "scope/group exactly and shown in brackets below each point where available. "
       "A point without brackets has no saved interval; -- is missing, not zero. "
       "N is unique_rows (distinct collocations); sites counts contributing sites; prediction-event "
       "counts n and all additional metrics remain in the full CSV. Display rounding: two decimals "
       "for rates/AUC, three for other metrics; counts are exact.",
    5: "Reader view only: variant=hierarchy, reference_mask=prior_joint, all regions and both "
       "references. The primary block shows every estimator on core_common; the separate E6 block "
       "shows E6 only on full_common_E6. E6 has different support: its N and reference mean are "
       "support-specific and its differences from core_common are NOT paired method contrasts. "
       "All seven saved numeric fields are displayed for every selected seasonal row. "
       "TableS05_seasonal.csv retains FULL source scope, all columns and unrounded values, including "
       "with_minimal_fallback, mean_only_reference_sensitivity, and all estimators on full_common_E6. "
       "Latitude-sensitivity regions remain explicitly labeled in this reader view. "
       "Display rounding: three decimals; counts are exact; -- is missing, not zero.",
}
HYPOTHESES = ("H1", "H2", "H2_RMSE", "H2_conditional_bias", "H3", "H4")
ROW = "source_record_index"
CAPTIONS = {
    2: "All saved pixel-level metrics by configuration, QA tier, population, scope, scheme and reference.",
    3: "Controlled retrieval-space decomposition and saved paired contrasts under both references.",
    4: "Preregistered hypothesis components, practical margins and saved decisions for all branches.",
    5: "Global and regional seasonal statistics against CALIOP for all saved estimators and supports.",
}
GROUPED_NOTE = (
    "All saved dimensions are retained, including overlapping overall, region, AOD-bin, platform "
    "and fold scopes where present. Do not sum counts across scopes. n counts prediction events; "
    "unique_rows counts collocations. Site CV pools both repeats, not averaged predictions. "
    "Own and common populations are distinct; unpaired population comparisons do not isolate SSA. "
    "Intervals are saved site-bootstrap bounds conditional on fitted predictions, not refitting or "
    "reference-calibration uncertainty. Missing intervals are not zero-width intervals. Deterministic "
    "baseline Brier scores are descriptive, not calibrated-probability claims. QA is the saved "
    "threshold, not a new threshold applied by this exporter."
)
NOTES = {
    2: GROUPED_NOTE + " Interval rows are exported separately with their exact matching dimensions. "
       "No AUC interval or other absent bound is synthesized.",
    3: GROUPED_NOTE + " The full8 comparator is separate from the matched two-predictor 2x2. "
       "The size target is SDA CMF500proxy, not compositional dust truth. Contrasts remain A minus B; "
       "negative favors A for error metrics. Point tables and paired-contrast tables can contain "
       "different saved QA grids; they are not silently expanded or averaged.",
    4: "H1 compares absolute site-month bias with E1; H2 includes RMSE and pooled suspected-mislabel "
       "absolute-bias components against E5; H3 is collocation noninferiority against min(E1,E5). "
       "H4 is a conjunction, not an independent experiment or a claim for every region. "
       "Stage B uses mean-over-repeat metrics, unlike the repeat-pooled grouped tables. "
       "Preserve minimum, level, branch, candidate, comparator and all event/site/row counts. "
       "For site-month tests, events are repeat-specific cells; for collocation tests they are "
       "prediction events. Counts for pooled conditional bias differ from overall support. "
       "Decisions and numeric margins are copied, never recomputed from rounded values or inferred "
       "from formulas. Aggregate H2/H4 margins missing from these sources are explicitly marked "
       "missing, not assigned zero or borrowed from another hypothesis. E8 recalibration contrasts "
       "are not H1-H4 and are excluded. No confidence interval is invented for aggregate decisions.",
    5: "All saved references, variants, regions, estimators, support intersections and reference masks "
       "are retained; primary and sensitivity rows remain distinguishable. n counts native 2x5 "
       "seasonal spatial cells, not retrievals or cell-months. Statistics are land-area weighted "
       "after equal means of three complete months on intersected monthly support. DJF is "
       "December 2016-February 2017; MAM/JJA/SON are in 2017. Frozen land-purity and fine-land "
       "coverage screens apply. CALIOP is transferred from 532 to 550 nm, not declared truth. "
       "full_common_E6 and mean_only_reference_sensitivity are not the primary core_common / "
       "prior_joint support. Primary boreal boxes end at 60N; 50-75N sensitivities remain labeled. "
       "No inferential intervals are supplied. Reference-scenario changes are not calibrated uncertainty.",
}
DEFINITIONS = {
    ROW: "Zero-based row position in the source table embedded in the registry; joins panels, not an observation count.",
    "reference": "Saved reference definition: experimental1020 or old675LR56; do not pool references.",
    "population": "Saved eligibility population (common, own or decomposition).",
    "model": "Saved predictor configuration or baseline identifier.",
    "scheme": "Saved held-out evaluation scheme; site_cv and region_out are distinct.",
    "QA": "Saved probability cutoff for the QA tier (0.4, 0.6 or 0.7); baseline rules remain as evaluated.",
    "scope": "Saved aggregation scope; scopes overlap.", "group": "Group within the saved scope.",
    "n": "Saved number of evaluated prediction events (grouped) or seasonal spatial cells (Stage C); see table notes.",
    "unique_rows": "Distinct collocations, not repeat prediction events.",
    "sites": "Distinct contributing sites on this row's support.",
    "eligible_rows": "Saved eligible-row denominator for this population/scope.",
    "coverage": "Saved evaluation coverage fraction relative to eligible rows; not global pixel availability.",
    "TP": "True-positive prediction-event count.", "FP": "False-positive prediction-event count.",
    "FN": "False-negative prediction-event count.", "TN": "True-negative prediction-event count.",
    "precision": "Positive predictive value.", "recall": "True-positive rate.",
    "FPR": "False-positive rate.", "F1": "F1 score of the saved classification decisions.",
    "AUC": "Area under the receiver operating characteristic curve.",
    "Brier": "Mean squared score error against the binary reference label.",
    "EE_descriptive": "Fraction within +/-(0.05 + 0.2 times reference DAOD); descriptive, not calibrated uncertainty.",
    "mean_reference_daod": "Mean reference DAOD550 using AERONET total AOD.",
    "mean_predicted_daod": "Mean predicted DAOD550 using DB total AOD.",
    "zero_fraction": "Saved proportion of exact-zero predictions.",
    "bootstrap_unit": "Saved resampling unit; no resampling is performed here.",
    "replicates": "Saved number of bootstrap replicates.",
    "bootstrap_replicates": "Saved number of paired site-bootstrap replicates.",
    "unit": "Verbatim saved paired-bootstrap unit description.",
    "direction": "Verbatim saved interpretation of the sign of a contrast.",
    "A": "First member of an A-minus-B paired contrast.",
    "B": "Second member of an A-minus-B paired contrast.",
    "n_prediction_events": "Prediction events with CV repeats retained.",
    "stage": "Saved evaluation stage.", "branch": "Branch and its population; not a common sample across branches.",
    "hypothesis": "Preregistered hypothesis or component identifier.",
    "candidate": "Saved candidate estimator, including fold-specific Eselected where applicable.",
    "comparator": "Saved comparator; min(E1,E5) is the registered H3 comparator, not a new selection here.",
    "metric": "Metric used for the saved contrast.",
    "level": "Collocation or site_month aggregation level.",
    "minimum": "Minimum distinct collocations per site-month where applicable; missing is not zero.",
    "margin": "Explicit saved practical-effect/noninferiority margin; blank if unavailable.",
    "mean_ref": "Saved reference mean used in the component's domain; no margin is derived from it here.",
    "candidate_value": "Saved candidate metric value.", "comparator_value": "Saved comparator metric value.",
    "prediction_events": "Saved event count; interpretation depends on level and repeat structure.",
    "repeats": "Number of CV repeats represented by the saved summary.",
    "unique_site_months": "Distinct represented site/year/month cells on the saved support.",
    "difference": "Saved candidate-minus-comparator metric contrast, not recalculated.",
    "ci_low": "Saved lower 95% site-bootstrap bound for the paired difference.",
    "ci_high": "Saved upper 95% site-bootstrap bound for the paired difference.",
    "valid_bootstraps": "Saved count of valid bootstrap replicates.",
    "decision": "Verbatim saved decision, never recomputed.",
    "margin_status": "Availability of an explicit scalar margin in the saved component source.",
    "component_source_record_index": "Source row supplying the margin; missing if no component margin exists.",
    "variant": "Saved deployment variant; hierarchy and minimal-fallback sensitivity are separate.",
    "support": "Saved estimator intersection; never compare unlike supports as paired results.",
    "reference_mask": "Saved joint-reference or mean-only sensitivity mask.",
    "period": "DJF, MAM, JJA or SON on the saved seasonal support.",
    "region": "Saved geographic box, including explicitly labeled latitude sensitivities.",
    "estimator": "Saved method: E1 QA2; Eselected EV selection; E5 type; E7 LG25; E8 AE refit; E6 screened LG25; Song/CALIOP external products.",
    "candidate_mean": "Area-weighted candidate mean DAOD550.",
    "reference_mean": "Area-weighted CALIOP mean DAOD550 on this exact support.",
    "bias": "Candidate-minus-reference mean DAOD550 difference.",
    "rmse": "Root mean squared DAOD550 error.", "mae": "Mean absolute DAOD550 error.",
    "r": "Pearson spatial correlation on this support.",
}
RATES = {"precision", "recall", "FPR", "F1", "AUC", "coverage", "EE_descriptive", "zero_fraction"}
COUNTS = {ROW, "component_source_record_index", "n", "unique_rows", "sites", "eligible_rows",
          "TP", "FP", "FN", "TN", "replicates", "bootstrap_replicates", "valid_bootstraps",
          "n_prediction_events", "prediction_events", "repeats", "unique_site_months", "minimum"}


@dataclass
class Part:
    name: str
    frame: pd.DataFrame
    sources: list[dict]
    transform: str = "Verbatim source columns and row order; source_record_index added."


def definition(column):
    if column in DEFINITIONS:
        return DEFINITIONS[column]
    if column in HYPOTHESES:
        return f"Verbatim saved {column} decision; not inferred from component outcomes."
    if column.endswith(("_lo", "_hi")):
        end = "lower" if column.endswith("_lo") else "upper"
        return f"Saved {end} 95% site-bootstrap bound for {column[:-3]}; missing remains missing."
    for prefix, domain in (("DAOD_", "DAOD550"), ("fraction_", "reference-defined fraction")):
        if column.startswith(prefix):
            metric = column[len(prefix):]
            meaning = {"bias": "mean prediction-minus-reference error", "MAE": "mean absolute error",
                       "RMSE": "root mean squared error", "r": "Pearson correlation",
                       "R2": "coefficient of determination"}.get(metric, metric)
            return f"Saved {domain} {meaning}."
    if column.startswith("delta_"):
        return "Saved paired A-minus-B error contrast; negative favors A."
    if column.startswith(("A_", "B_")):
        return "Saved MAE or RMSE of the named contrast member on paired support."
    return "Verbatim saved source field; no additional definition or units inferred by this exporter."


def _part(registry, name, table_name, filters=None, match_keys=None, allow_empty=False):
    table = registry.payload["tables"][table_name]
    frame = registry.table(table_name)
    if ROW in frame:
        raise ValueError(f"Reserved export column already exists: {ROW}")
    keys = table["filter_columns"]
    if not keys or frame.duplicated(keys).any():
        raise ValueError(f"Missing or ambiguous source dimensions: {table_name}")
    filters = filters or {}
    for field, values in filters.items():
        frame = frame[frame[field].isin(values)]
    if match_keys is not None:
        wanted = pd.MultiIndex.from_frame(match_keys[KEYS])
        frame = frame[pd.MultiIndex.from_frame(frame[KEYS]).isin(wanted)]
    if frame.empty and not allow_empty:
        raise ValueError(f"Empty required source selection: {table_name}")
    selected_rows = frame.index.tolist()
    evidence = {
        "table": table_name, "source_file": table["source_file"],
        "source_sha256": table["source_sha256"], "source_columns": table["columns"],
        "filter_columns": keys, "count_columns": table["count_columns"],
        "selection": filters,
        "key_intersection": None if match_keys is None else clean(match_keys[KEYS].to_dict("records")),
        "rows": [{"source_record_index": int(i),
                  "filter": clean({k: row[k] for k in keys}),
                  "N": clean({k: row[k] for k in table["count_columns"]})}
                 for i, row in frame.iterrows()],
        "semantics": table["semantics"],
    }
    frame = frame.copy()
    frame.insert(0, ROW, selected_rows)
    return Part(name, frame.reset_index(drop=True), [evidence])


def _decision_margins(decisions, components):
    rows = []
    common = ["reference", "branch", "scheme"]
    for _, saved in decisions.frame.iterrows():
        for hypothesis in HYPOTHESES:
            if hypothesis not in decisions.frame:
                continue
            matches = components.frame[components.frame.hypothesis.eq(hypothesis)]
            for field in common:
                matches = matches[matches[field].eq(saved[field])]
            if len(matches) > 1:
                raise ValueError("Ambiguous explicit margin for saved decision")
            margin = None
            component_row = None
            status = "missing_no_explicit_scalar_margin"
            if len(matches) == 1:
                component_row = int(matches.iloc[0][ROW])
                if "margin" in matches and pd.notna(matches.iloc[0]["margin"]):
                    margin = matches.iloc[0]["margin"]
                    status = "explicit_saved_component_margin"
            rows.append({ROW: int(saved[ROW]), **{k: saved[k] for k in common},
                         "hypothesis": hypothesis, "decision": saved[hypothesis],
                         "margin": margin, "margin_status": status,
                         "component_source_record_index": component_row})
    return Part("margin_availability", pd.DataFrame(rows), decisions.sources + components.sources,
                "Unpivot saved decision columns; copy margin only from an exactly matching "
                "reference/branch/scheme/hypothesis component. Missing margins remain missing. "
                "source_record_index refers to decision_table; component_source_record_index "
                "refers to paired_differences. No new decision, count, interval or margin inferred.")


def build_tables(registry):
    points = _part(registry, "decomposition", DECOMPOSITION)
    components = _part(registry, "components", COMPONENTS, {"hypothesis": list(HYPOTHESES)})
    decisions = _part(registry, "decisions", DECISIONS)
    return {
        2: [_part(registry, "metrics", METRICS), _part(registry, "intervals", INTERVALS, allow_empty=True)],
        3: [points, _part(registry, "intervals", INTERVALS, match_keys=points.frame, allow_empty=True),
            _part(registry, "paired_differences", CONTRASTS)],
        4: [components, decisions, _decision_margins(decisions, components)],
        5: [_part(registry, "seasonal", SEASONAL, {"period": ["DJF", "MAM", "JJA", "SON"]})],
    }


def tex(value):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                     "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
                     "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def typeset_views(index, parts):
    """Select display rows without modifying or narrowing the archival Parts."""
    if index == 2:
        points, intervals = (p.frame for p in parts)
        selected = points[points.scope.eq("overall") & points.group.eq("all")
                          & points.population.isin(["own", "common"])].copy()
        bounds = [c for c in intervals if c.endswith(("_lo", "_hi"))]
        joined = selected.merge(intervals[[ROW, *KEYS, *bounds]].rename(columns={ROW: CI_ROW}),
                                on=KEYS, how="left", sort=False, validate="one_to_one")
        return [("overall_own_common", joined,
                 {"scope": ["overall"], "group": ["all"], "population": ["own", "common"]})]
    if index == 5:
        frame = parts[0].frame
        common = {"variant": ["hierarchy"], "reference_mask": ["prior_joint"]}
        views = []
        for name, extra in (("primary_core", {"support": ["core_common"]}),
                            ("E6_separate_support", {"support": ["full_common_E6"], "estimator": ["E6"]})):
            selectors = {**common, **extra}
            selected = frame
            for field, values in selectors.items():
                selected = selected[selected[field].isin(values)]
            views.append((name, selected.copy(), selectors))
        return views
    return [(part.name, part.frame, {}) for part in parts]


def view_provenance(index, parts):
    result = []
    for name, frame, selectors in typeset_views(index, parts):
        record = {"view": name, "rows": len(frame), "selection": selectors,
                  "source_record_indices": frame[ROW].tolist(), "font_pt": 7}
        if index == 2:
            record.update(interval_join_keys=KEYS, matched_interval_rows=int(frame[CI_ROW].notna().sum()),
                          interval_source_record_indices=clean(frame[CI_ROW].tolist()),
                          displayed_metrics=[m for _, fields in S2_PANELS for m in fields],
                          displayed_counts=["unique_rows", "sites"])
        elif index == 5:
            record["displayed_numeric_fields"] = S5_VALUES
        result.append(record)
    return result


def _point_interval(row, metric):
    point = cell(row.get(metric), metric, {})
    low, high = row.get(metric + "_lo"), row.get(metric + "_hi")
    if pd.notna(low) or pd.notna(high):
        point += r"\newline [" + cell(low, metric, {}) + ", " + cell(high, metric, {}) + "]"
    return point


def _render_reader_table(index, parts):
    lines = [r"\clearpage", r"\section*{Table S" + str(index) + "}",
             r"\renewcommand{\thetable}{S" + str(index) + "}",
             tex("Overall pixel metrics" if index == 2 else "Primary seasonal statistics and separate E6 support"),
             r"\par\smallskip\noindent\textbf{" + tex(VIEW_NOTES[index]) + "}",
             r"\par\smallskip", tex(NOTES[index]),
             r"\begingroup\fontsize{7}{8}\selectfont\setlength{\tabcolsep}{2pt}\setlength{\LTcapwidth}{\linewidth}"]
    fields = (["model", "QA", "unique_rows", "sites"] + [m for _, ms in S2_PANELS for m in ms]
              if index == 2 else ["estimator", "period", *S5_VALUES])
    lines.append(_longtable(["Column", "Definition"], [(tex(c), tex(definition(c))) for c in fields],
                            "Reader-view column definitions; full CSV column definitions are in provenance."))
    for name, frame, _ in typeset_views(index, parts):
        if frame.empty:
            lines.append(tex(f"{name}: no saved rows available."))
            continue
        groups = ["reference", "population", "scheme"] if index == 2 else ["reference", "support"]
        for key, block in frame.groupby(groups, sort=True, dropna=False):
            heading = "; ".join(f"{k}={v}" for k, v in zip(groups, key))
            lines += [r"\clearpage", r"\subsection*{" + tex(heading) + "}"]
            if index == 2:
                block = block.sort_values(["model", "QA"], kind="stable")
                for panel, metrics in S2_PANELS:
                    rows = [[tex(row.model), cell(row.QA, "QA", {}),
                             cell(row.unique_rows, "unique_rows", {}), cell(row.sites, "sites", {})]
                            + [_point_interval(row, m) for m in metrics] for _, row in block.iterrows()]
                    lines.append(_longtable(["Model", "QA", "N", "Sites", *metrics], rows,
                                            f"{heading}; {panel}; overall/all; saved 95% intervals where available.",
                                            repeat_caption=True))
            else:
                lines.append(tex("hierarchy / prior_joint; " + ("E6 only; separate full_common_E6 support, not paired with core_common."
                                 if name == "E6_separate_support" else "All saved estimators on primary core_common support.")))
                for region, regional in block.groupby("region", sort=False, dropna=False):
                    regional = regional.assign(_season=regional.period.map({"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}))
                    regional = regional.sort_values(["estimator", "_season"], kind="stable")
                    rows = [[tex(row.estimator), tex(row.period)] + [cell(row[c], c, {}) for c in S5_VALUES]
                            for _, row in regional.iterrows()]
                    lines.append(_longtable(["Method", "Season", *S5_VALUES], rows,
                                            f"{heading}; hierarchy/prior_joint; region={region}.",
                                            repeat_caption=True))
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def cell(value, column, codes):
    if value is None or pd.isna(value):
        return "--"
    if isinstance(value, str):
        return codes[value]
    if isinstance(value, bool):
        return "true" if value else "false"
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite value in registry export")
    if column in COUNTS:
        if float(value) != int(value):
            raise ValueError(f"Noninteger saved count: {column}")
        return str(int(value))
    base = column[:-3] if column.endswith(("_lo", "_hi")) else column
    decimals = 6 if column == "margin" else 2 if base in RATES else 3
    return f"{value:.{decimals}f}"


def _longtable(headers, rows, caption, repeat_caption=False):
    width = f"{1 / len(headers):.8f}"
    col = r">{\raggedright\arraybackslash}p{\dimexpr" + width + r"\linewidth-2\tabcolsep\relax}"
    head = " & ".join(tex(h) for h in headers) + r" \\"
    continued = [r"\caption[]{" + tex(caption + " (continued)") + r"}\\"] if repeat_caption else []
    lines = [r"\begin{longtable}{" + col * len(headers) + "}",
             r"\caption{" + tex(caption) + r"}\\", r"\toprule", head,
             r"\midrule\endfirsthead", *continued, r"\toprule", head, r"\midrule\endhead",
             r"\midrule\endfoot", r"\bottomrule\endlastfoot"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def render_table(index, parts):
    if index in (2, 5):
        return _render_reader_table(index, parts)
    codes = {}
    for part in parts:
        for row in part.frame.itertuples(index=False, name=None):
            for value in row:
                if isinstance(value, str) and value not in codes:
                    codes[value] = f"K{len(codes) + 1}"
    lines = [r"\clearpage", r"\section*{Table S" + str(index) + "}",
             r"\renewcommand{\thetable}{S" + str(index) + "}", tex(CAPTIONS[index]),
             r"\par\smallskip", tex(NOTES[index]), r"\par\smallskip",
             tex("Candidate archival export: full scopes are intentionally extensive. CSVs retain "
                 "unrounded values. Display uses three decimals, two for rates/AUC, six for margins; "
                 "counts are exact integers. -- means missing, not zero. All categorical cells use "
                 "the codebook. Column panels join on source_record_index within each part; "
                 "margin_availability also requires hypothesis to identify a row."),
             r"\begingroup\fontsize{7}{8}\selectfont\setlength{\tabcolsep}{2pt}"]
    if codes:
        lines.append(_longtable(["Code", "Saved categorical value"],
                                [(code, tex(value)) for value, code in codes.items()],
                                "Categorical codebook; codes are display aliases only."))
    for part in parts:
        lines += [r"\subsection*{" + tex(part.name) + "}", tex(part.transform)]
        lines.append(_longtable(["Column", "Definition"],
                                [(tex(c), tex(definition(c))) for c in part.frame],
                                f"{part.name}: column definitions."))
        # Keep the saved row index in every panel; categorical aliases keep even
        # long region/model identifiers legible without dropping their dimensions.
        identifiers = [ROW, "hypothesis"] if part.name == "margin_availability" else [ROW]
        columns = [c for c in part.frame if c not in identifiers]
        for offset in range(0, len(columns), 8):
            fields = [*identifiers, *columns[offset:offset + 8]]
            rows = ([cell(value, column, codes) for column, value in zip(fields, values)]
                    for values in part.frame[fields].itertuples(index=False, name=None))
            lines.append(_longtable(fields, rows,
                                    f"{part.name}: column panel {offset // 8 + 1}; source rows retained."))
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def _guard_output(path):
    path = path.resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Unpublished outputs must be outside the public worktree")
    for parent in (path, *path.parents):
        if parent.name == "source" and (parent / "01_instructions").is_dir():
            raise ValueError("Do not write candidate outputs into author source folders")


def export_tables(registry_path, output_dir, compile_latex=False):
    registry_path, output_dir = Path(registry_path).resolve(), Path(output_dir).resolve()
    _guard_output(output_dir)
    registry_hash = digest(registry_path)
    tables = build_tables(Registry.read(registry_path))
    if digest(registry_path) != registry_hash:
        raise RuntimeError("Registry changed during export")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"registry_file": str(registry_path), "registry_sha256": registry_hash,
                "script_sha256": digest(Path(__file__)), "tables": {}}

    def write(name, content):
        path = output_dir / name
        _guard_output(path)
        if path.resolve() == registry_path:
            raise ValueError("Output would overwrite input registry")
        path.write_text(content)
        return path

    for index, parts in tables.items():
        prefix = f"TableS{index:02d}"
        outputs, descriptions = {}, []
        for part in parts:
            path = write(f"{prefix}_{part.name}.csv", part.frame.to_csv(index=False))
            outputs[path.name] = digest(path)
            descriptions.append({"part": part.name, "rows": len(part.frame), "columns": list(part.frame),
                                 "column_definitions": {c: definition(c) for c in part.frame},
                                 "transform": part.transform, "sources": part.sources})
        fragment = write(prefix + ".tex", render_table(index, parts))
        outputs[fragment.name] = digest(fragment)
        views = view_provenance(index, parts)
        notes = write(prefix + "_notes.md", f"# Table S{index}\n\n{CAPTIONS[index]}\n\n{NOTES[index]}\n\n"
                      + VIEW_NOTES.get(index, "All exported rows are typeset.") + "\n\n"
                      + "\n".join(f"- Typeset {v['view']}: {v['rows']} rows; selectors {v['selection']}" for v in views) + "\n\n"
                      + "\n".join(f"- {p.name}: {len(p.frame)} rows; {p.transform}" for p in parts) + "\n")
        outputs[notes.name] = digest(notes)
        wrapper = write(prefix + "_preview.tex", "\n".join([
            r"\documentclass{article}", r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}", r"\usepackage[a4paper,landscape,margin=16mm]{geometry}",
            r"\usepackage{longtable,booktabs,array}", r"\begin{document}",
            r"\input{" + fragment.name + "}", r"\end{document}", ""]))
        outputs[wrapper.name] = digest(wrapper)
        compilation = {"requested": compile_latex, "status": "not_run"}
        if compile_latex:
            executable = shutil.which("pdflatex")
            if executable is None:
                raise RuntimeError("pdflatex unavailable; no compilation claimed")
            for pass_number in (1, 2):
                result = subprocess.run(
                    [executable, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                    cwd=output_dir, capture_output=True, text=True, timeout=300,
                )
                write(f"{prefix}_compile_pass{pass_number}.txt", result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"{prefix}: LaTeX pass {pass_number} failed; see private compile log")
            compilation = {"requested": True, "status": "compiled", "passes": 2,
                           "engine": "pdflatex", "pdf": wrapper.with_suffix(".pdf").name}
            outputs[wrapper.with_suffix(".pdf").name] = digest(wrapper.with_suffix(".pdf"))
        provenance = {**{k: v for k, v in manifest.items() if k != "tables"},
                      "caption": CAPTIONS[index], "notes": NOTES[index], "parts": descriptions,
                      "outputs": outputs, "latex": compilation, "typeset_views": views,
                      "typeset_scope_note": VIEW_NOTES.get(index, "All exported rows are typeset.")}
        if index == 4:
            missing = parts[-1].frame
            provenance["missing_explicit_margins"] = clean(missing[missing.margin.isna()].to_dict("records"))
        path = write(prefix + "_provenance.json", json.dumps(provenance, allow_nan=False, indent=2) + "\n")
        manifest["tables"][prefix] = {"parts": {p.name: len(p.frame) for p in parts},
                                      "provenance": path.name, "provenance_sha256": digest(path),
                                      "latex": compilation,
                                      "typeset_rows": {v["view"]: v["rows"] for v in views}}
    if digest(registry_path) != registry_hash:
        raise RuntimeError("Registry changed during export")
    write("supplement_tables_manifest.json", json.dumps(manifest, allow_nan=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true", help="Compile private preview PDFs with pdflatex")
    args = parser.parse_args(argv)
    result = export_tables(args.registry, args.output_dir, args.compile)
    print(json.dumps(result["tables"], indent=2))


if __name__ == "__main__":
    main()
