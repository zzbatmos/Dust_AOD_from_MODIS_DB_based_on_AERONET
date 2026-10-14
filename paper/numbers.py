"""Build an auditable numbers registry without running inference or fitting.

Usage: python -m paper.numbers --source-root PATH --output-dir PRIVATE_PATH
The registry and its numerical contents are not intended for the public repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GROUPED = (
    "metrics", "site_bootstrap_95CI", "decomposition_2x2",
    "decomposition_paired_differences", "paired_site_bootstrap_differences",
    "population_counts", "ablation_collocation_coverage", "baselines_strict_paired",
    "reliability", "type_overrides", "site_inventory", "site_folds",
    "calibration_sites", "target_changes_stratified", "AOD440_target_distribution",
)
STAGE_B = (
    "collocation_metrics", "site_month_metrics", "stratum_bias", "paired_differences",
    "decision_table", "calibration_bins", "bias_decomposition_E1",
    "temporal_supplement_anomalies", "temporal_supplement_seasonal_sites",
    "temporal_supplement_seasonal_summary", "temporal_supplement_support",
)
STAGE_C = (
    "seasonal_metrics", "background_floor", "floor_selection_audit",
    "low_loading_contributions", "reference_mask_sensitivity",
)
STAGE_B_SUPPORT = ("inner_selection", "E8_coefficients", "f_pos_retransformation_stage_b")
DIMENSIONS = set("""reference target population model features scheme QA scope group A B
    stage branch pairing stratum estimator repeat_summary minimum hypothesis
    candidate comparator metric level variant support reference_mask period region
    component bin bin_low bin_high event site_key site Site repeat fold CV_region year lat lon wavelength
    row_id split_id point_index threshold row_order comparison_support selected_estimator figure panel axis_max
    case pixel_index swath_row swath_col role source_record_index date granule
    calibration_reserved sample platform variable month db_type algorithm aod_bin regime subset
    minimum_rows seasonal_minimum_calendar_months anomaly_minimum_years
    exploratory_below_sites expected_repeats""".split())
COUNTS = set("""n rows count sites unique_sites unique_rows unique_site_months
    prediction_events n_prediction_events eligible_rows evaluation_valid_rows
    train_valid_rows train_sites eligible_collocations target_valid finite_AOD440
    repeats n_calendar_months cell_months selected_rows selected_sites calibration_records
    eligible_sites overridden_sites parent_events excluded_events collocations
    training_rows training_sites training_positive_rows training_positive_sites
    evaluation_positive_rows qualifying_sites qualifying_cells input_rows paired_rows dropped_rows rows_2017""".split())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def units_for(metric, count_columns, value):
    if metric in count_columns:
        return "count"
    if metric.endswith("_sr") or metric.startswith("Lidar_Ratio["):
        return "sr"
    if metric in ("lat", "lat_lower", "lat_upper"):
        return "degrees_north"
    if metric in ("lon", "lon_lower", "lon_upper"):
        return "degrees_east"
    if metric == "wavelength" or metric.endswith("_wavelength_nm"):
        return "nm"
    if metric.endswith("_seconds"):
        return "s"
    return "1" if isinstance(value, (int, float)) else None


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def source_tables():
    for prefix, names in (("grouped_validation", GROUPED),
                          ("expected_value_test/stage_b", STAGE_B),
                          ("expected_value_test/stage_b_support", STAGE_B_SUPPORT),
                          ("expected_value_test/stage_c", STAGE_C)):
        for name in names:
            yield f"{prefix}/{name}.csv"
    yield "coverage/T1_coverage.csv"


def semantics(name):
    if "/stage_b_support/" in name:
        return ("Saved Stage B fold-level diagnostics and honest inner selection; "
                "preserve reference/population/branch/split_id/scheme/repeat. "
                "These are the Stage B nested artifacts, not earlier Stage A diagnostics. "
                "No new calibration, coefficient fit, model choice or bootstrap. "
                "Fold dispersion is not a confidence interval; deployment selection is separate.")
    if "/stage_b/" in name:
        return ("Preserve repeat_summary: mean_over_repeats means arithmetic mean "
                "of repeat-specific metrics, not metrics of averaged predictions. "
                "Intervals are saved site-bootstrap intervals where provided.")
    if "/stage_c/" in name:
        return ("Native 2x5 grid; preserve support/reference_mask. Seasonal means "
                "use the existing common-month support and equal-month means; "
                "regional metrics are area weighted. No inferential interval supplied.")
    if name.startswith("coverage/"):
        return ("Existing coverage audit, not automatically a full-year census. "
                "Its sample/month/region/type/algorithm/bin marginal rows overlap; "
                "do not sum across marginal totals.")
    return ("Original grouped evaluation pools held-out prediction events across "
            "repeats. n counts events; unique_rows counts distinct collocations. "
            "Preserve population, scheme, QA, scope and group.")


def ingest(path: Path, name: str):
    before = digest(path)
    df = pd.read_csv(path)
    rows = clean(df.to_dict("records"))
    if digest(path) != before:
        raise RuntimeError(f"Source changed during registry generation: {path}")
    return {
        "source_file": str(path.resolve()), "source_sha256": before,
        "source_rows": len(df), "columns": list(df.columns),
        "filter_columns": [c for c in df if c in DIMENSIONS],
        "count_columns": [c for c in df if c in COUNTS or c.startswith("n_")],
        "semantics": semantics(name), "records": rows,
    }


class Registry:
    """Exact selectors fail on empty or ambiguous matches; no nearest-row guesses."""

    def __init__(self, payload):
        self.payload = payload
        self._frames = {}

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def table(self, name, **filters):
        table = self.payload["tables"][name]
        if name not in self._frames:
            self._frames[name] = pd.DataFrame(table["records"], columns=table["columns"])
        frame = self._frames[name]
        for column, expected in filters.items():
            if column not in frame:
                raise KeyError(f"{name}: no filter column {column}")
            frame = frame[frame[column].isna() if expected is None
                          else frame[column].eq(expected)]
        return frame.copy()

    def number(self, name, value_column, **filters):
        metric = value_column
        table = self.payload["tables"][name]
        selected = self.table(name, **filters)
        if len(selected) != 1:
            raise ValueError(f"{name} {filters}: expected 1 row, found {len(selected)}")
        row = clean(selected.iloc[0].to_dict())
        if metric not in row:
            raise KeyError(f"{name}: no metric {metric}")
        interval = None
        interval_source = None
        low, high = f"{metric}_lo", f"{metric}_hi"
        if low in row and high in row:
            interval = [row[low], row[high]]
        elif metric in ("difference", "contribution") and "ci_low" in row:
            interval = [row["ci_low"], row["ci_high"]]
        elif metric.endswith("_A_minus_B"):
            stem = metric.removesuffix("_A_minus_B")
            if f"{stem}_lo" in row:
                interval = [row[f"{stem}_lo"], row[f"{stem}_hi"]]
        elif name in ("grouped_validation/metrics", "grouped_validation/decomposition_2x2"):
            ci_name = "grouped_validation/site_bootstrap_95CI"
            keys = ("reference", "population", "model", "scheme", "QA", "scope", "group")
            ci = self.table(ci_name, **{k: row[k] for k in keys})
            if len(ci) > 1:
                raise ValueError(f"Ambiguous interval for {name}: {filters}")
            if len(ci) == 1 and low in ci and high in ci:
                interval = clean([ci.iloc[0][low], ci.iloc[0][high]])
                interval_source = {
                    "file": self.payload["tables"][ci_name]["source_file"],
                    "sha256": self.payload["tables"][ci_name]["source_sha256"],
                    "filter": {k: row[k] for k in keys},
                }
        exact_filter = {k: row[k] for k in table["filter_columns"]}
        identifier = json.dumps([name, exact_filter, metric], sort_keys=True)
        if interval == [None, None]:
            interval = None
        return {
            "key": name + "/" + metric + "/" + hashlib.sha256(identifier.encode()).hexdigest()[:16],
            "value": row[metric], "interval": interval,
            "interval_type": "95% site bootstrap" if interval is not None else None,
            "interval_source": interval_source,
            "N": {k: row[k] for k in table["count_columns"]},
            "units": units_for(metric, table["count_columns"], row[metric]),
            "reference": row.get("reference", row.get("target")),
            "scheme": row.get("scheme"),
            "support": {k: row[k] for k in ("population", "branch", "pairing", "support",
                        "reference_mask", "stratum", "scope", "group", "minimum", "repeat_summary") if k in row},
            "source_file": table["source_file"], "source_sha256": table["source_sha256"],
            "source_record_index": int(selected.index[0]), "filter": exact_filter,
            "semantics": table["semantics"],
        }


def check_claims(registry, requests):
    results = []
    for request in requests:
        number = registry.number(request["table"], request["metric"], **request["filter"])
        expected = request.get("draft_value")
        decimals = request.get("decimals")
        value = number["value"]
        matches = None
        if expected is not None and value is not None:
            matches = (round(float(value), decimals) == round(float(expected), decimals)
                       if decimals is not None else value == expected)
        results.append({"claim_id": request["claim_id"], "location": request.get("location"),
                        "draft_value": expected, "matches_at_stated_precision": matches,
                        "number": number, "caveat": request.get("caveat")})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--claims", type=Path, action="append", default=[],
                        help="Private JSON list of exact claim selectors; repeat for additional lists")
    parser.add_argument("--derived-dir", type=Path, help="Audited plot arrays with per-file provenance JSON")
    parser.add_argument("--extra-table", action="append", default=[], metavar="NAME=PATH",
                        help="Explicit additional existing CSV, kept separate from primary source tables")
    parser.add_argument("--derive-claims", action="store_true", help="Include checked arithmetic with full operand provenance")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.is_relative_to(code_root):
        raise ValueError("Numerical outputs must be outside the public code worktree")
    tables = {}
    for relative in source_tables():
        path = args.source_root / relative
        tables[relative.removesuffix(".csv")] = ingest(path, relative)
    for entry in args.extra_table:
        name, path = entry.split("=", 1)
        if name in tables:
            raise ValueError(f"Duplicate table key: {name}")
        tables[name] = ingest(Path(path), name)
    if args.derived_dir:
        for path in sorted(args.derived_dir.glob("*.csv")):
            provenance = path.with_suffix(".provenance.json")
            metadata = json.loads(provenance.read_text())
            if digest(path) != metadata["output_sha256"]:
                raise ValueError(f"Derived array hash mismatch: {path}")
            table = ingest(path, "derived/" + path.stem)
            table["derivation"] = metadata
            table["semantics"] = metadata["semantics"]
            tables["derived/" + path.stem] = table
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit, "generator_sha256": digest(Path(__file__)),
        "primary_reference": "experimental1020", "tables": tables,
        "claims": [], "scope": "Existing frozen analysis tables; no inference or refitting",
        "remaining": "Derived plot arrays and claim selectors are added explicitly; this is not a completion claim.",
    }
    if args.claims:
        requests = [r for path in args.claims for r in json.loads(path.read_text())]
        if len({r["claim_id"] for r in requests}) != len(requests):
            raise ValueError("Duplicate claim identifiers")
        payload["claims"] = check_claims(Registry(payload), requests)
        payload["claims_sources"] = {str(path.resolve()): digest(path) for path in args.claims}
    if args.derive_claims:
        from paper.claim_arithmetic import derive_claims
        payload["derived_claims"] = derive_claims(Registry(payload))
        payload["arithmetic_script_sha256"] = digest(Path(__file__).with_name("claim_arithmetic.py"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_numbers.json").write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    lines = ["# Frozen manuscript numbers registry", "", f"Code: `{commit}`.", "",
             "No fits, predictions or new experimental comparisons were run.",
             "Every table retains its exact rows, filters, sample counts and source hash.",
             "Use `Registry.number` for a scalar with its saved interval and provenance.", "",
             "## Sources", "", "| Table | Rows | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| {name} | {t['source_rows']} | `{t['source_sha256']}` |" for name, t in tables.items()]
    lines += ["", "## Draft claim checks", ""]
    if not payload["claims"]:
        lines += ["Pending explicit claim selectors; do not treat the registry as fully reconciled."]
    for item in payload["claims"]:
        n = item["number"]
        lines += [f"- {item['claim_id']}: {n['value']}; interval {n['interval']}; "
                  f"N={n['N']}; draft match={item['matches_at_stated_precision']}. "
                  f"Source `{n['source_file']}`; filter `{n['filter']}`."]
    if payload.get("derived_claims"):
        lines += ["", "## Derived comparisons", "", "Exact operands, filters, counts and caveats are retained in JSON.", ""]
        lines += [f"- `{r['key']}`: {r['value']} {r['units']}; interval {r['interval']}; "
                  f"equation `{r['equation']}`." for r in payload["derived_claims"]]
    (output / "paper_numbers.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"tables": len(tables), "rows": sum(t["source_rows"] for t in tables.values()),
                      "claims": len(payload["claims"]), "derived_claims": len(payload.get("derived_claims", [])),
                      "output": str(output)}))


if __name__ == "__main__":
    main()
