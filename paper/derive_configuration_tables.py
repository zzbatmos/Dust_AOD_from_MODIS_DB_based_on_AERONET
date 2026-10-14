"""Source-hashed S6/S7 display records; no fits, predictions or training defaults."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paper.numbers import digest


S6 = "configuration_settings"
S7 = "monthly_variables"
UNRECORDED = ("gamma", "reg_alpha", "reg_lambda", "max_delta_step", "colsample_bylevel",
              "colsample_bynode", "max_bin", "grow_policy", "sampling_method",
              "eval_metric", "early_stopping_rounds", "monotone_constraints", "interaction_constraints")
SEMANTICS = {
    S6: "Frozen YAML and branch metadata copied with feature order preserved. Branch-level training "
        "config is not claimed to be separately recorded per learner. UBJ is loaded only to expose "
        "persistent inference-model JSON fields; never save_config/get_params, fitting or prediction. "
        "Absent training hyperparameters are NOT RECORDED, not XGBoost defaults. Empty serialized "
        "feature names are missing metadata; ordered YAML names are checked against metadata.json. "
        "Postprocessor arrays are summarized by saved shapes/dtypes and scalar constants only. "
        "The complete settings CSV and per-row evidence retain all extracted fields.",
    S7: "All configured monthly science/count variables, in configured order. Attributes, dtypes and "
        "encoding come from empty frozen writer snapshots on both configured grids; no data updates "
        "or inference. Definitions are verbatim DATA_FORMAT.md monthly-table entries. Retrieval-count "
        "weighting, not equal-day/equal-granule/area weighting. Counts are real zeros without fill; "
        "zero denominators give missing statistics. Contribution denominators are full n_ml, not subset counts.",
}


def literal(value):
    if isinstance(value, np.ndarray):
        return [literal(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return literal(value.item())
    if isinstance(value, dict):
        return {k: literal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [literal(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def json_text(value):
    return json.dumps(literal(value), ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def guard_output(output, root=None):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) or (root and output.is_relative_to(Path(root).resolve())):
        raise ValueError("Write private outputs outside the frozen code/source root")
    if any(p.name == "source" and (p / "01_instructions").is_dir() for p in (output, *output.parents)):
        raise ValueError("Do not write into author source folders")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def read_model(path):
    """Persistent model fields only: save_config would invent loaded training defaults."""
    from xgboost import Booster
    booster = Booster()
    booster.load_model(path)
    saved = json.loads(booster.save_raw(raw_format="json"))["learner"]
    gb = saved["gradient_booster"]
    return {"learner_model_param": saved["learner_model_param"], "objective": saved["objective"],
            "feature_names": saved["feature_names"], "feature_types": saved["feature_types"],
            "attributes": saved["attributes"], "gradient_booster_name": gb["name"],
            "gbtree_model_param": gb["model"]["gbtree_model_param"]}


def monthly_definitions(document, variables):
    section = document.split("## Monthly product (L3M)", 1)[1].split("### Coordinates", 1)[0]
    definitions = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0].startswith("`"):
            name = cells[0].strip("`")
            if name in definitions:
                raise ValueError(f"Duplicate documented monthly variable: {name}")
            definitions[name] = cells[1]
    if set(definitions) != set(variables) or len(variables) != len(set(variables)):
        raise ValueError("Documented monthly variable set differs from frozen YAML")
    return definitions


def build_settings(config, metadata, pairs, models, arrays, evidence, writer_attrs):
    records = []
    def add(scope, branch, component, metric, value, source, locator, note="", availability="explicit_saved"):
        # Protect literal scientific fill labels from pandas' default CSV NA parser.
        string_value = json_text(value) if isinstance(value, str) and value in {
            "NaN", "nan", "NA", "N/A", "None", "null", "NULL", "<NA>", ""} else value
        text = ("NOT RECORDED" if availability == "missing_metadata" else
                "none (saved null)" if value is None else string_value if isinstance(value, str) else json_text(value))
        records.append(dict(row_order=len(records), scope=scope, branch=branch, component=component,
                            metric=metric, value_text=text, value_json=json_text({"value": value}),
                            availability=availability, note=note or "Verbatim saved setting.",
                            evidence_json=json_text([{**evidence[source], "locator": locator}])))
    for metric, value in flatten(config):
        branch = metric.split(".")[1] if metric.startswith("branches.") else "global"
        add("yaml/" + metric.split(".")[0], branch, "configuration", metric, value, "config", metric)
    add("implementation", "global", "branch_assignment", "priority", list(config["branches"]),
        "branches", "assign_branches", "First input-valid configured branch; none=0. No model-output gating.")
    for branch, spec in config["branches"].items():
        meta = metadata[branch]
        if meta["feature_names"] != spec["features"] or meta["selected"] != spec["selected"] or meta["reference"] != config["reference"]["name"]:
            raise ValueError(f"YAML/metadata mismatch: {branch}")
        rule = (f"All features finite; AOD >= {config['inputs']['minimum_aod']}; "
                f"SSA within inclusive {config['inputs']['ssa_bounds']}; AE finite only.")
        add("implementation", branch, "branch_assignment", "validity", rule, "branches", "valid_features")
        for rank, feature in enumerate(spec["features"], 1):
            add("feature_order", branch, "model_input", f"feature_{rank:02d}", feature, "config",
                f"branches.{branch}.features[{rank-1}]", "One-based input position; metadata.json order checked identical.")
        for metric, value in flatten(meta):
            add("branch_metadata", branch, "metadata", metric, value, branch + "/metadata", metric,
                "Branch-level metadata; training config not independently asserted for each learner.")
        for metric in UNRECORDED:
            if metric not in meta.get("config", {}):
                add("missing_metadata", branch, "training_settings", metric, None, branch + "/metadata",
                    "config (key absent)", "No value inferred from library defaults or load-time configuration.", "missing_metadata")
        for metric, value in pairs[branch].items():
            add("pair_constants", branch, "pair", metric, value, branch + "/pair", metric)
        for role, saved in models[branch].items():
            if int(saved["learner_model_param"]["num_feature"]) != len(spec["features"]):
                raise ValueError(f"Serialized feature count mismatch: {branch}/{role}")
            if saved["feature_names"] and saved["feature_names"] != spec["features"]:
                raise ValueError(f"Serialized feature order mismatch: {branch}/{role}")
            for metric, value in flatten(saved):
                missing = metric in ("feature_names", "feature_types") and not value
                locator = ("learner/gradient_booster/model/" + metric.replace(".", "/") if metric.startswith("gbtree_model_param.")
                           else "learner/gradient_booster/name" if metric == "gradient_booster_name"
                           else "learner/" + metric.replace(".", "/"))
                add("serialized_model", branch, role, metric, value, branch + "/" + role, locator,
                    "Persistent inference-model field; not a recovered training default." if not missing else
                    "Empty serialized feature metadata; feature order is supplied by matching YAML and branch metadata.",
                    "missing_metadata" if missing else "explicit_saved")
        for artifact, saved in arrays[branch].items():
            for name, value in saved.items():
                add("postprocessing", branch, artifact, name, value, branch + "/" + artifact, name,
                    "Saved scalar, or array shape/dtype only; no fit or numerical re-estimation.")
    formulas = {
        "raw_probability": "P = classifier.predict_proba(X)[:,1]; QA uses P, not calibrated P.",
        "conditional_fraction": "f_pos = expit(positive_fraction.predict(X)); unsmeared fraction used for QA2.",
        "probability_calibrated": "P_cal = interp(P, iso_x, iso_y) using saved postprocessor knots.",
        "positive_fraction_star": "Saved smear-curve interpolation inside its logit range; empirical mean(expit(z+residual)) outside; no_positive_rows gives zero.",
        "E3c": "P_cal*f_pos_star + (1-P_cal)*f_neg (saved nondust mean).",
        "E4": "clip(direct_fraction.predict(X),0,1).",
        "DAOD_EV": "Selected branch EV fraction times DB AOD550; no QA gate.",
        "DAOD_QA2": f"Unsmeared f_pos times DB AOD550 where raw P >= {config['qa']['thresholds'][config['qa']['product_gate_index']]}; supported gate rejections are zero.",
    }
    for metric, value in formulas.items():
        add("implementation", "global", "output_forms", metric, value, "predictor", "BranchModel.predict / EVPredictor.predict_frame")
    add("implementation", "global", "qa", "tier_rule", "searchsorted(configured raw-P thresholds, P, side='right'); equality enters higher tier.",
        "branches", "qa_tiers")
    for metric in ("branch_rule", "screening", "missing_conventions", "weighting", "aggregation"):
        add("writer_contract", "global", "L3M", metric, writer_attrs[metric], "writer_l3", "MonthlyAccumulator.to_dataset/global_attrs")
    return pd.DataFrame(records)


def build_monthly(config, document, snapshots, evidence):
    variables = config["outputs"]["monthly"]
    definitions = monthly_definitions(document, variables)
    records, signatures = [], []
    for grid, ds in snapshots.items():
        signature = []
        for name in variables:
            v = ds[name]
            count = name.startswith("n_")
            denom = "not_applicable_count" if count else v.attrs.get("denominator")
            if v.dims != ("time", "lat", "lon") or v.attrs["units"] != config["outputs"]["units"]:
                raise ValueError(f"Monthly dimension/unit mismatch: {name}")
            if count:
                if str(v.dtype) != "int32" or v.encoding.get("_FillValue") is not None or not (v.values == 0).all():
                    raise ValueError(f"Count dtype/fill/empty-cell mismatch: {name}")
            elif str(v.dtype) != "float64" or not np.isnan(v.encoding.get("_FillValue")) or not np.isnan(v.values).all():
                raise ValueError(f"Statistic dtype/fill/empty-cell mismatch: {name}")
            expected = "n_valid_aod" if name == "frac_ml_missing" else "n_ml"
            if not count and denom != expected:
                raise ValueError(f"Monthly denominator mismatch: {name}")
            if name.startswith("frac_qa") and v.attrs["probability_threshold"] != config["qa"]["thresholds"][int(name[-1])-1]:
                raise ValueError(f"Raw QA threshold mismatch: {name}")
            if name.startswith("ev_contribution"):
                key = "low_aod_threshold" if "aod_lt" in name else "low_probability_threshold"
                if v.attrs["subset_threshold"] != config["diagnostics"][key]:
                    raise ValueError(f"Contribution threshold mismatch: {name}")
            missing = ("Real zero count in empty cells; no fill sentinel; int64 internal accumulation with checked int32 export."
                       if count else "Missing (NaN) when denominator is zero; supported zero values are real zeros.")
            if name == "frac_ml_missing":
                missing += " Valid-AOD-only cell: 1; entirely empty cell: missing."
            if name.startswith("ev_contribution"):
                missing += " No subset matches with ML support: zero; denominator is full n_ml, not subset count."
            if name == "daod550_qa2":
                missing += " Supported raw-P gate rejections contribute zeros, not missing values."
            record = dict(row_order=variables.index(name), variable=name, units=v.attrs["units"], dtype=str(v.dtype),
                          dimensions="time,lat,lon", denominator=denom, definition=definitions[name],
                          missing_vs_zero=missing, weighting=ds.attrs["weighting"],
                          cell_methods=v.attrs.get("cell_methods", "not_applicable_count"),
                          attrs_json=json_text(v.attrs), encoding_json=json_text(v.encoding),
                          evidence_json=json_text([
                              {**evidence["config"], "locator": f"outputs.monthly[{variables.index(name)}]"},
                              {**evidence["docs"], "locator": f"Monthly product (L3M)/Variable table/{name}"},
                              {**evidence["writer_l3"], "locator": f"MonthlyAccumulator.to_dataset/{name}"},
                              {**evidence["writer_l2"], "locator": "_attrs / _chunks / _normalise"}]))
            signature.append(record)
        signatures.append(signature)
        if not records:
            records = signature
    if not signatures or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Monthly variable metadata differs between grids")
    return pd.DataFrame(records)


def derive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    guard_output(output, root)
    paths = {"config": root / "configs/dbdust_v2.0.0.yaml", "docs": root / "docs/v2/DATA_FORMAT.md",
             "branches": root / "dbdust/branches.py", "predictor": root / "dbdust/predictor.py",
             "writer_l3": root / "dbdust/product_l3.py", "writer_l2": root / "dbdust/product_l2.py",
             "config_loader": root / "dbdust/config.py"}
    config = yaml.safe_load(paths["config"].read_text())
    model_root = root / config["artifacts"]["root"]
    manifest = model_root / config["artifacts"]["manifest"]
    if digest(manifest) != config["artifacts"]["manifest_sha256"]:
        raise ValueError("Frozen model manifest SHA mismatch")
    paths["manifest"] = manifest
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        p = (model_root / name).resolve()
        if not p.is_relative_to(model_root) or digest(p) != expected:
            raise ValueError(f"Manifest artifact mismatch: {name}")
        paths[name.removesuffix(p.suffix)] = p
    hashes = {str(p): digest(p) for p in paths.values()}
    evidence = {k: {"source_file": str(p), "source_sha256": hashes[str(p)]} for k, p in paths.items()}
    metadata, pairs, models, arrays = {}, {}, {}, {}
    for branch in config["branches"]:
        metadata[branch] = json.loads(paths[branch + "/metadata"].read_text())
        pairs[branch] = json.loads(paths[branch + "/pair"].read_text())
        models[branch] = {role: read_model(paths[branch + "/" + role])
                          for role in ("classifier", "positive_fraction", "direct_fraction")}
        arrays[branch] = {}
        for artifact in ("postprocessor", "smear_curve"):
            with np.load(paths[branch + "/" + artifact], allow_pickle=False) as data:
                arrays[branch][artifact] = {k: v.item() if v.ndim == 0 else
                                            {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()}
    from dbdust import product_l3, product_l2, config as frozen_config
    if Path(product_l3.__file__).resolve() != paths["writer_l3"] or Path(product_l2.__file__).resolve() != paths["writer_l2"]:
        raise ValueError("Imported writer is not the requested frozen source")
    if frozen_config.CONFIG != config:
        raise ValueError("Imported writer configuration differs from frozen YAML")
    snapshots = {grid: product_l3.MonthlyAccumulator(grid).to_dataset("2000-01") for grid in config["grids"]}
    try:
        first = next(iter(snapshots.values()))
        frames = {S6: build_settings(config, metadata, pairs, models, arrays, evidence, first.attrs),
                  S7: build_monthly(config, paths["docs"].read_text(), snapshots, evidence)}
        grid_metadata = {g: {"sizes": dict(ds.sizes), "attrs": literal(ds.attrs),
                             "coordinates": {c: {"dims": list(ds[c].dims), "attrs": literal(ds[c].attrs),
                                                   "encoding": literal(ds[c].encoding)}
                                             for c in ("lat", "lon", "time", "lat_bounds", "lon_bounds", "time_bounds")}}
                         for g, ds in snapshots.items()}
    finally:
        for ds in snapshots.values():
            ds.close()
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise RuntimeError("Frozen source changed during extraction")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        sidecar = {"sources": hashes, "output_sha256": digest(path), "script_sha256": digest(Path(__file__)),
                   "semantics": SEMANTICS[name], "rows": len(frame), "manifest_verified": True,
                   "grid_metadata": grid_metadata, "synthetic_month": "2000-01 (metadata-only empty snapshots)",
                   "versions": {n: importlib.metadata.version(n) for n in ("xgboost", "pandas", "numpy", "xarray")},
                   "recommended_filter_columns": ["scope", "branch", "component", "metric"] if name == S6 else ["variable"],
                   "DIMENSIONS_additions_requested": [] if name == S6 else ["variable"]}
        path.with_suffix(".provenance.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
    return {name: len(frame) for name, frame in frames.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Frozen release worktree, not analysis cache")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(derive(args.source_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
