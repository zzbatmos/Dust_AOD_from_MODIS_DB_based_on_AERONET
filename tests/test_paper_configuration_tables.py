"""Synthetic source/registry checks; no fits, predictions or unpublished fixtures."""
import copy
import json
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from paper import configuration_tables as tables
from paper import derive_configuration_tables as derive
from paper.numbers import Registry, digest


VARIABLES = ["daod550_ev", "daod550_qa2", "aod550_db", "n_valid_aod", "n_ml", "n_full8", "n_noSSA5",
             "frac_full8", "frac_ml_missing", "dust_probability_mean", "frac_qa1", "frac_qa2", "frac_qa3",
             "ev_contribution_aod_lt_0p2", "ev_contribution_p_lt_0p3"]


def synthetic():
    config = {"reference": {"name": "synthetic_reference"},
              "inputs": {"minimum_aod": 0, "ssa_bounds": [0, 1]},
              "branches": {"full8": {"code": 1, "features": ["DB_AOD550", "DB_AE", "DB_SSA412"], "selected": "E3c"},
                           "noSSA5": {"code": 2, "features": ["DB_AOD550", "DB_AE"], "selected": "E4"}},
              "qa": {"thresholds": [.4, .6, .7], "product_gate_index": 1},
              "diagnostics": {"low_aod_threshold": .2, "low_probability_threshold": .3},
              "outputs": {"monthly": VARIABLES, "units": "1", "missing_float": "NaN"}}
    evidence = {key: {"source_file": f"/not-opened/{key}", "source_sha256": "a" * 64}
                for key in ("config", "docs", "branches", "predictor", "writer_l3", "writer_l2")}
    metadata, pairs, models, arrays = {}, {}, {}, {}
    for branch, spec in config["branches"].items():
        metadata[branch] = {"feature_names": spec["features"].copy(), "selected": spec["selected"],
                            "reference": "synthetic_reference", "config": {"n_estimators": 3, "max_depth": 2},
                            "training_rows": 5}
        pairs[branch] = {"constant_probability": None, "constant_fraction": None}
        models[branch] = {role: {"learner_model_param": {"num_feature": str(len(spec["features"]))},
                                 "feature_names": [], "feature_types": [],
                                 "objective": {"name": "binary:logistic" if role == "classifier" else "reg:squarederror"},
                                 "gbtree_model_param": {"num_trees": "3"}}
                          for role in ("classifier", "positive_fraction", "direct_fraction")}
        arrays[branch] = {"postprocessor": {"f_neg": .03125, "residuals": {"shape": [2], "dtype": "float64"}}}
        for artifact in ("metadata", "pair", "classifier", "positive_fraction", "direct_fraction", "postprocessor"):
            evidence[branch + "/" + artifact] = {"source_file": f"/not-opened/{branch}/{artifact}", "source_sha256": "b" * 64}
    attrs = {key: "synthetic saved " + key for key in
             ("branch_rule", "screening", "missing_conventions", "weighting", "aggregation")}
    snapshots = {}
    for grid, width in (("1x1", 2), ("2x5", 3)):
        ds = xr.Dataset(attrs=attrs)
        for name in VARIABLES:
            count = name.startswith("n_")
            array = np.zeros((1, 1, width), dtype="int32") if count else np.full((1, 1, width), np.nan)
            metadata_attrs = {"units": "1", "long_name": name.replace("_", " ")}
            if not count:
                metadata_attrs.update(denominator="n_valid_aod" if name == "frac_ml_missing" else "n_ml",
                                      cell_methods="time: mean (weighted by retrieval count)")
            if name.startswith("frac_qa"):
                metadata_attrs["probability_threshold"] = config["qa"]["thresholds"][int(name[-1])-1]
            if name.startswith("ev_contribution"):
                metadata_attrs["subset_threshold"] = .2 if "aod_lt" in name else .3
            ds[name] = (("time", "lat", "lon"), array, metadata_attrs)
            ds[name].encoding.update(_FillValue=None if count else np.nan, zlib=True)
        snapshots[grid] = ds
    document = "## Monthly product (L3M)\n| Variable | Definition |\n| --- | --- |\n"
    document += "\n".join(f"| `{v}` | Saved definition of `{v}` with a & b. |" for v in VARIABLES)
    document += "\n### Coordinates, calendars, and metadata\n"
    return config, metadata, pairs, models, arrays, evidence, attrs, snapshots, document


def frames_and_registry():
    config, metadata, pairs, models, arrays, evidence, attrs, snapshots, document = synthetic()
    frames = {derive.S6: derive.build_settings(config, metadata, pairs, models, arrays, evidence, attrs),
              derive.S7: derive.build_monthly(config, document, snapshots, evidence)}
    sources = {v["source_file"]: v["source_sha256"] for v in evidence.values()}
    grid_metadata = {g: {"sizes": dict(ds.sizes), "attrs": dict(ds.attrs)} for g, ds in snapshots.items()}
    payload = {"tables": {}}
    for name, frame in frames.items():
        sidecar = {"output_sha256": "c" * 64, "sources": sources, "rows": len(frame),
                   "grid_metadata": grid_metadata, "semantics": derive.SEMANTICS[name],
                   "recommended_filter_columns": ["scope", "branch", "component", "metric"] if name == derive.S6 else ["variable"]}
        payload["tables"]["derived/" + name] = {"columns": list(frame), "records": frame.to_dict("records"),
                                                "source_file": f"/not-opened/{name}.csv", "source_sha256": "c" * 64,
                                                "source_rows": len(frame), "derivation": sidecar,
                                                "semantics": derive.SEMANTICS[name]}
    return frames, Registry(payload)


def test_s6_feature_order_raw_qa_forms_and_missing_parameters_are_explicit():
    frames, _ = frames_and_registry()
    frame = frames[derive.S6]
    ordered = frame[(frame.scope == "feature_order") & (frame.branch == "full8")]
    assert ordered.value_text.tolist() == ["DB_AOD550", "DB_AE", "DB_SSA412"]
    assert ordered.metric.tolist() == ["feature_01", "feature_02", "feature_03"]
    raw = frame[frame.metric == "qa.thresholds"].iloc[0]
    assert json.loads(raw.value_json)["value"] == [.4, .6, .7]
    missing = frame[frame.scope == "missing_metadata"]
    assert set(missing.metric) == set(derive.UNRECORDED)
    assert set(missing.value_text) == {"NOT RECORDED"}
    assert set(missing.availability) == {"missing_metadata"}
    assert all(json.loads(v)["value"] is None for v in missing.value_json)
    assert "not calibrated" in frame[frame.metric == "raw_probability"].iloc[0].value_text
    assert "(1-P_cal)*f_neg" in frame[frame.metric == "E3c"].iloc[0].value_text
    assert "clip(direct_fraction" in frame[frame.metric == "E4"].iloc[0].value_text
    assert not frame.duplicated(["scope", "branch", "component", "metric"]).any()
    serialized = frame[frame.metric == "gbtree_model_param.num_trees"]
    assert all(json.loads(e)[0]["locator"] == "learner/gradient_booster/model/gbtree_model_param/num_trees"
               for e in serialized.evidence_json)


@pytest.mark.parametrize("case", ["metadata_order", "model_order", "feature_count"])
def test_configuration_disagreements_fail_closed(case):
    config, metadata, pairs, models, arrays, evidence, attrs, _, _ = synthetic()
    if case == "metadata_order":
        metadata["full8"]["feature_names"].reverse()
    elif case == "model_order":
        models["full8"]["classifier"]["feature_names"] = ["wrong"]
    else:
        models["full8"]["classifier"]["learner_model_param"]["num_feature"] = "99"
    with pytest.raises(ValueError, match="mismatch"):
        derive.build_settings(config, metadata, pairs, models, arrays, evidence, attrs)


def test_model_inspection_never_calls_training_or_runtime_config(monkeypatch):
    calls = []
    saved = {"learner": {"learner_model_param": {"num_feature": "2"}, "objective": {"name": "reg:squarederror"},
                          "feature_names": [], "feature_types": [], "attributes": {},
                          "gradient_booster": {"name": "gbtree", "model": {"gbtree_model_param": {"num_trees": "3"}}}}}
    class ReadOnlyBooster:
        def load_model(self, path):
            calls.append(("load_model", path))
        def save_raw(self, *, raw_format):
            assert raw_format == "json"
            calls.append(("save_raw", raw_format))
            return json.dumps(saved).encode()
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(Booster=ReadOnlyBooster))
    result = derive.read_model(Path("fake.ubj"))
    assert result["gbtree_model_param"]["num_trees"] == "3"
    assert "max_depth" not in result and "reg_lambda" not in result
    assert [c[0] for c in calls] == ["load_model", "save_raw"]


def test_s7_all_fifteen_fields_exact_attrs_denominators_and_missing_vs_zero():
    frames, _ = frames_and_registry()
    frame = frames[derive.S7].set_index("variable")
    assert frame.index.tolist() == VARIABLES and len(frame) == 15
    assert (frame.loc[[v for v in VARIABLES if v.startswith("n_")], "dtype"] == "int32").all()
    assert (frame.loc[[v for v in VARIABLES if not v.startswith("n_")], "dtype"] == "float64").all()
    assert frame.loc["frac_ml_missing", "denominator"] == "n_valid_aod"
    for v in ("ev_contribution_aod_lt_0p2", "ev_contribution_p_lt_0p3"):
        assert frame.loc[v, "denominator"] == "n_ml"
        assert "not subset count" in frame.loc[v, "missing_vs_zero"]
    assert "Valid-AOD-only cell: 1" in frame.loc["frac_ml_missing", "missing_vs_zero"]
    assert "rejections contribute zeros" in frame.loc["daod550_qa2", "missing_vs_zero"]
    assert json.loads(frame.loc["n_ml", "encoding_json"])["_FillValue"] is None
    assert json.loads(frame.loc["daod550_ev", "encoding_json"])["_FillValue"] == "nan"
    assert json.loads(frame.loc["frac_qa2", "attrs_json"])["probability_threshold"] == .6
    assert frame.loc["n_ml", "definition"] == "Saved definition of `n_ml` with a & b."


@pytest.mark.parametrize("case", ["denominator", "dtype", "fill", "threshold", "docs", "grids"])
def test_monthly_schema_mismatches_fail_closed(case):
    config, _, _, _, _, evidence, _, snapshots, document = synthetic()
    ds = snapshots["1x1"]
    if case == "denominator":
        ds["ev_contribution_p_lt_0p3"].attrs["denominator"] = "subset_count"
    elif case == "dtype":
        ds["n_ml"] = ds.n_ml.astype("int64")
    elif case == "fill":
        ds.n_ml.encoding["_FillValue"] = -1
    elif case == "threshold":
        ds.frac_qa2.attrs["probability_threshold"] = .5
    elif case == "docs":
        document = document.replace("| `n_ml` |", "| `unlisted` |")
    else:
        snapshots["2x5"].daod550_ev.attrs["long_name"] = "different"
    with pytest.raises(ValueError):
        derive.build_monthly(config, document, snapshots, evidence)


def test_registry_only_export_preserves_every_csv_value_and_provenance(tmp_path, monkeypatch):
    frames, registry = frames_and_registry()
    source = tmp_path / "numbers" / "registry.json"
    source.parent.mkdir()
    source.write_text(json.dumps(registry.payload))
    before = digest(source)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("External source CSV opened"))
    output = tmp_path / "output"
    result = tables.export(source, output)
    assert digest(source) == before == result["registry_sha256"]
    for index, name in ((6, derive.S6), (7, derive.S7)):
        assert (output / f"TableS{index:02d}.csv").read_text() == frames[name].to_csv(index=False)
        provenance = json.loads((output / f"TableS{index:02d}_provenance.json").read_text())
        assert provenance["source"]["derivation"] == registry.payload["tables"]["derived/" + name]["derivation"]
        assert provenance["font_pt"] >= 7
        assert provenance["rows"] == len(frames[name])
        assert all(digest(output / p) == sha for p, sha in provenance["outputs"].items())
    assert "noSSA5 does not require SSA" in (output / "TableS06.tex").read_text()
    text = (output / "TableS07.tex").read_text()
    assert "float64" in text and "int32" in text and r"\&" in text
    assert "not subset counts" in text
    assert tables.escape("All configured science/count variables") in text


def test_private_preview_registry_is_explicit_and_does_not_overwrite(tmp_path):
    frames, registry = frames_and_registry()
    derived = tmp_path / "derived"
    derived.mkdir()
    for name, frame in frames.items():
        path = derived / f"{name}.csv"
        frame.to_csv(path, index=False)
        sidecar = copy.deepcopy(registry.payload["tables"]["derived/" + name]["derivation"])
        sidecar["output_sha256"] = digest(path)
        path.with_suffix(".provenance.json").write_text(json.dumps(sidecar))
    path = tmp_path / "preview" / "registry.json"
    tables.temporary_registry(derived, path)
    saved = Registry.read(path)
    assert "not canonical" in saved.payload["scope"]
    assert saved.payload["tables"]["derived/" + derive.S7]["filter_columns"] == ["variable"]
    for index in (6, 7):
        tables.load_table(saved, index)
    reloaded, _ = tables.load_table(saved, 6)
    assert reloaded.loc[reloaded.metric.eq("outputs.missing_float"), "value_text"].item() == '"NaN"'
    assert reloaded.to_csv(index=False) == frames[derive.S6].to_csv(index=False)
    with pytest.raises(ValueError, match="overwrite"):
        tables.temporary_registry(derived, path)


def test_tampered_registry_evidence_fails(tmp_path):
    _, registry = frames_and_registry()
    registry.payload["tables"][tables.TABLES[6]]["derivation"]["sources"] = {}
    with pytest.raises(ValueError, match="evidence mismatch"):
        tables.load_table(registry, 6)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex absent")
def test_readable_synthetic_tables_compile(tmp_path):
    _, registry = frames_and_registry()
    path = tmp_path / "numbers" / "registry.json"
    path.parent.mkdir()
    path.write_text(json.dumps(registry.payload))
    output = tmp_path / "output"
    result = tables.export(path, output, compile_latex=True)
    for table in result["tables"].values():
        assert table["latex"]["status"] == "compiled"
        assert (output / table["latex"]["pdf"]).exists()


def test_private_guards(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        derive.guard_output(Path(derive.__file__).parent / "output")
    root = tmp_path / "source"
    (root / "01_instructions").mkdir(parents=True)
    with pytest.raises(ValueError, match="author"):
        derive.guard_output(root / "output")
