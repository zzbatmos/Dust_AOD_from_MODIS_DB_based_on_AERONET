"""Optional private-cache OOF regression: inference and metrics, never fitting.

Set DBDUST_VALIDATION_ROOT to the existing revision_v2_work directory. Only
pytest temporary directories are written; source artifacts and result tables
are read in place. Deployment models are deliberately not used for OOF rows.
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from dbdust.config import CONFIG
from dbdust.predictor import BranchModel, file_hash


REFERENCE = "experimental1020"
KEYS = ["row_id", "split_id"]
EXPERIMENTS = [("common", "full8"), ("own", "full8"), ("own", "noSSA5")]
EV_ESTIMATORS = ["E1", "E3c", "E4", "Eselected"]
CORE = ["E0", "E1_QA1", "E1", "E1_QA3", "E2", "E3", "E5", "E7",
        "E3c", "E4", "E8", "Eselected"]


def required(path):
    assert path.is_file(), f"Incomplete validation cache: {path}"
    return path


def compare(actual, expected, label, tolerance=1e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=label)
    np.testing.assert_allclose(a, b, rtol=0, atol=tolerance,
                               equal_nan=True, err_msg=label)


def ordered(frame):
    assert not frame.duplicated(KEYS).any(), "Duplicate OOF event keys"
    return frame.set_index(KEYS).sort_index()


def source_functions(path, names, namespace=None):
    """Load only pure recipes, not source-module imports, writers, or mains."""
    tree = ast.parse(required(path).read_text(), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {n.id for n in node.targets if isinstance(n, ast.Name)}
            if targets & set(names):
                assert targets <= set(names)
                selected.append(node)
                found.update(targets)
    assert found == set(names), f"Metric recipe changed: {path}: {found}"
    scope = {"np": np, "pd": pd, **(namespace or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.fixture(scope="module")
def validation_root():
    value = os.environ.get("DBDUST_VALIDATION_ROOT")
    if not value:
        pytest.skip("Set DBDUST_VALIDATION_ROOT to the existing private revision_v2_work cache")
    root = Path(value).expanduser().resolve()
    if not root.exists():
        pytest.skip(f"Private validation cache is absent: {root}")
    assert root.is_dir(), f"Validation root is not a directory: {root}"
    required(root / "grouped_validation/rows.parquet")
    return root


@pytest.fixture(scope="module")
def reproduced(validation_root, tmp_path_factory):
    grouped = validation_root / "grouped_validation"
    ev = validation_root / "expected_value_test"
    rows = pd.read_parquet(grouped / "rows.parquet").set_index("row_id", drop=False)
    assert rows.index.is_unique
    assert CONFIG["reference"]["name"] == REFERENCE
    scratch = tmp_path_factory.mktemp("dbdust-oof")
    frames = {}
    source_hashes = {}

    def tracked(path):
        required(path)
        source_hashes[path] = file_hash(path)
        return path

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Fitting is forbidden in OOF reproduction")

    with pytest.MonkeyPatch.context() as patch:
        for cls in (XGBClassifier, XGBRegressor, IsotonicRegression):
            patch.setattr(cls, "fit", forbid_fit)
        for population, branch in EXPERIMENTS:
            tag = f"{REFERENCE}__{population}__{branch}"
            path = ev / "predictions" / f"stage_b_{REFERENCE}_{branch}_{population}.parquet"
            expected = pd.read_parquet(tracked(path))
            old = pd.read_parquet(tracked(grouped / "oof" / f"{tag}.parquet"))
            pd.testing.assert_index_equal(ordered(expected).index, ordered(old).index)
            assert not expected.empty
            assert set(expected.reference) == {REFERENCE}
            assert set(expected.branch) == {branch}
            assert set(expected.scheme) == {"site_cv", "region_out"}
            actual = expected.copy()

            # The saved split IDs are authoritative, including every region-out
            # fold. The original pair and Stage B correction live separately.
            for split, q in expected.groupby("split_id", sort=False):
                original = grouped / "models" / tag / split
                correction = ev / "stage_b_support/runs" / tag / split
                meta = json.loads(tracked(original / "metadata.json").read_text())
                assert meta["reference"] == meta["training_target"] == REFERENCE
                assert meta["features_inference"] == CONFIG["branches"][branch]["features"]
                assert meta["features_train"] == meta["features_inference"]
                fold_saved = pd.read_parquet(tracked(correction / "predictions.parquet"))
                # Concatenating E4-selected float32 and E3c-selected float64
                # folds promotes Eselected in the combined prediction table.
                fold_saved["Eselected"] = fold_saved.Eselected.astype(q.Eselected.dtype)
                pd.testing.assert_frame_equal(ordered(q), ordered(fold_saved), check_exact=True)
                decision = json.loads(tracked(correction / "selection.json").read_text())
                assert decision["selected"] in ("E3c", "E4")
                assert q.selected_estimator.eq(decision["selected"]).all()

                folder = scratch / tag / split
                folder.mkdir(parents=True)
                pair = {key: meta.get(key) for key in
                        ("constant_probability", "constant_fraction")}
                (folder / "pair.json").write_text(json.dumps(pair))
                for filename, constant in [("classifier.ubj", "constant_probability"),
                                           ("positive_fraction.ubj", "constant_fraction")]:
                    src = original / filename
                    if src.is_file():
                        (folder / filename).symlink_to(tracked(src))
                    else:
                        assert pair[constant] is not None, f"Missing fold model and constant: {src}"
                for filename in ("direct_fraction.ubj", "postprocessor.npz"):
                    (folder / filename).symlink_to(tracked(correction / filename))
                # Do not copy a deployment smear curve: outer folds used the
                # exact residual integral, and their Eselected is fold-specific.
                model = BranchModel(folder, branch)
                inputs = rows.loc[q.row_id]
                assert not inputs.calibration_reserved.any()
                assert np.isfinite(inputs[meta["features_inference"]]).all().all()
                compare(inputs[REFERENCE], q.truth_fraction, str(path), 1e-12)
                compare(inputs[REFERENCE].to_numpy() * inputs.AER_AOD550.to_numpy(),
                        q.truth_daod, f"{tag}/{split}/truth_daod", 1e-12)
                values = model.predict(inputs)
                p, f = values["probability"], values["positive_fraction"]
                fneg = float(model.post["f_neg"])
                for field, gate in zip(("E1_QA1", "E1", "E1_QA3"), CONFIG["qa"]["thresholds"]):
                    values[field] = np.where(p >= gate, f, 0)
                values["E2"] = p * f
                values["E3_Aunsmeared"] = p * f + (1 - p) * fneg
                values["E3"] = p * values["positive_fraction_star"] + (1 - p) * fneg
                values["f_neg"] = np.full(len(q), fneg)
                values["Eselected"] = values[decision["selected"]]
                for field, value in values.items():
                    compare(value, q[field], f"{tag}/{split}/{field}")
                    # Source tables retain float32 raw P/f_pos/E4. Preserve
                    # their storage arithmetic when reconstructing metrics.
                    actual.loc[q.index, field] = np.asarray(value, dtype=expected[field].dtype)
                compare(values["E1"] * q.DB_AOD550.to_numpy(), q.daod_QA2,
                        f"{tag}/{split}/QA2_DAOD")

            aligned = ordered(actual)
            for field in ("probability", "positive_fraction"):
                compare(aligned[field], ordered(old)[field], f"{tag}/{field}")
            frames[f"{branch}_{population}"] = actual

        full, no_ssa = frames["full8_own"], frames["noSSA5_own"]
        keep = ~pd.MultiIndex.from_frame(no_ssa[KEYS]).isin(pd.MultiIndex.from_frame(full[KEYS]))
        hierarchy = pd.concat([full, no_ssa.loc[keep]], ignore_index=True)
        expected = pd.read_parquet(tracked(ev / "predictions" / f"stage_b_{REFERENCE}_hierarchy_own.parquet"))
        a, b = ordered(hierarchy), ordered(expected)
        pd.testing.assert_index_equal(a.index, b.index)
        pd.testing.assert_series_equal(a.branch, b.branch)
        for field in (*EV_ESTIMATORS, "probability", "positive_fraction",
                      "probability_calibrated", "positive_fraction_star"):
            compare(a[field], b[field], f"hierarchy/{field}")
        frames["hierarchy_own"] = hierarchy

    for path, digest in source_hashes.items():
        assert file_hash(path) == digest, f"Frozen input changed during reproduction: {path}"
    return frames


@pytest.mark.parametrize("population,branch", EXPERIMENTS)
def test_packaged_all_outer_folds(reproduced, population, branch):
    # The shared fixture compares every saved split and every event, not a
    # convenience sample or just site_r0_f0.
    assert not reproduced[f"{branch}_{population}"].empty


def test_packaged_hierarchy(reproduced):
    assert not reproduced["hierarchy_own"].duplicated(KEYS).any()


def compare_summary(actual, expected, label):
    for field, value in actual.items():
        assert field in expected.index, f"Missing saved metric {label}/{field}"
        if isinstance(value, (int, np.integer)):
            assert value == expected[field], f"Count mismatch: {label}/{field}"
        else:
            compare(value, expected[field], f"{label}/{field}", 1e-12)


@pytest.mark.parametrize("level", ["collocation", "site_month"])
def test_ev_source_metrics_and_site_bootstrap(validation_root, reproduced, level):
    ev = validation_root / "expected_value_test"
    common = source_functions(ev / "common.py", ["paired", "monthly"])
    metrics = source_functions(ev / "metrics.py", ["METRICS", "_from_sums",
        "metric_distribution", "summarize_distribution", "counts"])
    saved = pd.read_csv(required(ev / "stage_b" / f"{level}_metrics.csv"))
    saved = saved.loc[(saved.reference == REFERENCE) & saved.branch.isin(reproduced)
                      & saved.estimator.isin(EV_ESTIMATORS) & (saved.stratum == "all")]
    assert not saved.empty
    assert set(saved.branch) == {"full8_common", "noSSA5_own", "hierarchy_own"}
    assert set(saved.pairing) == {"core", "including_E6"}
    monthly = level == "site_month"
    cache = {}
    for _, record in saved.iterrows():
        key = (record.branch, record.scheme, record.pairing, record.minimum, record.repeat_summary)
        if key not in cache:
            d = reproduced[record.branch]
            d = d.loc[d.scheme == record.scheme]
            es = CORE + (["E6"] if record.pairing == "including_E6" else [])
            q = (common["monthly"](d, es, int(record.minimum)) if monthly
                 else common["paired"](d, es))
            if record.repeat_summary != "mean_over_repeats":
                assert record.repeat_summary.startswith("repeat_")
                q = q.loc[q.repeat == int(record.repeat_summary.removeprefix("repeat_"))]
            assert not q.empty, f"Empty reconstructed metric population: {key}"
            cache[key] = q
        q = cache[key]
        distribution = metrics["metric_distribution"](q, record.estimator, monthly)
        summary = metrics["summarize_distribution"](distribution)
        summary.update(metrics["counts"](q, monthly))
        compare_summary(summary, record, f"{level}/{key}/{record.estimator}")


@pytest.mark.parametrize("table,recipe", [("metrics.csv", "statistics"),
                                         ("site_bootstrap_95CI.csv", "bootstrap")])
def test_grouped_source_metrics(validation_root, reproduced, table, recipe):
    grouped = validation_root / "grouped_validation"
    base = source_functions(grouped / "run.py", ["BASELINES"])["BASELINES"]
    functions = source_functions(grouped / "summarize.py", ["predictions", recipe],
                                 {"BASELINES": base, "roc_auc_score": roc_auc_score})
    saved = pd.read_csv(required(grouped / table))
    selected = saved.loc[(saved.reference == REFERENCE) & (saved.scope == "overall")]
    checked = 0
    for key, rebuilt in reproduced.items():
        branch, population = key.rsplit("_", 1)
        model = "hierarchy_prespecified" if branch == "hierarchy" else branch
        path = grouped / "oof" / f"{REFERENCE}__{population}__{model}.parquet"
        original = pd.read_parquet(required(path))
        # In particular, the legacy float32 Brier mean depends on reduction
        # order. Align predictions to the parquet order, not a sorted order.
        old = original.set_index(KEYS)
        new = ordered(rebuilt)
        pd.testing.assert_index_equal(new.index, ordered(original).index)
        new = new.reindex(old.index)
        for field in ("probability", "positive_fraction"):
            compare(new[field], old[field], f"{path}/{field}")
            old[field] = new[field].to_numpy(dtype=old[field].dtype)
        old = old.reset_index()
        records = selected.loc[(selected.population == population) & (selected.model == model)]
        assert not records.empty, f"No saved grouped metrics: {key}"
        assert set(records.scheme) == set(old.scheme)
        assert set(records.QA) == set(CONFIG["qa"]["thresholds"])
        for _, record in records.iterrows():
            cached = original.loc[original.scheme == record.scheme]
            cached_result = functions[recipe](cached, record.QA)
            assert cached_result, f"Empty original-only metric result: {key}"
            compare_summary(cached_result, record,
                            f"cached-original/{table}/{key}/{record.scheme}/{record.QA}")
            d = old.loc[old.scheme == record.scheme]
            result = functions[recipe](d, record.QA)
            assert result, f"Empty grouped metric result: {key}"
            compare_summary(result, record, f"{table}/{key}/{record.scheme}/{record.QA}")
            checked += 1
    assert checked > 0
