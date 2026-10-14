"""Synthetic export checks: no unpublished quantitative fixtures or source reads."""
import copy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from paper import supplement_tables as si
from paper.numbers import COUNTS, DIMENSIONS, Registry, digest


def synthetic_registry():
    tables = {}

    def add(name, rows):
        columns = list(dict.fromkeys(k for row in rows for k in row))
        tables[name] = {
            "source_file": f"/does-not-exist/{name}.csv", "source_sha256": "a" * 64,
            "columns": columns, "source_rows": len(rows),
            "filter_columns": [c for c in columns if c in DIMENSIONS],
            "count_columns": [c for c in columns if c in COUNTS or c.startswith("n_")],
            "semantics": "Synthetic saved source, never open its source_file",
            "records": rows,
        }

    grouped, intervals, decomposition, contrasts = [], [], [], []
    components, decisions, seasons = [], [], []
    for reference in ("experimental1020", "old675LR56"):
        for scheme in ("site_cv", "region_out"):
            for qa in (.4, .6, .7):
                for scope, group in (("overall", "all"), ("region", "a & b_%#{}~^\\test"),
                                     ("AOD_bin", ">=0.8"), ("platform", "Aqua"), ("fold", "f0")):
                    key = dict(reference=reference, population="own", model="full8",
                               scheme=scheme, QA=qa, scope=scope, group=group)
                    grouped.append({**key, "n": 8, "unique_rows": 4, "sites": 2,
                                    "TP": 1, "DAOD_RMSE": .1234567891234567,
                                    "precision": .25, "AUC": None, "custom_saved_value": .375,
                                    "recall": .5, "FPR": .125, "F1": .375, "Brier": .125,
                                    "DAOD_bias": -.0625, "DAOD_MAE": .125,
                                    "DAOD_R2": .5, "EE_descriptive": .75})
                    if scope == "overall":
                        intervals.append({**key, "bootstrap_unit": "site", "replicates": 9,
                                          "DAOD_RMSE_lo": .0625, "DAOD_RMSE_hi": .25})
            key = dict(reference=reference, population="decomposition", model="DB_size500proxy",
                       scheme=scheme, QA=.6, scope="overall", group="all")
            decomposition.append({**key, "n": 8, "unique_rows": 4, "sites": 2, "DAOD_RMSE": .125})
            intervals.append({**key, "bootstrap_unit": "site", "replicates": 9,
                              "DAOD_RMSE_lo": .0625, "DAOD_RMSE_hi": .25})
            for qa in (.4, .6, .7):
                contrasts.append(dict(reference=reference, population="decomposition",
                                      A="DB_size500proxy", B="AER_size500proxy", scheme=scheme,
                                      QA=qa, n_prediction_events=8, unique_rows=4, sites=2,
                                      delta_RMSE_A_minus_B=-.125, delta_RMSE_lo=-.25,
                                      delta_RMSE_hi=-.0625, direction="negative favors A"))
            for branch in ("hierarchy_own", "noSSA5_own"):
                key = dict(reference=reference, branch=branch, scheme=scheme)
                decisions.append({**key, "H1": "pass", "H2": "not_established",
                                  "H2_RMSE": "within_margin", "H2_conditional_bias": "pass",
                                  "H3": "pass", "H4": "not_established"})
                for hypothesis in ("H1", "H2_RMSE", "H2_conditional_bias", "H3", "E8_recalibration_contrast"):
                    components.append({**key, "stage": "b", "hypothesis": hypothesis,
                                       "candidate": "Eselected", "comparator": "E1",
                                       "metric": "rmse", "level": "site_month", "minimum": 3,
                                       "margin": .03125, "mean_ref": .5, "difference": -.125,
                                       "ci_low": -.25, "ci_high": -.0625, "decision": "unchanged_saved_status",
                                       "prediction_events": 8, "unique_rows": 12, "sites": 2,
                                       "repeats": 2, "unique_site_months": 4, "valid_bootstraps": 9})
        for variant in ("hierarchy", "with_minimal_fallback"):
            for support in ("core_common", "full_common_E6"):
                for mask in ("prior_joint", "mean_only_reference_sensitivity"):
                    for region in ("Land 60S-60N", "Sahara", "Siberia_50_75N_sensitivity"):
                        for period in ("DJF", "MAM", "JJA", "SON"):
                            for estimator in (["E1", "Song", "CALIOP"] if support == "core_common"
                                              else ["E1", "Song", "CALIOP", "E6"]):
                                seasons.append(dict(reference=reference, variant=variant, support=support,
                                                    reference_mask=mask, period=period, region=region,
                                                    estimator=estimator, n=4, candidate_mean=.125,
                                                    reference_mean=.25, bias=-.125, rmse=.25, mae=.125, r=.5))
    for rows in (grouped, intervals):
        own = [row for row in rows if row["population"] == "own"]
        for population, model in (("own", "noSSA5"), ("common", "full8"), ("common", "noSSA5")):
            rows.extend({**row, "population": population, "model": model} for row in own)
    # A valid non-all overall group must remain CSV-only, as must decomposition.
    grouped.append({**grouped[0], "group": "not_all"})
    grouped.extend({**grouped[0], **row} for row in decomposition)
    add(si.METRICS, grouped)
    add(si.INTERVALS, intervals)
    add(si.DECOMPOSITION, decomposition)
    add(si.CONTRASTS, contrasts)
    add(si.COMPONENTS, components)
    add(si.DECISIONS, decisions)
    add(si.SEASONAL, seasons)
    return Registry({"schema_version": 1, "tables": tables})


def test_s2_preserves_all_scopes_dimensions_columns_values_and_source_counts():
    registry = synthetic_registry()
    before = copy.deepcopy(registry.payload)
    parts = si.build_tables(registry)[2]
    for part, name in zip(parts, (si.METRICS, si.INTERVALS)):
        pd.testing.assert_frame_equal(part.frame.drop(columns=si.ROW), registry.table(name), check_exact=True)
        assert part.frame[si.ROW].tolist() == list(range(len(part.frame)))
        assert part.sources[0]["selection"] == {}
        for evidence in part.sources[0]["rows"]:
            original = registry.table(name).iloc[evidence[si.ROW]]
            assert evidence["filter"] == {k: original[k] for k in part.sources[0]["filter_columns"]}
    assert set(parts[0].frame.scope) == {"overall", "region", "AOD_bin", "platform", "fold"}
    assert "custom_saved_value" in parts[0].frame
    assert "AUC_lo" not in parts[1].frame
    assert registry.payload == before


def test_s3_keeps_saved_sign_qa_grid_and_exact_interval_key_intersection():
    registry = synthetic_registry()
    points, intervals, contrasts = si.build_tables(registry)[3]
    assert set(points.frame.QA) == {.6}
    assert set(contrasts.frame.QA) == {.4, .6, .7}
    assert (contrasts.frame.delta_RMSE_A_minus_B == -.125).all()
    assert (contrasts.frame.delta_RMSE_lo == -.25).all()
    assert set(map(tuple, intervals.frame[si.KEYS].to_numpy())) == set(map(tuple, points.frame[si.KEYS].to_numpy()))
    assert intervals.sources[0]["key_intersection"]


def test_s4_decisions_are_literal_and_aggregate_margins_remain_missing():
    components, decisions, margins = si.build_tables(synthetic_registry())[4]
    assert "E8_recalibration_contrast" not in set(components.frame.hypothesis)
    assert (components.frame.decision == "unchanged_saved_status").all()
    assert (decisions.frame.H2_RMSE == "within_margin").all()
    assert (margins.frame.loc[margins.frame.hypothesis.eq("H2_RMSE"), "decision"] == "within_margin").all()
    missing = margins.frame[margins.frame.hypothesis.isin(["H2", "H4"])]
    assert missing.margin.isna().all()
    assert missing.component_source_record_index.isna().all()
    assert (missing.margin_status == "missing_no_explicit_scalar_margin").all()
    explicit = margins.frame[~margins.frame.hypothesis.isin(["H2", "H4"])]
    assert (explicit.margin == .03125).all()
    assert not margins.frame.duplicated([si.ROW, "hypothesis"]).any()
    assert {"prediction_events", "unique_rows", "unique_site_months", "minimum", "repeats"}.issubset(components.frame)


def test_s4_missing_component_margin_column_is_reported_not_inferred():
    registry = synthetic_registry()
    table = registry.payload["tables"][si.COMPONENTS]
    table["columns"].remove("margin")
    for row in table["records"]:
        del row["margin"]
    margins = si.build_tables(registry)[4][-1].frame
    assert margins.margin.isna().all()
    assert (margins.margin_status == "missing_no_explicit_scalar_margin").all()


def test_s5_preserves_both_references_variants_regions_masks_and_supports():
    registry = synthetic_registry()
    part = si.build_tables(registry)[5][0]
    pd.testing.assert_frame_equal(part.frame.drop(columns=si.ROW), registry.table(si.SEASONAL), check_exact=True)
    for field in ("reference", "variant", "support", "reference_mask"):
        assert part.frame[field].nunique() == 2
    assert "Siberia_50_75N_sensitivity" in part.frame.region.values
    assert not any(c.endswith(("_lo", "_hi")) for c in part.frame)


def test_ambiguous_dimensions_fail_closed():
    registry = synthetic_registry()
    table = registry.payload["tables"][si.METRICS]
    table["records"].append(copy.deepcopy(table["records"][0]))
    with pytest.raises(ValueError, match="ambiguous source dimensions"):
        si.build_tables(registry)


def test_longtables_escape_saved_strings_and_include_counts_definitions_and_notes():
    tables = si.build_tables(synthetic_registry())
    text = si.render_table(2, tables[2])
    assert r"\begin{longtable}" in text and r"\endfirsthead" in text
    escaped = si.tex("a & b_%#{}~^\\test")
    assert r"\&" in escaped and r"\%" in escaped and r"\#" in escaped
    assert r"\textbackslash{}" in escaped and r"\textasciicircum{}" in escaped
    assert "Categorical codebook" not in text and "column definitions" in text
    assert "FULL source scope" in text and "TableS02" in text
    assert "reference=experimental1020; population=own; scheme=site" in text
    assert r"\caption[]{reference=experimental1020" in text and "(continued)" in text
    assert r"\fontsize{7}{8}" in text
    assert "not zero" in text
    assert si.cell(None, "AUC", {}) == "--"
    assert si.cell(0.0, "AUC", {}) == "0.00"
    assert si.cell(8.0, "n", {}) == "8"
    assert si.cell(.03125, "margin", {}) == "0.031250"
    text4 = si.render_table(4, tables[4])
    assert "margin" in text4 and "hypothesis" in text4


def test_s2_reader_selects_exact_scope_and_joins_every_key_without_borrowing_ci():
    registry = synthetic_registry()
    intervals = registry.payload["tables"][si.INTERVALS]["records"]
    # Unique bounds make any cross-reference/population/model/scheme/QA join visible.
    for i, row in enumerate(intervals):
        row["DAOD_RMSE_lo"] = i / 128
        row["DAOD_RMSE_hi"] = (i + 1) / 128
    removed = intervals.pop(0)
    intervals.reverse()
    parts = si.build_tables(registry)[2]
    original = parts[0].frame.copy(deep=True)
    name, view, selection = si.typeset_views(2, parts)[0]
    assert selection == {"scope": ["overall"], "group": ["all"], "population": ["own", "common"]}
    expected = original[original.scope.eq("overall") & original.group.eq("all")
                        & original.population.isin(["own", "common"])]
    pd.testing.assert_frame_equal(view[original.columns], expected.reset_index(drop=True), check_exact=True)
    assert set(view.population) == {"own", "common"}
    assert set(view.model) == {"full8", "noSSA5"}
    assert set(view.QA) == {.4, .6, .7}
    assert set(view.scheme) == {"site_cv", "region_out"}
    assert set(view.reference) == {"experimental1020", "old675LR56"}
    for _, row in view.iterrows():
        matches = [saved for saved in intervals if all(saved[k] == row[k] for k in si.KEYS)]
        if matches:
            assert row.DAOD_RMSE_lo == matches[0]["DAOD_RMSE_lo"]
            assert row.DAOD_RMSE_hi == matches[0]["DAOD_RMSE_hi"]
        else:
            assert all(row[k] == removed[k] for k in si.KEYS)
            assert pd.isna(row[si.CI_ROW]) and pd.isna(row.DAOD_RMSE_lo)
            assert "[" not in si._point_interval(row, "DAOD_RMSE")
        assert "[" not in si._point_interval(row, "AUC")
    pd.testing.assert_frame_equal(original, parts[0].frame, check_exact=True)
    evidence = si.view_provenance(2, parts)[0]
    assert evidence["interval_join_keys"] == si.KEYS
    assert evidence["matched_interval_rows"] == len(view) - 1
    assert evidence["source_record_indices"] == expected[si.ROW].tolist()
    assert evidence["font_pt"] >= 6.5


def test_s5_reader_has_all_primary_methods_regions_seasons_and_separate_e6_support():
    parts = si.build_tables(synthetic_registry())[5]
    original = parts[0].frame.copy(deep=True)
    core, e6 = si.typeset_views(5, parts)
    for name, view, selection in (core, e6):
        expected = original
        for field, values in selection.items():
            expected = expected[expected[field].isin(values)]
        pd.testing.assert_frame_equal(view, expected, check_exact=True)
        assert set(view.reference) == set(original.reference)
        assert set(view.region) == set(original.region)
        assert set(view.period) == {"DJF", "MAM", "JJA", "SON"}
        assert set(view.variant) == {"hierarchy"}
        assert set(view.reference_mask) == {"prior_joint"}
    assert set(core[1].estimator) == {"E1", "Song", "CALIOP"}
    assert set(core[1].support) == {"core_common"}
    assert set(e6[1].estimator) == {"E6"}
    assert set(e6[1].support) == {"full_common_E6"}
    assert all(v["displayed_numeric_fields"] == ["n", "candidate_mean", "reference_mean", "bias", "rmse", "mae", "r"]
               for v in si.view_provenance(5, parts))
    rendered = si.render_table(5, parts)
    assert "FULL source scope" in rendered and "TableS05" in rendered
    assert "NOT paired method contrasts" in rendered
    assert r"\caption[]{reference=experimental1020" in rendered and "(continued)" in rendered
    for field in si.S5_VALUES:
        assert si.tex(field) in rendered
    # Each selected seasonal row is rendered with all seven values, without panel duplication.
    numeric_tail = " & ".join(si.cell(original.iloc[0][c], c, {}) for c in si.S5_VALUES)
    assert rendered.count(numeric_tail) == len(core[1]) + len(e6[1])
    pd.testing.assert_frame_equal(original, parts[0].frame, check_exact=True)


def test_exports_registry_only_private_csv_and_provenance_without_mutation(tmp_path):
    registry = synthetic_registry()
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry.payload))
    before = digest(path)
    output = tmp_path / "candidate"
    manifest = si.export_tables(path, output)
    assert digest(path) == before == manifest["registry_sha256"]
    for index in (2, 3, 4, 5):
        provenance = json.loads((output / f"TableS{index:02d}_provenance.json").read_text())
        assert provenance["registry_sha256"] == before
        assert provenance["script_sha256"] == digest(Path(si.__file__))
        assert provenance["latex"]["status"] == "not_run"
        for name, sha in provenance["outputs"].items():
            assert digest(output / name) == sha
        for part in provenance["parts"]:
            assert set(part["columns"]) == set(part["column_definitions"])
            assert all(src["source_sha256"] == "a" * 64 for src in part["sources"])
    points = pd.read_csv(output / "TableS02_metrics.csv", float_precision="round_trip")
    expected = registry.table(si.METRICS)
    # CSV has no nullable-object dtype; an entirely absent numeric metric reads
    # as floating NaN. Nonmissing numeric values must still round-trip exactly.
    expected["AUC"] = float("nan")
    pd.testing.assert_frame_equal(points.drop(columns=si.ROW), expected, check_exact=True)
    missing = json.loads((output / "TableS04_provenance.json").read_text())["missing_explicit_margins"]
    assert {row["hypothesis"] for row in missing} == {"H2", "H4"}
    assert all(row["margin"] is None for row in missing)
    for index, parts in si.build_tables(registry).items():
        for part in parts:
            # Reader views must not change even a byte of the full CSV serialization.
            assert (output / f"TableS{index:02d}_{part.name}.csv").read_text() == part.frame.to_csv(index=False)
    for index in (2, 5):
        evidence = json.loads((output / f"TableS{index:02d}_provenance.json").read_text())
        assert evidence["typeset_views"] == si.view_provenance(index, si.build_tables(registry)[index])
        assert "FULL source scope" in (output / f"TableS{index:02d}_notes.md").read_text()


def test_rejects_public_and_author_folder_outputs(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(synthetic_registry().payload))
    with pytest.raises(ValueError, match="public worktree"):
        si.export_tables(path, Path(si.__file__).resolve().parent / "no_export")
    source = tmp_path / "source"
    (source / "01_instructions").mkdir(parents=True)
    with pytest.raises(ValueError, match="author source folders"):
        si.export_tables(path, source / "04_figures" / "tables")


def test_absent_decomposition_intervals_stay_empty_not_synthesized():
    registry = synthetic_registry()
    table = registry.payload["tables"][si.INTERVALS]
    table["records"] = [row for row in table["records"] if row["population"] != "decomposition"]
    intervals = si.build_tables(registry)[3][1]
    assert intervals.frame.empty
    assert intervals.sources[0]["rows"] == []


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_synthetic_latex_compiles_for_every_table(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(synthetic_registry().payload))
    output = tmp_path / "compiled"
    manifest = si.export_tables(path, output, compile_latex=True)
    assert all(row["latex"]["status"] == "compiled" for row in manifest["tables"].values())
    assert all((output / row["latex"]["pdf"]).is_file() for row in manifest["tables"].values())
