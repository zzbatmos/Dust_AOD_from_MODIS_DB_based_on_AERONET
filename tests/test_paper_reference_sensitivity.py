"""Synthetic paired-cohort and figure tests; no unpublished numerical fixtures."""
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

from paper import derive_reference_display as derive
from paper.numbers import Registry, digest


@pytest.fixture
def figure():
    pytest.importorskip("cmcrameri")
    pytest.importorskip("colorspacious")
    from paper.figures import figS03_reference_sensitivity
    return figS03_reference_sensitivity


def synthetic():
    rows = pd.DataFrame({
        "row_id": ["a", "reserved", "b", "c", "nonfinite", "d"],
        "site_key": ["s1", "s0", "s1", "s2", "s0", "s3"],
        "calibration_reserved": [False, True, False, False, False, False],
        "old675LR56": [.3, .9, .1, .2, np.nan, 0.],
        "experimental1020": [.1, .8, .2, .2, .3, 0.],
        "region": ["region_A", "unused", "region_A", "region_B", "unused", "region_B"],
        "platform": ["Terra", "Terra", "Aqua", "Terra", "Aqua", "Aqua"],
        "AOD_bin": ["0.2-0.4"] * 6,
        "experimental1020_native_raw": [1.1, .8, .2, .2, .3, 0.],
        "experimental1020_final_raw": [1.2, .8, .2, .2, .3, 0.],
    })
    pairs = rows.iloc[[0, 2, 3, 5]]

    def saved_summary(frame):
        # Independent fixture arithmetic, not the production summary helper.
        delta = frame.experimental1020.to_numpy() - frame.old675LR56.to_numpy()
        old, new = frame.old675LR56.to_numpy() >= .2, frame.experimental1020.to_numpy() >= .2
        return dict(rows=len(frame), label_changed=int(np.count_nonzero(old != new)),
                    old_to_new_positive=int(np.count_nonzero(~old & new)),
                    old_to_new_negative=int(np.count_nonzero(old & ~new)),
                    mean_fraction_change=float(np.mean(delta)), fraction_change_rmse=float(np.sqrt(np.mean(delta * delta))),
                    native_clip_high=int(np.count_nonzero(frame.experimental1020_native_raw > 1)),
                    final_clip_high=int(np.count_nonzero(frame.experimental1020_final_raw > 1)))

    summaries = [dict(population=derive.POPULATION, scope="overall", group="all", **saved_summary(pairs))]
    for scope in ("region", "AOD_bin", "platform"):
        for name, sub in pairs.groupby(scope):
            summaries.append(dict(population=derive.POPULATION, scope=scope, group=name, **saved_summary(sub)))
    # Decoy feature-complete population must never be mixed into this cohort.
    summaries.append({**summaries[0], "population": "full8_noncalibration", "rows": 1})
    table = {"columns": list(summaries[0]), "records": summaries,
             "filter_columns": ["population", "scope", "group"], "count_columns": ["rows"],
             "source_file": "/never/open/target_changes_stratified.csv", "source_sha256": "a" * 64,
             "semantics": "Synthetic saved summary"}
    metadata = dict(paired_rows=4, label_changed=2, old_positive=2, experimental_positive=2,
                    label_changed_fraction=.5, fraction_mean_change=summaries[0]["mean_fraction_change"],
                    fraction_RMSE_change=summaries[0]["fraction_change_rmse"], final_choice="not selected")
    return rows, Registry({"tables": {derive.TABLE: table}}), metadata


def embed_pairs(registry, pairs, sidecar, source_file="/never/open/reference_pairs.csv"):
    registry.payload["tables"]["derived/reference_pairs"] = {
        "source_file": str(source_file), "source_sha256": sidecar["output_sha256"],
        "source_rows": len(pairs), "columns": list(pairs), "records": pairs.to_dict("records"),
        "filter_columns": ["row_id", "population"], "count_columns": [],
        "semantics": derive.SEMANTICS, "derivation": copy.deepcopy(sidecar),
    }


def test_registry_native_loader_retains_values_and_historical_provenance(figure, monkeypatch):
    rows, registry, metadata = synthetic()
    pairs, checks = derive.extract(rows, registry, metadata)
    sidecar = {"registry_sha256": "0" * 64, "output_sha256": "b" * 64,
               "label_threshold": .2, "checks": checks, "sources": {"/never/open/old_registry.json": "0" * 64}}
    embed_pairs(registry, pairs, sidecar)
    before = copy.deepcopy(registry.payload)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("External CSV read"))
    loaded, provenance = figure.load_pairs(registry)
    pd.testing.assert_frame_equal(loaded, pairs, check_exact=True)
    assert provenance["derivation"] == sidecar
    assert provenance == {"table": figure.PAIRS_TABLE, **{
        k: v for k, v in registry.payload["tables"][figure.PAIRS_TABLE].items() if k != "records"}}
    provenance["derivation"]["checks"]["paired_rows"] = -1
    assert registry.payload == before


@pytest.mark.parametrize("mutation,match", [("sha", "SHA"), ("threshold", "threshold"),
                                          ("count", "row count"), ("population", "population"),
                                          ("index", "indices")])
def test_registry_native_loader_rejects_inconsistent_embedded_metadata(figure, mutation, match):
    rows, registry, metadata = synthetic()
    pairs, checks = derive.extract(rows, registry, metadata)
    embed_pairs(registry, pairs, {"registry_sha256": "0" * 64, "output_sha256": "b" * 64,
                                 "label_threshold": .2, "checks": checks})
    table = registry.payload["tables"][figure.PAIRS_TABLE]
    if mutation == "sha":
        table["source_sha256"] = "c" * 64
    elif mutation == "threshold":
        table["derivation"]["label_threshold"] = .3
    elif mutation == "count":
        table["source_rows"] += 1
    elif mutation == "population":
        table["records"][0]["population"] = "full8_noncalibration"
    else:
        table["records"][0]["source_record_index"] = -.5
    with pytest.raises(ValueError, match=match):
        figure.load_pairs(registry)


def test_standard_bundle_cli_requires_only_registry_and_output_dir(figure, monkeypatch, tmp_path, capsys):
    registry_path, output = tmp_path / "registry.json", tmp_path / "output"
    calls = []
    def fake_make(registry, destination):
        calls.append((registry, destination))
        return {"counts": {"plotted_pairs": 4}}
    monkeypatch.setattr(figure, "make", fake_make)
    figure.main(["--registry", str(registry_path), "--output-dir", str(output)])
    assert calls == [(registry_path, output)]
    assert json.loads(capsys.readouterr().out) == {"plotted_pairs": 4}


def test_exact_saved_cohort_order_fractions_and_counts_without_feature_screen():
    rows, registry, metadata = synthetic()
    original = rows.copy(deep=True)
    pairs, checks = derive.extract(rows, registry, metadata)
    assert pairs.row_id.tolist() == ["a", "b", "c", "d"]
    assert pairs.source_record_index.tolist() == [0, 2, 3, 5]
    pd.testing.assert_frame_equal(pairs[derive.COLUMNS], rows.iloc[[0, 2, 3, 5]][derive.COLUMNS].reset_index(drop=True), check_exact=True)
    assert checks["paired_rows"] == 4 and checks["paired_sites"] == 3
    assert checks["regions"] == 2 and checks["verified_strata"] == 6
    assert checks["counts_exact"]
    assert derive.summarize(pairs)["old_to_new_positive"] == 1
    assert derive.summarize(pairs)["old_to_new_negative"] == 1
    assert derive.LABEL_THRESHOLD == .2  # Equality is positive, not a retuned cutoff.
    pd.testing.assert_frame_equal(original, rows, check_exact=True)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "row_id"), ("reservation_string", "boolean"),
    ("count", "count mismatch"), ("mean", "summary mismatch"),
    ("missing_region", "strata"), ("duplicate_region", "Ambiguous saved"),
    ("unclipped", "already be finite and clipped"), ("metadata_count", "json count mismatch"),
])
def test_invalid_sources_fail_closed_without_repair(mutation, match):
    rows, registry, metadata = synthetic()
    records = registry.payload["tables"][derive.TABLE]["records"]
    if mutation == "duplicate":
        rows.loc[1, "row_id"] = "a"
    elif mutation == "reservation_string":
        rows["calibration_reserved"] = rows.calibration_reserved.astype(str)
    elif mutation == "count":
        records[0]["rows"] += 1
    elif mutation == "mean":
        records[0]["mean_fraction_change"] += 2e-12
    elif mutation == "missing_region":
        records.pop(1)
    elif mutation == "duplicate_region":
        records.append(copy.deepcopy(records[1]))
    elif mutation == "unclipped":
        rows.loc[0, "experimental1020"] = 1.1
    else:
        metadata["paired_rows"] += 1
    with pytest.raises(ValueError, match=match):
        derive.extract(rows, registry, metadata)


def test_regional_agreement_is_paired_count_arithmetic_not_repeat_or_site_weighting(figure):
    rows, registry, metadata = synthetic()
    pairs, _ = derive.extract(rows, registry, metadata)
    summaries, data, evidence, checks = figure.build_display(registry, pairs)
    assert summaries.group.tolist() == ["all", "region_A", "region_B"]
    assert summaries.agreement_percent.tolist() == [50., 0., 100.]
    assert summaries.rows.tolist() == [4, 2, 2]
    assert len(data[data.record_type == "paired_fraction"]) == 4
    assert len(data[data.record_type == "saved_stratum"]) == 3
    assert all(e["filter"]["population"] == derive.POPULATION and e["interval"] is None for e in evidence)
    assert all(e["source_sha256"] == "a" * 64 for e in evidence)
    assert not any("ci_" in c for c in data)
    assert "not OOF" in derive.SEMANTICS
    assert "not calibrated" in figure.CAPTION
    assert checks["paired_rows"] != checks["paired_sites"]


def test_draw_shows_all_pairs_and_all_regions_at_final_size(figure):
    rows, registry, metadata = synthetic()
    pairs, _ = derive.extract(rows, registry, metadata)
    summaries, _, _, _ = figure.build_display(registry, pairs)
    fig = figure.draw(pairs, summaries)
    try:
        fig.canvas.draw()
        np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets(), pairs[derive.TARGETS].to_numpy())
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_xdata(), [50., 0., 100.])
        assert len(fig.axes[1].get_yticklabels()) == 3
        np.testing.assert_allclose(fig.get_size_inches() / figure.style.MM, [190, 140])
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 6.5
        assert any("N=2" in t.get_text() for t in texts)
    finally:
        plt.close(fig)


def test_extraction_and_render_artifacts_hashes_and_immutable_inputs(tmp_path, monkeypatch, figure):
    rows, registry, metadata = synthetic()
    root = tmp_path / "frozen"
    grouped = root / "grouped_validation"
    grouped.mkdir(parents=True)
    rows.to_parquet(grouped / "rows.parquet", index=False)
    (grouped / "target_changes.json").write_text(json.dumps(metadata))
    source = grouped / "target_changes_stratified.csv"
    registry.table(derive.TABLE).to_csv(source, index=False)
    table = registry.payload["tables"][derive.TABLE]
    table.update(source_file=str(source), source_sha256=digest(source))
    for name in ("prepare.py", "finish.py"):
        (grouped / name).write_text("raise RuntimeError('Do not execute source mains')\n")
    registry_dir = tmp_path / "numbers"
    registry_dir.mkdir()
    registry_path = registry_dir / "registry.json"
    registry_path.write_text(json.dumps(registry.payload))
    inputs = [*grouped.iterdir(), registry_path]
    hashes = {p: digest(p) for p in inputs}
    derived = tmp_path / "derived"
    sidecar = derive.derive(root, registry_path, derived)
    pairs_path = derived / "reference_pairs.csv"
    assert digest(pairs_path) == sidecar["output_sha256"]
    assert sidecar["registry_sha256"] == digest(registry_path)
    pairs = pd.read_csv(pairs_path, float_precision="round_trip")
    embed_pairs(registry, pairs, sidecar, pairs_path)
    canonical = registry_dir / "canonical.json"
    canonical.write_text(json.dumps(registry.payload))
    canonical_hash = digest(canonical)
    assert canonical_hash != sidecar["registry_sha256"]
    # The custom renderer stays registry-native; only a read-only commit lookup is allowed.
    monkeypatch.setattr(figure.style, "export", lambda *a, **kw: pytest.fail("Shared git-export invoked"))
    git_reads = []
    def read_commit(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["text"] is True
        git_reads.append(command)
        return "a" * 40 + "\n"
    monkeypatch.setattr(figure.subprocess, "check_output", read_commit)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("Renderer opened external CSV"))
    output = tmp_path / "figures"
    provenance = figure.make(canonical, output)
    assert provenance["counts"] == {"plotted_pairs": 4, "plotted_strata": 3}
    assert provenance["visual_qa"]["canvas_bounds_pass"]
    assert provenance["code_commit"] == "a" * 40
    assert len(git_reads) == 1
    assert provenance["registry_sha256"] == canonical_hash
    assert provenance["derivation"] == sidecar
    assert provenance["paired_table"]["source_sha256"] == digest(pairs_path)
    assert provenance["inputs"] == {str(canonical): canonical_hash}
    for name, sha in provenance["outputs"].items():
        assert digest(output / name) == sha
    assert (output / f"{figure.STEM}.pdf").is_file()
    assert (output / f"{figure.STEM}.tif").is_file()
    assert len(list((output / "qa").glob("*.png"))) == 4
    assert hashes == {p: digest(p) for p in inputs}
    assert digest(canonical) == canonical_hash


def test_private_output_guards(tmp_path):
    with pytest.raises(ValueError, match="public worktree"):
        derive.guard_output(Path(derive.__file__).parent / "bad")
    author = tmp_path / "source"
    (author / "01_instructions").mkdir(parents=True)
    with pytest.raises(ValueError, match="author"):
        derive.guard_output(author / "figures")
    with pytest.raises(ValueError, match="input directories"):
        derive.guard_output(tmp_path / "frozen" / "bad", [tmp_path / "frozen" / "rows.parquet"])
