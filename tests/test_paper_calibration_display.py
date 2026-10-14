"""Synthetic saved-selection checks; no calibration fitting or raw-cache reads."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper import derive_calibration_display as display
from paper.numbers import digest


def synthetic_inputs():
    names = [f"synthetic_site_{i}" for i in range(100) if display.reserved_site(f"synthetic_site_{i}")][:2]
    saved = pd.DataFrame({
        display.LR: [10.0, 50.0, 90.0, 150.0],
        display.PLDR: [.3, .35, .3, .35],
        "date": ["2009-01-01", "2009-01-01", "2008-01-02", "2008-01-02"],
        "Site": [names[0], names[0], names[1], names[1]],
    })
    metadata = dict(
        hash_rule=display.HASH_RULE, selection=display.SELECTION,
        raw_cache_rows=100, raw_reserved_rows=20, selected_rows=4, selected_sites=2,
        median_LR1020_sr=70.0, endmembers=[.02, .3], CALIOP_used=False,
        all_reserved_collocation_rows=12, all_reserved_collocation_sites=3,
    )
    overview = pd.DataFrame({
        "site_key": names, "role": ["calibration"] * 2,
        "reference": [display.REFERENCE] * 2, "support": [display.SUPPORT] * 2,
        "lat": [10.0, -20.0], "lon": [30.0, -40.0], "calibration_records": [2, 2],
    })
    return saved, metadata, overview


def write_inputs(tmp_path):
    saved, metadata, overview = synthetic_inputs()
    root = tmp_path / "source"
    folder = root / "grouped_validation"
    folder.mkdir(parents=True)
    selected = folder / "calibration_selected.csv.gz"
    saved.to_csv(selected, index=False)
    metadata_path = folder / "calibration.json"
    metadata_path.write_text(json.dumps(metadata))
    overview_path = tmp_path / "site_overview.csv"
    overview.to_csv(overview_path, index=False)
    overview_path.with_suffix(".provenance.json").write_text(json.dumps({
        "output_sha256": digest(overview_path), "sources": {"/synthetic/cache": "b" * 64},
        "semantics": "Synthetic original-cache coordinates",
    }))
    return root, overview_path


def test_preserves_every_saved_row_and_exposes_metadata_without_refitting():
    saved, metadata, overview = synthetic_inputs()
    before = saved.copy(deep=True)
    before_meta = copy.deepcopy(metadata)
    records, summary, checks = display.validate_and_extract(saved, metadata, overview)
    pd.testing.assert_frame_equal(records[saved.columns], saved, check_exact=True)
    pd.testing.assert_frame_equal(saved, before)
    assert metadata == before_meta
    assert records.source_record_index.tolist() == [0, 1, 2, 3]
    assert records.lat.tolist() == [10.0, 10.0, -20.0, -20.0]
    assert records.lon.tolist() == [30.0, 30.0, -40.0, -40.0]
    assert records.calibration_reserved.all()
    for name in display.COUNT_FIELDS:
        assert summary.iloc[0][name] == metadata[name]
    assert summary.iloc[0].Sd_sr == summary.iloc[0].median_LR1020_sr == 70.0
    assert summary.iloc[0].endmember_nondust == .02
    assert summary.iloc[0].endmember_dust == .3
    assert not summary.iloc[0].CALIOP_used
    assert checks["saved_median_exact_match"]
    assert checks["selected_rows"] == 4 and checks["selected_sites"] == 2


def test_fraction_is_clipped_before_lr_scaling_and_lr_bounds_are_inclusive():
    saved, metadata, overview = synthetic_inputs()
    records, _, _ = display.validate_and_extract(saved, metadata, overview)
    raw = (saved[display.PLDR].to_numpy() - .02) * 1.30 / (.28 * (1 + saved[display.PLDR].to_numpy()))
    np.testing.assert_array_equal(records.backfraction1020_preLR, np.clip(raw, 0, 1))
    assert records.backfraction1020_preLR.iloc[1] == 1.0
    assert raw[1] > 1
    assert records.backfraction1020_preLR.iloc[-1] * metadata["median_LR1020_sr"] / 150 < .9
    assert records[display.LR].iloc[0] == 10 and records[display.LR].iloc[-1] == 150


def test_duplicate_observations_are_not_deduplicated_or_resampled():
    saved, metadata, overview = synthetic_inputs()
    saved = pd.concat([saved, saved.iloc[[1]]], ignore_index=True)
    metadata["selected_rows"] = 5
    metadata["median_LR1020_sr"] = 50.0
    overview.loc[0, "calibration_records"] = 3
    records, _, _ = display.validate_and_extract(saved, metadata, overview)
    pd.testing.assert_frame_equal(records[saved.columns], saved, check_exact=True)
    assert len(records) == 5


@pytest.mark.parametrize("field,value,message", [
    (display.LR, 9.99, "LR outside"),
    (display.LR, 150.01, "LR outside"),
    (display.PLDR, .02, "preLR fraction"),
    (display.PLDR, np.nan, "Nonfinite"),
    (display.LR, np.inf, "Nonfinite"),
    ("date", "2010-01-01", "pre2010"),
    ("date", None, "pre2010"),
    ("Site", None, "Missing or invalid"),
])
def test_bad_saved_row_fails_instead_of_filtering_it_out(field, value, message):
    saved, metadata, overview = synthetic_inputs()
    saved.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        display.validate_and_extract(saved, metadata, overview)


@pytest.mark.parametrize("field,value,message", [
    ("selected_rows", 5, "counts disagree"),
    ("selected_sites", 3, "counts disagree"),
    ("median_LR1020_sr", 70.0000000001, "Pooled record median differs"),
    ("median_LR1020_sr", None, "Invalid saved median"),
    ("endmembers", [.01, .3], "endmembers"),
    ("CALIOP_used", True, "CALIOP"),
    ("selection", "different selection", "selection"),
    ("hash_rule", "different rule", "hash rule"),
])
def test_metadata_mismatch_fails_without_silent_tolerance_or_new_selection(field, value, message):
    saved, metadata, overview = synthetic_inputs()
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        display.validate_and_extract(saved, metadata, overview)


def test_non_reserved_site_and_coordinate_problems_fail():
    saved, metadata, overview = synthetic_inputs()
    other = next(f"outside_{i}" for i in range(100) if not display.reserved_site(f"outside_{i}"))
    saved.loc[:1, "Site"] = other
    with pytest.raises(ValueError, match="non-reserved"):
        display.validate_and_extract(saved, metadata, overview)
    saved, metadata, overview = synthetic_inputs()
    for changed, message in (
        (overview.iloc[:1], "site set differs"),
        (pd.concat([overview, overview.iloc[[0]]]), "Ambiguous"),
        (overview.assign(lat=np.nan), "Invalid calibration coordinates"),
        (overview.assign(calibration_records=99), "counts disagree"),
    ):
        with pytest.raises(ValueError, match=message):
            display.validate_and_extract(saved, metadata, changed)


def test_cli_private_outputs_source_hashes_and_unmodified_inputs(tmp_path):
    root, overview = write_inputs(tmp_path)
    paths = [root / "grouped_validation" / name for name in
             ("calibration.json", "calibration_selected.csv.gz")]
    paths += [overview, overview.with_suffix(".provenance.json")]
    before = {str(path): digest(path) for path in paths}
    output = tmp_path / "derived"
    display.main(["--source-root", str(root), "--site-overview", str(overview),
                  "--output-dir", str(output)])
    for name in ("calibration_records", "calibration_summary"):
        path = output / (name + ".csv")
        provenance = json.loads(path.with_suffix(".provenance.json").read_text())
        assert provenance["sources"] == before
        assert provenance["output_sha256"] == digest(path)
        assert provenance["script_sha256"] == digest(Path(display.__file__))
        assert provenance["coordinate_provenance"]["sources"] == {"/synthetic/cache": "b" * 64}
        assert "No raw-cache search" in provenance["semantics"]
    assert before == {str(path): digest(path) for path in paths}
    records = pd.read_csv(output / "calibration_records.csv", float_precision="round_trip")
    saved = pd.read_csv(root / "grouped_validation/calibration_selected.csv.gz", float_precision="round_trip")
    pd.testing.assert_frame_equal(records[saved.columns], saved, check_exact=True)
    assert len(pd.read_csv(output / "calibration_summary.csv")) == 1


def test_cli_rejects_public_or_source_outputs_and_bad_coordinate_hash(tmp_path):
    root, overview = write_inputs(tmp_path)
    public = Path(display.__file__).resolve().parent
    for output in (public, root / "derived"):
        with pytest.raises(ValueError, match="public worktree|frozen source"):
            display.main(["--source-root", str(root), "--site-overview", str(overview),
                          "--output-dir", str(output)])
    overview.write_text(overview.read_text() + "\n")
    output = tmp_path / "bad_hash"
    with pytest.raises(ValueError, match="SHA256 differs"):
        display.main(["--source-root", str(root), "--site-overview", str(overview),
                      "--output-dir", str(output)])
    assert not output.exists()
