"""Synthetic fail-closed tests; opt-in read-only verification of a private archive."""
import gzip
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from paper import archive_minimal as archive


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reanchor_side(case, path):
    side_path = case["inventory_path"].with_suffix(".provenance.json")
    side = json.loads(side_path.read_text())
    side["sources"][str(path)] = archive.digest(path)
    write_json(side_path, side)


def reanchor_artifact(case, path):
    provenance = json.loads(case["provenance_path"].read_text())
    for artifact in provenance["artifacts"]:
        if path.name == artifact["row_site_membership"]:
            artifact["membership_sha256"] = archive.digest(path)
        for saved in artifact["files"]:
            if path.is_relative_to(case["source_root"]) and saved.endswith(str(path.relative_to(case["source_root"]))):
                artifact["files"][saved] = archive.digest(path)
    write_json(case["provenance_path"], provenance)


@pytest.fixture
def case(tmp_path):
    source, private = tmp_path / "source", tmp_path / "private"
    grouped = source / "grouped_validation"
    grouped.mkdir(parents=True)
    private.mkdir()
    rows = pd.DataFrame(dict(
        row_id=["r1", "r2", "r3", "unused", "reserved"],
        site=["Alpha", "Alpha", "Beta", "Gamma", "Cal"],
        site_key=["alpha", "alpha", "beta", "gamma", "cal"],
        granule=["g1", "g2", "g3", "g4", "g5"],
        platform=["One"] * 5, time=["2020-01-01"] * 5, region=["Region"] * 5,
        calibration_reserved=[False, False, False, False, True],
        source_file=["/private/cache/a.pkl", "b.pkl", "C:\\private\\c.pkl", "u.pkl", "r.pkl"],
        saved_reference=[0., 0.6, float("nan"), 0.4, 0.2],
        integer_field=[1, 2, 3, 4, 5]))
    rows.to_parquet(grouped / "rows.parquet", index=False)
    inventory = pd.DataFrame(dict(site_key=["alpha", "beta", "gamma", "cal"],
                                  calibration_reserved=[False, False, False, True]))
    inventory.to_csv(grouped / "site_inventory.csv", index=False)
    calibration = pd.DataFrame([[40., .3, "2001-01-01", "Cal"],
                                [40., .3, "2001-01-01", "Cal"],
                                [41., .3, "2001-01-01", "Cal"]], columns=archive.CAL_COLUMNS)
    calibration.to_csv(grouped / "calibration_selected.csv.gz", index=False)
    write_json(grouped / "calibration.json", dict(selected_rows=3, selected_sites=1))
    q = rows.iloc[:3][["row_id", *archive.IDENTITY]].copy()
    q = pd.concat([q.assign(repeat=0), q.assign(repeat=1)], ignore_index=True)
    q = q.assign(scheme="site_cv", reference="refA", population="own", model="hierarchy_prespecified")
    oof = grouped / "oof/cohort.parquet"
    oof.parent.mkdir()
    q.to_parquet(oof, index=False)
    header = "/private/headers/synthetic.all"
    header_hash = "1" * 64
    pi = json.dumps([dict(source_file=header, source_sha256=header_hash,
                          principal_investigators="Synthetic Investigator", header_sha256="2" * 64,
                          encoding="utf-8", field="PI= before semicolon")])
    attribution = pd.DataFrame(dict(
        site=["Alpha", "Beta", "Cal"], site_key=["alpha", "beta", "cal"],
        role=["evaluation", "evaluation", "calibration"], reference=["refA"] * 3,
        n_records=[2, 1, 3], collocations=["2", "1", ""], calibration_records=["", "", "3"],
        principal_investigators=["Synthetic Investigator"] * 3, pi_status=["local_header"] * 3,
        pi_evidence_json=[pi] * 3, region_status=["saved_inventory"] * 3))
    table = private / "site_inventory_table.csv"
    attribution.to_csv(table, index=False)
    sources = {str(p): archive.digest(p) for p in [oof, grouped / "site_inventory.csv",
               grouped / "calibration.json", grouped / "calibration_selected.csv.gz"]}
    sources[header] = header_hash
    write_json(table.with_suffix(".provenance.json"), dict(
        output_sha256=archive.digest(table), sources=sources, rows=3,
        overview_provenance={}, coverage=dict(rows=3, unique_collocations=3,
                                             evaluation_sites=2, calibration_records=3, calibration_sites=1)))
    scope = private / "scope.json"
    write_json(scope, dict(sha256=archive.digest(grouped / "rows.parquet"), rows=5, sites=4,
                           columns=list(rows.columns), site_inventory_rows=4,
                           site_inventory_sha256=archive.digest(grouped / "site_inventory.csv")))
    artifacts = []
    for reference in ("refA", "refB"):
        for branch in ("wide", "narrow", "small"):
            selected = rows.iloc[:2 if branch == "wide" else 3]
            folder = source / "expected_value_test/deploy_ev" / reference / branch
            folder.mkdir(parents=True)
            (folder / "model.ubj").write_bytes(b"synthetic opaque model; never load")
            memberships = dict(train=selected.row_id.tolist(), heldout=[], purged=[],
                               positive_regression=["r2"],
                               negative_mean=[r for r in selected.row_id if r != "r2"])
            audit = {}
            (folder / "membership").mkdir()
            for name, ids in memberships.items():
                with gzip.open(folder / f"membership/{name}_rows.json.gz", "wt") as stream:
                    json.dump(ids, stream)
                subset = rows.set_index("row_id").loc[ids]
                audit[name] = dict(rows=len(ids), sites=int(subset.site_key.nunique()),
                                   row_sha256=archive.saved_set_hash(ids),
                                   site_sha256=archive.saved_set_hash(subset.site_key.unique()),
                                   granule_sha256=archive.saved_set_hash(subset.granule.unique()))
            write_json(folder / "membership/audit.json", dict(memberships=audit))
            member = private / f"{reference}_{branch}_membership.csv"
            selected[["row_id", "site_key"]].to_csv(member, index=False)
            artifacts.append(dict(reference=reference, branch=branch, role="synthetic",
                                  row_site_membership=member.name, membership_sha256=archive.digest(member),
                                  files={"saved/" + str(p.relative_to(source)): archive.digest(p)
                                         for p in folder.rglob("*") if p.is_file()}))
    provenance = private / "artifact_provenance.json"
    write_json(provenance, dict(source_table_sha256=archive.digest(grouped / "rows.parquet"), artifacts=artifacts))
    return dict(source_root=source, provenance_path=provenance, scope_path=scope,
                inventory_path=table, oof_path=oof, reference="refA", expected_artifacts=6)


def test_prepare_and_independent_verification(case, tmp_path):
    before = {p: archive.digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "archive"
    report = archive.prepare(output, **case)
    assert report == archive.verify_archive(output, **case)
    assert report["collocations"] == 3
    assert report["source_file_values_sanitized"] == 2
    assert report["calibration_duplicate_records"] == 1
    assert report["calibration_repeated_site_date_records"] == 2
    assert pd.read_parquet(output / "collocations.parquet").source_file.tolist() == ["a.pkl", "b.pkl", "c.pkl"]
    assert not list(output.rglob("*.ubj"))
    assert len(list(output.glob("memberships/*.csv"))) == 6
    assert len(list(output.rglob("*_rows.json.gz"))) == 30
    assert all(archive.digest(p) == sha for p, sha in before.items())
    with pytest.raises(FileExistsError):
        archive.prepare(output, **case)


@pytest.mark.parametrize("target", ["rows", "membership", "original", "model", "calibration", "inventory", "oof"])
def test_corrupt_anchor_fails_before_output(case, tmp_path, target):
    paths = dict(rows=case["source_root"] / "grouped_validation/rows.parquet",
                 membership=case["provenance_path"].parent / "refA_wide_membership.csv",
                 original=case["source_root"] / "expected_value_test/deploy_ev/refA/wide/membership/train_rows.json.gz",
                 model=case["source_root"] / "expected_value_test/deploy_ev/refA/wide/model.ubj",
                 calibration=case["source_root"] / "grouped_validation/calibration_selected.csv.gz",
                 inventory=case["inventory_path"], oof=case["oof_path"])
    with paths[target].open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA mismatch"):
        archive.prepare(tmp_path / "archive", **case)
    assert not (tmp_path / "archive").exists()


@pytest.mark.parametrize("mutation", ["repeat", "identity", "subset"])
def test_oof_semantic_checks(case, mutation):
    q = pd.read_parquet(case["oof_path"])
    if mutation == "repeat":
        q = q.iloc[:-1]
    elif mutation == "identity":
        q.loc[q.row_id.eq("r1"), "site_key"] = "wrong"
    else:
        q = q.loc[q.row_id.ne("r3")]
    q.to_parquet(case["oof_path"], index=False)
    reanchor_side(case, case["oof_path"])
    with pytest.raises(ValueError, match="OOF"):
        archive.inspect_inputs(**case)


def test_wrong_membership_site_even_with_valid_hash(case):
    path = case["provenance_path"].parent / "refA_wide_membership.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "site_key"] = "beta"
    frame.to_csv(path, index=False)
    reanchor_artifact(case, path)
    with pytest.raises(ValueError, match="row/site"):
        archive.inspect_inputs(**case)


@pytest.mark.parametrize("ids", [["outside"], ["r2", "r2"], ["r1", "r2"]])
def test_positive_subset_linkage(case, ids):
    path = case["source_root"] / "expected_value_test/deploy_ev/refA/wide/membership/positive_regression_rows.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(ids, stream)
    reanchor_artifact(case, path)
    with pytest.raises(ValueError, match="membership"):
        archive.inspect_inputs(**case)


def test_reserved_calibration_overlap(case):
    path = case["source_root"] / "grouped_validation/calibration_selected.csv.gz"
    frame = pd.read_csv(path)
    frame["Site"] = "Alpha"
    frame.to_csv(path, index=False)
    reanchor_side(case, path)
    with pytest.raises(ValueError, match="not reserved"):
        archive.inspect_inputs(**case)


@pytest.mark.parametrize("mutation", ["count", "email", "header_anchor"])
def test_attribution_checks(case, mutation):
    path = case["inventory_path"]
    frame = pd.read_csv(path, keep_default_na=False)
    if mutation == "count":
        frame.loc[0, "n_records"] = 99
    else:
        evidence = json.loads(frame.loc[0, "pi_evidence_json"])
        if mutation == "email":
            evidence[0]["principal_investigators"] = "name@example.org"
            frame.loc[0, "principal_investigators"] = "name@example.org"
        else:
            evidence[0]["source_sha256"] = "3" * 64
        frame.loc[0, "pi_evidence_json"] = json.dumps(evidence)
    frame.to_csv(path, index=False)
    sidepath = path.with_suffix(".provenance.json")
    side = json.loads(sidepath.read_text())
    side["output_sha256"] = archive.digest(path)
    write_json(sidepath, side)
    with pytest.raises(ValueError):
        archive.inspect_inputs(**case)


@pytest.mark.parametrize("kind", ["source", "worktree", "symlink"])
def test_output_boundaries(case, tmp_path, kind):
    if kind == "source":
        output = case["source_root"] / "forbidden"
    elif kind == "worktree":
        root = tmp_path / "worktree"
        root.mkdir()
        (root / ".git").write_text("gitdir: elsewhere")
        output = root / "forbidden"
    else:
        root = tmp_path / "link"
        root.symlink_to(case["source_root"], target_is_directory=True)
        output = root / "forbidden"
    with pytest.raises(ValueError):
        archive.prepare(output, **case)
    assert not output.exists()


def test_traversal_artifact_rejected(case):
    provenance = json.loads(case["provenance_path"].read_text())
    provenance["artifacts"][0]["branch"] = "../outside"
    write_json(case["provenance_path"], provenance)
    with pytest.raises(ValueError, match="Unsafe"):
        archive.inspect_inputs(**case)


@pytest.mark.parametrize("mutation", ["extra", "data", "manifest", "symlink"])
def test_archive_corruption(case, tmp_path, mutation):
    output = tmp_path / "archive"
    archive.prepare(output, **case)
    if mutation == "extra":
        (output / "extra.txt").write_text("unexpected")
    elif mutation == "data":
        path = output / "collocations.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "integer_field"] += 1
        frame.to_parquet(path, index=False)
        # Even an updated manifest cannot disguise changed scientific data.
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["files"][path.name] = dict(sha256=archive.digest(path), bytes=path.stat().st_size)
        write_json(output / "manifest.json", manifest)
    elif mutation == "symlink":
        (output / "README.md").unlink()
        (output / "README.md").symlink_to(case["scope_path"])
    else:
        write_json(output / "manifest.json", dict(files={}))
    with pytest.raises(ValueError):
        archive.verify_archive(output, **case)


def test_source_mutation_during_copy_cleans_own_output(case, tmp_path, monkeypatch):
    original = archive.shutil.copyfileobj
    path = case["source_root"] / "expected_value_test/deploy_ev/refA/wide/model.ubj"

    def mutate(src, dst):
        original(src, dst)
        path.write_bytes(b"changed during packaging")

    monkeypatch.setattr(archive.shutil, "copyfileobj", mutate)
    with pytest.raises(ValueError, match="Source changed"):
        archive.prepare(tmp_path / "archive", **case)
    assert not (tmp_path / "archive").exists()


def test_actual_private_archive_read_only():
    """Supply JSON arguments via env; never store private paths/results in the repo."""
    config = os.environ.get("MINIMAL_ARCHIVE_VERIFY_JSON")
    if not config:
        pytest.skip("Private inputs not supplied; set MINIMAL_ARCHIVE_VERIFY_JSON")
    inputs = json.loads(config)
    report = archive.verify_archive(**inputs)
    assert report["status"] == "PRIVATE_PREPARATION_VERIFIED_NOT_PUBLICATION"
