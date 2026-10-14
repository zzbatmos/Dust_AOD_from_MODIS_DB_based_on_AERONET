"""Prepare a private, membership-bounded archive from saved evidence; never fit or publish.

All cohort sizes and hashes come from supplied private metadata, not this module.
Run as ``python -m paper.archive_minimal --help``. Verification is read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil

import pandas as pd


LISTS = ("train", "heldout", "purged", "positive_regression", "negative_mean")
CAL_COLUMNS = ["Lidar_Ratio[1020nm]", "Depolarization_Ratio[1020nm]", "date", "Site"]
IDENTITY = ["site", "site_key", "granule", "platform", "time", "region"]
LIMITS = [
    "Private preparation only; no publication, upload, tag, DOI or license approval.",
    "Membership means saved deployment training membership, not every upstream row.",
    "OOF is used only to check the primary own-population site-CV row/site identities; "
    "predictions and fold-level training artifacts are not included.",
    "All collocation columns and selected-row order are retained; only absolute "
    "source_file paths become basenames. Parquet container bytes are not preserved.",
    "Calibration CSV and metadata, membership CSVs and original compressed lists "
    "are byte-preserved. No calibration deduplication, reselection or renaming.",
    "PI names, header hashes and raw source hashes are inherited from the hash-checked "
    "inventory and its sidecar; raw inversion files are not reread or included. "
    "Names do not establish historical tenure, consultation or acknowledgment approval.",
    "Missing/conflicting attribution or regions remain explicitly recorded, never inferred.",
    "Models and other deployment files are verified locally and linked by relative path "
    "and SHA-256, not duplicated. Public access to those targets remains a release gate.",
    "This is not full raw-granule reconstruction or complete historical refitting proof; "
    "uncommitted original implementation and unavailable training settings remain limitations.",
    "Saved provenance/sidecars are trust anchors, not independently signed attestations. "
    "All consumed files and linked deployment files are checked before and after packaging.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def packed(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def saved_set_hash(values):
    """The saved membership audit's sorted pipe-separated string digest."""
    return hashlib.sha256("|".join(sorted(map(str, values))).encode()).hexdigest()


def component(value):
    require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))
            and value not in {".", ".."}, "Unsafe path component")
    return value


def basename(value):
    require(isinstance(value, str) and bool(value), "Missing source filename")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute():
        return windows.name
    if posix.is_absolute():
        return posix.name
    require(posix.name == value and windows.name == value and value not in {".", ".."},
            "Relative source directory is ambiguous; refusing to silently sanitize")
    return value


def private_text_check(value):
    text = str(value)
    require(not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text), "Email in export")
    require(not re.search(r"(?:^|[\s\"'=,])(?:/[A-Za-z]|[A-Za-z]:[\\/]|\\\\)", text),
            "Absolute path in export")


def frame_check(frame):
    for column in frame.columns:
        private_text_check(column)
        for value in frame[column].dropna().unique():
            if isinstance(value, str):
                private_text_check(value)


def equal(left, right, message):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                      check_exact=True)
    except AssertionError as error:
        raise ValueError(message) from error


def identities(frame, label):
    require(not frame.empty and not frame[["row_id", "site_key"]].isna().any().any()
            and not frame.row_id.duplicated().any(), f"Invalid {label} row identities")


@dataclass
class Plan:
    collocations: pd.DataFrame
    attribution: pd.DataFrame
    copies: dict
    sources: dict
    evidence: list
    mapping: dict
    report: dict

    def unchanged(self):
        for path, sha in self.sources.items():
            require(digest(path) == sha, f"Source changed: {path.name}")


def inspect_inputs(*, source_root, provenance_path, scope_path, inventory_path,
                   oof_path, reference, expected_artifacts):
    """Validate the complete bounded payload in memory, with no filesystem writes."""
    source_root = Path(source_root).resolve()
    provenance_path, scope_path, inventory_path, oof_path = map(
        lambda p: Path(p).resolve(), (provenance_path, scope_path, inventory_path, oof_path))
    sources, evidence, copies = {}, [], {}

    def anchor(path, label, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        require(expected is None or sha == expected, f"SHA mismatch: {label}")
        require(path not in sources or sources[path] == sha, f"Source changed: {label}")
        sources[path] = sha
        evidence.append(dict(source=label, sha256=sha,
                             anchor="saved_sha256" if expected else "preparation_snapshot"))
        return path

    def read(path, label, expected=None):
        return json.loads(anchor(path, label, expected).read_text())

    provenance = read(provenance_path, "provenance/artifact_provenance.json")
    scope = read(scope_path, "scope/collocation_scope_review.json")
    side = read(inventory_path.with_suffix(".provenance.json"), "attribution/source_sidecar.json")
    anchor(inventory_path, "attribution/source_inventory.csv", side["output_sha256"])
    require(scope["sha256"] == provenance["source_table_sha256"], "Conflicting row anchors")
    grouped = source_root / "grouped_validation"
    rows_path = anchor(grouped / "rows.parquet", "grouped_validation/rows.parquet", scope["sha256"])
    rows = pd.read_parquet(rows_path)
    identities(rows, "source")
    require(len(rows) == scope["rows"] and rows.site_key.nunique() == scope["sites"]
            and list(rows.columns) == scope["columns"], "Source scope mismatch")
    require(rows.site.str.strip().str.casefold().eq(rows.site_key).all(), "Source site/key mismatch")
    require(rows.calibration_reserved.isin([True, False]).all(), "Missing reservation flags")

    def side_anchor(path):
        path = Path(path).resolve()
        require(str(path) in side["sources"], f"Missing saved source anchor: {path.name}")
        return anchor(path, "grouped_validation/" + path.name, side["sources"][str(path)])

    events = pd.read_parquet(side_anchor(oof_path))
    cal_path = side_anchor(grouped / "calibration_selected.csv.gz")
    meta_path = side_anchor(grouped / "calibration.json")
    original_inventory = pd.read_csv(side_anchor(grouped / "site_inventory.csv"))
    require(digest(grouped / "site_inventory.csv") == scope["site_inventory_sha256"]
            and len(original_inventory) == scope["site_inventory_rows"], "Site inventory scope mismatch")
    require(not original_inventory.site_key.duplicated().any(), "Duplicate original site keys")
    require(original_inventory.calibration_reserved.isin([True, False]).all(), "Invalid site reservation")
    reserved = set(original_inventory.loc[original_inventory.calibration_reserved, "site_key"])
    require(set(rows.loc[rows.calibration_reserved, "site_key"]) == reserved,
            "Row/site reservation mismatch")
    require(not set(rows.loc[~rows.calibration_reserved, "site_key"]) & reserved,
            "Inconsistent row reservation")
    calibration = pd.read_csv(cal_path)
    meta = json.loads(meta_path.read_text())
    require(list(calibration.columns) == CAL_COLUMNS and not calibration.isna().any().any(),
            "Unexpected saved calibration schema")
    cal_keys = calibration.Site.str.strip().str.casefold()
    coverage = side["coverage"]
    require(len(calibration) == meta["selected_rows"] == coverage["calibration_records"]
            and cal_keys.nunique() == meta["selected_sites"] == coverage["calibration_sites"],
            "Calibration totals mismatch")
    require(pd.to_datetime(calibration.date, errors="raise").dt.year.lt(2010).all(),
            "Unexpected saved calibration year")
    require(not (set(cal_keys) & set(original_inventory.site_key)) - reserved,
            "Known calibration site is not reserved")
    copies["calibration/calibration_selected.csv.gz"] = cal_path
    copies["calibration/calibration.json"] = meta_path

    q = events.loc[events.scheme.eq("site_cv")].copy()
    require(not q.empty and not q[["row_id", "repeat", *IDENTITY]].isna().any().any(),
            "Missing OOF identity")
    for key, value in {"reference": reference, "population": "own", "model": "hierarchy_prespecified"}.items():
        require(q[key].eq(value).all(), f"Unexpected OOF {key}")
    repeats = side["overview_provenance"].get("repeats", [0, 1])
    require(set(q["repeat"]) == set(repeats) and not q.duplicated(["row_id", "repeat"]).any()
            and q.groupby("row_id").size().eq(len(repeats)).all(), "Incomplete OOF repeats")
    shared = [c for c in rows.columns if c in q.columns and c != "row_id"]
    require(q.groupby("row_id")[shared].nunique(dropna=False).eq(1).all().all(),
            "OOF identity/value changes across repeats")
    q = q.drop_duplicates("row_id")
    identities(q, "OOF")
    require(set(q.row_id) <= set(rows.row_id), "OOF rows absent from source")
    equal(q[shared], rows.set_index("row_id").loc[q.row_id, shared], "OOF/source identity mismatch")

    artifacts = provenance["artifacts"]
    require(expected_artifacts > 0 and len(artifacts) == expected_artifacts, "Artifact count mismatch")
    union, artifact_keys, mapping = set(), set(), []
    for artifact in artifacts:
        ref, branch = component(artifact["reference"]), component(artifact["branch"])
        key = f"{ref}/{branch}"
        require(key not in artifact_keys, "Duplicate artifact")
        artifact_keys.add(key)
        member_name = component(artifact["row_site_membership"])
        member_path = anchor(provenance_path.parent / member_name, "provenance/" + member_name,
                             artifact["membership_sha256"])
        members = pd.read_csv(member_path)
        require(list(members.columns) == ["row_id", "site_key"], "Membership schema mismatch")
        identities(members, "membership")
        ids = set(members.row_id)
        require(ids <= set(rows.row_id), "Membership row missing from source")
        equal(members, rows.set_index("row_id").loc[members.row_id].reset_index()[members.columns],
              "Membership row/site mismatch")
        union.update(ids)
        csv_target = "memberships/" + member_name
        require(csv_target not in copies, "Duplicate membership destination")
        copies[csv_target] = member_path
        lists, links, audit = {}, {}, None
        for saved, sha in artifact["files"].items():
            parts = PurePosixPath(saved).parts
            require(".." not in parts and parts.count("deploy_ev") == 1, "Unsafe artifact path")
            tail = PurePosixPath(*parts[parts.index("deploy_ev") + 1:])
            require(tail.parts[:2] == (ref, branch), "Artifact path/reference mismatch")
            path = source_root / "expected_value_test/deploy_ev" / tail
            require(path.resolve().is_relative_to(source_root), "Artifact escaped source root")
            anchor(path, "deploy_ev/" + str(tail), sha)
            links[str(tail.relative_to(PurePosixPath(ref) / branch))] = sha
            if len(tail.parts) == 4 and tail.parts[2] == "membership":
                target = "memberships/original/" + str(tail)
                copies[target] = path
                if path.name == "audit.json":
                    audit = json.loads(path.read_text())
                else:
                    require(path.name.endswith("_rows.json.gz"), "Unknown membership file")
                    name = path.name.removesuffix("_rows.json.gz")
                    require(name in LISTS and name not in lists, "Unknown/duplicate membership list")
                    with gzip.open(path, "rt") as stream:
                        values = json.load(stream)
                    require(isinstance(values, list) and all(isinstance(v, str) for v in values)
                            and len(set(values)) == len(values), "Invalid/duplicate original membership")
                    lists[name] = values
        require(set(lists) == set(LISTS) and audit is not None, "Missing original membership evidence")
        require(set(lists["train"]) == ids, "CSV/original training linkage mismatch")
        require(not lists["heldout"] and not lists["purged"], "Deployment heldout/purged scope is not empty")
        positive, negative = set(lists["positive_regression"]), set(lists["negative_mean"])
        require(positive <= ids and negative <= ids and not positive & negative
                and positive | negative == ids, "Positive/negative membership partition mismatch")
        for name, values in lists.items():
            selected = rows.set_index("row_id").loc[values]
            expected = dict(rows=len(values), sites=int(selected.site_key.nunique()),
                            row_sha256=saved_set_hash(values),
                            site_sha256=saved_set_hash(selected.site_key.unique()),
                            granule_sha256=saved_set_hash(selected.granule.unique()))
            require(audit["memberships"][name] == expected, f"Saved membership audit mismatch: {key}/{name}")
        mapping.append(dict(reference=ref, branch=branch, role=artifact["role"],
                            membership_csv=csv_target, membership_sha256=artifact["membership_sha256"],
                            original_membership_directory=f"memberships/original/{key}/membership",
                            rows=len(members), sites=int(members.site_key.nunique()),
                            list_counts={name: len(values) for name, values in lists.items()},
                            linked_deployment_base=f"deploy_ev/{key}", linked_files_sha256=links))

    require(union == set(q.row_id), "Exact membership union differs from primary OOF")
    selected = rows.loc[rows.row_id.isin(union)].copy().reset_index(drop=True)
    require(len(selected) == coverage["unique_collocations"]
            and selected.site_key.nunique() == coverage["evaluation_sites"], "Evaluation totals mismatch")
    require(not selected.calibration_reserved.any() and not set(selected.site_key) & (reserved | set(cal_keys)),
            "Reserved calibration site leaked into evaluation/training")
    clean = selected.copy()
    clean["source_file"] = clean.source_file.map(basename)
    equal(clean.drop(columns="source_file"), selected.drop(columns="source_file"), "Nonpath values changed")

    attribution = pd.read_csv(inventory_path, keep_default_na=False)
    require(len(attribution) == side["rows"] == coverage["rows"]
            and not attribution.site_key.duplicated().any()
            and attribution.role.isin(["evaluation", "calibration"]).all(), "Attribution inventory mismatch")
    require(attribution.site.str.strip().str.casefold().eq(attribution.site_key).all()
            and attribution.reference.eq(reference).all(), "Attribution identity mismatch")
    for role, frame, key_column, site_column in [
            ("evaluation", selected, "site_key", "site"),
            ("calibration", calibration.assign(site_key=cal_keys), "site_key", "Site")]:
        sites = attribution.loc[attribution.role.eq(role)]
        require(set(sites.site_key) == set(frame[key_column]), f"{role} attribution cohort mismatch")
        for row in sites.itertuples():
            group = frame.loc[frame[key_column].eq(row.site_key)]
            count_column = "collocations" if role == "evaluation" else "calibration_records"
            require(group[site_column].eq(row.site).all() and len(group) == int(row.n_records)
                    and len(group) == int(float(getattr(row, count_column))), "Attribution row/site count mismatch")
    evidence_fields = {"source_file", "source_sha256", "principal_investigators", "header_sha256", "encoding", "field"}
    sanitized = []
    for row in attribution.itertuples():
        entries = json.loads(row.pi_evidence_json)
        require(isinstance(entries, list), "Invalid PI evidence")
        names = set()
        for entry in entries:
            require(set(entry) == evidence_fields, "Unexpected PI evidence fields")
            require(side["sources"].get(entry["source_file"]) == entry["source_sha256"]
                    and bool(re.fullmatch(r"[0-9a-f]{64}", entry["header_sha256"])), "PI evidence anchor mismatch")
            if entry["principal_investigators"]:
                names.add(entry["principal_investigators"])
            entry["source_file"] = basename(entry["source_file"])
        status = "local_header" if len(names) == 1 else "conflicting_local_headers" if names else "missing_local_PI"
        require(row.pi_status == status and row.principal_investigators ==
                (next(iter(names)) if len(names) == 1 else "NOT RECORDED"), "PI name/status mismatch")
        sanitized.append(json.dumps(entries, ensure_ascii=True, separators=(",", ":")))
    attribution["pi_evidence_json"] = sanitized
    for frame in (clean, attribution, calibration):
        frame_check(frame)
    for target, path in copies.items():
        if path.suffix != ".gz":
            private_text_check(path.read_text())
    report = dict(status="PRIVATE_PREPARATION_VERIFIED_NOT_PUBLICATION", source_rows=len(rows),
                  collocations=len(clean), evaluation_sites=int(clean.site_key.nunique()),
                  calibration_records=len(calibration), calibration_sites=int(cal_keys.nunique()),
                  calibration_columns=list(calibration.columns),
                  calibration_duplicate_records=int(calibration.duplicated().sum()),
                  calibration_repeated_site_date_records=int(calibration.duplicated(["Site", "date"]).sum()),
                  attribution_sites=len(attribution), artifacts=len(artifacts),
                  source_file_values_sanitized=int(selected.source_file.ne(clean.source_file).sum()),
                  pi_status_counts={str(k): int(v) for k, v in attribution.pi_status.value_counts().items()},
                  missing_region_sites=attribution.loc[attribution.region_status.eq("missing_saved_region"), "site"].tolist(),
                  calibration_sites_absent_from_collocation_inventory=sorted(set(cal_keys) - set(original_inventory.site_key)),
                  oof_repeats=repeats, checks="Exact union, identities, partitions, saved audit hashes, "
                  "reserved disjointness, byte copies, exact data roundtrip and source before/after hashes",
                  assumptions_and_limits=LIMITS)
    plan = Plan(clean, attribution, copies, sources, evidence,
                dict(artifacts=mapping, models_included=False, publication_authorized=False), report)
    plan.unchanged()
    return plan


def output_guard(output, source_root, inputs):
    raw = Path(output).absolute()
    require(not any(p.is_symlink() for p in (raw, *raw.parents)), "Symlink output forbidden")
    output = raw.resolve()
    code_root = Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(code_root)
            and not any((p / ".git").is_file() for p in (output, *output.parents)),
            "Output must be outside the code/linked worktrees")
    require(not any(p.name == "source" and (p / "01_instructions").is_dir()
                    for p in (output, *output.parents)), "Output overlaps author source")
    source_root = Path(source_root).resolve()
    require(not output.is_relative_to(source_root) and not source_root.is_relative_to(output),
            "Output overlaps immutable source")
    require(not any(Path(p).resolve().is_relative_to(output) for p in inputs), "Output contains an input")
    return output


def expected_text(plan):
    readme = "# Minimal Reproducibility Preparation\n\nPRIVATE DRAFT. NOT APPROVED FOR PUBLICATION.\n\n"
    readme += "\n".join("- " + item for item in LIMITS) + "\n\n"
    readme += ("## Contents and Verification\n\n"
               "`collocations.parquet` is the exact union of the supplied artifact memberships. "
               "`site_attribution.csv` retains inventory fields, with basename-only PI evidence. "
               "`calibration/` and `memberships/` contain byte-preserved sources. "
               "`artifact_membership_mapping.json` links each artifact to copied membership and "
               "external deployment hashes. `source_evidence.json` records input anchors using "
               "logical paths, not private filesystem paths. `verification.json` records counts "
               "and limitations. `manifest.json` hashes every payload file except itself; it is "
               "written last, and is not a digital signature. Verify with the generic packager's "
               "`--verify-only` and the same private input arguments. No models are loaded.\n")
    return {"README.md": readme, "artifact_membership_mapping.json": packed(plan.mapping),
            "source_evidence.json": packed(dict(inputs=plan.evidence, packager_sha256=digest(__file__))),
            "verification.json": packed(plan.report)}


def verify_payload(output, plan):
    equal(pd.read_parquet(output / "collocations.parquet"), plan.collocations, "Collocation roundtrip changed data")
    equal(pd.read_csv(output / "site_attribution.csv", keep_default_na=False, float_precision="round_trip"),
          plan.attribution, "Attribution roundtrip changed data")
    for target, source in plan.copies.items():
        require(digest(output / target) == plan.sources[source.resolve()], f"Copy changed bytes: {target}")
    for target, content in expected_text(plan).items():
        require((output / target).read_text() == content, f"Metadata mismatch: {target}")
    plan.unchanged()


def verify_archive(output_dir, **inputs):
    """Revalidate sources, payload semantics, exact bytes, and the exhaustive manifest."""
    output = Path(output_dir).resolve()
    plan = inspect_inputs(**inputs)
    require(not any(p.is_symlink() for p in output.rglob("*")), "Archive contains a symlink")
    manifest = json.loads((output / "manifest.json").read_text())
    expected = {*plan.copies, *expected_text(plan), "collocations.parquet", "site_attribution.csv"}
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    require(actual == expected | {"manifest.json"} and set(manifest["files"]) == expected,
            "Archive manifest file set mismatch")
    for name in expected:
        path = output / name
        require(manifest["files"][name] == dict(sha256=digest(path), bytes=path.stat().st_size),
                f"Archive manifest mismatch: {name}")
    verify_payload(output, plan)
    return plan.report


def prepare(output_dir, **inputs):
    output = output_guard(output_dir, inputs["source_root"],
                          [inputs[k] for k in ("provenance_path", "scope_path", "inventory_path", "oof_path")])
    if output.exists():
        raise FileExistsError(output)
    plan = inspect_inputs(**inputs)
    output_guard(output, inputs["source_root"], plan.sources)
    output.mkdir(parents=True, exist_ok=False)
    try:
        plan.collocations.to_parquet(output / "collocations.parquet", index=False)
        plan.attribution.to_csv(output / "site_attribution.csv", index=False)
        for target, source in plan.copies.items():
            destination = output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        for target, content in expected_text(plan).items():
            private_text_check(content)
            with (output / target).open("x") as stream:
                stream.write(content)
        verify_payload(output, plan)
        files = {str(p.relative_to(output)): dict(sha256=digest(p), bytes=p.stat().st_size)
                 for p in sorted(output.rglob("*")) if p.is_file()}
        with (output / "manifest.json").open("x") as stream:
            stream.write(packed(dict(schema_version=1, files=files)))
    except BaseException:
        # Only this exclusively created directory is ours to remove on failure.
        shutil.rmtree(output)
        raise
    return plan.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "provenance-path", "scope-path", "inventory-path", "oof-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expected-artifacts", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = vars(parser.parse_args())
    verify = args.pop("verify_only")
    print(packed((verify_archive if verify else prepare)(**args)), end="")


if __name__ == "__main__":
    main()
