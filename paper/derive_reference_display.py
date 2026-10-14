"""Extract frozen paired fractions for S3; no target recomputation or fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import Registry, digest


TABLE = "grouped_validation/target_changes_stratified"
POPULATION = "all_noncalibration"
TARGETS = ["old675LR56", "experimental1020"]
LABEL_THRESHOLD = .2  # Frozen prepare.py/finish.py rule, not an adjustable plot parameter.
COLUMNS = ["row_id", "site_key", "calibration_reserved", *TARGETS, "region", "AOD_bin", "platform",
           "experimental1020_native_raw", "experimental1020_final_raw"]
COUNT_METRICS = ["rows", "label_changed", "old_to_new_positive", "old_to_new_negative",
                 "native_clip_high", "final_clip_high"]
FLOAT_METRICS = ["mean_fraction_change", "fraction_change_rmse"]
SEMANTICS = (
    "Identical paired collocations: saved calibration_reserved=False and both saved targets finite, "
    "as in prepare.py/finish.py. One row per row_id, not OOF prediction events, repeats, sites or "
    "independent observations. No MODIS-feature/AOD eligibility filter added. Saved final clipped "
    "fractions copied unchanged: old675LR56 is the frozen 675-nm/LR56 construction without a new "
    "wavelength transfer; experimental1020 includes the saved transfer to 550 nm. Existing >=0.2 "
    "labels checked only to reproduce saved summaries. Regions are the original first-match "
    "collocation partition, not Stage C climate boxes. Reference differences are sensitivity, "
    "not calibrated uncertainty or evidence selecting a reference. No bootstrap, fitting, new "
    "threshold, target recalculation or re-clipping."
)


def guard_output(output, inputs=()):
    output = Path(output).resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Unpublished output must be outside the public worktree")
    if any(p.name == "source" and (p / "01_instructions").is_dir() for p in (output, *output.parents)):
        raise ValueError("Do not write into author source folders")
    if any(output == Path(p).resolve() or output.is_relative_to(Path(p).resolve().parent) for p in inputs):
        raise ValueError("Output must be separate from frozen input directories")


def summarize(frame):
    old, new = (frame[c].ge(LABEL_THRESHOLD) for c in TARGETS)
    delta = frame.experimental1020 - frame.old675LR56
    return {"rows": len(frame), "label_changed": int((old != new).sum()),
            "old_to_new_positive": int((~old & new).sum()),
            "old_to_new_negative": int((old & ~new).sum()),
            "mean_fraction_change": float(delta.mean()),
            "fraction_change_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "native_clip_high": int(frame.experimental1020_native_raw.gt(1).sum()),
            "final_clip_high": int(frame.experimental1020_final_raw.gt(1).sum())}


def verify_pairs(pairs, registry, metadata=None):
    """Fail closed on cohort/count mismatches; float64 summary tolerance is 1e-12 absolute."""
    if pairs.empty or pairs.row_id.isna().any() or pairs.row_id.duplicated().any():
        raise ValueError("Empty or ambiguous paired row_id cohort")
    if pairs.calibration_reserved.isna().any() or not pairs.calibration_reserved.eq(False).all():
        raise ValueError("Paired cohort includes reserved calibration rows")
    if pairs[["site_key", "region", "AOD_bin", "platform"]].isna().any().any():
        raise ValueError("Missing paired-cohort identity/stratum")
    if not np.isfinite(pairs[TARGETS].to_numpy(float)).all() or not pairs[TARGETS].apply(
            lambda x: x.between(0, 1)).all().all():
        raise ValueError("Saved fractions must already be finite and clipped; do not re-clip")
    saved = registry.table(TABLE, population=POPULATION)
    if saved.duplicated(["scope", "group"]).any():
        raise ValueError("Ambiguous saved strata")
    expected = {("overall", "all")}
    expected.update((scope, str(group)) for scope in ("region", "AOD_bin", "platform")
                    for group in pairs[scope].unique())
    if set(zip(saved.scope, saved.group)) != expected:
        raise ValueError("Saved strata and paired cohort differ")
    maximum_error = 0.
    for _, row in saved.iterrows():
        sub = pairs if row.scope == "overall" else pairs[pairs[row.scope].eq(row.group)]
        observed = summarize(sub)
        for metric in COUNT_METRICS:
            if observed[metric] != row[metric]:
                raise ValueError(f"Saved count mismatch: {row.scope}/{row.group}/{metric}")
        for metric in FLOAT_METRICS:
            error = abs(observed[metric] - row[metric])
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError(f"Saved fraction summary mismatch: {row.scope}/{row.group}/{metric}")
            maximum_error = max(maximum_error, error)
    if metadata is not None:
        overall = summarize(pairs)
        checks = {"paired_rows": len(pairs), "label_changed": overall["label_changed"],
                  "old_positive": int(pairs.old675LR56.ge(LABEL_THRESHOLD).sum()),
                  "experimental_positive": int(pairs.experimental1020.ge(LABEL_THRESHOLD).sum())}
        for name, value in checks.items():
            if metadata[name] != value:
                raise ValueError(f"target_changes.json count mismatch: {name}")
        for name, value in {"label_changed_fraction": overall["label_changed"] / len(pairs),
                            "fraction_mean_change": overall["mean_fraction_change"],
                            "fraction_RMSE_change": overall["fraction_change_rmse"]}.items():
            if not np.isfinite(metadata[name]) or abs(metadata[name] - value) > 1e-12:
                raise ValueError(f"target_changes.json summary mismatch: {name}")
    return {"paired_rows": len(pairs), "paired_sites": int(pairs.site_key.nunique()),
            "regions": int(pairs.region.nunique()), "verified_strata": len(saved),
            "maximum_fraction_summary_absolute_error": maximum_error,
            "summary_absolute_tolerance": 1e-12, "counts_exact": True}


def extract(rows, registry, metadata):
    if not pd.api.types.is_bool_dtype(rows.calibration_reserved.dtype) or rows.calibration_reserved.isna().any():
        raise ValueError("Saved calibration reservation must be nonmissing boolean")
    if rows.row_id.isna().any() or rows.row_id.duplicated().any():
        raise ValueError("Cached row_id is missing or duplicated")
    mask = ~rows.calibration_reserved & np.isfinite(rows[TARGETS].to_numpy(dtype=float)).all(axis=1)
    pairs = rows.loc[mask, COLUMNS].copy()
    pairs.insert(0, "source_record_index", np.flatnonzero(mask))
    checks = verify_pairs(pairs, registry, metadata)
    pairs["population"] = POPULATION
    return pairs.reset_index(drop=True), checks


def derive(source_root, registry_path, output_dir):
    root = Path(source_root).resolve() / "grouped_validation"
    registry_path = Path(registry_path).resolve()
    inputs = [root / "rows.parquet", root / "target_changes.json", root / "target_changes_stratified.csv",
              root / "prepare.py", root / "finish.py", registry_path]
    output = Path(output_dir).resolve()
    guard_output(output, inputs)
    hashes = {str(path): digest(path) for path in inputs}
    registry = Registry.read(registry_path)
    table = registry.payload["tables"][TABLE]
    if table["source_sha256"] != hashes[str(root / "target_changes_stratified.csv")]:
        raise ValueError("Registry stratified source hash differs from frozen input")
    pairs, checks = extract(pd.read_parquet(inputs[0], columns=COLUMNS), registry,
                            json.loads(inputs[1].read_text()))
    if any(digest(path) != hashes[str(path)] for path in inputs):
        raise RuntimeError("Frozen input changed during extraction")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reference_pairs.csv"
    pairs.to_csv(path, index=False)
    sidecar = {"sources": hashes, "registry_sha256": hashes[str(registry_path)],
               "output_sha256": digest(path), "script_sha256": digest(Path(__file__)),
               "semantics": SEMANTICS, "selection": "~calibration_reserved & finite(old675LR56, experimental1020)",
               "label_threshold": LABEL_THRESHOLD, "checks": checks,
               "source_row_index": "Zero-based row position in grouped_validation/rows.parquet"}
    path.with_suffix(".provenance.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
    return sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(derive(args.source_root, args.registry, args.output_dir)["checks"], indent=2))


if __name__ == "__main__":
    main()
