"""S1 fixed-formula archive display, not a recalibration or validation experiment.

Read the saved inversion cache in chunks and retain distinct wavelength-specific
reference_valid cohorts. Verify source formula/masks without executing source
mains; compare native fractions against saved grouped rows and tail statistics
against the original CSV. Native 1020 only: no 550-nm transfer, fitting, or SDA
conversion. These corrected display arrays await author acceptance.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from dbdust.config import CONFIG, config_path
from paper.numbers import digest

WAVES = (675, 1020)
ATOL = 1e-12  # Numerical reproduction tolerance, not a scientific threshold.
SEMANTICS = {
    "coarse_fraction_display": "Corrected candidate S1, author acceptance pending. Distinct "
    "wavelength-specific reference_valid inversion archive cohorts, not paired wavelengths, "
    "the historical six-site cohort, or held-out ML validation. cmf is inversion coarse/total "
    "EXTINCTION AOD, not SDA. Native1020 uses frozen experimental1020 LR47.308 and PLDR "
    "0.02/0.30; native675 uses old LR56, explicitly sensitivity. Clip backscatter fraction "
    "FIRST, multiply by fixed/observed LR, then clip extinction fraction. No 550 transfer. "
    "Original CMF retained without clipping, including >0.85. source_row is the zero-based "
    "full-cache row, so IDs may occur in both distinct wavelength cohorts. No recalibration.",
    "coarse_fraction_tail": "Exactly the two saved reference_valid tail rows, independently "
    "reproduced from the full inversion cache using original wavelength-specific masks. "
    "Strict >0.85 before clipping. Not all_archive or six_site_paired. Counts, fraction, "
    "p95, p99 and maximum checked. This establishes a tail, not its physical attribution.",
    "coarse_fraction_formula_checks": "Reconstruction of saved grouped_validation/rows.parquet "
    "old675LR56, experimental1020_native_raw and experimental1020_native. This checks the "
    "fixed recipe on the separate collocation table; it is NOT a join or claim that the "
    "archive display cohort is held-out validation. All rows, including calibration-reserved "
    "rows, are used solely for arithmetic verification. No recalibration or 550 transfer.",
}


def assignments(tree, name):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if ((isinstance(target, ast.Name) and target.id == name)
                        or (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id == "df" and isinstance(target.slice, ast.Constant)
                            and target.slice.value == name)):
                    found.append(node.value)
    if len(found) != 1:
        raise ValueError("Missing/ambiguous source expression: " + name)
    return found[0]


def require_expression(node, expected, label):
    parsed = ast.parse(expected, mode="eval").body
    if ast.dump(node) != ast.dump(parsed):
        raise ValueError("Frozen source recipe changed: " + label)


def source_recipe(prepare_path, comparison_path, calibration):
    """Inspect AST only; never import the preparation/calibration source modules."""
    reference = CONFIG["reference"]
    if (calibration["median_LR1020_sr"] != reference["dust_lidar_ratio_sr"]
            or calibration["endmembers"] != [reference["nondust_pldr"], reference["dust_pldr"]]
            or not reference["clip_backscatter_before_lidar_conversion"]
            or reference["wavelength_nm"] != WAVES[1]):
        raise ValueError("Saved calibration/configuration disagreement")
    nondust, dust = reference["nondust_pldr"], reference["dust_pldr"]
    # Match source literal .28, rather than introducing subtraction roundoff.
    span = float(Decimal(str(dust)) - Decimal(str(nondust)))
    multiplier = float(Decimal(1) + Decimal(str(dust)))
    tree = ast.parse(Path(prepare_path).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "backfraction"]
    if len(functions) != 1 or len(functions[0].body) != 1 or not isinstance(functions[0].body[0], ast.Return):
        raise ValueError("Unrecognized source backfraction function")
    require_expression(functions[0].body[0].value,
        f"np.clip((delta-{nondust!r})*{multiplier!r}/({span!r}*(1+delta)),0,1)", "backfraction")
    old = assignments(tree, "old675LR56")
    try:
        old_lr = ast.literal_eval(old.args[0].left.right)
    except (AttributeError, ValueError, IndexError) as exc:
        raise ValueError("Unrecognized old675 fixed LR formula") from exc
    if old_lr <= 0:
        raise ValueError("Invalid historical fixed LR")
    require_expression(old, f"np.clip(backfraction(df.AER_PLDR675)*{old_lr!r}/df.AER_LR675,0,1)", "old675")
    require_expression(assignments(tree, "experimental1020_native_raw"),
                       "backfraction(df.AER_PLDR1020)*lr/df.AER_LR1020", "native1020 raw")
    require_expression(assignments(tree, "experimental1020_native"),
                       "df.experimental1020_native_raw.clip(0,1)", "native1020 clip")
    comparison = ast.parse(Path(comparison_path).read_text())
    validcmf = assignments(comparison, "validcmf")
    calls = [n for n in ast.walk(validcmf) if isinstance(n, ast.Call)]
    if len(calls) != 1:
        raise ValueError("Unrecognized reference_valid CMF mask")
    lower, upper = map(ast.literal_eval, calls[0].args)
    require_expression(validcmf, f"(total>0)&(coarse>=0)&cmf.between({lower!r},{upper!r})", "validcmf")
    valid = assignments(comparison, "reference_valid")
    bounds = {}
    for node in ast.walk(valid):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "between":
            key = ast.unparse(node.func.value)
            bounds["lr" if "Lidar_Ratio" in key else "pldr"] = tuple(map(ast.literal_eval, node.args))
    if set(bounds) != {"lr", "pldr"}:
        raise ValueError("Missing reference-valid LR/PLDR bounds")
    lrlo, lrhi = bounds["lr"]
    dlo, dhi = bounds["pldr"]
    require_expression(valid,
        f"validcmf & data[f'Lidar_Ratio[{{wave}}nm]'].between({lrlo!r},{lrhi!r}) & "
        f"data[f'Depolarization_Ratio[{{wave}}nm]'].between({dlo!r},{dhi!r})", "reference_valid")
    tail_records = [n for n in ast.walk(comparison) if isinstance(n, ast.Dict)
                    and any(isinstance(k, ast.Constant) and k.value == "n_gt_0p85" for k in n.keys)]
    if len(tail_records) != 1:
        raise ValueError("Missing/ambiguous saved tail statistics")
    record = {k.value: v for k, v in zip(tail_records[0].keys, tail_records[0].values) if isinstance(k, ast.Constant)}
    comparisons = [n for n in ast.walk(record["n_gt_0p85"]) if isinstance(n, ast.Compare)]
    if len(comparisons) != 1 or not isinstance(comparisons[0].ops[0], ast.Gt):
        raise ValueError("Tail count is not a strict exceedance")
    threshold = ast.literal_eval(comparisons[0].comparators[0])
    require_expression(record["n_gt_0p85"], f"int((values>{threshold!r}).sum())", "tail count")
    require_expression(record["fraction_gt_0p85"], f"float((values>{threshold!r}).mean())", "tail fraction")
    quantiles = {name: ast.literal_eval(record[name].args[0]) for name in ("p95", "p99")}
    return dict(nondust_pldr=nondust, dust_pldr=dust, backscatter_multiplier=multiplier,
                backscatter_span=span, primary_lr_sr=reference["dust_lidar_ratio_sr"], old_lr_sr=old_lr,
                cmf_lower=lower, cmf_upper=upper, lr_lower=lrlo, lr_upper=lrhi, pldr_lower=dlo, pldr_upper=dhi,
                tail_threshold=threshold, **quantiles)


def native_fraction(pldr, observed_lr, fixed_lr, recipe):
    """Return raw/clipped backscatter, raw/clipped native extinction fractions."""
    pldr, observed_lr = np.asarray(pldr, float), np.asarray(observed_lr, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_backscatter = ((pldr - recipe["nondust_pldr"]) * recipe["backscatter_multiplier"]
                           / (recipe["backscatter_span"] * (1 + pldr)))
        backscatter = np.clip(raw_backscatter, 0, 1)
        raw_extinction = backscatter * fixed_lr / observed_lr
    return raw_backscatter, backscatter, raw_extinction, np.clip(raw_extinction, 0, 1)


def derive(source_root, chunksize=100000):
    root = Path(source_root).resolve()
    group = root / "grouped_validation"
    sources, checks = {}, []

    def track(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = digest(path)
        return path

    def check(actual, expected, label, exact=False):
        good = actual == expected if exact else np.isclose(actual, expected, atol=ATOL, rtol=0, equal_nan=True)
        if not good:
            raise ValueError(f"Saved-data mismatch: {label}: {actual!r} != {expected!r}")
        checks.append(dict(check=label, status="PASS"))

    track(config_path())
    provenance = json.loads(track(group / "source_provenance.json").read_text())
    cache_paths = [Path(p) for p in provenance if Path(p).name == "aeronet_narrow_cache.csv.gz"]
    if len(cache_paths) != 1 or not cache_paths[0].is_absolute():
        raise ValueError("Require unique provenance-selected inversion cache")
    cache = track(cache_paths[0])
    if sources[str(cache)] != provenance[str(cache_paths[0])]["sha256"]:
        raise ValueError("Inversion cache hash mismatch")
    calibration = json.loads(track(group / "calibration.json").read_text())
    schema = json.loads(track(group / "schema.json").read_text())
    recipe = source_recipe(track(group / "prepare.py"), track(cache.parent.parent / "compare_shin.py"), calibration)
    saved_tail = pd.read_csv(track(cache.parent / "coarse_fraction_upper_tail.csv"), float_precision="round_trip")
    saved_tail = saved_tail.loc[saved_tail.subset.eq("reference_valid")].copy()
    if len(saved_tail) != len(WAVES) or saved_tail.wavelength.duplicated().any() or set(saved_tail.wavelength) != set(WAVES):
        raise ValueError("Require exact distinct reference_valid tail rows")
    target_names = ("old675LR56", "experimental1020_native_raw", "experimental1020_native")
    grouped = pd.read_parquet(track(group / "rows.parquet"), columns=["row_id", *target_names,
        "AER_PLDR675", "AER_LR675", "AER_PLDR1020", "AER_LR1020"])
    if grouped.row_id.duplicated().any():
        raise ValueError("Duplicate grouped-row identity")
    check(len(grouped), schema["rows"], "grouped/schema_rows", exact=True)
    old = native_fraction(grouped.AER_PLDR675, grouped.AER_LR675, recipe["old_lr_sr"], recipe)
    primary = native_fraction(grouped.AER_PLDR1020, grouped.AER_LR1020, recipe["primary_lr_sr"], recipe)
    formula_checks = []
    for target, actual in zip(target_names, (old[3], primary[2], primary[3])):
        expected = grouped[target].to_numpy(float)
        check(bool(np.array_equal(np.isnan(actual), np.isnan(expected))), True, target + "/missing", exact=True)
        if not np.allclose(actual, expected, rtol=0, atol=ATOL, equal_nan=True):
            raise ValueError("Grouped-row fixed formula mismatch: " + target)
        checks.append(dict(check=target + "/values", status="PASS"))
        finite = np.isfinite(expected)
        formula_checks.append(dict(saved_column=target, cohort="grouped_rows_formula_check_only", n=len(grouped),
            n_finite=int(finite.sum()), max_abs_error=float(np.max(abs(actual[finite] - expected[finite]))) if finite.any() else np.nan,
            author_acceptance="pending"))
    columns = ["Site", "date", "Time(hh:mm:ss)"] + [f"{field}[{w}nm]" for w in WAVES
        for field in ("AOD_Extinction-Total", "AOD_Extinction-Coarse", "Lidar_Ratio", "Depolarization_Ratio")]
    parts = []
    offset = 0
    for chunk in pd.read_csv(cache, usecols=columns, chunksize=chunksize):
        source_row = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        for wave in WAVES:
            total, coarse, lr, delta = (chunk[f"{name}[{wave}nm]"] for name in
                ("AOD_Extinction-Total", "AOD_Extinction-Coarse", "Lidar_Ratio", "Depolarization_Ratio"))
            with np.errstate(invalid="ignore", divide="ignore"):
                cmf = coarse / total
            use = ((total > 0) & (coarse >= 0) & cmf.between(recipe["cmf_lower"], recipe["cmf_upper"])
                   & lr.between(recipe["lr_lower"], recipe["lr_upper"])
                   & delta.between(recipe["pldr_lower"], recipe["pldr_upper"]))
            fixed_lr = recipe["primary_lr_sr"] if wave == WAVES[1] else recipe["old_lr_sr"]
            braw, b, fraw, f = native_fraction(delta[use], lr[use], fixed_lr, recipe)
            part = pd.DataFrame(dict(source_row=source_row[use], site=chunk.loc[use, "Site"].to_numpy(),
                date=chunk.loc[use, "date"].to_numpy(), time=chunk.loc[use, "Time(hh:mm:ss)"].to_numpy(),
                inversion_cmf=cmf[use].to_numpy(), pldr=delta[use].to_numpy(), observed_lr_sr=lr[use].to_numpy(),
                backscatter_raw=braw, backscatter_fraction=b, native_extinction_raw=fraw, native_extinction_fraction=f))
            parts.append(part.assign(wavelength=wave, reference="experimental1020_native" if wave == WAVES[1] else "old675LR56",
                role="primary_native" if wave == WAVES[1] else "sensitivity", cohort="reference_valid",
                fixed_lr_sr=fixed_lr, author_acceptance="pending"))
    check(offset, calibration["raw_cache_rows"], "archive/cache_rows", exact=True)
    points = pd.concat(parts, ignore_index=True).sort_values(["wavelength", "source_row"], ignore_index=True)
    if points.duplicated(["wavelength", "source_row"]).any():
        raise ValueError("Duplicate wavelength/source-row identity")
    for wave in WAVES:
        cmf = points.loc[points.wavelength.eq(wave), "inversion_cmf"]
        row = saved_tail.loc[saved_tail.wavelength.eq(wave)].iloc[0]
        measured = dict(n=len(cmf), n_gt_0p85=int((cmf > recipe["tail_threshold"]).sum()),
            fraction_gt_0p85=float((cmf > recipe["tail_threshold"]).mean()),
            p95=cmf.quantile(recipe["p95"]), p99=cmf.quantile(recipe["p99"]), maximum_before_clip=cmf.max())
        for name, value in measured.items():
            check(value, row[name], f"tail/{wave}/{name}", exact=name in ("n", "n_gt_0p85"))
    tail = saved_tail.assign(tail_threshold=recipe["tail_threshold"], cmf_upper=recipe["cmf_upper"],
                            sda_wavelength_nm=schema["SDA_wavelength_nm"], author_acceptance="pending")
    for path, fingerprint in sources.items():
        if digest(Path(path)) != fingerprint:
            raise RuntimeError("Source changed during extraction: " + path)
    return dict(coarse_fraction_display=points, coarse_fraction_tail=tail,
                coarse_fraction_formula_checks=pd.DataFrame(formula_checks)), sources, checks, recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args(argv)
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if output.is_relative_to(source) or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Write outside frozen sources/public worktree")
    targets = [output / (name + suffix) for name in SEMANTICS for suffix in (".csv", ".provenance.json")]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError("S1 outputs already exist; use a fresh private directory")
    tables, sources, checks, recipe = derive(source, args.chunk_size)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coarse-fraction-", dir=output) as temporary:
        for name, table in tables.items():
            path = Path(temporary) / (name + ".csv")
            table.to_csv(path, index=False)
            path.with_suffix(".provenance.json").write_text(json.dumps(dict(output_sha256=digest(path), sources=sources,
                semantics=SEMANTICS[name], script_sha256=digest(Path(__file__)), rows=len(table), checks=checks,
                fixed_recipe=recipe, numerical_check_atol=ATOL, author_acceptance="pending"), indent=2, allow_nan=False) + "\n")
        for path in sorted(Path(temporary).iterdir()):
            os.replace(path, output / path.name)
    print(f"S1 arrays: { {name: len(frame) for name, frame in tables.items()} }; {len(checks)} checks passed")


if __name__ == "__main__":
    main()
