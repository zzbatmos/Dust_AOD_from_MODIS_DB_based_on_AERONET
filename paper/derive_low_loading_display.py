"""S15 arrays from provenance-selected monthly aggregates, never raw pixels.

Verify the saved MODIS-only bitmap, reproduce original float64 weighted
cell-month summaries, and retain calendar-2017 available-month map means.
Contributions retain the FULL retrieval-count denominator; they are not
subset means. Amazon DJF and dual-reference low-dust selections are distinct.
No source-main imports, inference, fitting, bootstrap or downloads.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper import derive_seasonal_maps as base
from paper.numbers import digest

META = dict(reference=base.REFERENCE, variant="hierarchy", support="core_common")
SUBSETS = {"AOD550_lt_0p2": "low_aod_contribution", "P_lt_0p3": "low_p_contribution"}
REGIMES = ("Amazon_DJF", "dual_reference_low_dust")
SEMANTICS = {
    "low_loading_maps": "Eselected contribution on saved MODIS-only core_common support, Land "
    "60S-60N. Each native 2x5 cell is an arithmetic mean over its available calendar-2017 "
    "months; month_count is explicit, with no fixed-year intersection. Zero supported "
    "contribution remains zero; no months means missing. Monthly contributions use the full "
    "paired retrieval-count denominator, NOT a subset denominator. These spatial map means "
    "are NOT the global area-weighted cell-month summaries. The two subsets can overlap "
    "and must not be added. P<0.3 excludes missing P in the frozen input recipe.",
    "low_loading_summary": "All saved primary/hierarchy/core_common regional, estimator and "
    "subset rows from low_loading_contributions.csv, independently reproduced from monthly "
    "fields on saved MODIS-only calendar-2017 support with original float64 area weights. "
    "Quantiles describe cell-month distributions, not confidence intervals. No bootstrap.",
    "low_loading_floor_samples": "Individual saved estimator cell-month values on two "
    "DIFFERENT supports: Amazon_DJF is MODIS-only within recorded Amazonia_Cerrado bounds "
    "for Dec2016-Feb2017; dual_reference_low_dust is calendar2017, Land60S60N, saved "
    "reference-mean-only support with BOTH transferred CALIOP and Song below the frozen "
    "0.005 criterion. Neither is selected by QA2=0, and neither requires prior_joint ancillary "
    "coverage. CALIOP/Song are comparisons, not truth. All five core estimators retained.",
    "low_loading_floor_summary": "All ten saved primary core background_floor rows, checked "
    "against selected cell-months. Saved quantiles are distribution ranges, not uncertainty "
    "intervals. A single eligible cell-month is not a population-level floor test.",
    "low_loading_floor_audit": "Exact primary floor_selection_audit counts reproduced at "
    "each saved selection step. Spatial-cell counts differ from cell-month counts. Frozen "
    "low-dust and descriptive floor thresholds are read from the hash-verified comparison "
    "source, not newly selected. No QA2=0 filter.",
}


def frozen_recipe(script):
    """Read relevant existing literals without executing the source module."""
    tree = ast.parse(Path(script).read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "lowdust" for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("Ambiguous saved low-dust selection")
    limits = {}
    for node in ast.walk(assignments[0].value):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id in ("cal", "song") and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Lt)):
            limits[node.left.id] = ast.literal_eval(node.comparators[0])
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "distribution"]
    if len(functions) != 1 or set(limits) != {"cal", "song"} or limits["cal"] != limits["song"]:
        raise ValueError("Unrecognized frozen distribution/dual-reference recipe")
    flags, quantiles = [], []
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "x"
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt)):
            flags.append(ast.literal_eval(node.comparators[0]))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "quantile"):
            quantiles.append(ast.literal_eval(node.args[1]))
    if len(flags) != 1 or sorted(quantiles) != [.05, .95]:
        raise ValueError("Unexpected saved distribution statistics")
    return dict(low_dust_threshold=limits["cal"], descriptive_floor_threshold=flags[0],
                q05_probability=min(quantiles), q95_probability=max(quantiles))


def distribution(values, area, mask, recipe):
    """Same reduction order/weighted-quantile convention as compare_global."""
    x = values[mask]
    w = np.broadcast_to(area, values.shape)[mask]
    if not len(x):
        return {"cell_months": 0}
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid selected values or original area weights")
    order = np.argsort(x)
    sx, sw = x[order], w[order]
    return dict(cell_months=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q05=float(np.quantile(x, recipe["q05_probability"])),
                q95=float(np.quantile(x, recipe["q95_probability"])),
                area_weighted_mean=float(np.average(x, weights=w)),
                area_weighted_median=float(np.interp(.5, (np.cumsum(sw) - .5 * sw) / sw.sum(), sx)),
                fraction_above_0p01=float((x > recipe["descriptive_floor_threshold"]).mean()))


def region_mask(lat, lon, bounds):
    lo, hi, left, right = bounds
    return ((lat[:, None] >= lo) & (lat[:, None] < hi)
            & (lon[None, :] >= left) & (lon[None, :] < right))


def derive(source_root, land_shapefile=base.DEFAULT_LAND):
    root = Path(source_root).resolve()
    for prefix, key in (("AOD550_lt_", "low_aod_threshold"), ("P_lt_", "low_probability_threshold")):
        if prefix + str(CONFIG["diagnostics"][key]).replace(".", "p") not in SUBSETS:
            raise ValueError("Frozen contribution labels differ from configured thresholds")
    initial, sources, checks = base.derive(root, land_shapefile)

    def track(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = digest(path)
        return path

    track(base.__file__)
    stage = root / "expected_value_test/stage_c"
    provenance = json.loads((stage / "comparison_provenance.json").read_text())
    script = root / "expected_value_test/compare_global.py"
    recipe = frozen_recipe(script)
    periods, core = base.source_literals(script)
    template = initial["seasonal_maps"].loc[initial["seasonal_maps"].period.eq("DJF")].copy()
    lat, lon = template.lat.unique(), template.lon.unique()
    shape = (len(lat), len(lon))
    area = template.land_area_weight.to_numpy().reshape(shape)
    purity_ok = template.original_land_purity_eligible.to_numpy().reshape(shape)
    domain = template.in_domain.to_numpy().reshape(shape)
    with nc.Dataset(stage / "comparison_support.nc") as ds:
        months = json.loads(ds.months)
        group = ds.groups[base.GROUP]
        modis = base._mask(group, "modis_mask")
        mean_only = base._mask(group, "reference_mean_only_mask")
        counts, coverage = base.read(group["counts"]), base.read(group["coverage"])
        fields = {e: base.read(group[e]) for e in core}
        factor = base.spectral_factor(ds.CALIOP_spectral_transfer)
    expected_modis = purity_ok[None] & (coverage >= provenance["fine_land_coverage"]) & (counts > 0)
    for values in fields.values():
        expected_modis &= np.isfinite(values)
    if not np.array_equal(modis, expected_modis):
        raise ValueError("Saved MODIS-only bitmap does not match frozen support recipe")

    paths = {}
    for recorded, fingerprint in provenance["sources"].items():
        match = re.fullmatch(r"Aqua_EV_(\d{4}-\d{2})_2x5.nc", Path(recorded).name)
        if not match:
            continue
        month = match.group(1)
        if month in paths:
            raise ValueError("Duplicate provenance-selected monthly file")
        # Original receipts recorded some paths relative to the workspace root.
        path = Path(recorded)
        path = track(path if path.is_absolute() else root.parents[1] / path)
        if sources[str(path)] != fingerprint:
            raise ValueError("Monthly source hash mismatch")
        paths[month] = path
    if set(paths) != set(months):
        raise ValueError("Require exact Dec2016 plus calendar2017 monthly sources")
    low = {label: {e: [] for e in core} for label in SUBSETS}
    fingerprints = set()
    for index, month in enumerate(months):
        path = paths[month]
        receipt = json.loads(track(path.with_suffix(".sha256.json")).read_text())
        with nc.Dataset(path) as ds:
            if (ds.month != month or receipt["sha256"] != sources[str(path)]
                    or receipt["fingerprint"] != ds.fingerprint):
                raise ValueError("Monthly receipt/month/fingerprint mismatch")
            fingerprints.add(ds.fingerprint)
            if not np.array_equal(base.read(ds["lat"]), lat) or not np.array_equal(base.read(ds["lon"]), lon):
                raise ValueError("Monthly native coordinates differ")
            group = ds.groups[META["reference"]].groups[META["variant"]]
            for e in core:
                prefix = "core_common__" + e + "__"
                for suffix, expected in (("count", counts[index]), ("mean", fields[e][index])):
                    variable = group[prefix + suffix]
                    if variable.dimensions != ("lat", "lon") or not np.array_equal(base.read(variable), expected, equal_nan=True):
                        raise ValueError("Monthly/cache field or count mismatch: " + prefix + suffix)
                for label, suffix in SUBSETS.items():
                    variable = group[prefix + suffix]
                    if variable.dimensions != ("lat", "lon") or variable.denominator != prefix + "count":
                        raise ValueError("Contribution is not on the full paired count denominator")
                    value = base.read(variable)
                    selected = value[modis[index]]
                    if (not np.isfinite(selected).all() or (selected < 0).any()
                            or (selected > fields[e][index][modis[index]] + base.CHECK_ATOL).any()):
                        raise ValueError("Invalid contribution on saved MODIS support")
                    low[label][e].append(value)
    if len(fingerprints) != 1:
        raise ValueError("Mixed monthly estimator/code fingerprints")
    low = {label: {e: np.stack(v) for e, v in methods.items()} for label, methods in low.items()}
    cube = next(Path(p) for p in provenance["sources"] if Path(p).name == "matched_native_grid.nc")
    with nc.Dataset(cube) as ds:
        years = base.read(ds["year"])
        external = {}
        for name in ("cal_mean", "old"):
            external[name] = np.stack([np.asarray(np.ma.filled(
                ds[name][int(np.flatnonzero(years == int(m[:4]))[0]), int(m[5:]) - 1].astype(float), np.nan))
                for m in months])
    cal, song = external["cal_mean"] * factor, external["old"]
    external_valid = np.isfinite(cal) & np.isfinite(song) & (cal >= 0) & (song >= 0)
    if not np.array_equal(mean_only, modis & external_valid):
        raise ValueError("Saved mean-only support differs from frozen external availability")
    in2017 = np.array([m.startswith("2017-") for m in months])[:, None, None]
    external_low = (external_valid & (cal < recipe["low_dust_threshold"])
                    & (song < recipe["low_dust_threshold"]) & in2017 & domain[None])
    amazon = (modis & region_mask(lat, lon, provenance["regions"]["Amazonia_Cerrado"])[None]
              & np.array([m in periods["DJF"] for m in months])[:, None, None])
    regimes = dict(Amazon_DJF=amazon, dual_reference_low_dust=external_low & mean_only)

    def saved(name, keys, expected):
        frame = pd.read_csv(track(stage / (name + ".csv")), float_precision="round_trip")
        for key, value in META.items():
            frame = frame.loc[frame[key].eq(value)]
        if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError("Missing/duplicate/extra saved diagnostic rows: " + name)
        return frame.copy().reset_index(drop=True)

    loading = saved("low_loading_contributions", ["estimator", "region", "subset"],
                    {(e, r, s) for e in core for r in provenance["regions"] for s in SUBSETS})
    floors = saved("background_floor", ["estimator", "regime"], {(e, r) for e in core for r in REGIMES})
    audit = saved("floor_selection_audit", ["reference"], {(base.REFERENCE,)})

    def compare(stats, row, label):
        for key in ("cell_months", "mean", "median", "q05", "q95", "area_weighted_mean",
                    "area_weighted_median", "fraction_above_0p01"):
            base._check(stats.get(key, np.nan), row[key], label + "/" + key, checks, exact=key == "cell_months")

    for _, row in loading.iterrows():
        use = modis & in2017 & region_mask(lat, lon, provenance["regions"][row.region])[None]
        label = f"loading/{row.estimator}/{row.region}/{row.subset}"
        compare(distribution(low[row.subset][row.estimator], area, use, recipe), row, label)
        base._check(distribution(fields[row.estimator], area, use, recipe).get("area_weighted_mean", np.nan),
                    row.full_mean_DAOD, label + "/full_mean_DAOD", checks)
    samples = []
    for _, row in floors.iterrows():
        use = regimes[row.regime]
        stats = distribution(fields[row.estimator], area, use, recipe)
        compare(stats, row, f"floor/{row.estimator}/{row.regime}")
        base._check(stats.get("median", np.nan) > recipe["descriptive_floor_threshold"],
                    row.median_floor_flag, f"floor/{row.estimator}/{row.regime}/flag", checks, exact=True)
        t, iy, ix = np.where(use)
        samples.append(pd.DataFrame(dict(month=np.asarray(months)[t], lat_index=iy, lon_index=ix,
            lat=lat[iy], lon=lon[ix], value=fields[row.estimator][use], caliop=cal[use], song=song[use],
            land_area_weight=area[iy, ix])).assign(**META, estimator=row.estimator, regime=row.regime,
            reference_mask="modis_only" if row.regime == "Amazon_DJF" else "reference_mean_only_low_dust"))
    audit_values = dict(external_low_cells_before_land_modis=int(external_low.sum()),
        external_low_after_land_purity=int((external_low & purity_ok[None]).sum()),
        paired_low_dust_cell_months=int(regimes["dual_reference_low_dust"].sum()),
        paired_low_dust_spatial_cells=int(regimes["dual_reference_low_dust"].any(axis=0).sum()),
        Amazon_DJF_cell_months=int(amazon.sum()), Amazon_DJF_spatial_cells=int(amazon.any(axis=0).sum()))
    for key, value in audit_values.items():
        base._check(value, audit.iloc[0][key], "floor_audit/" + key, checks, exact=True)
    audit = audit.assign(**recipe)

    use = modis & in2017 & domain[None]
    month_count = use.sum(axis=0)
    geometry = ["lat_index", "lon_index", "lat", "lon", "lat_lower", "lat_upper", "lon_lower", "lon_upper",
                "in_domain", "has_land", "land_area_weight"]
    maps = []
    for subset in SUBSETS:
        values = low[subset]["Eselected"]
        mean = np.divide(np.where(use, values, 0).sum(axis=0), month_count,
                         out=np.full(shape, np.nan), where=month_count > 0)
        frame = template[geometry].copy()
        frame["value"], frame["month_count"] = mean.ravel(), month_count.ravel()
        frame["supported"] = month_count.ravel() > 0
        threshold = CONFIG["diagnostics"]["low_aod_threshold" if subset == "AOD550_lt_0p2" else "low_probability_threshold"]
        maps.append(frame.assign(**META, estimator="Eselected", subset=subset, threshold=threshold,
                                 period="2017", reference_mask="modis_only", denominator="full_paired_count"))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError("Frozen input changed during derivation: " + path)
    return dict(low_loading_maps=pd.concat(maps, ignore_index=True), low_loading_summary=loading,
                low_loading_floor_samples=pd.concat(samples, ignore_index=True), low_loading_floor_summary=floors,
                low_loading_floor_audit=audit), sources, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--land-shapefile", type=Path, default=base.DEFAULT_LAND)
    args = parser.parse_args(argv)
    output, source = args.output_dir.resolve(), args.source_root.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.is_relative_to(source):
        raise ValueError("Write private outputs outside frozen sources/public worktree")
    targets = [output / (name + suffix) for name in SEMANTICS for suffix in (".csv", ".provenance.json")]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError("S15 outputs already exist; choose a fresh output directory")
    tables, sources, checks = derive(source, args.land_shapefile)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".low-loading-", dir=output) as temporary:
        for name, frame in tables.items():
            path = Path(temporary) / (name + ".csv")
            frame.to_csv(path, index=False)
            path.with_suffix(".provenance.json").write_text(json.dumps(dict(output_sha256=digest(path),
                sources=sources, semantics=SEMANTICS[name], script_sha256=digest(Path(__file__)),
                rows=len(frame), checks=checks, numerical_check_atol=base.CHECK_ATOL), indent=2, allow_nan=False) + "\n")
        for path in sorted(Path(temporary).iterdir()):
            os.replace(path, output / path.name)
    print(f"Wrote S15 tables: { {name: len(frame) for name, frame in tables.items()} }; {len(checks)} checks passed")


if __name__ == "__main__":
    main()
