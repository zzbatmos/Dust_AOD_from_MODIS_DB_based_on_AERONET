"""Derive S13 native maps from saved Stage C caches; no source-main execution.

Reuse the Fig6 preflight and original-float64 geometry, then validate every
primary seasonal/global metric on both saved supports. Display core methods,
Song and CALIOP on core_common, and E6 only on full_common_E6. E6's CALIOP
reference is averaged on its own three-month intersection, never substituted
from the core map. Output names do not overwrite the completed Fig6 arrays.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd

from paper import derive_seasonal_maps as base
from paper.numbers import digest

CORE = ("E1", "Eselected", "E5", "E7", "E8")
SUPPORT_METHODS = {"core_common": CORE, "full_common_E6": CORE + ("E6",)}
DISPLAY = {"core_common": CORE + ("Song", "CALIOP"), "full_common_E6": ("E6",)}
SEMANTICS = {
    "all_seasonal_maps": "Primary experimental1020 hierarchy; full native 90x72 2x5 grid per "
    "estimator/season. E1/Eselected/E5/E7/E8/Song/CALIOP use core_common; E6 alone uses "
    "full_common_E6, NOT a paired comparison with the core methods. Each support retains its "
    "saved prior_joint all-three-month intersection within Land 60S-60N. Equal three-month "
    "means; DJF uses December 2016. Song is the saved old field already at 550nm. CALIOP "
    "cal_mean is promoted to float64 and transferred from 532nm before averaging. value is "
    "candidate DAOD550; caliop is the reference on THAT row's support; difference=value-caliop. "
    "Unsupported science fields remain missing; supported zeros remain zero. Geometry/mask "
    "counts are diagnostics, never replacement masks. No interpolation, fits or downloads.",
    "all_seasonal_map_metrics": "All 60 saved primary/hierarchy/prior_joint/global seasonal "
    "rows across core_common and full_common_E6, including full-support methods not displayed. "
    "All seven statistics checked with original float64 land-area weights reconstructed by "
    "the read-only Fig6 helper and verified against both cached float32 geometry arrays. "
    "No rounded-weight or cosine-latitude approximation. n counts spatial cells, not retrievals "
    "or cell-months. display_map distinguishes the 32 maps selected for S13.",
}


def derive(source_root, land_shapefile=base.DEFAULT_LAND):
    """Return two tables, immutable input hashes and 420 saved-statistic checks."""
    root = Path(source_root).resolve()
    # This performs the existing 28 Fig6 checks and exact geometry preflight.
    initial, sources, _ = base.derive(root, land_shapefile)
    sources[str(Path(base.__file__).resolve())] = digest(Path(base.__file__))
    stage = root / "expected_value_test/stage_c"
    provenance = json.loads((stage / "comparison_provenance.json").read_text())
    periods, core = base.source_literals(root / "expected_value_test/compare_global.py")
    if tuple(core) != CORE:
        raise ValueError("Unexpected frozen core methods")
    cube_paths = [Path(p) for p in provenance["sources"] if Path(p).name == "matched_native_grid.nc"]
    if len(cube_paths) != 1:
        raise ValueError("Require unique provenance-selected external cube")
    cube = cube_paths[0]
    template = initial["seasonal_maps"].loc[initial["seasonal_maps"].period.eq(next(iter(periods)))].copy()
    lat = template.lat.unique()
    lon = template.lon.unique()
    shape = (len(lat), len(lon))
    area = template.land_area_weight.to_numpy().reshape(shape)
    domain = template.in_domain.to_numpy().reshape(shape)
    geometry_columns = ["lat_index", "lon_index", "lat", "lon", "lat_lower", "lat_upper",
                        "lon_lower", "lon_upper", "in_domain", "land_fraction", "land_area_weight",
                        "has_land", "original_land_purity_eligible"]

    with nc.Dataset(stage / "comparison_support.nc") as ds:
        months = json.loads(ds.months)
        factor = base.spectral_factor(ds.CALIOP_spectral_transfer)
        groups = {}
        for support, methods in SUPPORT_METHODS.items():
            key = f"{base.REFERENCE}__hierarchy__{support}"
            if key not in ds.groups:
                raise ValueError("Missing exact comparison group: " + key)
            group = ds.groups[key]
            names = ("mask", "modis_mask", "reference_mean_only_mask", "counts") + methods
            if not set(names).issubset(group.variables):
                raise ValueError("Incomplete saved comparison group: " + key)
            if any(group[name].dimensions != ("time", "lat", "lon") for name in names):
                raise ValueError("Unexpected comparison-cache dimension order")
            values = {name: base._mask(group, name) if name.endswith("mask") else base.read(group[name])
                      for name in names}
            if any(v.shape != (len(months), *shape) for v in values.values()):
                raise ValueError("Comparison support/geometry shape mismatch")
            groups[support] = values

    with nc.Dataset(cube) as ds:
        years = base.read(ds["year"])
        external = {}
        for name in ("old", "cal_mean", "cal_low", "cal_high", "cal_total", "old_total"):
            if ds[name].dimensions != ("year", "month", "lat", "lon"):
                raise ValueError("Unexpected external cube dimensions")
            values = []
            for month in months:
                year, mon = map(int, month.split("-"))
                matches = np.flatnonzero(years == year)
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous external cube year")
                values.append(np.asarray(np.ma.filled(ds[name][int(matches[0]), mon - 1].astype(float), np.nan)))
            external[name] = np.stack(values)
    ancillary = np.logical_and.reduce([np.isfinite(external[k]) & (external[k] >= 0)
                                      for k in ("cal_low", "cal_high", "cal_total", "old_total")])
    caliop = external["cal_mean"] * factor
    for support, fields in groups.items():
        modis, mean_only, mask = (fields[k] for k in ("modis_mask", "reference_mean_only_mask", "mask"))
        if not np.array_equal(mask, mean_only & ancillary) or (mean_only & ~modis).any():
            raise ValueError("Saved prior_joint semantics violated: " + support)
        if (not np.isfinite(fields["counts"][modis]).all() or (fields["counts"][modis] <= 0).any()
                or any(not np.isfinite(fields[e][modis]).all() or (fields[e][modis] < 0).any()
                       for e in SUPPORT_METHODS[support])):
            raise ValueError("Invalid cached methods/counts on saved MODIS support")
        if any((mask & (~np.isfinite(v) | (v < 0))).any() for v in external.values()):
            raise ValueError("Invalid external data on saved prior_joint support")
    if (groups["full_common_E6"]["mask"] & ~groups["core_common"]["mask"]).any():
        raise ValueError("E6 support is not a subset of core support")

    saved = pd.read_csv(stage / "seasonal_metrics.csv", float_precision="round_trip")
    for key, value in dict(reference=base.REFERENCE, variant="hierarchy", reference_mask="prior_joint",
                           region=base.REGION).items():
        saved = saved.loc[saved[key].eq(value)]
    expected = {(s, p, e) for s, methods in SUPPORT_METHODS.items()
                for p in periods for e in methods + ("Song", "CALIOP")}
    keys = ["support", "period", "estimator"]
    if saved.duplicated(keys).any() or set(saved[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Require exactly all 60 primary seasonal method/support rows")
    maps, metrics, checks = [], [], []
    for support, methods in SUPPORT_METHODS.items():
        fields = groups[support]
        for period, dates in periods.items():
            indices = [months.index(date) for date in dates]
            use = fields["mask"][indices].all(axis=0) & domain
            means = {e: fields[e][indices].mean(axis=0) for e in methods}
            means.update(Song=external["old"][indices].mean(axis=0), CALIOP=caliop[indices].mean(axis=0))
            for estimator, candidate in means.items():
                statistics = base.weighted_stats(candidate, means["CALIOP"], area, use)
                row = saved.loc[saved.support.eq(support) & saved.period.eq(period) & saved.estimator.eq(estimator)].iloc[0]
                for name, value in statistics.items():
                    base._check(value, row[name], f"{support}/{period}/{estimator}/{name}", checks, exact=name == "n")
                meta = dict(reference=base.REFERENCE, variant="hierarchy", support=support,
                            reference_mask="prior_joint", region=base.REGION, estimator=estimator,
                            period=period, months="|".join(dates))
                displayed = estimator in DISPLAY[support]
                metrics.append(dict(meta, **statistics, display_map=displayed))
                if not displayed:
                    continue
                frame = template[geometry_columns].copy()
                frame["value"] = np.where(use, candidate, np.nan).ravel()
                frame["caliop"] = np.where(use, means["CALIOP"], np.nan).ravel()
                frame["difference"] = np.where(use, candidate - means["CALIOP"], np.nan).ravel()
                frame["supported"] = use.ravel()
                for column, mask_name in (("prior_joint_months", "mask"), ("modis_months", "modis_mask"),
                                          ("mean_only_months", "reference_mean_only_mask")):
                    frame[column] = fields[mask_name][indices].sum(axis=0).ravel()
                maps.append(frame.assign(**meta))
    for path, expected_hash in sources.items():
        if digest(Path(path)) != expected_hash:
            raise RuntimeError("Frozen source changed during derivation: " + path)
    return dict(all_seasonal_maps=pd.concat(maps, ignore_index=True),
                all_seasonal_map_metrics=pd.DataFrame(metrics)), sources, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--land-shapefile", type=Path, default=base.DEFAULT_LAND)
    args = parser.parse_args(argv)
    output, source = args.output_dir.resolve(), args.source_root.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.is_relative_to(source):
        raise ValueError("Write derived arrays outside public worktree and frozen sources")
    targets = [output / (name + suffix) for name in SEMANTICS for suffix in (".csv", ".provenance.json")]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("S13 arrays already exist; choose a fresh output directory")
    tables, sources, checks = derive(source, args.land_shapefile)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".all-seasonal-", dir=output) as temporary:
        for name, table in tables.items():
            path = Path(temporary) / (name + ".csv")
            table.to_csv(path, index=False)
            metadata = dict(output_sha256=digest(path), sources=sources, semantics=SEMANTICS[name],
                            script_sha256=digest(Path(__file__)), rows=len(table), checks=checks,
                            numerical_check_atol=base.CHECK_ATOL, grid_shape=[90, 72], seasons=4,
                            displayed_method_support_pairs=DISPLAY)
            path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        for path in sorted(Path(temporary).iterdir()):
            os.replace(path, output / path.name)
    print(f"Wrote S13 arrays: {len(tables['all_seasonal_maps'])} grid rows; "
          f"{len(tables['all_seasonal_map_metrics'])} metrics rows; {len(checks)} checks passed")


if __name__ == "__main__":
    main()
