"""Regional cycles on saved monthly joint support; no interpolation or new fits."""
import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from paper import derive_seasonal_maps as base
from paper.numbers import clean, digest

METHODS = ("E1", "Eselected", "E5", "E7", "E8", "Song", "CALIOP", "CALIOP_low", "CALIOP_high")
SEMANTICS = (
    "Primary experimental1020/hierarchy/core_common/prior_joint. December 2016 through November 2017. "
    "Original saved joint mask for each individual month, not a fixed annual mask or the three-month "
    "seasonal intersection. Original float64 NaturalEarth110m 0.5-degree land-center area weights, "
    "checked by the Fig6 preflight. CALIOP mean/low/high transferred 532 to 550nm using frozen dust AE. "
    "Low/high are retrieval scenarios, not confidence intervals. All core methods and references use "
    "identical support within each region-month. Counts refer to native spatial cells, not satellite "
    "pixels, days or independent samples. Exact source region boxes including separately labeled "
    "50-75N sensitivities are retained. Zero cells means missing means; supported zeros remain zero. "
    "No interpolation over missing months; no regression, bootstrap or refitting."
)


def regional_means(fields, mask, weights, lat, lon, bounds):
    lo, hi, left, right = bounds
    domain = (lat[:, None] >= lo) & (lat[:, None] < hi) & (lon[None, :] >= left) & (lon[None, :] < right)
    if mask.dtype != bool or mask.shape != weights.shape or mask.shape != domain.shape:
        raise ValueError("Invalid mask/weight geometry")
    use = mask & domain
    n = int(use.sum())
    if n == 0:
        return {name: np.nan for name in fields}, n, 0.
    w = weights[use]
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Invalid area weights on saved support")
    result = {}
    for name, values in fields.items():
        if values.shape != mask.shape or not np.isfinite(values[use]).all() or (values[use] < 0).any():
            raise ValueError("Invalid value on joint monthly support: " + name)
        result[name] = float(np.sum(values[use] * (w / w.sum())))
    if (result["CALIOP_low"] > result["CALIOP_high"] + 1e-12
            or not result["CALIOP_low"] - 1e-12 <= result["CALIOP"] <= result["CALIOP_high"] + 1e-12):
        raise ValueError("CALIOP scenario bounds do not bracket mean")
    return result, n, float(w.sum())


def derive(source_root, land_shapefile=base.DEFAULT_LAND):
    root = Path(source_root).resolve()
    original, sources, checks = base.derive(root, land_shapefile)
    sources[str(Path(base.__file__).resolve())] = digest(Path(base.__file__))
    stage = root / "expected_value_test/stage_c"
    provenance = json.loads((stage / "comparison_provenance.json").read_text())
    cube = next(Path(p) for p in provenance["sources"] if Path(p).name == "matched_native_grid.nc")
    template = original["seasonal_maps"].loc[original["seasonal_maps"].period.eq("DJF")]
    lat, lon = template.lat.unique(), template.lon.unique()
    weights = template.land_area_weight.to_numpy().reshape(len(lat), len(lon))
    with nc.Dataset(stage / "comparison_support.nc") as ds:
        months = json.loads(ds.months)
        factor = base.spectral_factor(ds.CALIOP_spectral_transfer)
        g = ds.groups[base.GROUP]
        mask = base._mask(g, "mask")
        methods = {k: base.read(g[k]) for k in METHODS[:5]}
    fields = dict(methods)
    with nc.Dataset(cube) as ds:
        years = base.read(ds["year"])
        for estimator, name in (("Song", "old"), ("CALIOP", "cal_mean"),
                                 ("CALIOP_low", "cal_low"), ("CALIOP_high", "cal_high")):
            arrays = []
            for month in months:
                year, mon = map(int, month.split("-"))
                y = np.flatnonzero(years == year)
                if len(y) != 1 or ds[name].dimensions != ("year", "month", "lat", "lon"):
                    raise ValueError("Invalid external cube calendar or dimensions")
                values = np.asarray(np.ma.filled(ds[name][int(y[0]), mon - 1].astype(float), np.nan))
                arrays.append(values if estimator == "Song" else values * factor)
            fields[estimator] = np.stack(arrays)
    selected_months = ["2016-12"] + [f"2017-{m:02d}" for m in range(1, 12)]
    records = []
    for index, date in enumerate(selected_months):
        m = months.index(date)
        for region, bounds in provenance["regions"].items():
            summary, count, area = regional_means({k: v[m] for k, v in fields.items()}, mask[m],
                                                 weights, lat, lon, bounds)
            for estimator, value in summary.items():
                records.append(dict(reference=base.REFERENCE, variant="hierarchy", support="core_common",
                    reference_mask="prior_joint", region=region, date=date, row_order=index, estimator=estimator,
                    mean=value, n=count, land_area_weight_sum=area, supported=count > 0,
                    sensitivity=region.endswith("_50_75N_sensitivity")))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError("Frozen source changed during derivation")
    return pd.DataFrame(records), sources, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(args.source_root.resolve()) or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Keep derived data outside frozen inputs and public code")
    path = output / "regional_monthly_cycles.csv"
    if path.exists() or path.with_suffix(".provenance.json").exists():
        raise FileExistsError("Preserve previous extraction")
    data, sources, checks = derive(args.source_root)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    path.with_suffix(".provenance.json").write_text(json.dumps(clean(dict(
        sources=sources, output_sha256=digest(path), script_sha256=digest(Path(__file__)),
        rows=len(data), semantics=SEMANTICS, preflight_checks=checks)), indent=2, allow_nan=False) + "\n")
    print("Monthly cycles:", len(data), "rows;", data.region.nunique(), "regions;", data.date.nunique(), "months")


if __name__ == "__main__":
    main()
