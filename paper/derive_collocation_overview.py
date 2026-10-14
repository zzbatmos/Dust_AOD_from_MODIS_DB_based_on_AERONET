"""Extract the saved collocation ledger and product comparisons without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper.numbers import clean, digest


REFERENCE = "experimental1020"
FEATURES5 = CONFIG["branches"]["noSSA5"]["features"]
FEATURES8 = CONFIG["branches"]["full8"]["features"]
VIEWS = {"aod": (0., 2.), "ae": (-.5, 3.)}
PAIR_FIELDS = {"aod": ("AER_AOD550", "DB_AOD550"), "ae": ("AER_AE440_675", "DB_AE")}
SEMANTICS = (
    "One count per saved collocation row_id, not OOF repeats or MODIS pixels. "
    "Ledger reproduces frozen primary-reference/noncalibration/finite-feature eligibility. "
    "Full8 and noSSA5 fallback partition the hierarchy; no extra AOD, AE, type, SSA or geographic "
    "screen is added. Broad product comparisons include reserved sites and do not require a "
    "dust reference or ML eligibility. AOD and AE use separate paired-valid masks. "
    "Saved AERONET AOD550 and AE440_675 are verified against the saved positive 440/675-nm AODs, "
    "not replaced by SDA AOD or AE. Comparison statistics use all valid pairs, including outside "
    "the stated display windows. Bias is DB minus AERONET. No fit, resampling, confidence "
    "interval or significance test is performed; collocations need not be independent."
)


def finite(d, columns):
    return np.isfinite(d[columns].to_numpy(float)).all(axis=1)


def extract(rows, population, events, schema):
    d = rows.copy()
    if d.empty or d.row_id.isna().any() or d.row_id.duplicated().any() or d.site_key.isna().any():
        raise ValueError("Missing or duplicate collocation identity")
    if not pd.api.types.is_bool_dtype(d.calibration_reserved.dtype) or d.calibration_reserved.isna().any():
        raise ValueError("Invalid saved calibration reservation")
    if set(d.platform) - {"Terra", "Aqua"}:
        raise ValueError("Unknown platform")
    dates = pd.to_datetime(d.time, errors="coerce")
    if dates.isna().any():
        raise ValueError("Missing collocation year; do not silently discard")
    d["year"] = dates.dt.year
    if len(d) != schema["rows"] or d.site_key.nunique() != schema["sites"]:
        raise ValueError("Saved schema counts disagree")
    wavelengths = ["AER_AOD440", "AER_AOD675"]
    observed = finite(d, wavelengths)
    if (d.loc[observed, wavelengths] <= 0).any().any():
        raise ValueError("Saved AERONET wavelengths must be positive")
    a440, a675 = (d.loc[observed, k].to_numpy(float) for k in wavelengths)
    expected_aod = np.exp(np.log(a440) + (np.log(a675) - np.log(a440)) * np.log(550 / 440) / np.log(675 / 440))
    expected_ae = -np.log(a440 / a675) / np.log(440 / 675)
    for column, expected in (("AER_AOD550", expected_aod), ("AER_AE440_675", expected_ae)):
        if not np.allclose(d.loc[observed, column], expected, atol=1e-12, rtol=1e-12):
            raise ValueError("Saved spectral derivation differs: " + column)
        if d.loc[~observed, column].notna().any():
            raise ValueError("Derived AERONET value without both observed wavelengths")
    target_finite = finite(d, [REFERENCE])
    if not d.loc[target_finite, REFERENCE].between(0, 1).all():
        raise ValueError("Saved dust fraction not already bounded")
    noncal = ~d.calibration_reserved
    ref = noncal & target_finite
    aod = finite(d, ["AER_AOD550", "DB_AOD550"]) & d.AER_AOD550.gt(0) & d.DB_AOD550.ge(0)
    ae = finite(d, ["AER_AE440_675", "DB_AE"])
    paired = ref & aod
    hierarchy = paired & finite(d, FEATURES5)
    full8 = hierarchy & finite(d, FEATURES8)
    fallback = hierarchy & ~full8
    ledger = []
    for order, (sample, mask) in enumerate([
        ("all_collocations", np.ones(len(d), bool)), ("noncalibration", noncal),
        ("finite_primary_reference", ref), ("reference_and_paired_aod", paired),
        ("hierarchy_eligible", hierarchy), ("full8_subset", full8), ("noSSA5_fallback_subset", fallback),
    ]):
        ledger.append(dict(reference=REFERENCE, sample=sample, row_order=order, rows=int(mask.sum()),
                           sites=int(d.loc[mask, "site_key"].nunique()),
                           relationship="nested_step" if order < 5 else "hierarchy_partition"))
    for feature, mask in (("full8", full8), ("noSSA5", hierarchy)):
        saved = population.loc[population.target.eq(REFERENCE) & population.features.eq(feature)]
        if len(saved) != 1:
            raise ValueError("Missing or ambiguous saved population")
        record = saved.iloc[0]
        if (record.train_valid_rows != mask.sum() or record.evaluation_valid_rows != mask.sum()
                or record.train_sites != d.loc[mask, "site_key"].nunique()):
            raise ValueError("Reconstructed eligibility differs from saved population")
    oof = events.loc[events.scheme.eq("site_cv")]
    if oof.duplicated(["row_id", "repeat"]).any() or set(oof["repeat"]) != {0, 1}:
        raise ValueError("Invalid hierarchy OOF event identity")
    wanted = set(d.loc[hierarchy, "row_id"])
    if any(set(q.row_id) != wanted for _, q in oof.groupby("repeat")):
        raise ValueError("Hierarchy row identities differ from saved held-out predictions")

    annual, cohorts = [], []
    years = range(int(d.year.min()), int(d.year.max()) + 1)
    for sample, mask in (("all_collocations", np.ones(len(d), bool)), ("hierarchy_eligible", hierarchy)):
        q = d.loc[mask]
        cohorts.append(dict(reference=REFERENCE, sample=sample, rows=len(q), sites=q.site_key.nunique(),
                            rows_2017=int(q.year.eq(2017).sum()), fraction_2017=float(q.year.eq(2017).mean()),
                            first_year=int(q.year.min()), last_year=int(q.year.max())))
        for year in years:
            for platform in ("Terra", "Aqua"):
                g = q.loc[q.year.eq(year) & q.platform.eq(platform)]
                annual.append(dict(reference=REFERENCE, sample=sample, year=year, platform=platform,
                                   rows=len(g), sites=g.site_key.nunique()))

    pairs = d.loc[aod | ae, ["row_id", "site_key", "platform", "year", "calibration_reserved",
                            "AER_AOD550", "DB_AOD550", "AER_AE440_675", "DB_AE"]].copy()
    pairs["source_record_index"] = np.flatnonzero(aod | ae)
    pairs["aod_pair_valid"] = np.asarray(aod)[aod | ae]
    pairs["ae_pair_valid"] = np.asarray(ae)[aod | ae]
    pairs["hierarchy_eligible"] = np.asarray(hierarchy)[aod | ae]
    statistics = []
    for panel, (reference_column, db_column) in PAIR_FIELDS.items():
        q = pairs.loc[pairs[f"{panel}_pair_valid"]]
        if q.empty:
            raise ValueError("No paired-valid values for " + panel)
        x, y = (q[k].to_numpy(float) for k in (reference_column, db_column))
        low, high = VIEWS[panel]
        inside = (x >= low) & (x <= high) & (y >= low) & (y <= high)
        r = np.corrcoef(x, y)[0, 1] if len(q) >= 2 and np.var(x) > 0 and np.var(y) > 0 else np.nan
        statistics.append(dict(panel=panel, sample="broad_product_comparison", rows=len(q),
            sites=q.site_key.nunique(), n_inside_view=int(inside.sum()), n_outside_view=int((~inside).sum()),
            view_min=low, view_max=high, bias=float(np.mean(y - x)), rmse=float(np.sqrt(np.mean((y - x) ** 2))),
            r=float(r), r_defined=bool(np.isfinite(r)), mean_aeronet=float(x.mean()), mean_db=float(y.mean()),
            minimum_aeronet=float(x.min()), maximum_aeronet=float(x.max()),
            minimum_db=float(y.min()), maximum_db=float(y.max())))
    return {"collocation_ledger": pd.DataFrame(ledger),
            "collocation_year_platform": pd.DataFrame(annual),
            "collocation_cohort_summary": pd.DataFrame(cohorts),
            "collocation_product_pairs": pairs.reset_index(drop=True),
            "collocation_product_statistics": pd.DataFrame(statistics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root / "grouped_validation"
    paths = [root / "rows.parquet", root / "population_counts.csv", root / "schema.json",
             root / "oof/experimental1020__own__hierarchy_prespecified.parquet",
             root / "prepare.py", Path(__file__).resolve().parents[1] / "configs/dbdust_v2.0.0.yaml"]
    output = args.output_dir.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.is_relative_to(args.source_root.resolve()):
        raise ValueError("Derived tables must be separate from public code and frozen inputs")
    hashes = {str(p.resolve()): digest(p) for p in paths}
    tables = extract(pd.read_parquet(paths[0]), pd.read_csv(paths[1]),
                     pd.read_parquet(paths[3], columns=["row_id", "scheme", "repeat"]),
                     json.loads(paths[2].read_text()))
    if any(digest(p) != hashes[str(p.resolve())] for p in paths):
        raise RuntimeError("Source changed during extraction")
    output.mkdir(parents=True, exist_ok=True)
    for name in tables:
        if (output / f"{name}.csv").exists():
            raise FileExistsError("Preserve previous derived outputs: " + name)
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        path.with_suffix(".provenance.json").write_text(json.dumps(clean(dict(
            sources=hashes, output_sha256=digest(path), script_sha256=digest(Path(__file__)),
            semantics=SEMANTICS, reference=REFERENCE, feature_lists=dict(full8=FEATURES8, noSSA5=FEATURES5),
            rows=len(frame), display_windows=VIEWS,
            checks="Schema, population counts, spectral derivation, and both OOF repeat row-identity sets verified")),
            indent=2, allow_nan=False) + "\n")
    print(tables["collocation_ledger"].to_string(index=False))
    print(tables["collocation_product_statistics"].to_string(index=False))


if __name__ == "__main__":
    main()
