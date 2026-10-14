"""Extract site-map counts and calibration locations without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import digest


def evaluation_sites(events, inventory):
    q = events.loc[events.scheme.eq("site_cv")].copy()
    if q.duplicated(["row_id", "repeat"]).any():
        raise ValueError("Duplicate held-out events")
    if set(q["repeat"].unique()) != {0, 1} or not q.groupby("row_id").size().eq(2).all():
        raise ValueError("Expected two events per unique collocation")
    identity = ["site_key", "site", "region", "time", "platform"]
    if not q.groupby("row_id")[identity].nunique(dropna=False).eq(1).all().all():
        raise ValueError("Collocation identity changes across repeats")
    q = q.drop_duplicates("row_id")
    q["year"] = pd.to_datetime(q.time).dt.year
    result = q.groupby("site_key").agg(
        site=("site", "first"), region=("region", "first"),
        collocations=("row_id", "size"), first_year=("year", "min"),
        last_year=("year", "max"), platforms=("platform", lambda x: ";".join(sorted(x.unique()))),
    ).reset_index()
    result = result.merge(inventory[["site_key", "lat", "lon", "calibration_reserved", "CV_region"]],
                          on="site_key", how="left", validate="one_to_one")
    if result.calibration_reserved.isna().any() or result.calibration_reserved.any():
        raise ValueError("Calibration reservation violated or site missing")
    if not result.region.eq(result.CV_region).all():
        raise ValueError("Region identity differs from site inventory")
    if not np.isfinite(result[["lat", "lon"]]).all().all():
        raise ValueError("Missing evaluation coordinates")
    result["role"] = "evaluation"
    result["reference"] = "experimental1020"
    result["scheme"] = "site_cv"
    result["support"] = "hierarchy_own"
    return result.drop(columns="CV_region")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--aeronet-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Unpublished arrays must remain outside public worktree")
    root = args.source_root / "grouped_validation"
    files = [root / "oof/experimental1020__own__hierarchy_prespecified.parquet",
             root / "site_inventory.csv", root / "calibration_sites.csv",
             root / "calibration.json", args.aeronet_cache, root / "population_counts.csv"]
    hashes = {str(p.resolve()): digest(p) for p in files}
    result = evaluation_sites(pd.read_parquet(files[0]), pd.read_csv(files[1]))
    counts = pd.read_csv(files[-1])
    expected = counts.loc[counts.target.eq("experimental1020") & counts.features.eq("noSSA5")].squeeze()
    if result.collocations.sum() != expected.train_valid_rows or len(result) != expected.train_sites:
        raise ValueError("Hierarchy counts differ from expected noSSA5-eligible population")
    cal = pd.read_csv(files[2])
    metadata = json.loads(files[3].read_text())
    if len(cal) != metadata["selected_sites"] or cal["count"].sum() != metadata["selected_rows"]:
        raise ValueError("Calibration selection counts differ")
    coordinates = []
    for chunk in pd.read_csv(args.aeronet_cache, usecols=["Site", "Latitude(Degrees)", "Longitude(Degrees)"],
                             chunksize=100000):
        coordinates.append(chunk.loc[chunk.Site.isin(cal.Site)])
    raw = pd.concat(coordinates, ignore_index=True)
    coords = raw.groupby("Site")[["Latitude(Degrees)", "Longitude(Degrees)"]].median().reset_index()
    cal = cal.merge(coords, on="Site", how="left", validate="one_to_one").rename(
        columns={"Site": "site", "Latitude(Degrees)": "lat", "Longitude(Degrees)": "lon",
                 "count": "calibration_records", "median": "median_LR1020"})
    if not np.isfinite(cal[["lat", "lon"]]).all().all():
        raise ValueError("Missing calibration coordinates")
    cal["site_key"] = cal.site.str.strip().str.casefold()
    if set(cal.site_key) & set(result.site_key):
        raise ValueError("Calibration/evaluation site overlap")
    cal["role"] = "calibration"
    cal["reference"] = "experimental1020"
    cal["region"] = "calibration only"
    cal["support"] = "pre2010_reserved_dust_calibration"
    cal["collocations"] = np.nan
    output = args.output_dir / "site_overview.csv"
    if output.exists():
        raise FileExistsError(output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([result, cal], ignore_index=True).to_csv(output, index=False)
    for path, sha in hashes.items():
        if digest(Path(path)) != sha:
            raise RuntimeError(f"Source changed: {path}")
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "sources": hashes, "output_sha256": digest(output), "script_sha256": digest(Path(__file__)),
        "evaluation_sites": len(result), "unique_collocations": int(result.collocations.sum()),
        "calibration_sites": len(cal), "calibration_records": int(cal.calibration_records.sum()),
        "semantics": "Hierarchy own-population held-out site CV; one count per row_id, not per repeat. "
        "Evaluation locations/regions from saved site inventory. Calibration locations are per-site "
        "median coordinates in the original AERONET narrow cache, not inferred from nearby sites. "
        "Calibration sites are disjoint, fixed by saved selection, never reselected."
    }, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
