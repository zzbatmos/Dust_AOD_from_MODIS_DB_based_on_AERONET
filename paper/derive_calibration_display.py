"""Extract saved calibration records for display, without selecting or fitting.

Inputs are grouped_validation/calibration_selected.csv.gz, calibration.json,
and the existing private site_overview.csv with its provenance sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import digest


LR = "Lidar_Ratio[1020nm]"
PLDR = "Depolarization_Ratio[1020nm]"
REFERENCE = "experimental1020"
SUPPORT = "pre2010_reserved_dust_calibration"
SELECTION = "reserved sites AND year<2010 AND backfraction1020>0.9 AND 10<=LR1020<=150"
HASH_RULE = "int(sha256('grouped_validation_v2_calibration|' + site.strip().casefold())[:16],16)/2**64 < 0.2"
COUNT_FIELDS = (
    "raw_cache_rows", "raw_reserved_rows", "selected_rows", "selected_sites",
    "all_reserved_collocation_rows", "all_reserved_collocation_sites",
)
SEMANTICS = (
    "All saved calibration records retained in original order, including repeated site/date rows. "
    "Validation only: frozen reservation hash, year<2010, clipped backscatter fraction>0.9 "
    "BEFORE any lidar-ratio scaling, and inclusive LR1020 bounds [10,150]. "
    "Sd is the saved pooled-record median, not a mean of site medians. "
    "No raw-cache search, reselection, recalibration, fitting or bootstrap. "
    "Coordinates are joined from the existing site overview, whose provenance records "
    "per-site median locations from the original cache. Other raw/reserved counts are "
    "copied from calibration.json, not independently reconstructed."
)


def reserved_site(site):
    """Check membership of an already saved site; never select new records."""
    key = site.strip().casefold()
    value = int(hashlib.sha256(("grouped_validation_v2_calibration|" + key).encode()).hexdigest()[:16], 16)
    return value / 2**64 < 0.2


def validate_and_extract(saved, metadata, overview):
    """Return record/summary frames after fail-closed checks of the frozen inputs."""
    if set(saved.columns) != {LR, PLDR, "date", "Site"} or saved.empty:
        raise ValueError("Unexpected or empty saved calibration schema")
    if metadata.get("selection") != SELECTION or metadata.get("hash_rule") != HASH_RULE:
        raise ValueError("Unexpected frozen selection or hash rule")
    if metadata.get("endmembers") != [0.02, 0.30] or metadata.get("CALIOP_used") is not False:
        raise ValueError("Unexpected endmembers or CALIOP calibration flag")
    for name in COUNT_FIELDS:
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid saved count: {name}")
    sd = metadata.get("median_LR1020_sr")
    if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not np.isfinite(sd):
        raise ValueError("Invalid saved median_LR1020_sr")
    if saved.Site.isna().any() or not saved.Site.map(
            lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise ValueError("Missing or invalid calibration site")
    keys = saved.Site.str.strip().str.casefold()
    if keys.nunique() != saved.Site.nunique():
        raise ValueError("Site-name aliases would change the saved site count")
    if len(saved) != metadata["selected_rows"] or saved.Site.nunique() != metadata["selected_sites"]:
        raise ValueError("Saved calibration record/site counts disagree with metadata")
    if not saved.Site.map(reserved_site).all():
        raise ValueError("Saved calibration contains a non-reserved site")
    values = saved[[LR, PLDR]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite LR or PLDR in saved calibration")
    dates = pd.to_datetime(saved.date, errors="raise")
    if dates.isna().any() or not (dates.dt.year < 2010).all():
        raise ValueError("Saved calibration must be strictly pre2010")
    if not saved[LR].between(10, 150, inclusive="both").all():
        raise ValueError("Saved calibration LR outside inclusive [10,150]")
    # Preserve prepare.py's literal coefficients and operation order, including
    # clipping before (and without) any Sd/LR extinction-fraction conversion.
    with np.errstate(divide="ignore", invalid="ignore"):
        backfraction = np.clip((values[:, 1] - .02) * 1.30 / (.28 * (1 + values[:, 1])), 0, 1)
    if not np.isfinite(backfraction).all() or not (backfraction > .9).all():
        raise ValueError("Saved calibration clipped preLR fraction must be >0.9")
    observed_median = float(saved[LR].median())
    if observed_median != sd:
        raise ValueError("Pooled record median differs from saved Sd; no rounding or refitting allowed")

    required = {"site_key", "role", "reference", "support", "lat", "lon", "calibration_records"}
    if not required.issubset(overview.columns):
        raise ValueError("Site overview lacks calibration coordinate/count metadata")
    sites = overview.loc[(overview.role == "calibration") & (overview.reference == REFERENCE)
                         & (overview.support == SUPPORT)].copy()
    if sites.site_key.isna().any() or sites.site_key.duplicated().any():
        raise ValueError("Ambiguous calibration coordinates")
    if set(sites.site_key) != set(keys):
        raise ValueError("Calibration coordinate site set differs from saved selection")
    sites = sites.set_index("site_key")
    coords = sites[["lat", "lon"]].to_numpy(dtype=float)
    if (not np.isfinite(coords).all() or not sites.lat.between(-90, 90).all()
            or not sites.lon.between(-180, 180).all()):
        raise ValueError("Invalid calibration coordinates")
    counts = keys.value_counts()
    if not np.array_equal(sites.loc[counts.index, "calibration_records"].to_numpy(), counts.to_numpy()):
        raise ValueError("Site overview calibration counts disagree with saved records")

    records = saved.copy().reset_index(drop=True)
    records["source_record_index"] = np.arange(len(records))
    records["site_key"] = keys.to_numpy()
    records["year"] = dates.dt.year.to_numpy()
    records["backfraction1020_preLR"] = backfraction
    records["calibration_reserved"] = True
    records["reference"] = REFERENCE
    records["support"] = SUPPORT
    records["lat"] = records.site_key.map(sites.lat)
    records["lon"] = records.site_key.map(sites.lon)
    summary = pd.DataFrame([{
        "reference": REFERENCE, "support": SUPPORT, "scope": "calibration",
        **{name: metadata[name] for name in COUNT_FIELDS},
        "median_LR1020_sr": sd, "Sd_sr": sd,
        "endmember_nondust": metadata["endmembers"][0],
        "endmember_dust": metadata["endmembers"][1],
        "CALIOP_used": metadata["CALIOP_used"],
        "selection": metadata["selection"], "hash_rule": metadata["hash_rule"],
    }])
    checks = {
        "selected_rows": len(records), "selected_sites": records.Site.nunique(),
        "pooled_median_LR1020_sr": observed_median,
        "saved_median_exact_match": True, "all_records_pass_saved_selection": True,
        "source_row_order_preserved": True, "coordinates_complete": True,
        "observed_min_LR1020_sr": float(saved[LR].min()),
        "observed_max_LR1020_sr": float(saved[LR].max()),
        "observed_latest_year": int(dates.dt.year.max()),
        "observed_min_backfraction1020_preLR": float(backfraction.min()),
        "selection_bounds": {"LR_min_inclusive": 10, "LR_max_inclusive": 150,
                             "year_upper_exclusive": 2010, "backfraction_lower_exclusive": .9},
    }
    return records, summary, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--site-overview", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Write unpublished arrays outside the public worktree")
    root = args.source_root.resolve() / "grouped_validation"
    selected = root / "calibration_selected.csv.gz"
    metadata_path = root / "calibration.json"
    overview_path = args.site_overview.resolve()
    overview_sidecar = overview_path.with_suffix(".provenance.json")
    inputs = [selected, metadata_path, overview_path, overview_sidecar]
    outputs = [output_dir / f"calibration_{name}.csv" for name in ("records", "summary")]
    destinations = [path.resolve() for path in outputs]
    destinations += [path.with_suffix(".provenance.json").resolve() for path in outputs]
    if set(destinations) & set(inputs) or output_dir.is_relative_to(args.source_root.resolve()):
        raise ValueError("Output must not overwrite frozen source inputs")
    sources = {str(path): digest(path) for path in inputs}
    overview_provenance = json.loads(overview_sidecar.read_text())
    if overview_provenance.get("output_sha256") != sources[str(overview_path)]:
        raise ValueError("Site overview SHA256 differs from its provenance sidecar")
    metadata = json.loads(metadata_path.read_text())
    records, summary, checks = validate_and_extract(
        pd.read_csv(selected, float_precision="round_trip"), metadata,
        pd.read_csv(overview_path, float_precision="round_trip"),
    )
    for path in inputs:
        if digest(path) != sources[str(path)]:
            raise RuntimeError(f"Source changed during extraction: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(outputs, (records, summary)):
        frame.to_csv(path, index=False)
        path.with_suffix(".provenance.json").write_text(json.dumps({
            "sources": sources, "output_sha256": digest(path),
            "script_sha256": digest(Path(__file__)), "semantics": SEMANTICS,
            "checks": checks, "coordinate_provenance": overview_provenance,
            "source_row_index": "Zero-based row position in calibration_selected.csv.gz",
        }, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": [str(path) for path in outputs], **checks}))


if __name__ == "__main__":
    main()
