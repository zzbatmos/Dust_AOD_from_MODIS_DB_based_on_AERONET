"""Table S1 from fixed cohorts and local AERONET headers, never new selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from paper.derive_configuration_tables import guard_output
from paper.numbers import digest


NAME = "site_inventory_table"
TABLE = "derived/" + NAME
FILTERS = ["site_key", "site", "role", "reference", "support", "region"]
SEMANTICS = (
    "Fixed site_overview membership and roles. Evaluation years/counts use site_cv held-out "
    "OOF rows deduplicated by row_id after validating both saved repeats and invariant identity. "
    "Calibration years/counts use every saved calibration_selected record, including repeated "
    "site/date records; these are AERONET inversions, not MODIS collocations. Years list only "
    "observed cohort years, not all years between endpoints. Evaluation coordinates/regions and "
    "calibration coordinates retain site_overview values; calibration regions join the original "
    "site inventory where available, otherwise NOT RECORDED. PI names are verbatim decoded "
    "local inversion-file header strings, not verified historical tenure or current contacts. "
    "Header decoding is UTF-8 when valid, otherwise explicit Windows-1252 (including smart "
    "apostrophes); raw header/file hashes preserve byte provenance. No emails exported, network "
    "lookups, fits, scientific computations, site selection or author-source edits."
)


def packed(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def read_pi_header(path, site):
    """Read only six header lines; never scan observation rows for names."""
    with Path(path).open("rb") as stream:
        raw = b"".join(stream.readline() for _ in range(6))
    try:
        header, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        header, encoding = raw.decode("cp1252"), "windows-1252"
    lines = header.splitlines()
    if len(lines) < 5 or lines[0].strip() != "AERONET Version 3" or lines[1].strip() != site:
        raise ValueError(f"Local header/site mismatch: {path}")
    matches = re.findall(r"\bPI=([^;\r\n]*)", "\n".join(lines))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous PI field: {path}")
    name = matches[0].strip() if matches else ""
    if name.casefold() in {"", "n/a", "na", "none", "null", "unknown", "not available"}:
        name = ""
    return {"principal_investigators": name, "header_sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": encoding, "field": "first six lines; exact site in line 2; PI= before semicolon"}


def pi_metadata(sites, header_dir, evidence):
    indexed = {}
    for path in sorted(Path(header_dir).glob("*.all")):
        match = re.fullmatch(r"\d{8}_\d{8}_(.+)\.all", path.name)
        if match:
            indexed.setdefault(match.group(1), []).append(path.resolve())
    result = {}
    for site in sites:
        candidates = []
        for path in indexed.get(site, []):
            before = digest(path)
            item = read_pi_header(path, site)
            if digest(path) != before:
                raise RuntimeError(f"PI source changed: {path}")
            evidence[str(path)] = before
            candidates.append(dict(source_file=str(path), source_sha256=before, **item))
        names = sorted({c["principal_investigators"] for c in candidates if c["principal_investigators"]})
        status = "local_header" if len(names) == 1 else "conflicting_local_headers" if names else "missing_local_PI"
        result[site] = {"principal_investigators": names[0] if len(names) == 1 else "NOT RECORDED",
                        "pi_status": status, "pi_evidence_json": packed(candidates)}
    return result


def cohort_summary(overview, events, calibration, inventory, expected):
    """Join identities exactly, fail on changed cohorts, retain unavailable fields."""
    if overview.empty or overview.site_key.isna().any() or overview.site_key.duplicated().any():
        raise ValueError("Missing or ambiguous overview site keys")
    if not overview.role.isin(["evaluation", "calibration"]).all():
        raise ValueError("Unknown overview role")
    if not overview.site.str.strip().str.casefold().eq(overview.site_key).all():
        raise ValueError("Overview name/key mismatch")
    if not overview.reference.eq("experimental1020").all():
        raise ValueError("Unexpected overview reference")
    if not np.isfinite(overview[["lat", "lon"]].to_numpy(float)).all():
        raise ValueError("Missing coordinates")
    if inventory.site_key.duplicated().any():
        raise ValueError("Ambiguous source region inventory")
    q = events.loc[events.scheme.eq("site_cv")].copy()
    identity = ["site", "site_key", "time", "platform", "region"]
    if q.empty or q[["row_id", "repeat", *identity]].isna().any().any():
        raise ValueError("Missing OOF identities")
    if (q.duplicated(["row_id", "repeat"]).any() or set(q["repeat"]) != {0, 1}
            or not q.groupby("row_id").size().eq(2).all()):
        raise ValueError("Expected exactly two distinct site_cv repeats per collocation")
    if not q.groupby("row_id")[identity].nunique(dropna=False).eq(1).all().all():
        raise ValueError("OOF identity changes across repeats")
    for key, value in {"reference": "experimental1020", "population": "own",
                       "model": "hierarchy_prespecified"}.items():
        if not q[key].eq(value).all():
            raise ValueError(f"Unexpected OOF {key}")
    q = q.drop_duplicates("row_id").copy()
    q["year"] = pd.to_datetime(q.time, errors="raise").dt.year
    if q.year.isna().any():
        raise ValueError("Missing OOF year")
    cal = calibration.copy()
    if cal.Site.isna().any() or cal.empty:
        raise ValueError("Missing calibration identities")
    cal["site_key"] = cal.Site.str.strip().str.casefold()
    cal["year"] = pd.to_datetime(cal.date, errors="raise").dt.year
    if cal.year.isna().any() or not cal.year.lt(2010).all():
        raise ValueError("Invalid saved calibration years")
    for role, data in [("evaluation", q), ("calibration", cal)]:
        sites = overview.loc[overview.role.eq(role)]
        if set(sites.site_key) != set(data.site_key):
            raise ValueError(f"{role} cohort membership changed")
        if len(sites) != expected[role + "_sites"]:
            raise ValueError(f"{role} site count differs from saved provenance")
    if len(q) != expected["unique_collocations"] or len(cal) != expected["calibration_records"]:
        raise ValueError("Cohort row count differs from saved provenance")
    regions = inventory.set_index("site_key")
    rows = []
    for _, site in overview.iterrows():
        evaluation = site.role == "evaluation"
        group = (q if evaluation else cal).loc[lambda d: d.site_key.eq(site.site_key)]
        name_column = "site" if evaluation else "Site"
        if not group[name_column].eq(site.site).all():
            raise ValueError("Site display name changed")
        count_column = "collocations" if evaluation else "calibration_records"
        if len(group) != site[count_column]:
            raise ValueError("Per-site count differs from overview")
        years = sorted(int(y) for y in group.year.unique())
        if evaluation:
            if (site.first_year != years[0] or site.last_year != years[-1]
                    or not group.region.eq(site.region).all()
                    or site.platforms != ";".join(sorted(group.platform.unique()))):
                raise ValueError("Evaluation overview metadata differs from cohort")
            region, region_status = site.region, "saved_overview"
        elif site.site_key in regions.index and pd.notna(regions.loc[site.site_key, "CV_region"]):
            if not bool(regions.loc[site.site_key, "calibration_reserved"]):
                raise ValueError("Calibration region belongs to nonreserved site")
            region, region_status = regions.loc[site.site_key, "CV_region"], "saved_inventory"
        else:
            region, region_status = "NOT RECORDED", "missing_saved_region"
        year_counts = {str(y): int(group.year.eq(y).sum()) for y in years}
        rows.append(dict(site_key=site.site_key, site=site.site, role=site.role,
                         reference=site.reference, support=site.support, region=region,
                         region_status=region_status, overview_region=site.region,
                         lat=site.lat, lon=site.lon, collocations=len(group) if evaluation else None,
                         calibration_records=None if evaluation else len(group), n_records=len(group),
                         record_kind="unique_MODIS_collocations" if evaluation else "saved_AERONET_calibration_records",
                         first_year=years[0], last_year=years[-1], years=";".join(map(str, years)),
                         n_years=len(years), year_counts_json=packed(year_counts),
                         platforms=site.platforms if evaluation else "NOT APPLICABLE",
                         cohort_filter_json=packed({"scheme": "site_cv", "site_key": site.site_key,
                                                   "deduplicate": "row_id", "repeats": [0, 1]}
                                                  if evaluation else {"Site": site.site, "selection": "all saved records"})))
    return pd.DataFrame(rows)


def derive(overview_path, source_root, header_dir, output_dir):
    overview_path, source_root = Path(overview_path).resolve(), Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    guard_output(output_dir, source_root)
    guard_output(output_dir, header_dir)
    output = output_dir / (NAME + ".csv")
    if output.exists() or output.with_suffix(".provenance.json").exists():
        raise FileExistsError(output)
    sidepath = overview_path.with_suffix(".provenance.json")
    side = json.loads(sidepath.read_text())
    if digest(overview_path) != side["output_sha256"]:
        raise ValueError("Overview SHA mismatch")
    files = [overview_path, sidepath, source_root / "oof/experimental1020__own__hierarchy_prespecified.parquet",
             source_root / "calibration_selected.csv.gz", source_root / "site_inventory.csv",
             source_root / "calibration.json"]
    sources = {str(p): digest(p) for p in files}
    for path in files[2:]:
        if str(path) in side["sources"] and sources[str(path)] != side["sources"][str(path)]:
            raise ValueError("Original cohort source SHA mismatch")
    overview = pd.read_csv(overview_path)
    cal_meta = json.loads(files[-1].read_text())
    if (cal_meta["selected_rows"] != side["calibration_records"]
            or cal_meta["selected_sites"] != side["calibration_sites"]):
        raise ValueError("Saved calibration totals differ from overview provenance")
    frame = cohort_summary(overview, pd.read_parquet(files[2]), pd.read_csv(files[3]),
                           pd.read_csv(files[4]), side)
    pi = pi_metadata(frame.site, header_dir, sources)
    for column in ("principal_investigators", "pi_status", "pi_evidence_json"):
        frame[column] = frame.site.map(lambda site: pi[site][column])
    coverage = dict(rows=len(frame), evaluation_sites=int(frame.role.eq("evaluation").sum()),
                    calibration_sites=int(frame.role.eq("calibration").sum()),
                    unique_collocations=int(frame.collocations.sum()), calibration_records=int(frame.calibration_records.sum()),
                    pi_available=int(frame.pi_status.eq("local_header").sum()),
                    pi_missing_sites=frame.loc[frame.pi_status.eq("missing_local_PI"), "site"].tolist(),
                    pi_conflicting_sites=frame.loc[frame.pi_status.eq("conflicting_local_headers"), "site"].tolist(),
                    region_missing_sites=frame.loc[frame.region_status.eq("missing_saved_region"), "site"].tolist())
    for path, sha in sources.items():
        if digest(Path(path)) != sha:
            raise RuntimeError(f"Source changed during derivation: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".provenance.json").write_text(json.dumps(dict(
        sources=sources, output_sha256=digest(output), script_sha256=digest(Path(__file__)),
        rows=len(frame), coverage=coverage, semantics=SEMANTICS, overview_provenance=side,
        recommended_filter_columns=FILTERS, required_dimension_additions=[],
        pi_search=dict(directory=str(Path(header_dir).resolve()), network_used=False,
                       matching="Exact site suffix AND exact site in header line 2; no fuzzy joins"),
        unresolved="Historical PI tenure/author acknowledgment approval not verified. Missing saved regions are not assigned geographically."
    ), indent=2, allow_nan=False) + "\n")
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-overview", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Frozen grouped_validation directory")
    parser.add_argument("--header-dir", type=Path, required=True, help="Existing local AERONET *.all files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.site_overview, args.source_root, args.header_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
