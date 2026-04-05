#!/usr/bin/env python3
"""Build a single all-site MODIS DB collocation file with QA and SDA data.

This merges:
1. QA-augmented per-site collocation pickles in
   `AERONET_MODIS_DB_collocation_files_with_QA/`
2. The original Terra/Aqua collocation dictionaries that already include
   nearest AERONET SDA matches.

The QA-augmented records already contain the AERONET inversion fields, so the
only expected missing block is the nearest SDA metadata/data pair. The output
keeps the nested structure:

    combined[site_name][granule_name] = record

and preserves both Terra (`MOD04_L2`) and Aqua (`MYD04_L2`) granules.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parent = here.parent

    parser = argparse.ArgumentParser(
        description=(
            "Merge QA-augmented per-site collocation files with the original "
            "SDA-enhanced Terra/Aqua collocation dictionaries."
        )
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=here / "AERONET_MODIS_DB_collocation_files_with_QA",
        help="Directory containing per-site *_with_QA.pkl files.",
    )
    parser.add_argument(
        "--mod04-sda",
        type=Path,
        default=parent / "AERONET_MOD04_L2_collocation_with_SDA_data.pkl",
        help="Terra collocation dictionary that already includes nearest SDA data.",
    )
    parser.add_argument(
        "--myd04-sda",
        type=Path,
        default=parent / "AERONET_MYD04_L2_collocation_with_SDA_data.pkl",
        help="Aqua collocation dictionary that already includes nearest SDA data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl",
        help="Path to the merged all-site output pickle.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=here / "AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA_summary.json",
        help="Path to the merge summary JSON.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_site_name(filename: str) -> tuple[str, str]:
    name = Path(filename).name
    if name.startswith("AERONET_MOD04_L2_collocation_") and name.endswith("_with_QA.pkl"):
        return "MOD04_L2", name[len("AERONET_MOD04_L2_collocation_") : -len("_with_QA.pkl")]
    if name.startswith("AERONET_MYD04_L2_collocation_") and name.endswith("_with_QA.pkl"):
        return "MYD04_L2", name[len("AERONET_MYD04_L2_collocation_") : -len("_with_QA.pkl")]
    raise ValueError(f"Unrecognized QA file name pattern: {filename}")


def ensure_sda_fields(record: dict, sda_record: dict | None) -> tuple[bool, bool]:
    """Add SDA fields if they are not already present.

    Returns `(added_meta, added_data)`.
    """

    added_meta = False
    added_data = False

    if "nearest_AERONET_SDA_L1.5_data_meta" not in record:
        record["nearest_AERONET_SDA_L1.5_data_meta"] = (
            sda_record.get("nearest_AERONET_SDA_L1.5_data_meta", {"found": False})
            if sda_record
            else {"found": False}
        )
        added_meta = True

    if "nearest_AERONET_SDA_L1.5_data" not in record:
        record["nearest_AERONET_SDA_L1.5_data"] = (
            sda_record.get("nearest_AERONET_SDA_L1.5_data", {}) if sda_record else {}
        )
        added_data = True

    return added_meta, added_data


def main() -> None:
    args = parse_args()

    mod04_sda = load_pickle(args.mod04_sda)
    myd04_sda = load_pickle(args.myd04_sda)
    sda_lookup = {
        "MOD04_L2": mod04_sda,
        "MYD04_L2": myd04_sda,
    }

    combined: dict[str, dict[str, dict]] = {}
    summary = {
        "qa_dir": str(args.qa_dir),
        "mod04_sda": str(args.mod04_sda),
        "myd04_sda": str(args.myd04_sda),
        "output": str(args.output),
        "n_site_files": 0,
        "n_sites": 0,
        "n_records": 0,
        "n_mod04_records": 0,
        "n_myd04_records": 0,
        "sda_meta_added": 0,
        "sda_data_added": 0,
        "sda_matches_found": 0,
        "sda_matches_missing": 0,
        "site_level_summary": {},
        "missing_sda_examples": [],
    }

    qa_files = sorted(args.qa_dir.glob("*.pkl"))
    summary["n_site_files"] = len(qa_files)

    for qa_file in qa_files:
        platform, site_name = parse_site_name(qa_file.name)
        qa_site_dict = load_pickle(qa_file)
        combined.setdefault(site_name, {})
        site_summary = summary["site_level_summary"].setdefault(
            site_name,
            {
                "platforms": set(),
                "n_records": 0,
                "n_sda_matches_found": 0,
                "n_sda_matches_missing": 0,
            },
        )
        site_summary["platforms"].add(platform)

        sda_site_dict = sda_lookup.get(platform, {}).get(site_name, {})

        for granule_name, record in qa_site_dict.items():
            sda_record = sda_site_dict.get(granule_name)
            if sda_record is not None:
                summary["sda_matches_found"] += 1
                site_summary["n_sda_matches_found"] += 1
            else:
                summary["sda_matches_missing"] += 1
                site_summary["n_sda_matches_missing"] += 1
                if len(summary["missing_sda_examples"]) < 20:
                    summary["missing_sda_examples"].append(
                        {
                            "site_name": site_name,
                            "platform": platform,
                            "granule_name": granule_name,
                        }
                    )

            merged_record = dict(record)
            added_meta, added_data = ensure_sda_fields(merged_record, sda_record)
            if added_meta:
                summary["sda_meta_added"] += 1
            if added_data:
                summary["sda_data_added"] += 1

            combined[site_name][granule_name] = merged_record
            summary["n_records"] += 1
            site_summary["n_records"] += 1
            if platform == "MOD04_L2":
                summary["n_mod04_records"] += 1
            else:
                summary["n_myd04_records"] += 1

    for site_summary in summary["site_level_summary"].values():
        site_summary["platforms"] = sorted(site_summary["platforms"])

    summary["n_sites"] = len(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary_json": str(args.summary_json),
                "n_sites": summary["n_sites"],
                "n_records": summary["n_records"],
                "n_mod04_records": summary["n_mod04_records"],
                "n_myd04_records": summary["n_myd04_records"],
                "sda_matches_found": summary["sda_matches_found"],
                "sda_matches_missing": summary["sda_matches_missing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
