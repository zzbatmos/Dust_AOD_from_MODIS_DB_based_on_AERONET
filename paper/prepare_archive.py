"""Prepare a private archive candidate, not a Zenodo deposit or publication.

No authentication, remote mutations, tag creation, or source-data redistribution.
The collocation table is inventoried, not copied into the upload candidate until
its scope and attribution have been reviewed by the authors.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import requests

from paper.numbers import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--collocations", type=Path, required=True)
    parser.add_argument("--site-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-ref", default="release/v2.0.0")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.is_relative_to(repo) or output.exists():
        raise ValueError("Choose a new private output directory outside the public worktree")
    commit = subprocess.check_output(["git", "rev-parse", args.release_ref], cwd=repo, text=True).strip()
    response = requests.get("https://zenodo.org/api/records/21223044", timeout=(15, 60))
    response.raise_for_status()
    predecessor = response.json()
    if str(predecessor["id"]) != "21223044" or predecessor["metadata"]["version"] != "v1.0.0":
        raise ValueError("Unexpected archive predecessor identity")
    products = sorted(args.product_dir.glob("DBDust_v2.0.0_Aqua_L3M_*.nc"))
    if len(products) != 26:
        raise ValueError("Expected 13 months on each of two grids")
    output.mkdir(parents=True)
    (output / "predecessor_public_metadata.json").write_text(json.dumps(predecessor, indent=2) + "\n")
    payload = output / "candidate_payload"
    payload.mkdir()
    archive = payload / "DBDust_v2_source_release_candidate.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--output={archive}", commit], cwd=repo, check=True)
    product_output = payload / "monthly_products"
    product_output.mkdir()
    manifest = [dict(component="source", path=str(archive.relative_to(output)),
                     source=commit, sha256=digest(archive), bytes=archive.stat().st_size,
                     status="release candidate, not author-approved tag")]
    for source in products:
        destination = product_output / source.name
        before = digest(source)
        shutil.copy2(source, destination)
        if digest(destination) != before or digest(source) != before:
            raise ValueError("Monthly product copy hash mismatch")
        manifest.append(dict(component="L3", path=str(destination.relative_to(output)),
                             source=str(source.resolve()), sha256=before, bytes=destination.stat().st_size,
                             status="derived research product; not operational NASA product"))
    for name in ("DATA_FORMAT.md", "MODEL_CARD.md", "ALGORITHM.md", "FREEZE_REPORT.md"):
        content = subprocess.check_output(["git", "show", f"{commit}:docs/v2/{name}"], cwd=repo)
        destination = payload / name
        destination.write_bytes(content)
        manifest.append(dict(component="documentation", path=str(destination.relative_to(output)),
                             source=f"{commit}:docs/v2/{name}", sha256=digest(destination),
                             bytes=len(content), status="release-candidate documentation"))
    pd.DataFrame(manifest).to_csv(output / "candidate_manifest.csv", index=False)
    sites = pd.read_csv(args.site_inventory)
    table = pd.read_parquet(args.collocations)
    (output / "collocation_scope_review.json").write_text(json.dumps({
        "source_file": str(args.collocations.resolve()), "sha256": digest(args.collocations),
        "bytes": args.collocations.stat().st_size, "rows": len(table),
        "sites": int(table.site_key.nunique()), "columns": list(table.columns),
        "site_inventory_sha256": digest(args.site_inventory), "site_inventory_rows": len(sites),
        "PI_mapping_in_inventory": any("principal" in s.lower() or s.lower() in ("pi", "co_pi") for s in sites),
        "copied_into_payload": False,
        "pending": "Author-approved row/column scope and complete source/site/PI attribution."
    }, indent=2) + "\n")
    (output / "metadata_proposal.json").write_text(json.dumps({
        "title": "DBDust: Retrieval-space dust optical depth estimation from MODIS Deep Blue",
        "version": "2.0.0", "resource_type_proposal": "software",
        "description": "Frozen full8/noSSA5 dust-probability, QA2 and expected-value DAOD estimators; "
        "Aqua monthly demonstration products for calendar 2017 plus December 2016 for DJF context.",
        "predecessor_record": predecessor["id"], "predecessor_doi": predecessor["doi"],
        "predecessor_parent": predecessor["links"]["parent"],
        "creators_proposal": ["Zhibo Zhang", "Yingxi Shi", "QianQian Song", "Hongbin Yu", "Ronghao Wang", "Jianyu Zheng"],
        "license_decision": "Pending component-specific author review; predecessor CC-BY-4.0 is not automatic clearance.",
        "new_doi": None, "remote_draft_created": False, "published": False,
        "source_commit": commit, "source_is_author_approved_tag": False,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Private archive preparation\n\n"
        "This is a local candidate, not a Zenodo draft, deposited archive, reserved DOI, or release. "
        "The predecessor was verified through the public Zenodo API; its metadata is saved here. "
        "No remote state was changed.\n\n"
        "The source ZIP is from the frozen release branch commit, not an approved version tag. "
        "It includes model artifacts and configuration. Monthly NetCDF copies are byte-verified. "
        "December 2016 accompanies the twelve calendar-2017 months solely as seasonal context.\n\n"
        f"The {len(table):,}-row upstream/derived collocation table is inventoried but deliberately not "
        "included in the candidate payload. Authors must approve the actual archived scope and "
        "complete the site/PI attribution and AERONET usage review. Do not claim that all of these "
        "rows are training observations. See collocation_scope_review.json for actual source counts.\n\n"
        "Next gates: authenticated Zenodo editing access; creator/license/attribution approval; "
        "approved release merge/tag; finalize the payload and create the linked version draft. "
        "DOI reservation is authorized by the task if feasible, but has not happened here. "
        "Publication still requires explicit author approval. Never substitute the predecessor DOI "
        "for a new version DOI.\n")
    print(json.dumps({"output": str(output), "candidate_files": len(manifest),
                      "candidate_bytes": sum(row["bytes"] for row in manifest),
                      "remote_draft_created": False}))


if __name__ == "__main__":
    main()
