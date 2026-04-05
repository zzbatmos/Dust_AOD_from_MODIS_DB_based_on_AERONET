from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(".").resolve()))

import earthaccess  # type: ignore

from augment_collocation_with_db_qa import augment_site_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill DB QA-augmented per-site collocation files. The run is resumable: "
            "existing output files are skipped."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../AERONET_MODIS_DB_collocation_files"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("AERONET_MODIS_DB_collocation_files_with_QA"),
    )
    parser.add_argument(
        "--short-name",
        choices=["MOD04_L2", "MYD04_L2", "both"],
        default="both",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--radius-km", type=float, default=25.0)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("./temp_downloads"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if the QA-augmented file already exists.",
    )
    return parser.parse_args()


def gather_site_files(input_dir: Path, short_name: str) -> list[Path]:
    patterns = []
    if short_name in {"MOD04_L2", "both"}:
        patterns.append("AERONET_MOD04_L2_collocation_*.pkl")
    if short_name in {"MYD04_L2", "both"}:
        patterns.append("AERONET_MYD04_L2_collocation_*.pkl")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_dir.glob(pattern)))
    return sorted(files)


def output_name(site_file: Path) -> str:
    return site_file.stem + "_with_QA.pkl"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    all_files = gather_site_files(args.input_dir, args.short_name)
    if args.shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")

    shard_files = [p for i, p in enumerate(all_files) if i % args.shard_count == args.shard_index]
    print(
        f"[INFO] short_name={args.short_name} total_files={len(all_files)} "
        f"shard_index={args.shard_index} shard_count={args.shard_count} shard_files={len(shard_files)}"
    )

    earthaccess.login(strategy="netrc")

    completed = 0
    skipped = 0
    failed = 0
    for idx, site_file in enumerate(shard_files, start=1):
        out_file = args.output_dir / output_name(site_file)
        if out_file.exists() and not args.overwrite:
            skipped += 1
            print(f"[SKIP {idx}/{len(shard_files)}] {site_file.name} -> {out_file.name}")
            continue
        print(f"[RUN  {idx}/{len(shard_files)}] {site_file.name}")
        try:
            augment_site_file(
                site_file=site_file,
                output_file=out_file,
                download_dir=args.download_dir,
                radius_km=args.radius_km,
                keep_downloads=False,
            )
            completed += 1
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"[FAIL {idx}/{len(shard_files)}] {site_file.name}: {exc}")

    print(
        f"[SUMMARY] completed={completed} skipped={skipped} failed={failed} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
