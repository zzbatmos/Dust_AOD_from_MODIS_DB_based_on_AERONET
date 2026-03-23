#!/usr/bin/env python3

from __future__ import annotations

import argparse
import calendar
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class MethodSpec:
    name: str
    script: Path
    output_dir: Path
    filename_template: str
    extra_args: list[str]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run resumable monthly backfill for MODIS L3 DAOD products."
    )
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--short-name", choices=["MOD08_D3", "MYD08_D3"], default="MOD08_D3")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["li_ginoux", "xgboost"],
        default=["li_ginoux", "xgboost"],
        help="Which DAOD methods to run.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=here / "modis_l3_daod_backfill",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any monthly job fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned monthly jobs without running them.",
    )
    parser.add_argument(
        "--month-retries",
        type=int,
        default=3,
        help="How many times to retry a failed monthly subprocess before moving on.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=int,
        default=30,
        help="Sleep between monthly retries.",
    )
    return parser.parse_args()


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def month_range(start_year: int, end_year: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append((year, month))
    return months


def build_method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    here = Path(__file__).resolve().parent
    short_name = args.short_name
    specs: list[MethodSpec] = []
    if "li_ginoux" in args.methods:
        specs.append(
            MethodSpec(
                name="li_ginoux",
                script=here / "DAOD_from_DB_L3_LiGinoux.py",
                output_dir=args.base_output_dir / f"{short_name}_LiGinoux",
                filename_template=f"{short_name}_DustAOD_LiGinoux" + "_{date}.nc",
                extra_args=["--platform", "MOD" if short_name.startswith("MOD") else "MYD"],
            )
        )
    if "xgboost" in args.methods:
        specs.append(
            MethodSpec(
                name="xgboost",
                script=here / "DAOD_from_DB_L3_XGBoost.py",
                output_dir=args.base_output_dir / f"{short_name}_XGBoost",
                filename_template=f"{short_name}_DustAOD_XGB" + "_{date}.nc",
                extra_args=[],
            )
        )
    return specs


def expected_monthly_files(spec: MethodSpec, year: int, month: int) -> list[Path]:
    count = days_in_month(year, month)
    return [
        spec.output_dir / spec.filename_template.format(date=date(year, month, day).isoformat())
        for day in range(1, count + 1)
    ]


def month_complete(spec: MethodSpec, year: int, month: int) -> bool:
    expected = expected_monthly_files(spec, year, month)
    return all(path.exists() for path in expected)


def run_month(
    spec: MethodSpec,
    short_name: str,
    year: int,
    month: int,
    dry_run: bool = False,
    month_retries: int = 1,
    retry_sleep_seconds: int = 30,
) -> int:
    start = date(year, month, 1).isoformat()
    end = date(year, month, days_in_month(year, month)).isoformat()
    cmd = [
        sys.executable,
        str(spec.script),
        "--start",
        start,
        "--end",
        end,
        "--short-name",
        short_name,
        "--output-dir",
        str(spec.output_dir),
        *spec.extra_args,
    ]
    print(f"[RUN] {spec.name} {start}..{end}")
    if dry_run:
        print(" ".join(cmd))
        return 0
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    attempts = max(1, month_retries)
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, check=False)
        if int(result.returncode) == 0 and month_complete(spec, year, month):
            return 0
        if attempt < attempts:
            print(
                f"[RETRY] {spec.name} {year:04d}-{month:02d} attempt {attempt}/{attempts} failed; "
                f"sleeping {retry_sleep_seconds}s"
            )
            time.sleep(retry_sleep_seconds)
    return int(result.returncode)


def main() -> None:
    args = parse_args()
    specs = build_method_specs(args)
    if not specs:
        raise ValueError("No methods selected.")

    failures: list[str] = []
    for year, month in month_range(args.start_year, args.end_year):
        for spec in specs:
            if month_complete(spec, year, month):
                print(f"[SKIP complete] {spec.name} {year:04d}-{month:02d}")
                continue
            code = run_month(
                spec,
                args.short_name,
                year,
                month,
                dry_run=args.dry_run,
                month_retries=args.month_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            if code != 0:
                message = f"{spec.name} failed for {year:04d}-{month:02d} with exit code {code}"
                print(f"[ERROR] {message}")
                failures.append(message)
                if args.stop_on_error:
                    raise SystemExit(1)

    if failures:
        print("[SUMMARY] Completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("[SUMMARY] Backfill completed successfully.")


if __name__ == "__main__":
    main()
