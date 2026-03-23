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
class JobSpec:
    session_name: str
    short_name: str
    method: str
    start_year: int
    end_year: int
    base_output_dir: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Watch MODIS L3 DAOD backfill jobs, restart missing ones, and stop when complete."
    )
    parser.add_argument("--interval-seconds", type=int, default=900, help="Polling interval; 900 s = 15 minutes.")
    parser.add_argument("--base-output-dir", type=Path, default=here / "modis_l3_daod_backfill")
    parser.add_argument("--run-script", type=Path, default=here / "run_modis_l3_daod_backfill.py")
    return parser.parse_args()


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def month_range(start_year: int, end_year: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append((year, month))
    return months


def filename_prefix(short_name: str, method: str) -> str:
    if method == "li_ginoux":
        return f"{short_name}_DustAOD_LiGinoux_"
    if method == "xgboost":
        return f"{short_name}_DustAOD_XGB_"
    raise ValueError(f"Unsupported method: {method}")


def output_dir(base_output_dir: Path, short_name: str, method: str) -> Path:
    suffix = "LiGinoux" if method == "li_ginoux" else "XGBoost"
    return base_output_dir / f"{short_name}_{suffix}"


def month_complete(job: JobSpec, year: int, month: int) -> bool:
    out_dir = output_dir(job.base_output_dir, job.short_name, job.method)
    prefix = filename_prefix(job.short_name, job.method)
    expected = [
        out_dir / f"{prefix}{date(year, month, day).isoformat()}.nc"
        for day in range(1, days_in_month(year, month) + 1)
    ]
    return all(path.exists() for path in expected)


def all_complete(job: JobSpec) -> bool:
    return all(month_complete(job, year, month) for year, month in month_range(job.start_year, job.end_year))


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", session_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def tmux_session_running_job(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_current_command}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    commands = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not commands:
        return False
    # A finished tmux job drops back to an interactive shell; treat that as not alive.
    return any(cmd not in {"bash", "sh", "zsh"} for cmd in commands)


def start_job(job: JobSpec, run_script: Path) -> None:
    cmd = (
        f"cd {Path(__file__).resolve().parent} && "
        f"python {run_script} "
        f"--start-year {job.start_year} --end-year {job.end_year} "
        f"--short-name {job.short_name} --methods {job.method} "
        f"--base-output-dir {job.base_output_dir} "
        f"2>&1 | tee {job.log_path}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", job.session_name, cmd], check=True)
    print(f"[STARTED] {job.session_name}")


def count_files(job: JobSpec) -> int:
    out_dir = output_dir(job.base_output_dir, job.short_name, job.method)
    if not out_dir.exists():
        return 0
    return sum(1 for _ in out_dir.glob("*.nc"))


def main() -> None:
    args = parse_args()
    jobs = [
        JobSpec(
            session_name="daod_li_ginoux_2002_2024",
            short_name="MOD08_D3",
            method="li_ginoux",
            start_year=2002,
            end_year=2024,
            base_output_dir=args.base_output_dir,
            log_path=Path(__file__).resolve().parent / "modis_l3_daod_backfill_li_ginoux.log",
        ),
        JobSpec(
            session_name="daod_xgboost_2002_2024",
            short_name="MOD08_D3",
            method="xgboost",
            start_year=2002,
            end_year=2024,
            base_output_dir=args.base_output_dir,
            log_path=Path(__file__).resolve().parent / "modis_l3_daod_backfill_xgboost.log",
        ),
        JobSpec(
            session_name="daod_li_ginoux_aqua_2022_2024",
            short_name="MYD08_D3",
            method="li_ginoux",
            start_year=2022,
            end_year=2024,
            base_output_dir=args.base_output_dir,
            log_path=Path(__file__).resolve().parent / "modis_l3_daod_backfill_li_ginoux_aqua_2022_2024.log",
        ),
        JobSpec(
            session_name="daod_xgboost_aqua_2022_2024",
            short_name="MYD08_D3",
            method="xgboost",
            start_year=2022,
            end_year=2024,
            base_output_dir=args.base_output_dir,
            log_path=Path(__file__).resolve().parent / "modis_l3_daod_backfill_xgboost_aqua_2022_2024.log",
        ),
    ]

    while True:
        all_done = True
        print("[CHECK] supervisor pass")
        for job in jobs:
            complete = all_complete(job)
            nfiles = count_files(job)
            session_exists = tmux_session_exists(job.session_name)
            alive = session_exists and tmux_session_running_job(job.session_name)
            print(
                f"[STATUS] {job.session_name}: complete={complete} alive={alive} "
                f"files={nfiles}"
            )
            if complete:
                continue
            all_done = False
            if not alive:
                if session_exists:
                    subprocess.run(
                        ["tmux", "kill-session", "-t", job.session_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                start_job(job, args.run_script)
        if all_done:
            print("[SUMMARY] all jobs complete")
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
