#!/usr/bin/env python3
"""Run limited daily regional tests for the high-AOD dust-rescue rule."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apply_two_stage_high_aod_dust_rescue_test import (
    ROOT,
    TYPE_LABELS,
    apply_high_aod_rescue,
    compute_two_stage_qa2,
    infer_li_ae_screen,
    type_labels_to_codes,
    wrap_lon,
)


OUT_DIR = (
    Path(__file__).resolve().parent
    / "two_stage_high_aod_dust_rescue_test"
    / "regional_daily_top_setting"
)


@dataclass(frozen=True)
class RegionalTest:
    name: str
    label: str
    start: date
    end: date
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    min_pixels_per_granule: int = 20


REGIONAL_TESTS = {
    "aug2017_north_america": RegionalTest(
        name="aug2017_north_america",
        label="August 2017 North America smoke stress test",
        start=date(2017, 8, 1),
        end=date(2017, 8, 31),
        lon_min=-132.0,
        lon_max=-85.0,
        lat_min=32.0,
        lat_max=60.0,
        min_pixels_per_granule=20,
    ),
    "aug2017_north_africa": RegionalTest(
        name="aug2017_north_africa",
        label="August 2017 North Africa/Sahara dust stress test",
        start=date(2017, 8, 1),
        end=date(2017, 8, 31),
        lon_min=-20.0,
        lon_max=60.0,
        lat_min=0.0,
        lat_max=40.0,
        min_pixels_per_granule=50,
    ),
    "dec2017_north_africa": RegionalTest(
        name="dec2017_north_africa",
        label="December 2017 North Africa/Sahara-Sahel dust stress test",
        start=date(2017, 12, 1),
        end=date(2017, 12, 31),
        lon_min=-20.0,
        lon_max=60.0,
        lat_min=0.0,
        lat_max=40.0,
        min_pixels_per_granule=50,
    ),
    "dec2017_sahel": RegionalTest(
        name="dec2017_sahel",
        label="December 2017 West/Central Sahel focused test",
        start=date(2017, 12, 1),
        end=date(2017, 12, 31),
        lon_min=-15.0,
        lon_max=25.0,
        lat_min=5.0,
        lat_max=18.0,
        min_pixels_per_granule=20,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--test",
        action="append",
        choices=sorted(REGIONAL_TESTS),
        help="Regional test to run. May be repeated. Defaults to all tests.",
    )
    parser.add_argument("--qa2-threshold", type=float, default=0.6)
    parser.add_argument("--fmf-max", type=float, default=0.7)
    parser.add_argument("--high-aod-threshold", type=float, default=2.0)
    parser.add_argument("--li-dust-threshold", type=float, default=0.8)
    parser.add_argument("--min-two-stage-probability", type=float, default=0.25)
    parser.add_argument("--rescue-cap-fraction", type=float, default=0.95)
    parser.add_argument("--allowed-db-types", default="Dust")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def date_range(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def finite_mean(values: list[np.ndarray]) -> float:
    if not values:
        return float("nan")
    merged = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def count_type(values: list[np.ndarray], code: int) -> int:
    if not values:
        return 0
    merged = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    return int(np.count_nonzero(merged == code))


def collect_day(
    day: date,
    test: RegionalTest,
    args: argparse.Namespace,
    allowed_db_type_codes: set[int],
) -> dict[str, object]:
    day_dir = args.input_root / day.isoformat() / "granules"
    row: dict[str, object] = {
        "test": test.name,
        "label": test.label,
        "date": day.isoformat(),
        "lon_min": test.lon_min,
        "lon_max": test.lon_max,
        "lat_min": test.lat_min,
        "lat_max": test.lat_max,
        "n_granules": 0,
        "n_valid_aod_pixels": 0,
        "rescue_candidate_pixels": 0,
        "rescued_pixels": 0,
    }
    if not day_dir.exists():
        return row

    aod_values: list[np.ndarray] = []
    li_values: list[np.ndarray] = []
    two_values: list[np.ndarray] = []
    rescue_values: list[np.ndarray] = []
    increment_values: list[np.ndarray] = []
    probability_values: list[np.ndarray] = []
    fraction_values: list[np.ndarray] = []
    aerosol_type_values: list[np.ndarray] = []

    for npz_path in sorted(day_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            lat = np.asarray(data["lat"], dtype=float)
            lon = wrap_lon(np.asarray(data["lon"], dtype=float))
            aod = np.asarray(data["aod550"], dtype=float)
            li_original = np.asarray(data["li_dust_aod"], dtype=float)
            probability = np.asarray(data["two_stage_probability"], dtype=float)
            fraction = np.asarray(data["two_stage_fraction"], dtype=float)
            aerosol_type = np.asarray(data["aerosol_type_code"], dtype=float)

        mask = (
            (lat >= test.lat_min)
            & (lat <= test.lat_max)
            & (lon >= test.lon_min)
            & (lon <= test.lon_max)
            & np.isfinite(aod)
        )
        n_pixels = int(np.count_nonzero(mask))
        if n_pixels < test.min_pixels_per_granule:
            continue

        li_ae142 = infer_li_ae_screen(li_original, aod, args.fmf_max)
        two_stage_qa2 = compute_two_stage_qa2(aod, probability, fraction, args.qa2_threshold)
        rescued, flag, candidate, increment = apply_high_aod_rescue(
            aod=aod,
            li_ae_screened=li_ae142,
            two_stage_qa2=two_stage_qa2,
            probability=probability,
            aerosol_type=aerosol_type,
            high_aod_threshold=args.high_aod_threshold,
            li_dust_threshold=args.li_dust_threshold,
            min_probability=args.min_two_stage_probability,
            cap_fraction=args.rescue_cap_fraction,
            allowed_db_type_codes=allowed_db_type_codes,
        )

        row["n_granules"] = int(row["n_granules"]) + 1
        row["n_valid_aod_pixels"] = int(row["n_valid_aod_pixels"]) + n_pixels
        row["rescue_candidate_pixels"] = int(row["rescue_candidate_pixels"]) + int(np.count_nonzero(candidate[mask] > 0))
        row["rescued_pixels"] = int(row["rescued_pixels"]) + int(np.count_nonzero(flag[mask] > 0))

        aod_values.append(aod[mask])
        li_values.append(li_ae142[mask])
        two_values.append(two_stage_qa2[mask])
        rescue_values.append(rescued[mask])
        increment_values.append(increment[mask])
        probability_values.append(probability[mask])
        fraction_values.append(fraction[mask])
        aerosol_type_values.append(aerosol_type[mask])

    row["mean_aod550"] = finite_mean(aod_values)
    row["mean_li_ginoux_ae142"] = finite_mean(li_values)
    row["mean_two_stage_qa2"] = finite_mean(two_values)
    row["mean_rescued_qa2"] = finite_mean(rescue_values)
    row["mean_rescue_increment"] = finite_mean(increment_values)
    row["mean_two_stage_probability"] = finite_mean(probability_values)
    row["mean_two_stage_fraction"] = finite_mean(fraction_values)
    n = int(row["n_valid_aod_pixels"])
    row["rescued_pixel_fraction"] = float(row["rescued_pixels"] / n) if n else float("nan")
    row["rescue_candidate_pixel_fraction"] = float(row["rescue_candidate_pixels"] / n) if n else float("nan")
    for code, label in TYPE_LABELS.items():
        row[f"aerosol_type_{label.lower()}_pixels"] = count_type(aerosol_type_values, code)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_test(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if int(row["n_valid_aod_pixels"]) > 0]
    if not valid:
        return {}
    df = pd.DataFrame(valid)
    for col in [
        "mean_aod550",
        "mean_li_ginoux_ae142",
        "mean_two_stage_qa2",
        "mean_rescued_qa2",
        "mean_rescue_increment",
        "rescued_pixel_fraction",
        "rescue_candidate_pixel_fraction",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    max_row = df.loc[df["mean_rescue_increment"].idxmax()]
    return {
        "test": str(valid[0]["test"]),
        "label": str(valid[0]["label"]),
        "n_days_with_data": int(len(valid)),
        "daily_mean_aod550": float(df["mean_aod550"].mean(skipna=True)),
        "daily_mean_li_ginoux_ae142": float(df["mean_li_ginoux_ae142"].mean(skipna=True)),
        "daily_mean_two_stage_qa2": float(df["mean_two_stage_qa2"].mean(skipna=True)),
        "daily_mean_rescued_qa2": float(df["mean_rescued_qa2"].mean(skipna=True)),
        "daily_mean_rescue_increment": float(df["mean_rescue_increment"].mean(skipna=True)),
        "daily_mean_rescued_pixel_fraction": float(df["rescued_pixel_fraction"].mean(skipna=True)),
        "max_increment_date": str(max_row["date"]),
        "max_increment": float(max_row["mean_rescue_increment"]),
        "max_increment_rescued_pixel_fraction": float(max_row["rescued_pixel_fraction"]),
    }


def plot_test(rows: list[dict[str, object]], out_path: Path, args: argparse.Namespace) -> None:
    valid = [row for row in rows if int(row["n_valid_aod_pixels"]) > 0]
    if not valid:
        return
    df = pd.DataFrame(valid)
    df["date"] = pd.to_datetime(df["date"])
    for col in [
        "mean_li_ginoux_ae142",
        "mean_two_stage_qa2",
        "mean_rescued_qa2",
        "mean_rescue_increment",
        "rescued_pixel_fraction",
        "n_valid_aod_pixels",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(df["date"], df["mean_li_ginoux_ae142"], label="Li-Ginoux AE<1.42", color="#444444", linewidth=2)
    axes[0].plot(df["date"], df["mean_two_stage_qa2"], label="Two-stage QA2", color="#2b6cb0", linewidth=2)
    axes[0].plot(df["date"], df["mean_rescued_qa2"], label="Two-stage QA2 + rescue", color="#d97706", linewidth=2)
    axes[0].set_ylabel("Mean DAOD550")
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].bar(df["date"], df["mean_rescue_increment"], color="#d97706", width=0.8)
    axes[1].set_ylabel("Rescue increment")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["date"], df["rescued_pixel_fraction"], color="#7c2d12", marker="o", markersize=3)
    axes[2].set_ylabel("Rescued pixel fraction")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    title = str(valid[0]["label"])
    thresholds = (
        f"AOD>={args.high_aod_threshold:g}, Li DAOD>={args.li_dust_threshold:g}, "
        f"Pdust>={args.min_two_stage_probability:g}, cap={args.rescue_cap_fraction:g}*AOD, "
        f"DB types={args.allowed_db_types}"
    )
    fig.suptitle(f"{title}\n{thresholds}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_db_type_codes = type_labels_to_codes(args.allowed_db_types)
    selected = args.test or sorted(REGIONAL_TESTS)

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for name in selected:
        test = REGIONAL_TESTS[name]
        rows = [
            collect_day(day, test, args, allowed_db_type_codes)
            for day in date_range(test.start, test.end)
        ]
        all_rows.extend(rows)
        summary = summarize_test(rows)
        if summary:
            summary["daily_csv"] = str(args.output_dir / name / f"{name}_daily_summary.csv")
            summary["timeseries_png"] = str(args.output_dir / name / f"{name}_timeseries.png")
            summary_rows.append(summary)
        write_csv(args.output_dir / name / f"{name}_daily_summary.csv", rows)
        plot_test(rows, args.output_dir / name / f"{name}_timeseries.png", args)
        print(f"[OK] {name}: {args.output_dir / name / f'{name}_daily_summary.csv'}")

    write_csv(args.output_dir / "regional_daily_all_tests.csv", all_rows)
    write_csv(args.output_dir / "regional_daily_summary.csv", summary_rows)
    metadata = {
        "thresholds": {
            "qa2_threshold": args.qa2_threshold,
            "fmf_max": args.fmf_max,
            "high_aod_threshold": args.high_aod_threshold,
            "li_dust_threshold": args.li_dust_threshold,
            "min_two_stage_probability": args.min_two_stage_probability,
            "rescue_cap_fraction": args.rescue_cap_fraction,
            "allowed_db_types": args.allowed_db_types,
        },
        "tests": [REGIONAL_TESTS[name].__dict__ for name in selected],
        "outputs": {
            "all_daily_csv": str(args.output_dir / "regional_daily_all_tests.csv"),
            "summary_csv": str(args.output_dir / "regional_daily_summary.csv"),
        },
    }
    with (args.output_dir / "regional_daily_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f"Summary: {args.output_dir / 'regional_daily_summary.csv'}")


if __name__ == "__main__":
    main()
