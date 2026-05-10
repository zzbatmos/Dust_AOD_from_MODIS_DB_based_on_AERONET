#!/usr/bin/env python3
"""Plot daily maps for high-impact high-AOD dust-rescue days."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - cartopy is optional.
    ccrs = None

from apply_two_stage_high_aod_dust_rescue_test import (
    ROOT,
    TYPE_LABELS,
    apply_high_aod_rescue,
    compute_two_stage_qa2,
    infer_li_ae_screen,
    type_labels_to_codes,
    wrap_lon,
)
from run_high_aod_rescue_regional_daily_test import REGIONAL_TESTS


DAILY_SUMMARY = (
    Path(__file__).resolve().parent
    / "two_stage_high_aod_dust_rescue_test"
    / "regional_daily_top_setting"
    / "regional_daily_all_tests.csv"
)
OUT_DIR = (
    Path(__file__).resolve().parent
    / "two_stage_high_aod_dust_rescue_test"
    / "regional_daily_top_setting"
    / "high_impact_daily_maps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--daily-summary", type=Path, default=DAILY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--test",
        action="append",
        choices=sorted(REGIONAL_TESTS),
        default=None,
        help="Regional test to plot. Defaults to the December high-impact tests.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--qa2-threshold", type=float, default=0.6)
    parser.add_argument("--fmf-max", type=float, default=0.7)
    parser.add_argument("--high-aod-threshold", type=float, default=2.0)
    parser.add_argument("--li-dust-threshold", type=float, default=0.8)
    parser.add_argument("--min-two-stage-probability", type=float, default=0.25)
    parser.add_argument("--rescue-cap-fraction", type=float, default=0.95)
    parser.add_argument("--allowed-db-types", default="Dust")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def robust_max(arrays: list[np.ndarray], percentile: float, minimum: float) -> float:
    parts = [arr[np.isfinite(arr)].ravel() for arr in arrays if np.any(np.isfinite(arr))]
    if not parts:
        return minimum
    merged = np.concatenate(parts)
    if merged.size == 0:
        return minimum
    return float(max(np.nanpercentile(merged, percentile), minimum))


def add_map_context(ax, test_name: str) -> None:
    test = REGIONAL_TESTS[test_name]
    if ccrs:
        ax.set_extent([test.lon_min, test.lon_max, test.lat_min, test.lat_max], crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="110m", linewidth=0.7)
        except Exception:
            pass
        ax.gridlines(draw_labels=True, linewidth=0.3, color="0.4", alpha=0.5)
    else:
        ax.set_xlim(test.lon_min, test.lon_max)
        ax.set_ylim(test.lat_min, test.lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)


def collect_day_arrays(
    test_name: str,
    day: str,
    args: argparse.Namespace,
    allowed_db_type_codes: set[int],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    test = REGIONAL_TESTS[test_name]
    day_dir = args.input_root / day / "granules"
    if not day_dir.exists():
        raise FileNotFoundError(f"Missing granule directory: {day_dir}")

    pieces: dict[str, list[np.ndarray]] = {
        "lat": [],
        "lon": [],
        "aod550": [],
        "li_ae142": [],
        "two_stage_qa2": [],
        "two_stage_qa2_rescue": [],
        "rescue_increment": [],
        "rescue_flag": [],
        "two_stage_probability": [],
        "aerosol_type": [],
    }
    rows: list[dict[str, object]] = []

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
        rescued, flag, _candidate, increment = apply_high_aod_rescue(
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

        rows.append(
            {
                "date": day,
                "test": test_name,
                "granule": npz_path.stem,
                "path": str(npz_path),
                "n_pixels": n_pixels,
                "rescued_pixels": int(np.count_nonzero(flag[mask] > 0)),
            }
        )
        arrays = {
            "lat": lat,
            "lon": lon,
            "aod550": aod,
            "li_ae142": li_ae142,
            "two_stage_qa2": two_stage_qa2,
            "two_stage_qa2_rescue": rescued,
            "rescue_increment": increment,
            "rescue_flag": flag,
            "two_stage_probability": probability,
            "aerosol_type": aerosol_type,
        }
        for key, arr in arrays.items():
            pieces[key].append(arr[mask])

    if not rows:
        raise RuntimeError(f"No selected granules for {test_name} {day}")
    return rows, {key: np.concatenate(parts) for key, parts in pieces.items()}


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


def plot_day(test_name: str, day: str, data: dict[str, np.ndarray], args: argparse.Namespace, out_path: Path) -> None:
    test = REGIONAL_TESTS[test_name]
    aod_vmax = robust_max([data["aod550"]], 98.0, 0.2)
    daod_vmax = robust_max([data["li_ae142"], data["two_stage_qa2"], data["two_stage_qa2_rescue"]], 98.0, 0.05)
    inc_vmax = robust_max([data["rescue_increment"]], 99.0, 0.02)
    panels = [
        ("aod550", "DB AOD550", "viridis", 0.0, aod_vmax),
        ("li_ae142", "Li-Ginoux DAOD, AE<1.42", "plasma", 0.0, daod_vmax),
        ("two_stage_qa2", "Two-stage QA2 DAOD", "plasma", 0.0, daod_vmax),
        ("two_stage_qa2_rescue", "Two-stage QA2 + rescue", "plasma", 0.0, daod_vmax),
        ("rescue_increment", "Rescue increment", "magma", 0.0, inc_vmax),
        ("rescue_flag", "Rescued pixels", "gray_r", 0.0, 1.0),
        ("two_stage_probability", "Two-stage dust probability", "viridis", 0.0, 1.0),
        ("aerosol_type", "DB aerosol type", "tab10", -0.5, 3.5),
    ]

    projection = ccrs.PlateCarree() if ccrs else None
    subplot_kw = {"projection": projection} if projection else {}
    transform = ccrs.PlateCarree() if ccrs else None
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), subplot_kw=subplot_kw, constrained_layout=True)
    for ax, (key, title, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        add_map_context(ax, test_name)
        kwargs = {
            "c": data[key],
            "s": 3,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "linewidths": 0,
            "alpha": 0.9,
        }
        if transform:
            kwargs["transform"] = transform
        mesh = ax.scatter(data["lon"], data["lat"], **kwargs)
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
        if key == "aerosol_type":
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels([TYPE_LABELS[i] for i in range(4)])
        elif key == "rescue_flag":
            cbar.set_ticks([0, 1])

    thresholds = (
        f"AOD>={args.high_aod_threshold:g}, Li DAOD>={args.li_dust_threshold:g}, "
        f"Pdust>={args.min_two_stage_probability:g}, cap={args.rescue_cap_fraction:g}*AOD, "
        f"DB types={args.allowed_db_types}"
    )
    fig.suptitle(f"{test.label} | {day}\n{thresholds}", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def selected_days(args: argparse.Namespace) -> pd.DataFrame:
    tests = args.test or ["dec2017_north_africa", "dec2017_sahel"]
    df = pd.read_csv(args.daily_summary)
    df = df[df["test"].isin(tests)].copy()
    df = df[pd.to_numeric(df["n_valid_aod_pixels"], errors="coerce") > 0]
    df["mean_rescue_increment"] = pd.to_numeric(df["mean_rescue_increment"], errors="coerce")
    selected = []
    for test_name, group in df.groupby("test"):
        selected.append(group.sort_values("mean_rescue_increment", ascending=False).head(args.top_n))
    return pd.concat(selected, ignore_index=True).sort_values(["test", "mean_rescue_increment"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_db_type_codes = type_labels_to_codes(args.allowed_db_types)
    selected = selected_days(args)
    summary_rows: list[dict[str, object]] = []

    for row in selected.to_dict(orient="records"):
        test_name = str(row["test"])
        day = str(row["date"])
        rows, data = collect_day_arrays(test_name, day, args, allowed_db_type_codes)
        fig_path = args.output_dir / test_name / f"{test_name}_{day}_high_aod_rescue_map.png"
        plot_day(test_name, day, data, args, fig_path)
        granule_csv = args.output_dir / test_name / f"{test_name}_{day}_selected_granules.csv"
        write_csv(granule_csv, rows)
        summary = dict(row)
        summary["figure"] = str(fig_path)
        summary["selected_granules_csv"] = str(granule_csv)
        summary_rows.append(summary)
        print(f"[OK] {test_name} {day}: {fig_path}")

    write_csv(args.output_dir / "selected_high_impact_daily_maps.csv", summary_rows)
    print(f"Summary: {args.output_dir / 'selected_high_impact_daily_maps.csv'}")


if __name__ == "__main__":
    main()
