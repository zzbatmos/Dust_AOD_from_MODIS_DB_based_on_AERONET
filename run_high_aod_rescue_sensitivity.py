#!/usr/bin/env python3
"""Run a threshold sensitivity sweep for the high-AOD dust-rescue rule."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from apply_two_stage_high_aod_dust_rescue_test import (
    CASES,
    ROOT,
    apply_high_aod_rescue,
    compute_two_stage_qa2,
    finite_mean,
    finite_percentile,
    infer_li_ae_screen,
    region_mask,
    type_labels_to_codes,
    wrap_lon,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "two_stage_high_aod_dust_rescue_test"
    / "sensitivity"
)


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def parse_type_sets(text: str) -> list[str]:
    values = [x.strip() for x in text.split(";") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one DB type set.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qa2-threshold", type=float, default=0.6)
    parser.add_argument("--fmf-max", type=float, default=0.7)
    parser.add_argument("--aod-thresholds", type=parse_float_list, default=parse_float_list("1.2,1.5,2.0"))
    parser.add_argument("--li-thresholds", type=parse_float_list, default=parse_float_list("0.8,1.0,1.2"))
    parser.add_argument("--prob-thresholds", type=parse_float_list, default=parse_float_list("0.25,0.35,0.45"))
    parser.add_argument("--cap-fractions", type=parse_float_list, default=parse_float_list("0.85,0.90,0.95"))
    parser.add_argument(
        "--db-type-sets",
        type=parse_type_sets,
        default=parse_type_sets("Dust;Dust,Mixed"),
        help="Semicolon-separated DB type sets, e.g. 'Dust;Dust,Mixed'.",
    )
    parser.add_argument("--max-aug-full-increment", type=float, default=0.002)
    parser.add_argument("--max-aug-central-increment", type=float, default=0.005)
    parser.add_argument("--min-dec7-core-recovery", type=float, default=0.5)
    return parser.parse_args()


def load_case_base(case_name: str, input_root: Path, qa2_threshold: float, fmf_max: float) -> dict[str, np.ndarray]:
    case = CASES[case_name]
    granule_dir = input_root / case.date / "granules"
    if not granule_dir.exists():
        raise FileNotFoundError(f"Missing granule directory: {granule_dir}")

    pieces: dict[str, list[np.ndarray]] = {
        "lat": [],
        "lon": [],
        "aod550": [],
        "li_ae142": [],
        "two_stage_probability": [],
        "two_stage_fraction": [],
        "two_stage_qa2": [],
        "aerosol_type": [],
    }

    for npz_path in sorted(granule_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            lat = np.asarray(data["lat"], dtype=float)
            lon = wrap_lon(np.asarray(data["lon"], dtype=float))
            aod = np.asarray(data["aod550"], dtype=float)
            li_original = np.asarray(data["li_dust_aod"], dtype=float)
            probability = np.asarray(data["two_stage_probability"], dtype=float)
            fraction = np.asarray(data["two_stage_fraction"], dtype=float)
            aerosol_type = np.asarray(data["aerosol_type_code"], dtype=float)

        mask = (
            (lat >= case.lat_min)
            & (lat <= case.lat_max)
            & (lon >= case.lon_min)
            & (lon <= case.lon_max)
            & np.isfinite(aod)
        )
        if int(np.count_nonzero(mask)) < case.min_pixels:
            continue

        li_ae142 = infer_li_ae_screen(li_original, aod, fmf_max=fmf_max)
        two_stage_qa2 = compute_two_stage_qa2(aod, probability, fraction, qa2_threshold)
        arrays = {
            "lat": lat,
            "lon": lon,
            "aod550": aod,
            "li_ae142": li_ae142,
            "two_stage_probability": probability,
            "two_stage_fraction": fraction,
            "two_stage_qa2": two_stage_qa2,
            "aerosol_type": aerosol_type,
        }
        for key, arr in arrays.items():
            pieces[key].append(arr[mask])

    if not pieces["aod550"]:
        raise RuntimeError(f"No valid granules found for {case_name}")
    return {key: np.concatenate(parts) for key, parts in pieces.items()}


def summarize_roi(
    case_name: str,
    roi_name: str,
    mask: np.ndarray,
    data: dict[str, np.ndarray],
    rescued: np.ndarray,
    rescue_flag: np.ndarray,
    rescue_candidate: np.ndarray,
    increment: np.ndarray,
) -> dict[str, object]:
    n = int(np.count_nonzero(mask))
    standard = finite_mean(data["two_stage_qa2"][mask])
    rescue_mean = finite_mean(rescued[mask])
    li_mean = finite_mean(data["li_ae142"][mask])
    increase = rescue_mean - standard
    denominator = li_mean - standard
    recovery_ratio = float(increase / denominator) if np.isfinite(denominator) and denominator > 0 else float("nan")
    return {
        "case": case_name,
        "roi": roi_name,
        "n_valid_aod_pixels": n,
        "mean_aod550": finite_mean(data["aod550"][mask]),
        "mean_li_ginoux_ae142": li_mean,
        "mean_two_stage_qa2": standard,
        "mean_rescued_qa2": rescue_mean,
        "mean_rescue_increment": increase,
        "recovery_ratio_to_li": recovery_ratio,
        "rescued_pixels": int(np.count_nonzero(mask & (rescue_flag > 0))),
        "rescue_candidate_pixels": int(np.count_nonzero(mask & (rescue_candidate > 0))),
        "rescued_pixel_fraction": float(np.count_nonzero(mask & (rescue_flag > 0)) / n) if n else float("nan"),
        "p90_rescue_increment": finite_percentile(increment[mask], 90.0),
    }


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_data = {
        name: load_case_base(name, args.input_root, args.qa2_threshold, args.fmf_max)
        for name in ("dec7_north_africa", "aug29_north_america")
    }
    roi_masks: dict[tuple[str, str], np.ndarray] = {}
    for case_name, data in case_data.items():
        case = CASES[case_name]
        for roi_name, lon_min, lon_max, lat_min, lat_max in case.rois:
            roi_masks[(case_name, roi_name)] = region_mask(data, lon_min, lon_max, lat_min, lat_max)

    roi_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    grid = itertools.product(
        args.aod_thresholds,
        args.li_thresholds,
        args.prob_thresholds,
        args.cap_fractions,
        args.db_type_sets,
    )
    run_id = 0
    for aod_threshold, li_threshold, prob_threshold, cap_fraction, db_type_set in grid:
        run_id += 1
        allowed_codes = type_labels_to_codes(db_type_set)
        selected: dict[tuple[str, str], dict[str, object]] = {}
        for case_name, data in case_data.items():
            rescued, flag, candidate, increment = apply_high_aod_rescue(
                aod=data["aod550"],
                li_ae_screened=data["li_ae142"],
                two_stage_qa2=data["two_stage_qa2"],
                probability=data["two_stage_probability"],
                aerosol_type=data["aerosol_type"],
                high_aod_threshold=aod_threshold,
                li_dust_threshold=li_threshold,
                min_probability=prob_threshold,
                cap_fraction=cap_fraction,
                allowed_db_type_codes=allowed_codes,
            )
            for roi_name, *_ in CASES[case_name].rois:
                row = summarize_roi(
                    case_name,
                    roi_name,
                    roi_masks[(case_name, roi_name)],
                    data,
                    rescued,
                    flag,
                    candidate,
                    increment,
                )
                row.update(
                    {
                        "run_id": run_id,
                        "aod_threshold": aod_threshold,
                        "li_dust_threshold": li_threshold,
                        "min_two_stage_probability": prob_threshold,
                        "cap_fraction": cap_fraction,
                        "db_type_set": db_type_set,
                    }
                )
                roi_rows.append(row)
                selected[(case_name, roi_name)] = row

        dec7_core = selected[("dec7_north_africa", "high_li_plume_core")]
        dec7_western = selected[("dec7_north_africa", "western_plume")]
        aug_full = selected[("aug29_north_america", "full_domain")]
        aug_central = selected[("aug29_north_america", "central_false_dust_candidate")]
        aug_west = selected[("aug29_north_america", "upwind_smoke_west")]

        passes = (
            dec7_core["recovery_ratio_to_li"] >= args.min_dec7_core_recovery
            and aug_full["mean_rescue_increment"] <= args.max_aug_full_increment
            and aug_central["mean_rescue_increment"] <= args.max_aug_central_increment
        )
        score = (
            float(dec7_core["recovery_ratio_to_li"])
            + 0.5 * float(dec7_western["recovery_ratio_to_li"])
            - 50.0 * max(float(aug_full["mean_rescue_increment"]), 0.0)
            - 20.0 * max(float(aug_central["mean_rescue_increment"]), 0.0)
            - 20.0 * max(float(aug_west["mean_rescue_increment"]), 0.0)
        )
        ranking_rows.append(
            {
                "run_id": run_id,
                "aod_threshold": aod_threshold,
                "li_dust_threshold": li_threshold,
                "min_two_stage_probability": prob_threshold,
                "cap_fraction": cap_fraction,
                "db_type_set": db_type_set,
                "passes_initial_screen": bool(passes),
                "score": score,
                "dec7_core_recovery_ratio": dec7_core["recovery_ratio_to_li"],
                "dec7_core_increment": dec7_core["mean_rescue_increment"],
                "dec7_core_rescued_fraction": dec7_core["rescued_pixel_fraction"],
                "dec7_western_recovery_ratio": dec7_western["recovery_ratio_to_li"],
                "dec7_western_increment": dec7_western["mean_rescue_increment"],
                "aug_full_increment": aug_full["mean_rescue_increment"],
                "aug_full_rescued_fraction": aug_full["rescued_pixel_fraction"],
                "aug_central_increment": aug_central["mean_rescue_increment"],
                "aug_central_rescued_fraction": aug_central["rescued_pixel_fraction"],
                "aug_west_increment": aug_west["mean_rescue_increment"],
                "aug_west_rescued_fraction": aug_west["rescued_pixel_fraction"],
            }
        )

    ranking_rows.sort(key=lambda row: (not row["passes_initial_screen"], -float(row["score"])))
    write_csv(args.output_dir / "high_aod_rescue_sensitivity_roi_metrics.csv", roi_rows)
    write_csv(args.output_dir / "high_aod_rescue_sensitivity_candidate_ranking.csv", ranking_rows)

    summary = {
        "threshold_grid": {
            "aod_thresholds": args.aod_thresholds,
            "li_thresholds": args.li_thresholds,
            "prob_thresholds": args.prob_thresholds,
            "cap_fractions": args.cap_fractions,
            "db_type_sets": args.db_type_sets,
        },
        "screening_criteria": {
            "min_dec7_core_recovery": args.min_dec7_core_recovery,
            "max_aug_full_increment": args.max_aug_full_increment,
            "max_aug_central_increment": args.max_aug_central_increment,
        },
        "n_runs": len(ranking_rows),
        "n_passing_runs": int(sum(bool(row["passes_initial_screen"]) for row in ranking_rows)),
        "top_10": ranking_rows[:10],
        "outputs": {
            "roi_metrics_csv": str(args.output_dir / "high_aod_rescue_sensitivity_roi_metrics.csv"),
            "candidate_ranking_csv": str(args.output_dir / "high_aod_rescue_sensitivity_candidate_ranking.csv"),
        },
    }
    with (args.output_dir / "high_aod_rescue_sensitivity_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Completed {len(ranking_rows)} sensitivity runs.")
    print(f"Passing initial screen: {summary['n_passing_runs']}")
    print(f"Ranking CSV: {summary['outputs']['candidate_ranking_csv']}")
    if ranking_rows:
        top = ranking_rows[0]
        print(
            "Top run: "
            f"id={top['run_id']}, AOD>={top['aod_threshold']}, Li>={top['li_dust_threshold']}, "
            f"P>={top['min_two_stage_probability']}, cap={top['cap_fraction']}, "
            f"types={top['db_type_set']}, score={top['score']:.3f}"
        )


if __name__ == "__main__":
    main()
