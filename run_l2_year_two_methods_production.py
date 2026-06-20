#!/usr/bin/env python3
"""Production-freeze MODIS DB L2 dust-AOD driver.

This driver implements the manuscript/mass-production definition only:

- Li-Ginoux dust AOD with the SSA412 < SSA470 condition.
- Li-Ginoux recommended field additionally screened by inferred FMF <= 0.7
  (approximately AE <= 1.42 for the Li-Ginoux parameterization).
- Two-stage XGBoost dust probability plus raw regressor dust fraction.
- Separated two-stage QA1/QA2/QA3 dust AOD products.

It intentionally excludes the DB-type-aware, suspect-veto, and high-AOD-rescue
experimental branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from apply_dust_model_to_modis_db import (
    clip_fraction_grid,
    enforce_daod_output_semantics,
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    summarize_grid,
)
from apply_two_stage_dust_model_to_modis_db import inv_logit_transform


TWO_STAGE_THRESHOLDS = {
    "qa1": 0.4,
    "qa2": 0.6,
    "qa3": 0.7,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--short-name", choices=["MOD04_L2", "MYD04_L2"], default="MYD04_L2")
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD start date.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD end date.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-days", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=here / "modis_l2_production_freeze_v1",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=here / "modis_l2_production_freeze_v1_scratch",
    )
    parser.add_argument(
        "--two-stage-model-dir",
        type=Path,
        default=here / "two_stage_tuning" / "best_model",
    )
    parser.add_argument(
        "--li-ginoux-fmf-max",
        type=float,
        default=0.7,
        help="Maximum Li-Ginoux inferred fine-mode fraction for the recommended Li-Ginoux field.",
    )
    return parser.parse_args()


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def load_two_stage_models(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    with (model_dir / "two_stage_metadata.json").open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(model_dir / meta["regressor_model"])
    return classifier, regressor, meta


def predict_two_stage_raw(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    meta: dict,
    features_2d: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dust probability and raw regressor fraction without QA thresholding."""
    feature_order = meta["feature_names"]
    h, w = np.asarray(next(iter(features_2d.values()))).shape
    x_flat = np.column_stack([np.asarray(features_2d[col], dtype=float).ravel() for col in feature_order])
    valid = np.isfinite(x_flat).all(axis=1)

    prob_flat = np.full(h * w, np.nan, dtype=np.float32)
    frac_flat = np.full(h * w, np.nan, dtype=np.float32)
    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=feature_order)
        prob_flat[valid] = classifier.predict_proba(x_valid)[:, 1].astype(np.float32)
        frac_flat[valid] = inv_logit_transform(regressor.predict(x_valid)).astype(np.float32)

    return (
        prob_flat.reshape(h, w),
        clip_fraction_grid(frac_flat.reshape(h, w)),
        valid.reshape(h, w),
    )


def two_stage_qa_dust_aod(
    *,
    aod550: np.ndarray,
    probability: np.ndarray,
    fraction_raw: np.ndarray,
    threshold: float,
) -> np.ndarray:
    out = np.zeros(aod550.shape, dtype=float)
    keep = (
        np.isfinite(aod550)
        & np.isfinite(probability)
        & np.isfinite(fraction_raw)
        & (probability >= threshold)
    )
    out[keep] = np.clip(fraction_raw[keep], 0.0, 1.0) * aod550[keep]
    out[~np.isfinite(aod550)] = np.nan
    return out


def li_ginoux_fmf_screen(
    *,
    li_raw_dust_aod: np.ndarray,
    li_fmf: np.ndarray,
    aod550: np.ndarray,
    fmf_max: float,
) -> np.ndarray:
    keep = np.isfinite(li_raw_dust_aod) & np.isfinite(li_fmf) & (li_fmf <= fmf_max)
    out = np.where(keep, li_raw_dust_aod, np.nan)
    total_valid = np.isfinite(aod550)
    out[total_valid & ~np.isfinite(out)] = 0.0
    out[~total_valid] = np.nan
    return enforce_daod_output_semantics(out, aod550)


def save_granule_npz(
    output_path: Path,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    aod550: np.ndarray,
    li_raw_dust_aod: np.ndarray,
    li_fmf: np.ndarray,
    li_ae_screen_dust_aod: np.ndarray,
    two_stage_probability: np.ndarray,
    two_stage_fraction_raw: np.ndarray,
    two_stage_qa: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        output_path,
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        aod550=np.asarray(aod550, dtype=np.float32),
        li_dust_aod=np.asarray(li_raw_dust_aod, dtype=np.float32),
        li_ginoux_fmf=np.asarray(li_fmf, dtype=np.float32),
        li_ginoux_ae_screen_dust_aod=np.asarray(li_ae_screen_dust_aod, dtype=np.float32),
        two_stage_probability=np.asarray(two_stage_probability, dtype=np.float32),
        two_stage_fraction_raw=np.asarray(two_stage_fraction_raw, dtype=np.float32),
        two_stage_qa1_dust_aod=np.asarray(two_stage_qa["qa1"], dtype=np.float32),
        two_stage_qa2_dust_aod=np.asarray(two_stage_qa["qa2"], dtype=np.float32),
        two_stage_qa3_dust_aod=np.asarray(two_stage_qa["qa3"], dtype=np.float32),
    )


def search_and_download_day(short_name: str, day: date, download_dir: Path) -> list[Path]:
    start = f"{day.isoformat()} 00:00:00"
    end = f"{day.isoformat()} 23:59:59"
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            earthaccess.login(strategy="netrc")
            granules = earthaccess.search_data(short_name=short_name, temporal=(start, end))
            if not granules:
                return []
            download_dir.mkdir(parents=True, exist_ok=True)
            downloaded = earthaccess.download(granules, local_path=str(download_dir))
            return [Path(p).resolve() for p in downloaded]
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] {day.isoformat()} download attempt {attempt}/5 failed: {exc}", flush=True)
            time.sleep(min(60, 5 * attempt))
    assert last_exc is not None
    raise last_exc


def weighted_mean(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float | None:
    values: list[float] = []
    weights: list[float] = []
    for row in rows:
        try:
            value = float(row[value_key])
            weight = float(row[weight_key])
        except Exception:
            continue
        if np.isfinite(value) and np.isfinite(weight) and weight > 0:
            values.append(value)
            weights.append(weight)
    if not values:
        return None
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def process_day(
    *,
    day: date,
    short_name: str,
    day_output_dir: Path,
    day_download_dir: Path,
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    model_meta: dict,
    li_ginoux_fmf_max: float,
) -> dict[str, object]:
    granule_dir = day_output_dir / "granules"
    granule_dir.mkdir(parents=True, exist_ok=True)

    downloaded = search_and_download_day(short_name, day, day_download_dir)
    summary_rows: list[dict[str, object]] = []
    skip_rows: list[dict[str, object]] = []

    if not downloaded:
        meta = {
            "date": day.isoformat(),
            "short_name": short_name,
            "n_granules": 0,
            "n_completed": 0,
            "n_skipped": 0,
            "status": "no_granules_found",
        }
        with (day_output_dir / "run_metadata.json").open("w") as handle:
            json.dump(meta, handle, indent=2)
        return meta

    for idx, granule_path in enumerate(sorted(downloaded), start=1):
        print(f"[{day.isoformat()} {idx}/{len(downloaded)}] {granule_path.name}", flush=True)
        try:
            features_2d, lat2d, lon2d = read_granule_features(granule_path)
        except Exception as exc:
            skip_rows.append({"granule": granule_path.name, "status": "read_failed", "message": str(exc)})
            continue

        aod550 = np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
        if not np.isfinite(aod550).any():
            skip_rows.append({"granule": granule_path.name, "status": "no_finite_db_aod550"})
            continue

        try:
            li_fmf, _, li_raw_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)
            li_ae_screen_dust_aod = li_ginoux_fmf_screen(
                li_raw_dust_aod=li_raw_dust_aod,
                li_fmf=li_fmf,
                aod550=aod550,
                fmf_max=li_ginoux_fmf_max,
            )
            probability, fraction_raw, _ = predict_two_stage_raw(
                classifier,
                regressor,
                model_meta,
                features_2d,
            )
            two_stage_qa = {
                qa_name: two_stage_qa_dust_aod(
                    aod550=aod550,
                    probability=probability,
                    fraction_raw=fraction_raw,
                    threshold=threshold,
                )
                for qa_name, threshold in TWO_STAGE_THRESHOLDS.items()
            }
        except Exception as exc:
            skip_rows.append({"granule": granule_path.name, "status": "compute_failed", "message": str(exc)})
            continue

        save_granule_npz(
            granule_dir / f"{granule_path.stem}.npz",
            lat=lat2d,
            lon=lon2d,
            aod550=aod550,
            li_raw_dust_aod=li_raw_dust_aod,
            li_fmf=li_fmf,
            li_ae_screen_dust_aod=li_ae_screen_dust_aod,
            two_stage_probability=probability,
            two_stage_fraction_raw=fraction_raw,
            two_stage_qa=two_stage_qa,
        )

        aod_summary = summarize_grid("aod550", aod550)
        li_raw_summary = summarize_grid("li_ginoux_raw_dust_aod", li_raw_dust_aod)
        li_screen_summary = summarize_grid("li_ginoux_ae_screen_dust_aod", li_ae_screen_dust_aod)
        qa_summaries = {qa: summarize_grid(f"two_stage_{qa}_dust_aod", arr) for qa, arr in two_stage_qa.items()}
        row: dict[str, object] = {
            "granule": granule_path.name,
            "aod550_mean": aod_summary.get("mean"),
            "aod550_valid": aod_summary.get("valid_pixels"),
            "li_ginoux_raw_mean": li_raw_summary.get("mean"),
            "li_ginoux_raw_valid": li_raw_summary.get("valid_pixels"),
            "li_ginoux_ae_screen_mean": li_screen_summary.get("mean"),
            "li_ginoux_ae_screen_valid": li_screen_summary.get("valid_pixels"),
        }
        for qa_name, summary in qa_summaries.items():
            row[f"two_stage_{qa_name}_mean"] = summary.get("mean")
            row[f"two_stage_{qa_name}_valid"] = summary.get("valid_pixels")
        summary_rows.append(row)

    write_csv(summary_rows, day_output_dir / "two_method_summary.csv")
    write_csv(skip_rows, day_output_dir / "two_method_skips.csv")
    meta = {
        "date": day.isoformat(),
        "short_name": short_name,
        "n_granules": len(downloaded),
        "n_completed": len(summary_rows),
        "n_skipped": len(skip_rows),
        "status": "completed",
        "aod550_weighted_mean": weighted_mean(summary_rows, "aod550_mean", "aod550_valid"),
        "li_ginoux_raw_weighted_mean": weighted_mean(summary_rows, "li_ginoux_raw_mean", "li_ginoux_raw_valid"),
        "li_ginoux_ae_screen_weighted_mean": weighted_mean(
            summary_rows,
            "li_ginoux_ae_screen_mean",
            "li_ginoux_ae_screen_valid",
        ),
        "two_stage_qa1_weighted_mean": weighted_mean(summary_rows, "two_stage_qa1_mean", "two_stage_qa1_valid"),
        "two_stage_qa2_weighted_mean": weighted_mean(summary_rows, "two_stage_qa2_mean", "two_stage_qa2_valid"),
        "two_stage_qa3_weighted_mean": weighted_mean(summary_rows, "two_stage_qa3_mean", "two_stage_qa3_valid"),
        "two_stage_thresholds": TWO_STAGE_THRESHOLDS,
        "li_ginoux_fmf_max": li_ginoux_fmf_max,
        "product_version": "production-freeze-l2-daod-v1",
    }
    with (day_output_dir / "run_metadata.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else date(args.year, 1, 1)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else date(args.year, 12, 31)
    all_days = iter_dates(start, end)
    shard_days = [d for idx, d in enumerate(all_days) if idx % args.num_shards == args.shard_index]
    if args.max_days is not None:
        shard_days = shard_days[: args.max_days]

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    run_root = args.output_root / f"{args.short_name}_{start.isoformat()}_{end.isoformat()}"
    run_root.mkdir(parents=True, exist_ok=True)

    classifier, regressor, model_meta = load_two_stage_models(args.two_stage_model_dir)
    earthaccess.login(strategy="netrc")

    overall_rows: list[dict[str, object]] = []
    for day in shard_days:
        day_dir = run_root / day.isoformat()
        meta_path = day_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except Exception:
                existing = None
            if isinstance(existing, dict) and existing.get("status") in {"completed", "no_granules_found"}:
                print(f"[SKIP] {day.isoformat()} already completed", flush=True)
                overall_rows.append(existing)
                continue

        download_dir = args.scratch_root / f"{args.short_name}_{day.isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DAY] {day.isoformat()} start", flush=True)
        try:
            meta = process_day(
                day=day,
                short_name=args.short_name,
                day_output_dir=day_dir,
                day_download_dir=download_dir,
                classifier=classifier,
                regressor=regressor,
                model_meta=model_meta,
                li_ginoux_fmf_max=args.li_ginoux_fmf_max,
            )
        finally:
            if download_dir.exists():
                shutil.rmtree(download_dir, ignore_errors=True)
        overall_rows.append(meta)
        print(f"[DAY] {day.isoformat()} done: {meta['status']}", flush=True)

    shard_summary = run_root / f"shard_{args.shard_index:02d}_summary.csv"
    write_csv(overall_rows, shard_summary)
    shard_meta = {
        "year": args.year,
        "short_name": args.short_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_days_assigned": len(shard_days),
        "n_days_processed": len(overall_rows),
        "run_root": str(run_root),
        "shard_summary_csv": str(shard_summary),
        "product_version": "production-freeze-l2-daod-v1",
    }
    with (run_root / f"shard_{args.shard_index:02d}_metadata.json").open("w") as handle:
        json.dump(shard_meta, handle, indent=2)
    print(json.dumps(shard_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
