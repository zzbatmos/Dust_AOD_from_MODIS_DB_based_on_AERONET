#!/usr/bin/env python3
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
from pyhdf.SD import SD, SDC
from xgboost import XGBClassifier, XGBRegressor

from apply_db_type_conditional_dust_model import build_pixel_features as build_dbtype_pixel_features, decode_db_qa
from apply_dust_model_to_modis_db import (
    enforce_daod_output_semantics,
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    summarize_grid,
)
from apply_two_stage_dust_model_to_modis_db import predict_two_stage


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Process a full year of MOD04/MYD04 L2 granules day by day without plotting, "
            "applying Li-Ginoux and tuned two-stage methods."
        )
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--short-name", default="MYD04_L2")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=here / "modis_l2_2023_dust_aod",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=here / "modis_l2_2023_dust_aod_scratch",
    )
    parser.add_argument(
        "--two-stage-model-dir",
        type=Path,
        default=here / "two_stage_tuning" / "best_model",
    )
    parser.add_argument(
        "--veto-model-dir",
        type=Path,
        default=here / "two_stage_suspect_veto",
    )
    parser.add_argument("--veto-threshold", type=float, default=0.7)
    parser.add_argument(
        "--high-aod-rescue-aod-threshold",
        type=float,
        default=2.0,
        help="Minimum DB AOD550 required for the optional high-AOD dust rescue.",
    )
    parser.add_argument(
        "--high-aod-rescue-li-threshold",
        type=float,
        default=0.8,
        help="Minimum AE-screened Li-Ginoux dust AOD required for the optional high-AOD dust rescue.",
    )
    parser.add_argument(
        "--high-aod-rescue-probability-threshold",
        type=float,
        default=0.25,
        help="Minimum two-stage dust probability required for the optional high-AOD dust rescue.",
    )
    parser.add_argument(
        "--high-aod-rescue-cap-fraction",
        type=float,
        default=0.95,
        help="Upper cap for rescued dust AOD as a fraction of DB AOD550.",
    )
    parser.add_argument(
        "--high-aod-rescue-fmf-max",
        type=float,
        default=0.7,
        help="Maximum inferred Li-Ginoux fine-mode fraction used for AE-screened rescue input.",
    )
    parser.add_argument("--max-days", type=int)
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


def load_two_stage_models(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    with (model_dir / "two_stage_metadata.json").open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(model_dir / meta["regressor_model"])
    return classifier, regressor, meta


def load_veto_model(model_dir: Path) -> tuple[XGBClassifier, dict]:
    with (model_dir / "metadata.json").open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["model"])
    return classifier, meta


def decode_algorithm_flag(granule_path: Path) -> np.ndarray:
    handle = SD(str(granule_path), SDC.READ)
    algorithm_flag = np.asarray(handle.select("Deep_Blue_Algorithm_Flag_Land").get(), dtype=np.int16)
    handle.end()
    return algorithm_flag


def compute_two_stage_qa_dust_aod(
    *,
    aod550: np.ndarray,
    two_stage_probability: np.ndarray,
    two_stage_fraction: np.ndarray,
    probability_threshold: float,
) -> np.ndarray:
    out = np.zeros(aod550.shape, dtype=float)
    keep = (
        np.isfinite(aod550)
        & np.isfinite(two_stage_probability)
        & np.isfinite(two_stage_fraction)
        & (two_stage_probability >= probability_threshold)
    )
    out[keep] = np.clip(two_stage_fraction[keep], 0.0, 1.0) * aod550[keep]
    out[~np.isfinite(aod550)] = np.nan
    return out


def infer_li_ginoux_fmf_screen(
    *,
    li_dust_aod: np.ndarray,
    aod550: np.ndarray,
    fmf_max: float,
) -> np.ndarray:
    coarse_fraction = np.full(aod550.shape, np.nan, dtype=float)
    valid = np.isfinite(li_dust_aod) & np.isfinite(aod550) & (aod550 > 0.0)
    np.divide(li_dust_aod, aod550, out=coarse_fraction, where=valid)
    coarse_fraction = np.clip(coarse_fraction, 0.0, 1.0)
    fmf = 1.0 - coarse_fraction
    keep = valid & np.isfinite(fmf) & (fmf <= fmf_max)

    out = np.where(keep, li_dust_aod, np.nan)
    total_valid = np.isfinite(aod550) & (aod550 > 0.0)
    out[total_valid & ~np.isfinite(out)] = 0.0
    out[~np.isfinite(aod550)] = np.nan
    return out


def apply_high_aod_rescue(
    *,
    aod550: np.ndarray,
    li_dust_aod_ae_screened: np.ndarray,
    two_stage_qa2_dust_aod: np.ndarray,
    two_stage_probability: np.ndarray,
    aerosol_type_code: np.ndarray,
    aod_threshold: float,
    li_threshold: float,
    probability_threshold: float,
    cap_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = (
        np.isfinite(aod550)
        & np.isfinite(li_dust_aod_ae_screened)
        & np.isfinite(two_stage_qa2_dust_aod)
        & np.isfinite(two_stage_probability)
        & (aod550 >= aod_threshold)
        & (li_dust_aod_ae_screened >= li_threshold)
        & (two_stage_probability >= probability_threshold)
        & (np.asarray(aerosol_type_code, dtype=int) == 1)
    )
    capped_li = np.minimum(li_dust_aod_ae_screened, cap_fraction * aod550)
    rescued = np.asarray(two_stage_qa2_dust_aod, dtype=float).copy()
    rescue_flag = candidate & np.isfinite(capped_li) & (capped_li > rescued)
    rescued[rescue_flag] = capped_li[rescue_flag]
    increment = rescued - two_stage_qa2_dust_aod
    increment[~np.isfinite(aod550)] = np.nan
    return (
        enforce_daod_output_semantics(rescued, aod550),
        rescue_flag.astype(np.int8),
        candidate.astype(np.int8),
        increment.astype(np.float32),
    )


def save_granule_npz(
    output_path: Path,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    aod550: np.ndarray,
    li_dust_aod: np.ndarray,
    two_stage_probability: np.ndarray,
    two_stage_fraction: np.ndarray,
    two_stage_dust_aod: np.ndarray,
    veto_probability: np.ndarray,
    suspect_mask: np.ndarray,
    two_stage_veto_qa2_dust_aod: np.ndarray,
    li_ginoux_ae_screen_dust_aod: np.ndarray,
    two_stage_qa2_dust_aod: np.ndarray,
    two_stage_high_aod_rescue_qa2_dust_aod: np.ndarray,
    high_aod_rescue_flag: np.ndarray,
    high_aod_rescue_candidate: np.ndarray,
    high_aod_rescue_increment: np.ndarray,
    aerosol_type_code: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        aod550=np.asarray(aod550, dtype=np.float32),
        li_dust_aod=np.asarray(li_dust_aod, dtype=np.float32),
        two_stage_probability=np.asarray(two_stage_probability, dtype=np.float32),
        two_stage_fraction=np.asarray(two_stage_fraction, dtype=np.float32),
        two_stage_dust_aod=np.asarray(two_stage_dust_aod, dtype=np.float32),
        veto_probability=np.asarray(veto_probability, dtype=np.float32),
        suspect_mask=np.asarray(suspect_mask, dtype=np.int8),
        two_stage_veto_qa2_dust_aod=np.asarray(two_stage_veto_qa2_dust_aod, dtype=np.float32),
        li_ginoux_ae_screen_dust_aod=np.asarray(li_ginoux_ae_screen_dust_aod, dtype=np.float32),
        two_stage_qa2_dust_aod=np.asarray(two_stage_qa2_dust_aod, dtype=np.float32),
        two_stage_high_aod_rescue_qa2_dust_aod=np.asarray(
            two_stage_high_aod_rescue_qa2_dust_aod,
            dtype=np.float32,
        ),
        high_aod_rescue_flag=np.asarray(high_aod_rescue_flag, dtype=np.int8),
        high_aod_rescue_candidate=np.asarray(high_aod_rescue_candidate, dtype=np.int8),
        high_aod_rescue_increment=np.asarray(high_aod_rescue_increment, dtype=np.float32),
        aerosol_type_code=np.asarray(aerosol_type_code, dtype=np.int8),
    )


def iter_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


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
            print(
                f"[WARN] {day.isoformat()} download attempt {attempt}/5 failed: {exc}",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
    assert last_exc is not None
    raise last_exc


def weighted_mean(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float | None:
    values: list[float] = []
    weights: list[float] = []
    for row in rows:
        val = row.get(value_key)
        wt = row.get(weight_key)
        if val is None or wt is None:
            continue
        try:
            v = float(val)
            w = float(wt)
        except Exception:
            continue
        if np.isfinite(v) and np.isfinite(w) and w > 0:
            values.append(v)
            weights.append(w)
    if not values:
        return None
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def process_day(
    *,
    day: date,
    short_name: str,
    day_output_dir: Path,
    day_download_dir: Path,
    two_stage_classifier: XGBClassifier,
    two_stage_regressor: XGBRegressor,
    two_stage_meta: dict,
    veto_classifier: XGBClassifier,
    veto_meta: dict,
    veto_threshold: float,
    high_aod_rescue_aod_threshold: float,
    high_aod_rescue_li_threshold: float,
    high_aod_rescue_probability_threshold: float,
    high_aod_rescue_cap_fraction: float,
    high_aod_rescue_fmf_max: float,
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
            skip_rows.append(
                {"granule": granule_path.name, "status": "read_failed", "message": str(exc)}
            )
            continue

        aod550 = np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
        if not np.isfinite(aod550).any():
            skip_rows.append({"granule": granule_path.name, "status": "no_finite_db_aod550"})
            continue

        try:
            _, _, li_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)
            two_stage_probability, two_stage_fraction, _ = predict_two_stage(
                two_stage_classifier,
                two_stage_regressor,
                two_stage_meta,
                features_2d,
            )
            two_stage_dust_aod = enforce_daod_output_semantics(
                np.asarray(two_stage_fraction, dtype=float) * aod550,
                aod550,
            )
            base_qa2_dust_aod = compute_two_stage_qa_dust_aod(
                aod550=aod550,
                two_stage_probability=two_stage_probability,
                two_stage_fraction=two_stage_fraction,
                probability_threshold=0.6,
            )
            li_ginoux_ae_screen_dust_aod = infer_li_ginoux_fmf_screen(
                li_dust_aod=li_dust_aod,
                aod550=aod550,
                fmf_max=high_aod_rescue_fmf_max,
            )

            qa = decode_db_qa(granule_path)
            (
                two_stage_high_aod_rescue_qa2_dust_aod,
                high_aod_rescue_flag,
                high_aod_rescue_candidate,
                high_aod_rescue_increment,
            ) = apply_high_aod_rescue(
                aod550=aod550,
                li_dust_aod_ae_screened=li_ginoux_ae_screen_dust_aod,
                two_stage_qa2_dust_aod=base_qa2_dust_aod,
                two_stage_probability=two_stage_probability,
                aerosol_type_code=qa["aerosol_type_code"],
                aod_threshold=high_aod_rescue_aod_threshold,
                li_threshold=high_aod_rescue_li_threshold,
                probability_threshold=high_aod_rescue_probability_threshold,
                cap_fraction=high_aod_rescue_cap_fraction,
            )

            algorithm_flag = decode_algorithm_flag(granule_path)
            dbtype_pixel_features = build_dbtype_pixel_features(features_2d, qa)
            suspect_mask = (
                (np.asarray(qa["aerosol_type_code"], dtype=int) == 1)
                & (np.asarray(algorithm_flag, dtype=int) == 0)
                & np.isfinite(aod550)
                & (aod550 > 0.0)
                & np.isfinite(two_stage_probability)
                & (two_stage_probability >= 0.6)
            )
            veto_probability = np.full(aod550.shape, np.nan, dtype=np.float32)
            if np.any(suspect_mask):
                feature_columns = []
                for feature_name in veto_meta["veto_features"]:
                    feature_columns.append(np.asarray(dbtype_pixel_features[feature_name], dtype=float)[suspect_mask])
                design = np.column_stack(feature_columns)
                finite_rows = np.isfinite(design).all(axis=1)
                if np.any(finite_rows):
                    predicted = veto_classifier.predict_proba(design[finite_rows])[:, 1].astype(np.float32)
                    suspect_indices = np.flatnonzero(suspect_mask.ravel())
                    veto_probability.ravel()[suspect_indices[finite_rows]] = predicted
            two_stage_veto_qa2_dust_aod = np.asarray(base_qa2_dust_aod, dtype=float).copy()
            veto_fail = suspect_mask & np.isfinite(veto_probability) & (veto_probability < veto_threshold)
            two_stage_veto_qa2_dust_aod[veto_fail] = 0.0
            two_stage_veto_qa2_dust_aod = enforce_daod_output_semantics(two_stage_veto_qa2_dust_aod, aod550)
        except Exception as exc:
            skip_rows.append(
                {"granule": granule_path.name, "status": "compute_failed", "message": str(exc)}
            )
            continue

        save_granule_npz(
            granule_dir / f"{granule_path.stem}.npz",
            lat=lat2d,
            lon=lon2d,
            aod550=aod550,
            li_dust_aod=li_dust_aod,
            two_stage_probability=two_stage_probability,
            two_stage_fraction=two_stage_fraction,
            two_stage_dust_aod=two_stage_dust_aod,
            veto_probability=veto_probability,
            suspect_mask=suspect_mask,
            two_stage_veto_qa2_dust_aod=two_stage_veto_qa2_dust_aod,
            li_ginoux_ae_screen_dust_aod=li_ginoux_ae_screen_dust_aod,
            two_stage_qa2_dust_aod=base_qa2_dust_aod,
            two_stage_high_aod_rescue_qa2_dust_aod=two_stage_high_aod_rescue_qa2_dust_aod,
            high_aod_rescue_flag=high_aod_rescue_flag,
            high_aod_rescue_candidate=high_aod_rescue_candidate,
            high_aod_rescue_increment=high_aod_rescue_increment,
            aerosol_type_code=qa["aerosol_type_code"],
        )

        li_summary = summarize_grid("li_ginoux_dust_aod", li_dust_aod)
        two_summary = summarize_grid("two_stage_dust_aod", two_stage_dust_aod)
        veto_summary = summarize_grid("two_stage_veto_qa2_dust_aod", two_stage_veto_qa2_dust_aod)
        rescue_summary = summarize_grid(
            "two_stage_high_aod_rescue_qa2_dust_aod",
            two_stage_high_aod_rescue_qa2_dust_aod,
        )
        aod_summary = summarize_grid("aod550", aod550)
        summary_rows.append(
            {
                "granule": granule_path.name,
                "li_ginoux_mean": li_summary.get("mean"),
                "li_ginoux_valid": li_summary.get("valid_pixels"),
                "two_stage_mean": two_summary.get("mean"),
                "two_stage_valid": two_summary.get("valid_pixels"),
                "two_stage_veto_qa2_mean": veto_summary.get("mean"),
                "two_stage_veto_qa2_valid": veto_summary.get("valid_pixels"),
                "two_stage_high_aod_rescue_qa2_mean": rescue_summary.get("mean"),
                "two_stage_high_aod_rescue_qa2_valid": rescue_summary.get("valid_pixels"),
                "aod550_mean": aod_summary.get("mean"),
                "aod550_valid": aod_summary.get("valid_pixels"),
                "suspect_pixels": int(np.count_nonzero(suspect_mask)),
                "vetoed_pixels": int(np.count_nonzero(veto_fail)),
                "high_aod_rescue_candidate_pixels": int(np.count_nonzero(high_aod_rescue_candidate)),
                "high_aod_rescued_pixels": int(np.count_nonzero(high_aod_rescue_flag)),
            }
        )

    write_csv(summary_rows, day_output_dir / "two_method_summary.csv")
    write_csv(skip_rows, day_output_dir / "two_method_skips.csv")
    meta = {
        "date": day.isoformat(),
        "short_name": short_name,
        "n_granules": len(downloaded),
        "n_completed": len(summary_rows),
        "n_skipped": len(skip_rows),
        "status": "completed",
        "li_ginoux_weighted_mean": weighted_mean(summary_rows, "li_ginoux_mean", "li_ginoux_valid"),
        "two_stage_weighted_mean": weighted_mean(summary_rows, "two_stage_mean", "two_stage_valid"),
        "two_stage_veto_qa2_weighted_mean": weighted_mean(
            summary_rows, "two_stage_veto_qa2_mean", "two_stage_veto_qa2_valid"
        ),
        "two_stage_high_aod_rescue_qa2_weighted_mean": weighted_mean(
            summary_rows,
            "two_stage_high_aod_rescue_qa2_mean",
            "two_stage_high_aod_rescue_qa2_valid",
        ),
        "aod550_weighted_mean": weighted_mean(summary_rows, "aod550_mean", "aod550_valid"),
        "high_aod_rescue_parameters": {
            "aod_threshold": high_aod_rescue_aod_threshold,
            "li_dust_aod_threshold": high_aod_rescue_li_threshold,
            "two_stage_probability_threshold": high_aod_rescue_probability_threshold,
            "cap_fraction": high_aod_rescue_cap_fraction,
            "fmf_max": high_aod_rescue_fmf_max,
            "db_aerosol_type_required": "Dust",
        },
    }
    with (day_output_dir / "run_metadata.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else date(args.year, 1, 1)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else date(args.year, 12, 31)
    all_days = iter_dates(start, end)
    shard_days = [d for i, d in enumerate(all_days) if i % args.num_shards == args.shard_index]
    if args.max_days is not None:
        shard_days = shard_days[: args.max_days]

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    run_root = args.output_root / f"{args.short_name}_{start.isoformat()}_{end.isoformat()}"
    run_root.mkdir(parents=True, exist_ok=True)

    two_stage_classifier, two_stage_regressor, two_stage_meta = load_two_stage_models(
        args.two_stage_model_dir
    )
    veto_classifier, veto_meta = load_veto_model(args.veto_model_dir)

    overall_rows: list[dict[str, object]] = []
    earthaccess.login(strategy="netrc")

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
                two_stage_classifier=two_stage_classifier,
                two_stage_regressor=two_stage_regressor,
                two_stage_meta=two_stage_meta,
                veto_classifier=veto_classifier,
                veto_meta=veto_meta,
                veto_threshold=args.veto_threshold,
                high_aod_rescue_aod_threshold=args.high_aod_rescue_aod_threshold,
                high_aod_rescue_li_threshold=args.high_aod_rescue_li_threshold,
                high_aod_rescue_probability_threshold=args.high_aod_rescue_probability_threshold,
                high_aod_rescue_cap_fraction=args.high_aod_rescue_cap_fraction,
                high_aod_rescue_fmf_max=args.high_aod_rescue_fmf_max,
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
    }
    with (run_root / f"shard_{args.shard_index:02d}_metadata.json").open("w") as handle:
        json.dump(shard_meta, handle, indent=2)
    print(json.dumps(shard_meta, indent=2))


if __name__ == "__main__":
    main()
