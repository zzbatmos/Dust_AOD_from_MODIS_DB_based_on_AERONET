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
from xgboost import XGBClassifier, XGBRegressor

from apply_db_type_conditional_dust_model import (
    build_pixel_features,
    decode_db_qa,
    load_models as load_db_type_models,
    predict_routed,
    qa_to_daod,
)
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
            "applying Li-Ginoux, tuned two-stage, and DB-type-aware methods."
        )
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--short-name", default="MYD04_L2")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=here / "l2_year_three_methods_noplot",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=here / "l2_year_three_methods_scratch",
    )
    parser.add_argument(
        "--two-stage-model-dir",
        type=Path,
        default=here / "two_stage_tuning" / "best_model",
    )
    parser.add_argument(
        "--db-type-model-dir",
        type=Path,
        default=here / "modis_db_type_conditional_detection_from_qa",
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
    db_type_probability: np.ndarray,
    db_type_fraction: np.ndarray,
    db_type_qa: np.ndarray,
    db_type_qa1_dust_aod: np.ndarray,
    db_type_qa2_dust_aod: np.ndarray,
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
        db_type_probability=np.asarray(db_type_probability, dtype=np.float32),
        db_type_fraction=np.asarray(db_type_fraction, dtype=np.float32),
        db_type_qa=np.asarray(db_type_qa, dtype=np.float32),
        db_type_qa1_dust_aod=np.asarray(db_type_qa1_dust_aod, dtype=np.float32),
        db_type_qa2_dust_aod=np.asarray(db_type_qa2_dust_aod, dtype=np.float32),
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
    db_type_classifiers: dict,
    db_type_regressors: dict,
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

            qa = decode_db_qa(granule_path)
            pixel_features = build_pixel_features(features_2d, qa)
            db_type_probability, db_type_fraction, db_type_qa = predict_routed(
                db_type_classifiers,
                db_type_regressors,
                pixel_features,
                qa,
            )
            db_type_qa1_dust_aod = qa_to_daod(db_type_fraction, db_type_qa, aod550, minimum_qa=1)
            db_type_qa2_dust_aod = qa_to_daod(db_type_fraction, db_type_qa, aod550, minimum_qa=2)
        except Exception as exc:
            skip_rows.append({"granule": granule_path.name, "status": "compute_failed", "message": str(exc)})
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
            db_type_probability=db_type_probability,
            db_type_fraction=db_type_fraction,
            db_type_qa=db_type_qa,
            db_type_qa1_dust_aod=db_type_qa1_dust_aod,
            db_type_qa2_dust_aod=db_type_qa2_dust_aod,
            aerosol_type_code=qa["aerosol_type_code"],
        )

        li_summary = summarize_grid("li_ginoux_dust_aod", li_dust_aod)
        two_summary = summarize_grid("two_stage_dust_aod", two_stage_dust_aod)
        qa1_summary = summarize_grid("db_type_qa1_dust_aod", db_type_qa1_dust_aod)
        qa2_summary = summarize_grid("db_type_qa2_dust_aod", db_type_qa2_dust_aod)
        aod_summary = summarize_grid("aod550", aod550)
        summary_rows.append(
            {
                "granule": granule_path.name,
                "li_ginoux_mean": li_summary.get("mean"),
                "li_ginoux_valid": li_summary.get("valid_pixels"),
                "two_stage_mean": two_summary.get("mean"),
                "two_stage_valid": two_summary.get("valid_pixels"),
                "db_type_qa1_mean": qa1_summary.get("mean"),
                "db_type_qa1_valid": qa1_summary.get("valid_pixels"),
                "db_type_qa2_mean": qa2_summary.get("mean"),
                "db_type_qa2_valid": qa2_summary.get("valid_pixels"),
                "aod550_mean": aod_summary.get("mean"),
                "aod550_valid": aod_summary.get("valid_pixels"),
                "dust_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 1)),
                "smoke_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 2)),
                "mixed_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 0)),
                "sulfate_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 3)),
            }
        )

    write_csv(summary_rows, day_output_dir / "three_method_summary.csv")
    write_csv(skip_rows, day_output_dir / "three_method_skips.csv")
    meta = {
        "date": day.isoformat(),
        "short_name": short_name,
        "n_granules": len(downloaded),
        "n_completed": len(summary_rows),
        "n_skipped": len(skip_rows),
        "status": "completed",
        "li_ginoux_weighted_mean": weighted_mean(summary_rows, "li_ginoux_mean", "li_ginoux_valid"),
        "two_stage_weighted_mean": weighted_mean(summary_rows, "two_stage_mean", "two_stage_valid"),
        "db_type_qa1_weighted_mean": weighted_mean(summary_rows, "db_type_qa1_mean", "db_type_qa1_valid"),
        "db_type_qa2_weighted_mean": weighted_mean(summary_rows, "db_type_qa2_mean", "db_type_qa2_valid"),
        "aod550_weighted_mean": weighted_mean(summary_rows, "aod550_mean", "aod550_valid"),
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

    two_stage_classifier, two_stage_regressor, two_stage_meta = load_two_stage_models(args.two_stage_model_dir)
    db_type_classifiers, db_type_regressors = load_db_type_models(args.db_type_model_dir)

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
                db_type_classifiers=db_type_classifiers,
                db_type_regressors=db_type_regressors,
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
