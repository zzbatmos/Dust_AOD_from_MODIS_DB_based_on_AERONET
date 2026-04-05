#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
            "Apply Li-Ginoux, tuned two-stage, and DB-type-conditional methods "
            "to a day of Aqua MYD04_L2 granules without plotting."
        )
    )
    parser.add_argument("--date", default="2017-04-17")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=here / "aqua_l2_2017-04-17_downloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "aqua_l2_2017-04-17_three_methods_noplot",
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


def main() -> None:
    args = parse_args()
    granule_dir = args.output_dir / "granules"
    granule_dir.mkdir(parents=True, exist_ok=True)

    granules = sorted(args.download_dir.glob("*.hdf"))
    if not granules:
        raise FileNotFoundError(f"No HDF files found in {args.download_dir}")

    two_stage_classifier, two_stage_regressor, two_stage_meta = load_two_stage_models(args.two_stage_model_dir)
    db_type_classifiers, db_type_regressors = load_db_type_models(args.db_type_model_dir)

    summary_rows: list[dict[str, object]] = []
    skip_rows: list[dict[str, object]] = []

    for idx, granule_path in enumerate(granules, start=1):
        print(f"[{idx}/{len(granules)}] {granule_path.name}", flush=True)
        try:
            features_2d, lat2d, lon2d = read_granule_features(granule_path)
        except Exception as exc:
            skip_rows.append({"granule": granule_path.name, "status": "read_failed", "message": str(exc)})
            print(f"  skip: read failed: {exc}", flush=True)
            continue

        aod550 = np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
        if not np.isfinite(aod550).any():
            skip_rows.append({"granule": granule_path.name, "status": "no_finite_db_aod550"})
            print("  skip: no finite Deep Blue land AOD", flush=True)
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
            print(f"  skip: compute failed: {exc}", flush=True)
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

        summary_rows.append(
            {
                "granule": granule_path.name,
                "li_ginoux_mean": summarize_grid("li_ginoux_dust_aod", li_dust_aod).get("mean"),
                "li_ginoux_valid": summarize_grid("li_ginoux_dust_aod", li_dust_aod).get("valid_pixels"),
                "two_stage_mean": summarize_grid("two_stage_dust_aod", two_stage_dust_aod).get("mean"),
                "two_stage_valid": summarize_grid("two_stage_dust_aod", two_stage_dust_aod).get("valid_pixels"),
                "db_type_qa1_mean": summarize_grid("db_type_qa1_dust_aod", db_type_qa1_dust_aod).get("mean"),
                "db_type_qa1_valid": summarize_grid("db_type_qa1_dust_aod", db_type_qa1_dust_aod).get("valid_pixels"),
                "db_type_qa2_mean": summarize_grid("db_type_qa2_dust_aod", db_type_qa2_dust_aod).get("mean"),
                "db_type_qa2_valid": summarize_grid("db_type_qa2_dust_aod", db_type_qa2_dust_aod).get("valid_pixels"),
                "aod550_mean": summarize_grid("aod550", aod550).get("mean"),
                "dust_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 1)),
                "smoke_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 2)),
                "mixed_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 0)),
                "sulfate_pixels_db_type": int(np.count_nonzero(qa["aerosol_type_code"] == 3)),
            }
        )

    write_csv(summary_rows, args.output_dir / f"MYD04_L2_{args.date}_three_method_summary.csv")
    write_csv(skip_rows, args.output_dir / f"MYD04_L2_{args.date}_three_method_skips.csv")
    with (args.output_dir / f"MYD04_L2_{args.date}_run_metadata.json").open("w") as handle:
        json.dump(
            {
                "date": args.date,
                "download_dir": str(args.download_dir),
                "output_dir": str(args.output_dir),
                "n_granules": len(granules),
                "n_completed": len(summary_rows),
                "n_skipped": len(skip_rows),
                "two_stage_model_dir": str(args.two_stage_model_dir),
                "db_type_model_dir": str(args.db_type_model_dir),
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "date": args.date,
                "n_granules": len(granules),
                "n_completed": len(summary_rows),
                "n_skipped": len(skip_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
