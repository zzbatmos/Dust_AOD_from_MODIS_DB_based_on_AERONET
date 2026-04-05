#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from apply_dust_model_to_modis_db import (
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    save_comparison_panel,
    summarize_grid,
)
from apply_two_stage_dust_model_to_modis_db import predict_two_stage
from apply_db_type_conditional_dust_model import (
    build_pixel_features,
    compute_qa_class,
    decode_db_qa,
    load_models as load_db_type_models,
)
from train_modis_db_type_conditional_from_qa_xgb import AEROSOL_GROUP_MAP, FEATURES_BY_GROUP


TYPE_LABELS = {
    0: "Mixed",
    1: "Dust",
    2: "Smoke",
    3: "Sulfate",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare tuned two-stage QA with DB-type-conditional QA on MODIS L2 granules.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def load_two_stage_models(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    with (model_dir / "two_stage_metadata.json").open("r") as f:
        meta = json.load(f)
    clf = XGBClassifier()
    clf.load_model(model_dir / meta["classifier_model"])
    reg = XGBRegressor()
    reg.load_model(model_dir / meta["regressor_model"])
    return clf, reg, meta


def predict_db_type(
    classifiers: dict[str, XGBClassifier],
    regressors: dict[str, XGBRegressor | None],
    pixel_features: dict[str, np.ndarray],
    qa: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = np.asarray(next(iter(pixel_features.values()))).shape
    aerosol_code = np.asarray(qa["aerosol_type_code"], dtype=int)
    probability = np.full((h, w), np.nan, dtype=np.float32)
    fraction = np.full((h, w), np.nan, dtype=np.float32)
    qa_class = np.full((h, w), np.nan, dtype=np.float32)
    valid_geo = np.isfinite(pixel_features["MODIS_DB_AOD550"]) & (pixel_features["MODIS_DB_AOD550"] > 0.0)
    for code, label in TYPE_LABELS.items():
        group = AEROSOL_GROUP_MAP.get(label)
        if group is None or group not in classifiers:
            continue
        mask = valid_geo & (aerosol_code == code)
        if not np.any(mask):
            continue
        feats = FEATURES_BY_GROUP[group]
        x = np.column_stack([np.asarray(pixel_features[f], dtype=float)[mask] for f in feats])
        finite = np.isfinite(x).all(axis=1)
        if not np.any(finite):
            continue
        rows = x[finite]
        import pandas as pd
        rows_df = pd.DataFrame(rows, columns=feats)
        prob = classifiers[group].predict_proba(rows_df)[:, 1].astype(np.float32)
        pred_frac = np.zeros_like(prob, dtype=np.float32)
        reg = regressors.get(group)
        if reg is not None:
            pred_frac = np.clip(reg.predict(rows_df), 0.0, 1.0).astype(np.float32)
        qa_vals = compute_qa_class(label, prob)
        mask_idx = np.flatnonzero(mask)
        valid_idx = mask_idx[finite]
        probability.ravel()[valid_idx] = prob
        fraction.ravel()[valid_idx] = pred_frac
        qa_class.ravel()[valid_idx] = qa_vals[np.isfinite(qa_vals)]
    return probability, fraction, qa_class


def qa_to_daod(fraction: np.ndarray, qa_class: np.ndarray, aod550: np.ndarray, minimum_qa: int) -> np.ndarray:
    out = np.asarray(fraction, dtype=float).copy() * np.asarray(aod550, dtype=float)
    out[np.asarray(qa_class, dtype=float) < float(minimum_qa)] = np.nan
    from apply_dust_model_to_modis_db import enforce_daod_output_semantics
    return enforce_daod_output_semantics(out, aod550)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_2d, lat2d, lon2d = read_granule_features(args.input_file)
    aod550 = np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)

    two_clf, two_reg, two_meta = load_two_stage_models(args.two_stage_model_dir)
    two_prob, two_frac, _ = predict_two_stage(two_clf, two_reg, two_meta, features_2d)
    two_daod = qa_to_daod(two_frac, np.where(np.isfinite(two_frac) & (two_prob >= float(two_meta["probability_threshold"])), 2.0, 0.0), aod550, minimum_qa=2)

    qa = decode_db_qa(args.input_file)
    pixel_features = build_pixel_features(features_2d, qa)
    db_clf, db_reg = load_db_type_models(args.db_type_model_dir)
    db_prob, db_frac, db_qa = predict_db_type(db_clf, db_reg, pixel_features, qa)
    db_daod = qa_to_daod(db_frac, db_qa, aod550, minimum_qa=2)

    _, _, li_daod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)

    vmax = float(
        np.nanpercentile(
            np.concatenate(
                [
                    two_daod[np.isfinite(two_daod)],
                    db_daod[np.isfinite(db_daod)],
                    li_daod[np.isfinite(li_daod)],
                ]
            ),
            98,
        )
    )
    save_comparison_panel(
        lat2d=lat2d,
        lon2d=lon2d,
        panels=[
            {"data": two_daod, "title": "Two-stage QA DAOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": db_daod, "title": "DB-type QA DAOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": li_daod, "title": "Li-Ginoux DAOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": db_daod - two_daod, "title": "DB-type QA minus two-stage", "label": "DAOD difference", "cmap": "RdBu_r"},
        ],
        output_path=args.output_dir / f"{args.input_file.stem}_two_stage_vs_db_type_comparison.png",
        figure_title=f"{args.input_file.name}: two-stage QA vs DB-type QA",
    )

    save_comparison_panel(
        lat2d=lat2d,
        lon2d=lon2d,
        panels=[
            {"data": two_prob, "title": "Two-stage dust probability", "label": "Probability", "cmap": "inferno", "vmin": 0.0, "vmax": 1.0},
            {"data": db_prob, "title": "DB-type dust probability", "label": "Probability", "cmap": "inferno", "vmin": 0.0, "vmax": 1.0},
            {"data": np.where(np.isfinite(db_qa), db_qa, np.nan), "title": "DB-type QA class", "label": "QA class", "cmap": "viridis", "vmin": 0.0, "vmax": 3.0},
            {"data": db_prob - two_prob, "title": "DB-type minus two-stage probability", "label": "Probability difference", "cmap": "RdBu_r"},
        ],
        output_path=args.output_dir / f"{args.input_file.stem}_two_stage_vs_db_type_probability.png",
        figure_title=f"{args.input_file.name}: two-stage vs DB-type probabilities",
    )

    summary = {
        "granule": args.input_file.name,
        "two_stage_daod_summary": summarize_grid("two_stage_daod", two_daod),
        "db_type_daod_summary": summarize_grid("db_type_daod", db_daod),
        "li_ginoux_daod_summary": summarize_grid("li_ginoux_daod", li_daod),
        "two_stage_probability_summary": summarize_grid("two_stage_probability", two_prob),
        "db_type_probability_summary": summarize_grid("db_type_probability", db_prob),
        "db_type_qa_summary": summarize_grid("db_type_qa", db_qa),
    }
    with (args.output_dir / f"{args.input_file.stem}_two_stage_vs_db_type_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
