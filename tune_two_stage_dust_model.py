from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from aeronet_dpr_utils import write_json
from apply_dust_model_to_modis_db import (
    enforce_daod_output_semantics,
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
)
from extract_modis_db_dust_model import build_training_dataframe, combine_arrays


FEATURE_COLUMNS = [
    "MODIS_DB_AOD550",
    "MODIS_DB_AOD412",
    "MODIS_DB_AOD470",
    "MODIS_DB_AOD660",
    "MODIS_DB_AE",
    "MODIS_DB_SSA412",
    "MODIS_DB_SSA470",
    "MODIS_DB_SSA660",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep dust-label thresholds and classifier probability cutoffs for the two-stage dust model."
    )
    parser.add_argument(
        "--mod04",
        type=Path,
        default=Path("../AERONET_MOD04_L2_collocation_with_SDA_data.pkl"),
    )
    parser.add_argument(
        "--myd04",
        type=Path,
        default=Path("../AERONET_MYD04_L2_collocation_with_SDA_data.pkl"),
    )
    parser.add_argument("--lr-dust-675", type=float, default=56.0)
    parser.add_argument(
        "--dust-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
    )
    parser.add_argument(
        "--probability-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--dust-granule",
        type=Path,
        default=Path("dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638.hdf"),
    )
    parser.add_argument(
        "--smoke-granule",
        type=Path,
        default=Path("dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127.hdf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("two_stage_tuning"),
    )
    return parser.parse_args()


def logit_transform(values: np.ndarray, eps: float = 1.0e-4) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def inv_logit_transform(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def summarize_case(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    probability_threshold: float,
    features_2d: dict[str, np.ndarray],
) -> dict[str, float]:
    h, w = np.asarray(next(iter(features_2d.values()))).shape
    x_flat = np.column_stack([np.asarray(features_2d[col], dtype=float).ravel() for col in FEATURE_COLUMNS])
    valid = np.isfinite(x_flat).all(axis=1)
    prob_flat = np.full(h * w, np.nan, dtype=float)
    frac_flat = np.full(h * w, np.nan, dtype=float)
    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=FEATURE_COLUMNS)
        prob = classifier.predict_proba(x_valid)[:, 1]
        frac = inv_logit_transform(regressor.predict(x_valid))
        frac = np.where(prob >= probability_threshold, frac, 0.0)
        prob_flat[valid] = prob
        frac_flat[valid] = frac

    prob_grid = prob_flat.reshape(h, w)
    frac_grid = frac_flat.reshape(h, w)
    daod_grid = frac_grid * np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
    daod_grid = enforce_daod_output_semantics(daod_grid, features_2d["MODIS_DB_AOD550"])

    valid_prob = np.isfinite(prob_grid)
    valid_daod = np.isfinite(daod_grid)
    return {
        "model_valid_pixels": int(np.sum(valid_prob)),
        "prob_mean": float(np.nanmean(prob_grid)) if np.any(valid_prob) else np.nan,
        "frac_mean": float(np.nanmean(frac_grid[valid_prob])) if np.any(valid_prob) else np.nan,
        "daod_mean": float(np.nanmean(daod_grid[valid_daod])) if np.any(valid_daod) else np.nan,
        "daod_p98": float(np.nanpercentile(daod_grid[valid_daod], 98)) if np.any(valid_daod) else np.nan,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_arrays(args.mod04, args.myd04)
    df = build_training_dataframe(combined, lr_dust_675=args.lr_dust_675)
    X = df[FEATURE_COLUMNS].astype(float)
    y_fraction = df["dust_fraction_675"].astype(float).to_numpy()
    total_aod550 = df["MODIS_DB_AOD550"].astype(float).to_numpy()
    y_daod550 = y_fraction * total_aod550

    dust_features, _, _ = read_granule_features(args.dust_granule)
    smoke_features, _, _ = read_granule_features(args.smoke_granule)
    _, _, dust_li = li_ginoux_dust_aod_with_ssa_constraint(dust_features)
    _, _, smoke_li = li_ginoux_dust_aod_with_ssa_constraint(smoke_features)
    dust_li_mean = float(np.nanmean(dust_li[np.isfinite(dust_li)]))
    smoke_li_mean = float(np.nanmean(smoke_li[np.isfinite(smoke_li)]))

    results = []
    trained_models: dict[float, tuple[XGBClassifier, XGBRegressor, dict]] = {}

    for dust_threshold in args.dust_thresholds:
        y_label = (y_fraction >= dust_threshold).astype(int)
        (
            X_train,
            X_tmp,
            y_label_train,
            y_label_tmp,
            y_fraction_train,
            y_fraction_tmp,
            y_daod_train,
            y_daod_tmp,
        ) = train_test_split(
            X,
            y_label,
            y_fraction,
            y_daod550,
            test_size=0.30,
            random_state=42,
            stratify=y_label,
        )
        (
            X_val,
            X_test,
            y_label_val,
            y_label_test,
            y_fraction_val,
            y_fraction_test,
            y_daod_val,
            y_daod_test,
        ) = train_test_split(
            X_tmp,
            y_label_tmp,
            y_fraction_tmp,
            y_daod_tmp,
            test_size=0.50,
            random_state=42,
            stratify=y_label_tmp,
        )

        pos = float(np.sum(y_label_train == 1))
        neg = float(np.sum(y_label_train == 0))
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        classifier = XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=120,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(X_train, y_label_train, eval_set=[(X_val, y_label_val)], verbose=False)

        train_mask = y_label_train == 1
        val_mask = y_label_val == 1
        test_mask = y_label_test == 1

        regressor = XGBRegressor(
            n_estimators=1200,
            early_stopping_rounds=150,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        regressor.fit(
            X_train.loc[train_mask],
            logit_transform(y_fraction_train[train_mask]),
            eval_set=[(X_val.loc[val_mask], logit_transform(y_fraction_val[val_mask]))],
            verbose=False,
        )

        prob_test = classifier.predict_proba(X_test)[:, 1]
        reg_fraction_test = inv_logit_transform(regressor.predict(X_test))
        base_meta = {
            "dust_threshold": dust_threshold,
            "n_total": int(len(X)),
            "n_dusty": int(np.sum(y_label)),
            "roc_auc": float(roc_auc_score(y_label_test, prob_test)),
        }
        trained_models[dust_threshold] = (classifier, regressor, base_meta)

        for prob_threshold in args.probability_thresholds:
            pred_label = (prob_test >= prob_threshold).astype(int)
            pred_fraction = np.where(pred_label == 1, reg_fraction_test, 0.0)
            pred_daod = pred_fraction * X_test["MODIS_DB_AOD550"].to_numpy()

            dust_case = summarize_case(classifier, regressor, prob_threshold, dust_features)
            smoke_case = summarize_case(classifier, regressor, prob_threshold, smoke_features)
            contrast_ratio = dust_case["daod_mean"] / (smoke_case["daod_mean"] + 1.0e-4)
            dust_retention = min(dust_case["daod_mean"] / max(dust_li_mean, 1.0e-6), 1.0)
            optimization_score = contrast_ratio * dust_retention

            row = {
                "dust_threshold": dust_threshold,
                "probability_threshold": prob_threshold,
                "roc_auc": float(roc_auc_score(y_label_test, prob_test)),
                "precision": float(precision_score(y_label_test, pred_label, zero_division=0)),
                "recall": float(recall_score(y_label_test, pred_label, zero_division=0)),
                "f1": float(f1_score(y_label_test, pred_label, zero_division=0)),
                "daod_rmse": float(np.sqrt(mean_squared_error(y_daod_test, pred_daod))),
                "daod_mae": float(mean_absolute_error(y_daod_test, pred_daod)),
                "daod_r2": float(r2_score(y_daod_test, pred_daod)),
                "dust_case_daod_mean": dust_case["daod_mean"],
                "dust_case_daod_p98": dust_case["daod_p98"],
                "dust_case_prob_mean": dust_case["prob_mean"],
                "smoke_case_daod_mean": smoke_case["daod_mean"],
                "smoke_case_daod_p98": smoke_case["daod_p98"],
                "smoke_case_prob_mean": smoke_case["prob_mean"],
                "li_dust_case_daod_mean": dust_li_mean,
                "li_smoke_case_daod_mean": smoke_li_mean,
                "dust_smoke_ratio": contrast_ratio,
                "dust_retention_vs_li": dust_case["daod_mean"] / max(dust_li_mean, 1.0e-6),
                "smoke_fraction_of_li": smoke_case["daod_mean"] / max(smoke_li_mean, 1.0e-6),
                "optimization_score": optimization_score,
            }
            results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        ["optimization_score", "roc_auc", "f1"],
        ascending=[False, False, False],
    )
    results_df.to_csv(args.output_dir / "two_stage_parameter_sweep.csv", index=False)

    best = results_df.iloc[0].to_dict()
    best_threshold = float(best["dust_threshold"])
    best_prob = float(best["probability_threshold"])
    classifier, regressor, base_meta = trained_models[best_threshold]

    best_dir = args.output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(best_dir / "two_stage_dust_classifier.json")
    regressor.save_model(best_dir / "two_stage_dust_fraction_regressor.json")
    best_meta = {
        "feature_names": FEATURE_COLUMNS,
        "classifier_model": "two_stage_dust_classifier.json",
        "regressor_model": "two_stage_dust_fraction_regressor.json",
        "regressor_target_transform": "logit",
        "target_name": "dust_fraction_675",
        "derived_dust_aod_name": "dust_AOD_550_from_fraction_times_MODIS_DB_AOD550",
        "dust_label_definition": f"dust_fraction_675 >= {best_threshold}",
        "probability_threshold": best_prob,
        "lr_dust_675": args.lr_dust_675,
        "selection_objective": "maximize dust/smoke contrast while retaining dust-case mean DAOD",
    }
    write_json(best_dir / "two_stage_metadata.json", best_meta)
    write_json(best_dir / "best_selection_summary.json", best)

    top5 = results_df.head(5)
    top5.to_csv(args.output_dir / "two_stage_parameter_sweep_top5.csv", index=False)

    print("Top parameter settings:")
    print(top5.to_string(index=False))
    print("\nRecommended setting:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
