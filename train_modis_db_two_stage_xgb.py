from __future__ import annotations

import argparse
import json
import os
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
from extract_modis_db_dust_model import (
    build_training_dataframe,
    combine_arrays,
)


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
        description=(
            "Train a two-stage MODIS DB dust model: "
            "(1) dust detection classifier and (2) dust-fraction regressor."
        )
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
        "--dust-threshold",
        type=float,
        default=0.2,
        help="Dusty label threshold applied to dust_fraction_675.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Classifier probability threshold used at inference time.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modis_db_two_stage_xgb"),
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_arrays(args.mod04, args.myd04)
    df = build_training_dataframe(combined, lr_dust_675=args.lr_dust_675)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["dust_label"] = (df["dust_fraction_675"] >= args.dust_threshold).astype(int)
    df.to_csv(args.output_dir / "two_stage_training_dataframe.csv", index=False)

    X = df[FEATURE_COLUMNS].astype(float)
    y_fraction = df["dust_fraction_675"].astype(float).to_numpy()
    y_label = df["dust_label"].astype(int).to_numpy()
    total_aod550 = df["MODIS_DB_AOD550"].astype(float).to_numpy()
    y_daod550 = y_fraction * total_aod550

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
        n_estimators=1200,
        early_stopping_rounds=150,
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
    classifier.fit(X_train, y_label_train, eval_set=[(X_val, y_label_val)], verbose=200)

    train_mask = y_label_train == 1
    val_mask = y_label_val == 1
    test_mask = y_label_test == 1

    regressor = XGBRegressor(
        n_estimators=1500,
        early_stopping_rounds=200,
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
        verbose=200,
    )

    prob_test = classifier.predict_proba(X_test)[:, 1]
    pred_label = (prob_test >= args.probability_threshold).astype(int)
    pred_fraction_positive = inv_logit_transform(regressor.predict(X_test))
    pred_fraction_final = np.where(pred_label == 1, pred_fraction_positive, 0.0)
    pred_daod550_final = pred_fraction_final * X_test["MODIS_DB_AOD550"].to_numpy()

    metrics = {
        "detection_threshold": args.dust_threshold,
        "probability_threshold": args.probability_threshold,
        "n_samples_total": int(len(df)),
        "n_samples_dusty": int(np.sum(y_label)),
        "classification": {
            "ROC_AUC": float(roc_auc_score(y_label_test, prob_test)),
            "precision": float(precision_score(y_label_test, pred_label, zero_division=0)),
            "recall": float(recall_score(y_label_test, pred_label, zero_division=0)),
            "f1": float(f1_score(y_label_test, pred_label, zero_division=0)),
        },
        "regression_overall_fraction": {
            "RMSE": float(np.sqrt(mean_squared_error(y_fraction_test, pred_fraction_final))),
            "MAE": float(mean_absolute_error(y_fraction_test, pred_fraction_final)),
            "R2": float(r2_score(y_fraction_test, pred_fraction_final)),
            "bias": float(np.mean(pred_fraction_final - y_fraction_test)),
        },
        "regression_dusty_only_fraction": {
            "RMSE": float(np.sqrt(mean_squared_error(y_fraction_test[test_mask], pred_fraction_final[test_mask]))),
            "MAE": float(mean_absolute_error(y_fraction_test[test_mask], pred_fraction_final[test_mask])),
            "R2": float(r2_score(y_fraction_test[test_mask], pred_fraction_final[test_mask])),
            "bias": float(np.mean(pred_fraction_final[test_mask] - y_fraction_test[test_mask])),
        },
        "regression_overall_daod550": {
            "RMSE": float(np.sqrt(mean_squared_error(y_daod_test, pred_daod550_final))),
            "MAE": float(mean_absolute_error(y_daod_test, pred_daod550_final)),
            "R2": float(r2_score(y_daod_test, pred_daod550_final)),
            "bias": float(np.mean(pred_daod550_final - y_daod_test)),
        },
        "regression_dusty_only_daod550": {
            "RMSE": float(np.sqrt(mean_squared_error(y_daod_test[test_mask], pred_daod550_final[test_mask]))),
            "MAE": float(mean_absolute_error(y_daod_test[test_mask], pred_daod550_final[test_mask])),
            "R2": float(r2_score(y_daod_test[test_mask], pred_daod550_final[test_mask])),
            "bias": float(np.mean(pred_daod550_final[test_mask] - y_daod_test[test_mask])),
        },
    }

    classifier.save_model(args.output_dir / "two_stage_dust_classifier.json")
    regressor.save_model(args.output_dir / "two_stage_dust_fraction_regressor.json")

    classifier_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    classifier_importance.to_csv(args.output_dir / "two_stage_classifier_feature_importance.csv", index=False)

    regressor_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    regressor_importance.to_csv(args.output_dir / "two_stage_regressor_feature_importance.csv", index=False)

    metadata = {
        "feature_names": FEATURE_COLUMNS,
        "classifier_model": "two_stage_dust_classifier.json",
        "regressor_model": "two_stage_dust_fraction_regressor.json",
        "regressor_target_transform": "logit",
        "target_name": "dust_fraction_675",
        "derived_dust_aod_name": "dust_AOD_550_from_fraction_times_MODIS_DB_AOD550",
        "dust_label_definition": f"dust_fraction_675 >= {args.dust_threshold}",
        "probability_threshold": args.probability_threshold,
        "lr_dust_675": args.lr_dust_675,
    }
    write_json(args.output_dir / "two_stage_metadata.json", metadata)
    write_json(args.output_dir / "two_stage_metrics.json", metrics)

    print(f"Training rows: {len(df)}")
    print(
        "Classifier:",
        f"ROC_AUC={metrics['classification']['ROC_AUC']:.4f}",
        f"precision={metrics['classification']['precision']:.4f}",
        f"recall={metrics['classification']['recall']:.4f}",
        f"f1={metrics['classification']['f1']:.4f}",
    )
    print(
        "Overall dust_AOD_550:",
        f"RMSE={metrics['regression_overall_daod550']['RMSE']:.4f}",
        f"MAE={metrics['regression_overall_daod550']['MAE']:.4f}",
        f"R2={metrics['regression_overall_daod550']['R2']:.4f}",
    )


if __name__ == "__main__":
    main()
