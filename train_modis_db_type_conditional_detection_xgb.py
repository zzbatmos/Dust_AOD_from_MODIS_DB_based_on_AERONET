from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from aeronet_dpr_utils import write_json
from extract_modis_db_dust_model import combine_arrays, compute_aeronet_dust_fraction


BASE_COLUMNS = [
    "MODIS_DB_AOD550",
    "MODIS_DB_AOD412",
    "MODIS_DB_AOD470",
    "MODIS_DB_AOD660",
    "MODIS_DB_AE",
    "MODIS_DB_SSA412",
    "MODIS_DB_SSA470",
    "MODIS_DB_SSA660",
]

REGIME_FEATURES = {
    "dust": [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "AOD412_over_470",
        "AOD470_over_660",
        "AOD550_over_660",
        "MODIS_DB_SSA412",
        "MODIS_DB_SSA470",
        "MODIS_DB_SSA660",
        "SSA470_minus_412",
        "SSA660_minus_470",
        "has_ssa412",
        "has_ssa470",
        "has_ssa660",
    ],
    "smoke": [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "MODIS_DB_AE",
        "AOD412_over_470",
        "AOD470_over_660",
        "AOD550_over_660",
        "MODIS_DB_SSA412",
        "MODIS_DB_SSA470",
        "MODIS_DB_SSA660",
        "SSA470_minus_412",
        "SSA660_minus_470",
        "has_ssa412",
        "has_ssa470",
        "has_ssa660",
    ],
    "mixed": [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "MODIS_DB_AE",
        "AOD412_over_470",
        "AOD470_over_660",
        "AOD550_over_660",
        "MODIS_DB_SSA412",
        "MODIS_DB_SSA470",
        "MODIS_DB_SSA660",
        "SSA470_minus_412",
        "SSA660_minus_470",
        "has_ssa412",
        "has_ssa470",
        "has_ssa660",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train regime-conditional dust-detection XGBoost classifiers using "
            "AERONET DPR/LR dust labels and DB aerosol-regime proxies inferred "
            "from collocated MODIS Deep Blue outputs."
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
    parser.add_argument("--dust-threshold", type=float, default=0.2)
    parser.add_argument(
        "--dust-ae-max",
        type=float,
        default=0.5,
        help="AE threshold below which the collocation is routed to the DB dust branch proxy.",
    )
    parser.add_argument(
        "--smoke-ae-min",
        type=float,
        default=1.2,
        help="AE threshold above which the collocation is routed to the DB smoke branch proxy.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modis_db_type_conditional_detection"),
    )
    return parser.parse_args()


def infer_db_regime_proxy(ae: pd.Series, dust_ae_max: float, smoke_ae_min: float) -> pd.Series:
    regime = pd.Series(index=ae.index, dtype="object")
    regime.loc[ae <= dust_ae_max] = "dust"
    regime.loc[ae >= smoke_ae_min] = "smoke"
    regime.loc[(ae > dust_ae_max) & (ae < smoke_ae_min)] = "mixed"
    return regime


def build_detection_dataframe(
    combined: dict[str, np.ndarray],
    lr_dust_675: float,
    dust_threshold: float,
    dust_ae_max: float,
    smoke_ae_min: float,
) -> pd.DataFrame:
    dust_fraction_675, rd_675 = compute_aeronet_dust_fraction(
        combined["AERONET_DPR675"],
        combined["AERONET_LR675"],
        lr_dust=lr_dust_675,
    )
    df = pd.DataFrame(
        {
            "dust_fraction_675": dust_fraction_675,
            "Rd_675": rd_675,
            "dust_label": (dust_fraction_675 >= dust_threshold).astype(int),
            "MODIS_DB_AOD550": combined["MODIS_DB_AOD550"],
            "MODIS_DB_AOD412": combined["MODIS_DB_AOD412"],
            "MODIS_DB_AOD470": combined["MODIS_DB_AOD470"],
            "MODIS_DB_AOD660": combined["MODIS_DB_AOD660"],
            "MODIS_DB_AE": combined["MODIS_DB_AE"],
            "MODIS_DB_SSA412": combined["MODIS_DB_SSA412"],
            "MODIS_DB_SSA470": combined["MODIS_DB_SSA470"],
            "MODIS_DB_SSA660": combined["MODIS_DB_SSA660"],
            "AERONET_SDA_AOD550": combined["AERONET_SDA_AOD550"],
            "AERONET_SDA_AE550": combined["AERONET_SDA_AE550"],
            "AERONET_SDA_FMF550": combined["AERONET_SDA_FMF550"],
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["dust_fraction_675", "MODIS_DB_AOD550", "MODIS_DB_AOD412", "MODIS_DB_AOD470", "MODIS_DB_AOD660", "MODIS_DB_AE"])
    for col in ["MODIS_DB_AOD550", "MODIS_DB_AOD412", "MODIS_DB_AOD470", "MODIS_DB_AOD660"]:
        df = df[df[col] > 0.0]

    for col in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        df[f"has_{col.lower().replace('modis_db_', '')}"] = np.isfinite(df[col]).astype(int)
        df[col] = df[col].clip(0.0, 1.0)
        df[col] = df[col].fillna(-0.05)

    df["AOD412_over_470"] = df["MODIS_DB_AOD412"] / df["MODIS_DB_AOD470"]
    df["AOD470_over_660"] = df["MODIS_DB_AOD470"] / df["MODIS_DB_AOD660"]
    df["AOD550_over_660"] = df["MODIS_DB_AOD550"] / df["MODIS_DB_AOD660"]
    df["SSA470_minus_412"] = df["MODIS_DB_SSA470"] - df["MODIS_DB_SSA412"]
    df["SSA660_minus_470"] = df["MODIS_DB_SSA660"] - df["MODIS_DB_SSA470"]
    df["db_regime_proxy"] = infer_db_regime_proxy(df["MODIS_DB_AE"], dust_ae_max, smoke_ae_min)
    df = df.dropna(subset=["db_regime_proxy"])
    return df


def make_stratify_key(df: pd.DataFrame) -> pd.Series:
    key = df["db_regime_proxy"].astype(str) + "_" + df["dust_label"].astype(str)
    counts = key.value_counts()
    rare = counts[counts < 2].index
    return key.where(~key.isin(rare), other=df["dust_label"].map({0: "other_0", 1: "other_1"}))


def fit_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    random_state: int,
) -> XGBClassifier:
    y_train = train_df["dust_label"].to_numpy()
    y_val = val_df["dust_label"].to_numpy()
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=100,
    )
    model.fit(
        train_df[features],
        y_train,
        eval_set=[(val_df[features], y_val)],
        verbose=False,
    )
    return model


def summarize_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_arrays(args.mod04, args.myd04)
    df = build_detection_dataframe(
        combined=combined,
        lr_dust_675=args.lr_dust_675,
        dust_threshold=args.dust_threshold,
        dust_ae_max=args.dust_ae_max,
        smoke_ae_min=args.smoke_ae_min,
    )
    df.to_csv(args.output_dir / "type_conditional_detection_training_dataframe.csv", index=False)

    stratify_key = make_stratify_key(df)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=args.random_state,
        stratify=stratify_key,
    )
    train_key = make_stratify_key(train_df)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=args.random_state,
        stratify=train_key,
    )

    models: dict[str, XGBClassifier] = {}
    regime_rows: list[dict[str, object]] = []
    test_prob = pd.Series(index=test_df.index, dtype=float)

    for regime, features in REGIME_FEATURES.items():
        train_regime = train_df[train_df["db_regime_proxy"] == regime].copy()
        val_regime = val_df[val_df["db_regime_proxy"] == regime].copy()
        test_regime = test_df[test_df["db_regime_proxy"] == regime].copy()
        if len(train_regime) < 50 or train_regime["dust_label"].nunique() < 2:
            regime_rows.append(
                {
                    "regime": regime,
                    "n_train": int(len(train_regime)),
                    "n_val": int(len(val_regime)),
                    "n_test": int(len(test_regime)),
                    "n_dust_train": int(train_regime["dust_label"].sum()),
                    "status": "skipped_insufficient_training_data",
                }
            )
            continue
        if len(val_regime) < 10 or val_regime["dust_label"].nunique() < 2:
            val_regime = train_regime.sample(frac=0.20, random_state=args.random_state)
            train_regime = train_regime.drop(val_regime.index)

        model = fit_classifier(train_regime, val_regime, features, args.random_state)
        models[regime] = model
        val_prob = model.predict_proba(val_regime[features])[:, 1]
        test_prob_regime = model.predict_proba(test_regime[features])[:, 1]
        test_prob.loc[test_regime.index] = test_prob_regime
        metrics = summarize_metrics(test_regime["dust_label"].to_numpy(), test_prob_regime)
        regime_rows.append(
            {
                "regime": regime,
                "n_train": int(len(train_regime)),
                "n_val": int(len(val_regime)),
                "n_test": int(len(test_regime)),
                "n_dust_train": int(train_regime["dust_label"].sum()),
                "n_dust_test": int(test_regime["dust_label"].sum()),
                "status": "trained",
                **metrics,
            }
        )
        model.save_model(args.output_dir / f"{regime}_dust_detector.json")
        pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            args.output_dir / f"{regime}_feature_importance.csv",
            index=False,
        )

    trained_regimes = set(models)
    test_eval = test_df[test_df["db_regime_proxy"].isin(trained_regimes)].copy()
    test_eval["predicted_probability"] = test_prob.loc[test_eval.index].to_numpy()

    overall_metrics = summarize_metrics(
        test_eval["dust_label"].to_numpy(),
        test_eval["predicted_probability"].to_numpy(),
    )

    crosstab = (
        df.groupby(["db_regime_proxy", "dust_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non_dust", 1: "dust"})
        .reset_index()
    )
    crosstab.to_csv(args.output_dir / "regime_vs_aeronet_dust_crosstab.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(args.output_dir / "regime_test_metrics.csv", index=False)
    test_eval.to_csv(args.output_dir / "regime_conditional_test_predictions.csv", index=False)

    metadata = {
        "dust_threshold": args.dust_threshold,
        "db_regime_proxy_definition": {
            "dust": f"MODIS_DB_AE <= {args.dust_ae_max}",
            "smoke": f"MODIS_DB_AE >= {args.smoke_ae_min}",
            "mixed": f"{args.dust_ae_max} < MODIS_DB_AE < {args.smoke_ae_min}",
        },
        "per_regime_features": REGIME_FEATURES,
        "n_total_samples": int(len(df)),
        "n_train_samples": int(len(train_df)),
        "n_val_samples": int(len(val_df)),
        "n_test_samples": int(len(test_df)),
        "overall_test_metrics": overall_metrics,
        "note": (
            "This first-pass implementation uses a DB aerosol-regime proxy inferred "
            "from collocated MODIS DB Angstrom exponent because the saved "
            "collocation pickles do not include Quality_Assurance_Land aerosol-type bits."
        ),
    }
    write_json(args.output_dir / "type_conditional_detection_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
