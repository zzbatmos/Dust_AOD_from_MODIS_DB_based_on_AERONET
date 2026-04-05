from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

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
from extract_modis_db_dust_model import compute_aeronet_dust_fraction
from train_aeronet_dust_aod500_xgb import (
    compute_dust_lidar_ratio_stats,
    loglog_interpolate_target_500,
)


AEROSOL_GROUP_MAP = {
    "Dust": "dust",
    "Mixed": "mixed",
    "Smoke": "smoke_sulfate",
    "Sulfate": "smoke_sulfate",
}

BASE_FEATURES = [
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
    "has_ae",
    "qa_useful_fraction",
    "qa_conf0_fraction_all",
    "qa_conf1_fraction_all",
    "qa_conf2_fraction_all",
    "qa_conf3_fraction_all",
    "qa_aerosol_mixed_fraction_all",
    "qa_aerosol_dust_fraction_all",
    "qa_aerosol_smoke_fraction_all",
    "qa_aerosol_sulfate_fraction_all",
]

FEATURES_BY_GROUP = {
    "dust": BASE_FEATURES,
    "mixed": BASE_FEATURES + ["MODIS_DB_AE"],
    "smoke_sulfate": BASE_FEATURES + ["MODIS_DB_AE"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DB aerosol-type-conditional XGBoost dust models using the "
            "QA-augmented all-site collocation file."
        )
    )
    parser.add_argument(
        "--combined-pickle",
        type=Path,
        default=Path("AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl"),
    )
    parser.add_argument("--dust-threshold", type=float, default=0.2)
    parser.add_argument(
        "--target-method",
        choices=["loglog_440_675", "loglog_fit_440_675_1020"],
        default="loglog_440_675",
    )
    parser.add_argument("--lr-stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--dust-dominant-threshold", type=float, default=0.89)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modis_db_type_conditional_detection_from_qa"),
    )
    return parser.parse_args()


def frac_from_hist(hist: dict, key: str) -> float:
    entry = hist.get(key, {})
    value = entry.get("fraction", np.nan)
    try:
        return float(value)
    except Exception:
        return float("nan")


def dominant_aerosol_type(summary: dict) -> str | None:
    label = summary.get("dominant_aerosol_type_useful")
    if label is None:
        label = summary.get("dominant_aerosol_type_all")
    if label in AEROSOL_GROUP_MAP:
        return label
    return None


def dominant_algorithm_flag(summary: dict) -> str:
    label = summary.get("dominant_algorithm_flag_useful")
    if label is None or label == "-999":
        label = summary.get("dominant_algorithm_flag_all")
    if label in {"DeepBlue", "Vegetated", "Mixed"}:
        return str(label)
    return "Unknown"


def derive_dust_aod_550_from_rows(
    df: pd.DataFrame,
    dust_lr_by_wavelength: dict[int, float],
    target_method: str,
) -> np.ndarray:
    dust_fraction_440, _ = compute_aeronet_dust_fraction(
        df["AERONET_DPR440"].to_numpy(dtype=float),
        df["AERONET_LR440"].to_numpy(dtype=float),
        lr_dust=dust_lr_by_wavelength[440],
    )
    dust_fraction_675, _ = compute_aeronet_dust_fraction(
        df["AERONET_DPR675"].to_numpy(dtype=float),
        df["AERONET_LR675"].to_numpy(dtype=float),
        lr_dust=dust_lr_by_wavelength[675],
    )
    dust_fraction_1020, _ = compute_aeronet_dust_fraction(
        df["AERONET_DPR1020"].to_numpy(dtype=float),
        df["AERONET_LR1020"].to_numpy(dtype=float),
        lr_dust=dust_lr_by_wavelength[1020],
    )
    dust_aod_440 = df["AERONET_AOD440"].to_numpy(dtype=float) * dust_fraction_440
    dust_aod_675 = df["AERONET_AOD675"].to_numpy(dtype=float) * dust_fraction_675
    dust_aod_1020 = df["AERONET_AOD1020"].to_numpy(dtype=float) * dust_fraction_1020
    dust_aod_500 = loglog_interpolate_target_500(
        dust_aod_440=dust_aod_440,
        dust_aod_675=dust_aod_675,
        dust_aod_1020=dust_aod_1020,
        method=target_method,
    )
    dust_aod_550 = np.full_like(dust_aod_500, np.nan, dtype=float)
    mask = (
        np.isfinite(dust_aod_440)
        & np.isfinite(dust_aod_675)
        & np.isfinite(dust_aod_500)
        & (dust_aod_440 > 0.0)
        & (dust_aod_675 > 0.0)
        & (dust_aod_500 > 0.0)
    )
    alpha_440_675 = -np.log(dust_aod_440[mask] / dust_aod_675[mask]) / np.log(440.0 / 675.0)
    dust_aod_550[mask] = dust_aod_500[mask] * (550.0 / 500.0) ** (-alpha_440_675)
    return dust_aod_550


def build_dataframe(combined_pickle: Path) -> pd.DataFrame:
    import pickle

    with open(combined_pickle, "rb") as f:
        combined = pickle.load(f)

    rows: list[dict] = []
    for site_name, site_dict in combined.items():
        for granule_name, rec in site_dict.items():
            db = rec.get("Deep_Blue_data_with_QA") or rec.get("Deep_Blue_data")
            if not isinstance(db, dict):
                continue
            stats_full = db.get("stats_full")
            if not isinstance(stats_full, dict):
                continue
            qa = db.get("qa", {})
            qa_summary = qa.get("summary", {}) if isinstance(qa, dict) else {}
            aerosol_type = dominant_aerosol_type(qa_summary) if qa_summary else None
            if aerosol_type is None:
                continue

            sda_meta = rec.get("nearest_AERONET_SDA_L1.5_data_meta", {})
            sda = rec.get("nearest_AERONET_SDA_L1.5_data", {})
            spec_aod = stats_full.get("Deep_Blue_Spectral_Aerosol_Optical_Depth_Land", {}).get("mean")
            spec_ssa = stats_full.get("Deep_Blue_Spectral_Single_Scattering_Albedo_Land", {}).get("mean")
            ae_mean = stats_full.get("Deep_Blue_Angstrom_Exponent_Land", {}).get("mean")
            aod550_mean = db.get("aod550", {}).get("mean")

            if spec_aod is None or spec_ssa is None or ae_mean is None or aod550_mean is None:
                continue

            row = {
                "site_name": site_name,
                "granule_name": granule_name,
                "platform": "MOD04_L2" if granule_name.startswith("MOD04") else "MYD04_L2",
                "modis_time": rec.get("modis_time"),
                "AERONET_time": rec.get("AERONET_time"),
                "delta_minutes": float(rec.get("delta_minutes", np.nan)),
                "db_aerosol_type": aerosol_type,
                "db_algorithm_flag": dominant_algorithm_flag(qa_summary) if qa_summary else "Unknown",
                "MODIS_DB_AOD550": float(aod550_mean),
                "MODIS_DB_AOD412": float(spec_aod[0]),
                "MODIS_DB_AOD470": float(spec_aod[1]),
                "MODIS_DB_AOD660": float(spec_aod[2]),
                "MODIS_DB_AE": float(ae_mean),
                "MODIS_DB_SSA412": float(spec_ssa[0]),
                "MODIS_DB_SSA470": float(spec_ssa[1]),
                "MODIS_DB_SSA660": float(spec_ssa[2]),
                "AERONET_AOD440": float(rec.get("AOD_Coincident_Input[440nm]", np.nan)),
                "AERONET_AOD675": float(rec.get("AOD_Coincident_Input[675nm]", np.nan)),
                "AERONET_AOD1020": float(rec.get("AOD_Coincident_Input[1020nm]", np.nan)),
                "AERONET_DPR440": float(rec.get("Depolarization_Ratio[440nm]", np.nan)),
                "AERONET_DPR675": float(rec.get("Depolarization_Ratio[675nm]", np.nan)),
                "AERONET_DPR1020": float(rec.get("Depolarization_Ratio[1020nm]", np.nan)),
                "AERONET_LR440": float(rec.get("Lidar_Ratio[440nm]", np.nan)),
                "AERONET_LR675": float(rec.get("Lidar_Ratio[675nm]", np.nan)),
                "AERONET_LR1020": float(rec.get("Lidar_Ratio[1020nm]", np.nan)),
                "AERONET_SDA_AOD550": float(sda.get("Total_AOD_500nm[tau_a]", np.nan)),
                "AERONET_SDA_AE550": float(sda.get("Angstrom_Exponent(AE)-Total_500nm[alpha]", np.nan)),
                "AERONET_SDA_FMF550": float(sda.get("FineModeFraction_500nm[eta]", np.nan)),
                "AERONET_SDA_found": bool(sda_meta.get("found", False)),
                "qa_useful_fraction": frac_from_hist(qa_summary.get("usefulness_histogram", {}), "useful"),
                "qa_conf0_fraction_all": frac_from_hist(qa_summary.get("confidence_histogram_all", {}), "conf_0"),
                "qa_conf1_fraction_all": frac_from_hist(qa_summary.get("confidence_histogram_all", {}), "conf_1"),
                "qa_conf2_fraction_all": frac_from_hist(qa_summary.get("confidence_histogram_all", {}), "conf_2"),
                "qa_conf3_fraction_all": frac_from_hist(qa_summary.get("confidence_histogram_all", {}), "conf_3"),
                "qa_aerosol_mixed_fraction_all": frac_from_hist(qa_summary.get("aerosol_type_histogram_all", {}), "Mixed"),
                "qa_aerosol_dust_fraction_all": frac_from_hist(qa_summary.get("aerosol_type_histogram_all", {}), "Dust"),
                "qa_aerosol_smoke_fraction_all": frac_from_hist(qa_summary.get("aerosol_type_histogram_all", {}), "Smoke"),
                "qa_aerosol_sulfate_fraction_all": frac_from_hist(qa_summary.get("aerosol_type_histogram_all", {}), "Sulfate"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        out[f"has_{col.lower().replace('modis_db_', '')}"] = np.isfinite(out[col]).astype(int)
        out[col] = out[col].clip(0.0, 1.0).fillna(-0.05)
    out["has_ae"] = np.isfinite(out["MODIS_DB_AE"]).astype(int)
    out["MODIS_DB_AE"] = out["MODIS_DB_AE"].fillna(-0.05)
    out["AOD412_over_470"] = out["MODIS_DB_AOD412"] / out["MODIS_DB_AOD470"]
    out["AOD470_over_660"] = out["MODIS_DB_AOD470"] / out["MODIS_DB_AOD660"]
    out["AOD550_over_660"] = out["MODIS_DB_AOD550"] / out["MODIS_DB_AOD660"]
    out["SSA470_minus_412"] = out["MODIS_DB_SSA470"] - out["MODIS_DB_SSA412"]
    out["SSA660_minus_470"] = out["MODIS_DB_SSA660"] - out["MODIS_DB_SSA470"]
    out["db_type_group"] = out["db_aerosol_type"].map(AEROSOL_GROUP_MAP)
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in ["MODIS_DB_AOD550", "MODIS_DB_AOD412", "MODIS_DB_AOD470", "MODIS_DB_AOD660"]:
        out = out[np.isfinite(out[col]) & (out[col] > 0.0)]
    out = out.dropna(subset=["db_type_group"])
    return out


def fit_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], random_state: int) -> XGBClassifier:
    y_train = train_df["dust_label"].to_numpy(dtype=int)
    y_val = val_df["dust_label"].to_numpy(dtype=int)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=700,
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
        early_stopping_rounds=75,
    )
    model.fit(train_df[features], y_train, eval_set=[(val_df[features], y_val)], verbose=False)
    return model


def fit_fraction_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], random_state: int) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=75,
    )
    model.fit(
        train_df[features],
        train_df["dust_fraction_550_target"].to_numpy(dtype=float),
        eval_set=[(val_df[features], val_df["dust_fraction_550_target"].to_numpy(dtype=float))],
        verbose=False,
    )
    return model


def summarize_classifier(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, prob)) if np.unique(y_true).size > 1 else float("nan")
    return metrics


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if np.unique(y_true).size > 1 else float("nan"),
        "bias": float(np.mean(y_pred - y_true)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.combined_pickle)

    dust_lr_by_wl, dust_lr_stats_df = compute_dust_lidar_ratio_stats(
        combined={
            "AERONET_DPR440": df["AERONET_DPR440"].to_numpy(dtype=float),
            "AERONET_LR440": df["AERONET_LR440"].to_numpy(dtype=float),
            "AERONET_DPR675": df["AERONET_DPR675"].to_numpy(dtype=float),
            "AERONET_LR675": df["AERONET_LR675"].to_numpy(dtype=float),
            "AERONET_DPR1020": df["AERONET_DPR1020"].to_numpy(dtype=float),
            "AERONET_LR1020": df["AERONET_LR1020"].to_numpy(dtype=float),
        },
        dust_dominant_threshold=args.dust_dominant_threshold,
        lr_stat=args.lr_stat,
    )
    dust_lr_stats_df.to_csv(args.output_dir / "dust_dominant_lidar_ratio_stats.csv", index=False)

    dust_fraction_675, _ = compute_aeronet_dust_fraction(
        df["AERONET_DPR675"].to_numpy(dtype=float),
        df["AERONET_LR675"].to_numpy(dtype=float),
        lr_dust=dust_lr_by_wl[675],
    )
    dust_aod_550 = derive_dust_aod_550_from_rows(df, dust_lr_by_wl, args.target_method)
    sda_aod550 = df["AERONET_SDA_AOD550"].to_numpy(dtype=float)
    dust_fraction_550_target = np.clip(dust_aod_550 / sda_aod550, 0.0, 1.0)

    df["dust_fraction_675"] = dust_fraction_675
    df["dust_aod_550_target"] = dust_aod_550
    df["dust_fraction_550_target"] = dust_fraction_550_target
    df["dust_label"] = (df["dust_fraction_675"] >= args.dust_threshold).astype(int)
    df = add_engineered_features(df)
    df = df.dropna(
        subset=[
            "dust_fraction_675",
            "dust_aod_550_target",
            "dust_fraction_550_target",
            "AERONET_SDA_AOD550",
        ]
    )
    df = df[np.isfinite(df["dust_fraction_550_target"])]
    df = df[(df["dust_fraction_550_target"] >= 0.0) & (df["dust_fraction_550_target"] <= 1.0)]
    df.to_csv(args.output_dir / "type_conditional_from_qa_training_dataframe.csv", index=False)

    crosstab = (
        pd.crosstab(df["db_aerosol_type"], df["dust_label"])
        .rename(columns={0: "non_dust", 1: "dust"})
        .reset_index()
    )
    crosstab.to_csv(args.output_dir / "db_aerosol_type_vs_aeronet_dust_crosstab.csv", index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=args.random_state,
        stratify=df["db_type_group"].astype(str) + "_" + df["dust_label"].astype(str),
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=args.random_state,
        stratify=train_df["db_type_group"].astype(str) + "_" + train_df["dust_label"].astype(str),
    )

    regime_rows = []
    importance_rows = []
    metadata = {
        "combined_pickle": str(args.combined_pickle),
        "dust_threshold": args.dust_threshold,
        "target_method": args.target_method,
        "lr_stat": args.lr_stat,
        "dust_dominant_threshold": args.dust_dominant_threshold,
        "groups": {},
    }

    for group_name in ["dust", "mixed", "smoke_sulfate"]:
        features = FEATURES_BY_GROUP[group_name]
        train_g = train_df[train_df["db_type_group"] == group_name].copy()
        val_g = val_df[val_df["db_type_group"] == group_name].copy()
        test_g = test_df[test_df["db_type_group"] == group_name].copy()
        if train_g.empty or val_g.empty or test_g.empty:
            continue
        if train_g["dust_label"].nunique() < 2 or val_g["dust_label"].nunique() < 2:
            continue

        clf = fit_classifier(train_g, val_g, features, args.random_state)
        prob_test = clf.predict_proba(test_g[features])[:, 1]
        clf_metrics = summarize_classifier(test_g["dust_label"].to_numpy(dtype=int), prob_test)

        reg_train = train_g[train_g["dust_label"] == 1].copy()
        reg_val = val_g[val_g["dust_label"] == 1].copy()
        reg_test = test_g[test_g["dust_label"] == 1].copy()
        reg_metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "bias": float("nan")}
        if len(reg_train) >= 25 and len(reg_val) >= 10 and len(reg_test) >= 10:
            reg = fit_fraction_regressor(reg_train, reg_val, features, args.random_state)
            pred_fraction = np.clip(reg.predict(reg_test[features]), 0.0, 1.0)
            reg_metrics = summarize_regression(
                reg_test["dust_fraction_550_target"].to_numpy(dtype=float),
                pred_fraction,
            )
            for feat, imp in zip(features, reg.feature_importances_):
                importance_rows.append(
                    {"group": group_name, "model": "fraction_regressor", "feature": feat, "importance": float(imp)}
                )
        else:
            reg = None

        for feat, imp in zip(features, clf.feature_importances_):
            importance_rows.append(
                {"group": group_name, "model": "classifier", "feature": feat, "importance": float(imp)}
            )

        row = {
            "group": group_name,
            "n_train": int(len(train_g)),
            "n_val": int(len(val_g)),
            "n_test": int(len(test_g)),
            "n_train_dust": int(train_g["dust_label"].sum()),
            **clf_metrics,
            **{f"fraction_{k}": v for k, v in reg_metrics.items()},
        }
        regime_rows.append(row)
        metadata["groups"][group_name] = row
        clf.save_model(str(args.output_dir / f"{group_name}_classifier.json"))
        if reg is not None:
            reg.save_model(str(args.output_dir / f"{group_name}_fraction_regressor.json"))

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(args.output_dir / "regime_metrics.csv", index=False)
    pd.DataFrame(importance_rows).sort_values(
        ["group", "model", "importance"], ascending=[True, True, False]
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)
    write_json(args.output_dir / "metadata.json", metadata)

    summary = {
        "n_records": int(len(df)),
        "n_sites": int(df["site_name"].nunique()),
        "db_aerosol_type_counts": df["db_aerosol_type"].value_counts().to_dict(),
        "db_type_group_counts": df["db_type_group"].value_counts().to_dict(),
        "sda_found_count": int(df["AERONET_SDA_found"].sum()),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
