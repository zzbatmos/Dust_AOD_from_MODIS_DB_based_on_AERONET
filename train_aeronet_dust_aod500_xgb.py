from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from extract_modis_db_dust_model import (
    compute_aeronet_dust_fraction,
    extract_arrays_from_collocation_dict,
)
from aeronet_dpr_utils import dust_fraction_from_pldr, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AERONET-on-AERONET XGBoost model for dust AOD at 500 nm."
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
    parser.add_argument(
        "--target-method",
        choices=["loglog_440_675", "loglog_fit_440_675_1020"],
        default="loglog_440_675",
        help="How to derive dust AOD at 500 nm from the 440/675/1020 nm dust-AOD spectrum.",
    )
    parser.add_argument(
        "--lr-stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic used for the dust-dominant LR_DUST values.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["compare", "without_dpr_lr", "with_dpr_lr"],
        default="compare",
        help="Train without DPR/LR, with DPR/LR, or both for side-by-side comparison.",
    )
    parser.add_argument("--dust-dominant-threshold", type=float, default=0.89)
    parser.add_argument("--output-dir", type=Path, default=Path("aeronet_dust_aod500_xgb"))
    return parser.parse_args()


def load_combined_arrays(mod04_path: Path, myd04_path: Path) -> dict[str, np.ndarray]:
    with mod04_path.open("rb") as handle:
        dic_mod04 = pickle.load(handle)
    with myd04_path.open("rb") as handle:
        dic_myd04 = pickle.load(handle)
    arr_mod04 = extract_arrays_from_collocation_dict(dic_mod04)
    arr_myd04 = extract_arrays_from_collocation_dict(dic_myd04)
    return {key: np.concatenate([arr_mod04[key], arr_myd04[key]]) for key in arr_mod04}


def compute_dust_lidar_ratio_stats(
    combined: dict[str, np.ndarray],
    dust_dominant_threshold: float,
    lr_stat: str,
) -> tuple[dict[int, float], pd.DataFrame]:
    rd_1020 = dust_fraction_from_pldr(combined["AERONET_DPR1020"])
    stats_records = []
    dust_lr_by_wavelength: dict[int, float] = {}
    for wl in [440, 675, 1020]:
        lr = np.asarray(combined[f"AERONET_LR{wl}"], dtype=float)
        mask = np.isfinite(rd_1020) & np.isfinite(lr) & (lr > 0.0) & (rd_1020 > dust_dominant_threshold)
        values = lr[mask]
        if values.size == 0:
            raise ValueError(f"No dust-dominant LR samples found for {wl} nm.")
        dust_lr_by_wavelength[wl] = float(np.mean(values) if lr_stat == "mean" else np.median(values))
        stats_records.append(
            {
                "wavelength_nm": wl,
                "dust_dominant_count": int(values.size),
                "dust_lidar_ratio_mean": float(np.mean(values)),
                "dust_lidar_ratio_median": float(np.median(values)),
                "dust_lidar_ratio_std": float(np.std(values)),
                "dust_lidar_ratio_selected": dust_lr_by_wavelength[wl],
            }
        )
    return dust_lr_by_wavelength, pd.DataFrame(stats_records)


def loglog_interpolate_target_500(
    dust_aod_440: np.ndarray,
    dust_aod_675: np.ndarray,
    dust_aod_1020: np.ndarray,
    method: str,
) -> np.ndarray:
    target = np.full_like(np.asarray(dust_aod_675, dtype=float), np.nan, dtype=float)
    tau440 = np.asarray(dust_aod_440, dtype=float)
    tau675 = np.asarray(dust_aod_675, dtype=float)
    tau1020 = np.asarray(dust_aod_1020, dtype=float)
    if method == "loglog_440_675":
        mask = np.isfinite(tau440) & np.isfinite(tau675) & (tau440 > 0.0) & (tau675 > 0.0)
        alpha = -np.log(tau440[mask] / tau675[mask]) / np.log(440.0 / 675.0)
        target[mask] = tau440[mask] * (500.0 / 440.0) ** (-alpha)
        return target
    if method == "loglog_fit_440_675_1020":
        mask = (
            np.isfinite(tau440)
            & np.isfinite(tau675)
            & np.isfinite(tau1020)
            & (tau440 > 0.0)
            & (tau675 > 0.0)
            & (tau1020 > 0.0)
        )
        if np.any(mask):
            x = np.log(np.array([440.0, 675.0, 1020.0]))
            y = np.log(np.stack([tau440[mask], tau675[mask], tau1020[mask]], axis=1))
            slopes, intercepts = np.polyfit(x, y.T, 1)
            target[mask] = np.exp(intercepts + slopes * np.log(500.0))
        return target
    raise ValueError(f"Unsupported target interpolation method: {method}")


def build_training_dataframe(
    combined: dict[str, np.ndarray],
    dust_lr_by_wavelength: dict[int, float],
    target_method: str,
) -> pd.DataFrame:
    aeronet_cmf440 = combined["AERONET_EXT_COARSE440"] / combined["AERONET_EXT_TOTAL440"]
    aeronet_cm_aod440 = combined["AERONET_AOD440"] * aeronet_cmf440
    aeronet_cmf675 = combined["AERONET_EXT_COARSE675"] / combined["AERONET_EXT_TOTAL675"]
    aeronet_cm_aod675 = combined["AERONET_AOD675"] * aeronet_cmf675
    aeronet_cmf1020 = combined["AERONET_EXT_COARSE1020"] / combined["AERONET_EXT_TOTAL1020"]
    aeronet_cm_aod1020 = combined["AERONET_AOD1020"] * aeronet_cmf1020

    dust_fraction_440, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR440"], combined["AERONET_LR440"], lr_dust=dust_lr_by_wavelength[440]
    )
    dust_fraction_675, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR675"], combined["AERONET_LR675"], lr_dust=dust_lr_by_wavelength[675]
    )
    dust_fraction_1020, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR1020"], combined["AERONET_LR1020"], lr_dust=dust_lr_by_wavelength[1020]
    )

    dust_aod_440 = combined["AERONET_AOD440"] * dust_fraction_440
    dust_aod_675 = combined["AERONET_AOD675"] * dust_fraction_675
    dust_aod_1020 = combined["AERONET_AOD1020"] * dust_fraction_1020
    dust_aod_500 = loglog_interpolate_target_500(
        dust_aod_440=dust_aod_440,
        dust_aod_675=dust_aod_675,
        dust_aod_1020=dust_aod_1020,
        method=target_method,
    )

    aeronet_cmf550 = 1.0 - combined["AERONET_SDA_FMF550"]
    aeronet_cm_aod550 = aeronet_cmf550 * combined["AERONET_SDA_AOD550"]

    df = pd.DataFrame(
        {
            "dust_AOD_500": dust_aod_500,
            "AERONET_SDA_AE550": combined["AERONET_SDA_AE550"],
            "AERONET_SDA_AOD550": combined["AERONET_SDA_AOD550"],
            "AERONET_SDA_FMF550": combined["AERONET_SDA_FMF550"],
            "AERONET_AOD440": combined["AERONET_AOD440"],
            "AERONET_AOD675": combined["AERONET_AOD675"],
            "AERONET_AOD1020": combined["AERONET_AOD1020"],
            "AERONET_CM_AOD440": aeronet_cm_aod440,
            "AERONET_CM_AOD675": aeronet_cm_aod675,
            "AERONET_CM_AOD1020": aeronet_cm_aod1020,
            "AERONET_CMF440": aeronet_cmf440,
            "AERONET_CMF675": aeronet_cmf675,
            "AERONET_CMF1020": aeronet_cmf1020,
            "AERONET_SSA440": combined["AERONET_SSA440"],
            "AERONET_SSA675": combined["AERONET_SSA675"],
            "AERONET_SSA1020": combined["AERONET_SSA1020"],
            "AERONET_CM_AOD550_SDA": aeronet_cm_aod550,
            "AERONET_DPR440": combined["AERONET_DPR440"],
            "AERONET_DPR675": combined["AERONET_DPR675"],
            "AERONET_DPR1020": combined["AERONET_DPR1020"],
            "AERONET_LR440": combined["AERONET_LR440"],
            "AERONET_LR675": combined["AERONET_LR675"],
            "AERONET_LR1020": combined["AERONET_LR1020"],
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["dust_AOD_500"] >= 0.0]
    for column in [
        "AERONET_SDA_AOD550",
        "AERONET_AOD440",
        "AERONET_AOD675",
        "AERONET_AOD1020",
        "AERONET_CM_AOD440",
        "AERONET_CM_AOD675",
        "AERONET_CM_AOD1020",
        "AERONET_CM_AOD550_SDA",
        "AERONET_DPR440",
        "AERONET_DPR675",
        "AERONET_DPR1020",
        "AERONET_LR440",
        "AERONET_LR675",
        "AERONET_LR1020",
    ]:
        df = df[df[column] > 0.0]
    df["AERONET_SDA_FMF550"] = df["AERONET_SDA_FMF550"].clip(0.0, 1.0)
    for column in ["AERONET_SSA440", "AERONET_SSA675", "AERONET_SSA1020"]:
        df[column] = df[column].clip(0.0, 1.0)
    return df


def log1p_transform(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))


def inv_log1p_transform(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(values, dtype=float)), 0.0, None)


def base_feature_columns() -> list[str]:
    return [
        "AERONET_SDA_AE550",
        "AERONET_SDA_AOD550",
        "AERONET_SDA_FMF550",
        "AERONET_AOD440",
        "AERONET_AOD675",
        "AERONET_AOD1020",
        "AERONET_CM_AOD440",
        "AERONET_CM_AOD675",
        "AERONET_CM_AOD1020",
        "AERONET_CMF440",
        "AERONET_CMF675",
        "AERONET_CMF1020",
        "AERONET_SSA440",
        "AERONET_SSA675",
        "AERONET_SSA1020",
        "AERONET_CM_AOD550_SDA",
    ]


def dpr_lr_feature_columns() -> list[str]:
    return [
        "AERONET_DPR440",
        "AERONET_DPR675",
        "AERONET_DPR1020",
        "AERONET_LR440",
        "AERONET_LR675",
        "AERONET_LR1020",
    ]


def sahp_parity_plot_log(
    y_true_lin: np.ndarray,
    y_pred_lin: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    lo: float = 1e-4,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true_lin, dtype=float)
    y_pred = np.asarray(y_pred_lin, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0) & (y_pred > 0.0)
    yt = y_true[mask]
    yp = y_pred[mask]
    hi = float(np.nanmax([yt.max(), yp.max()]) * 1.05)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))
    n = int(yt.size)

    txt = f"N = {n}\nBias = {bias:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR2 = {r2:.4f}"
    xx = np.logspace(np.log10(lo), np.log10(hi), 200)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    axes[0].scatter(yt, yp, s=10, alpha=0.35, linewidths=0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].plot(xx, xx, linestyle="--", linewidth=1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} (scatter)")
    axes[0].text(
        0.02,
        0.98,
        txt,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    hb = axes[1].hexbin(yt, yp, gridsize=70, mincnt=1, xscale="log", yscale="log")
    fig.colorbar(hb, ax=axes[1], label="count")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].plot(xx, xx, linestyle="--", linewidth=1)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"{title} (density)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return {"N": n, "bias": bias, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_one_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    experiment_name: str,
    output_dir: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    y_lin = df["dust_AOD_500"].astype(float).to_numpy()
    X = df[feature_columns].astype(float)
    y_log = log1p_transform(y_lin)

    X_train, X_tmp, y_train_log, y_tmp_log, y_train_lin, y_tmp_lin = train_test_split(
        X, y_log, y_lin, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val_log, y_test_log, y_val_lin, y_test_lin = train_test_split(
        X_tmp, y_tmp_log, y_tmp_lin, test_size=0.50, random_state=42
    )

    model = XGBRegressor(
        n_estimators=1500,
        early_stopping_rounds=200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    y_pred_log = model.predict(X_test)
    y_pred_lin = inv_log1p_transform(y_pred_log)
    metrics = {
        "experiment": experiment_name,
        "RMSE_linear": float(np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))),
        "MAE_linear": float(mean_absolute_error(y_test_lin, y_pred_lin)),
        "R2_linear": float(r2_score(y_test_lin, y_pred_lin)),
        "mean_bias_linear": float(np.mean(y_pred_lin - y_test_lin)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    metrics["parity_positive_only"] = sahp_parity_plot_log(
        y_true_lin=y_test_lin,
        y_pred_lin=y_pred_lin,
        output_path=output_dir / f"{experiment_name}_parity.png",
        title=f"{experiment_name}: Observed vs Predicted dust AOD500",
        xlabel="Observed dust AOD (500 nm)",
        ylabel="Predicted dust AOD (500 nm)",
    )

    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    (
        importance.rename_axis("feature")
        .reset_index(name="importance")
        .to_csv(output_dir / f"{experiment_name}_feature_importance.csv", index=False)
    )

    test_predictions = X_test.copy()
    test_predictions["y_true_dust_AOD500"] = y_test_lin
    test_predictions["y_pred_dust_AOD500"] = y_pred_lin
    test_predictions.to_csv(output_dir / f"{experiment_name}_test_predictions.csv", index=False)

    model_path = output_dir / f"{experiment_name}_xgb.json"
    model.save_model(model_path)
    write_json(output_dir / f"{experiment_name}_metrics.json", metrics)
    write_json(
        output_dir / f"{experiment_name}_metadata.json",
        {
            **metadata,
            "feature_columns": feature_columns,
            "training_rows": int(len(df)),
            "experiment": experiment_name,
        },
    )
    return {"experiment": experiment_name, "metrics": metrics, "model_path": str(model_path)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined_arrays(args.mod04, args.myd04)
    dust_lr_by_wl, dust_lr_stats_df = compute_dust_lidar_ratio_stats(
        combined=combined,
        dust_dominant_threshold=args.dust_dominant_threshold,
        lr_stat=args.lr_stat,
    )
    dust_lr_stats_df.to_csv(args.output_dir / "dust_dominant_lidar_ratio_stats.csv", index=False)

    df = build_training_dataframe(
        combined=combined,
        dust_lr_by_wavelength=dust_lr_by_wl,
        target_method=args.target_method,
    )
    df.to_csv(args.output_dir / "aeronet_dust_aod500_training_dataframe.csv", index=False)
    metadata = {
        "mod04_path": str(args.mod04),
        "myd04_path": str(args.myd04),
        "dust_lidar_ratio_by_wavelength": dust_lr_by_wl,
        "target_method": args.target_method,
        "lr_stat": args.lr_stat,
        "dust_dominant_threshold": args.dust_dominant_threshold,
    }

    experiments: list[tuple[str, list[str]]] = []
    base_cols = base_feature_columns()
    if args.feature_mode in {"compare", "without_dpr_lr"}:
        experiments.append(("without_dpr_lr", base_cols))
    if args.feature_mode in {"compare", "with_dpr_lr"}:
        experiments.append(("with_dpr_lr", base_cols + dpr_lr_feature_columns()))

    results = []
    for experiment_name, feature_columns in experiments:
        results.append(
            train_one_experiment(
                df=df,
                feature_columns=feature_columns,
                experiment_name=experiment_name,
                output_dir=args.output_dir,
                metadata=metadata,
            )
        )

    comparison_df = pd.DataFrame([result["metrics"] for result in results])[
        ["experiment", "feature_count", "RMSE_linear", "MAE_linear", "R2_linear", "mean_bias_linear"]
    ].sort_values("RMSE_linear")
    comparison_df.to_csv(args.output_dir / "aeronet_dust_aod500_model_comparison.csv", index=False)
    write_json(
        args.output_dir / "aeronet_dust_aod500_comparison_summary.json",
        {
            **metadata,
            "training_rows": int(len(df)),
            "comparison": comparison_df.to_dict(orient="records"),
            "best_experiment": str(comparison_df.iloc[0]["experiment"]),
        },
    )

    print(f"Training dataframe rows: {len(df)}")
    for _, row in comparison_df.iterrows():
        print(
            f"{row['experiment']}: RMSE={row['RMSE_linear']:.4f}, "
            f"MAE={row['MAE_linear']:.4f}, "
            f"R2={row['R2_linear']:.4f}, "
            f"bias={row['mean_bias_linear']:.4f}"
        )
    print(f"Target method: {args.target_method}")
    print(f"Comparison: {args.output_dir / 'aeronet_dust_aod500_model_comparison.csv'}")


if __name__ == "__main__":
    main()
