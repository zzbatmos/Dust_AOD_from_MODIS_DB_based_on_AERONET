from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from extract_modis_db_dust_model import (
    combine_arrays,
    compute_aeronet_dust_fraction,
)
from train_aeronet_dust_aod500_xgb import (
    compute_dust_lidar_ratio_stats,
    loglog_interpolate_target_500,
)
from aeronet_dpr_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to predict AERONET-derived dust AOD at 550 nm from collocated MODIS Deep Blue features."
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
        help="How to interpolate AERONET-derived dust AOD to 550 nm from 440/675/1020 nm.",
    )
    parser.add_argument(
        "--lr-stat",
        choices=["mean", "median"],
        default="mean",
        help="Dust-dominant lidar-ratio statistic used in the DPR/LR dust-AOD derivation.",
    )
    parser.add_argument("--dust-dominant-threshold", type=float, default=0.89)
    parser.add_argument("--output-dir", type=Path, default=Path("modis_db_dust_aod550_xgb"))
    return parser.parse_args()


def log1p_transform(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))


def inv_log1p_transform(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(values, dtype=float)), 0.0, None)


def derive_dust_aod_550(
    combined: dict[str, np.ndarray],
    dust_lr_by_wavelength: dict[int, float],
    target_method: str,
) -> np.ndarray:
    dust_fraction_440, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR440"],
        combined["AERONET_LR440"],
        lr_dust=dust_lr_by_wavelength[440],
    )
    dust_fraction_675, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR675"],
        combined["AERONET_LR675"],
        lr_dust=dust_lr_by_wavelength[675],
    )
    dust_fraction_1020, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR1020"],
        combined["AERONET_LR1020"],
        lr_dust=dust_lr_by_wavelength[1020],
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

    # 500 and 550 nm are close, but keep the wavelength change explicit through AE.
    # Estimate dust AE from 440/675 and then step from 500 to 550.
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


def build_training_dataframe(
    combined: dict[str, np.ndarray],
    dust_lr_by_wavelength: dict[int, float],
    target_method: str,
) -> pd.DataFrame:
    dust_aod_550 = derive_dust_aod_550(combined, dust_lr_by_wavelength, target_method)
    df = pd.DataFrame(
        {
            "dust_AOD_550": dust_aod_550,
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["dust_AOD_550"] >= 0.0]
    for column in ["MODIS_DB_AOD550", "MODIS_DB_AOD412", "MODIS_DB_AOD470", "MODIS_DB_AOD660"]:
        df = df[df[column] > 0.0]
    for column in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        df[column] = df[column].clip(0.0, 1.0)
    df["AERONET_SDA_FMF550"] = df["AERONET_SDA_FMF550"].clip(0.0, 1.0)
    return df


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
    if yt.size == 0:
        raise ValueError("No valid positive samples for parity plot.")
    hi = float(np.nanmax([yt.max(), yp.max()]) * 1.05)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))
    n = int(yt.size)
    xx = np.logspace(np.log10(lo), np.log10(hi), 200)
    txt = f"N = {n}\nBias = {bias:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR2 = {r2:.4f}"

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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"N": n, "bias": bias, "MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_arrays(args.mod04, args.myd04)
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
    df.to_csv(args.output_dir / "modis_db_dust_aod550_training_dataframe.csv", index=False)

    feature_columns = [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "MODIS_DB_AE",
        "MODIS_DB_SSA412",
        "MODIS_DB_SSA470",
        "MODIS_DB_SSA660",
    ]
    X = df[feature_columns].astype(float)
    y_lin = df["dust_AOD_550"].astype(float).to_numpy()
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
    model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=200)

    y_pred_log = model.predict(X_test)
    y_pred_lin = inv_log1p_transform(y_pred_log)
    metrics = {
        "target_wavelength_nm": 550,
        "target_method": args.target_method,
        "target_transform": "log1p",
        "lr_stat": args.lr_stat,
        "dust_dominant_threshold": args.dust_dominant_threshold,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "RMSE_linear": float(np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))),
        "MAE_linear": float(mean_absolute_error(y_test_lin, y_pred_lin)),
        "R2_linear": float(r2_score(y_test_lin, y_pred_lin)),
        "mean_bias_linear": float(np.mean(y_pred_lin - y_test_lin)),
        "feature_names": feature_columns,
    }
    metrics["parity_positive_only"] = sahp_parity_plot_log(
        y_true_lin=y_test_lin,
        y_pred_lin=y_pred_lin,
        output_path=args.output_dir / "modis_db_dust_aod550_parity.png",
        title="MODIS DB dust AOD550: Observed vs Predicted",
        xlabel="Observed AERONET-derived dust AOD (550 nm)",
        ylabel="Predicted dust AOD (550 nm)",
    )

    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    importance.rename_axis("feature").reset_index(name="importance").to_csv(
        args.output_dir / "modis_db_dust_aod550_feature_importance.csv",
        index=False,
    )

    test_predictions = X_test.copy()
    test_predictions["y_true_dust_AOD550"] = y_test_lin
    test_predictions["y_pred_dust_AOD550"] = y_pred_lin
    test_predictions.to_csv(args.output_dir / "modis_db_dust_aod550_test_predictions.csv", index=False)

    model_path = args.output_dir / "modis_db_dust_aod550_xgb.json"
    model.save_model(model_path)
    write_json(args.output_dir / "modis_db_dust_aod550_metrics.json", metrics)
    write_json(
        args.output_dir / "modis_db_dust_aod550_metadata.json",
        {
            "mod04_path": str(args.mod04),
            "myd04_path": str(args.myd04),
            "dust_lidar_ratio_by_wavelength": dust_lr_by_wl,
            "target_method": args.target_method,
            "target_wavelength_nm": 550,
            "feature_columns": feature_columns,
            "training_rows": int(len(df)),
        },
    )

    print(f"Training dataframe rows: {len(df)}")
    print(
        f"RMSE={metrics['RMSE_linear']:.4f}, "
        f"MAE={metrics['MAE_linear']:.4f}, "
        f"R2={metrics['R2_linear']:.4f}, "
        f"bias={metrics['mean_bias_linear']:.4f}"
    )
    print(f"Target method: {args.target_method}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
