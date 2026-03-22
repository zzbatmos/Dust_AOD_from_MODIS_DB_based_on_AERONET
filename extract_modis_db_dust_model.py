#!/usr/bin/env python3
"""Train the notebook's MODIS Deep Blue dust-fraction XGBoost model from CLI.

This script is extracted from the later analysis/training section of
`AERONET_DPR_analysis.ipynb` (execution counts 123-157). It:

1. Loads MOD04 and MYD04 collocation dictionaries.
2. Extracts the AERONET and MODIS Deep Blue arrays used in the notebook.
3. Derives AERONET dust fraction from DPR and lidar ratio.
4. Trains the final log-target XGBoost model on MODIS Deep Blue features.
5. Saves the trained model and metadata to disk.

By default, the input pickle files are resolved from the parent directory
because the notebook was copied into the current workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# CLI and runtime configuration
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parent = here.parent

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the standalone MODIS Deep Blue XGBoost training section "
            "from AERONET_DPR_analysis.ipynb."
        )
    )
    parser.add_argument(
        "--mod04",
        type=Path,
        default=parent / "AERONET_MOD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the Terra MOD04 collocation pickle.",
    )
    parser.add_argument(
        "--myd04",
        type=Path,
        default=parent / "AERONET_MYD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the Aqua MYD04 collocation pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Directory for saved model, metadata, plots, and tables.",
    )
    parser.add_argument(
        "--lr-dust-675",
        type=float,
        default=56.0,
        help=(
            "Dust lidar ratio used to convert AERONET DPR to dust fraction at "
            "675 nm. The notebook sourced this from earlier derived statistics."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation/test splits.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Save parity/diagnostic plots reproduced from the notebook section.",
    )
    parser.add_argument(
        "--run-shap",
        action="store_true",
        help="Generate SHAP summary plots if the shap package is available.",
    )
    return parser.parse_args()


def extract_arrays_from_collocation_dict(colloc_dic: dict) -> dict[str, np.ndarray]:
    out = {
        "AERONET_AOD440": [],
        "AERONET_DPR440": [],
        "AERONET_LR440": [],
        "AERONET_SSA440": [],
        "AERONET_EXT_TOTAL440": [],
        "AERONET_EXT_COARSE440": [],
        "AERONET_AOD675": [],
        "AERONET_DPR675": [],
        "AERONET_LR675": [],
        "AERONET_SSA675": [],
        "AERONET_EXT_TOTAL675": [],
        "AERONET_EXT_COARSE675": [],
        "AERONET_AOD1020": [],
        "AERONET_DPR1020": [],
        "AERONET_LR1020": [],
        "AERONET_SSA1020": [],
        "AERONET_EXT_TOTAL1020": [],
        "AERONET_EXT_COARSE1020": [],
        "AERONET_SDA_AOD550": [],
        "AERONET_SDA_AE550": [],
        "AERONET_SDA_FMF550": [],
        "MODIS_DB_AOD550": [],
        "MODIS_DB_AOD412": [],
        "MODIS_DB_AOD470": [],
        "MODIS_DB_AOD660": [],
        "MODIS_DB_AE": [],
        "MODIS_DB_SSA412": [],
        "MODIS_DB_SSA470": [],
        "MODIS_DB_SSA660": [],
    }

    for site_name in colloc_dic:
        for granule in colloc_dic[site_name]:
            granule_data = colloc_dic[site_name][granule]
            db_data = granule_data.get("Deep_Blue_data")
            if not db_data or "aod550" not in db_data:
                continue

            out["AERONET_AOD440"].append(granule_data["AOD_Coincident_Input[440nm]"])
            out["AERONET_DPR440"].append(granule_data["Depolarization_Ratio[440nm]"])
            out["AERONET_LR440"].append(granule_data["Lidar_Ratio[440nm]"])
            out["AERONET_SSA440"].append(granule_data["Single_Scattering_Albedo[440nm]"])
            out["AERONET_EXT_TOTAL440"].append(granule_data["AOD_Extinction-Total[440nm]"])
            out["AERONET_EXT_COARSE440"].append(
                granule_data["AOD_Extinction-Coarse[440nm]"]
            )

            out["AERONET_AOD675"].append(granule_data["AOD_Coincident_Input[675nm]"])
            out["AERONET_DPR675"].append(granule_data["Depolarization_Ratio[675nm]"])
            out["AERONET_LR675"].append(granule_data["Lidar_Ratio[675nm]"])
            out["AERONET_SSA675"].append(granule_data["Single_Scattering_Albedo[675nm]"])
            out["AERONET_EXT_TOTAL675"].append(granule_data["AOD_Extinction-Total[675nm]"])
            out["AERONET_EXT_COARSE675"].append(
                granule_data["AOD_Extinction-Coarse[675nm]"]
            )

            out["AERONET_AOD1020"].append(granule_data["AOD_Coincident_Input[1020nm]"])
            out["AERONET_DPR1020"].append(granule_data["Depolarization_Ratio[1020nm]"])
            out["AERONET_LR1020"].append(granule_data["Lidar_Ratio[1020nm]"])
            out["AERONET_SSA1020"].append(
                granule_data["Single_Scattering_Albedo[1020nm]"]
            )
            out["AERONET_EXT_TOTAL1020"].append(
                granule_data["AOD_Extinction-Total[1020nm]"]
            )
            out["AERONET_EXT_COARSE1020"].append(
                granule_data["AOD_Extinction-Coarse[1020nm]"]
            )

            meta = granule_data.get("nearest_AERONET_SDA_L1.5_data_meta", {})
            if meta.get("found", False):
                sda = granule_data["nearest_AERONET_SDA_L1.5_data"]
                out["AERONET_SDA_AOD550"].append(
                    sda.get("Total_AOD_500nm[tau_a]", np.nan)
                )
                out["AERONET_SDA_AE550"].append(
                    sda.get("Angstrom_Exponent(AE)-Total_500nm[alpha]", np.nan)
                )
                out["AERONET_SDA_FMF550"].append(
                    sda.get("FineModeFraction_500nm[eta]", np.nan)
                )
            else:
                out["AERONET_SDA_AOD550"].append(np.nan)
                out["AERONET_SDA_AE550"].append(np.nan)
                out["AERONET_SDA_FMF550"].append(np.nan)

            stats_full = db_data["stats_full"]
            out["MODIS_DB_AOD550"].append(db_data["aod550"]["mean"])
            out["MODIS_DB_AOD412"].append(
                stats_full["Deep_Blue_Spectral_Aerosol_Optical_Depth_Land"]["mean"][0]
            )
            out["MODIS_DB_AOD470"].append(
                stats_full["Deep_Blue_Spectral_Aerosol_Optical_Depth_Land"]["mean"][1]
            )
            out["MODIS_DB_AOD660"].append(
                stats_full["Deep_Blue_Spectral_Aerosol_Optical_Depth_Land"]["mean"][2]
            )
            out["MODIS_DB_SSA412"].append(
                stats_full["Deep_Blue_Spectral_Single_Scattering_Albedo_Land"]["mean"][0]
            )
            out["MODIS_DB_SSA470"].append(
                stats_full["Deep_Blue_Spectral_Single_Scattering_Albedo_Land"]["mean"][1]
            )
            out["MODIS_DB_SSA660"].append(
                stats_full["Deep_Blue_Spectral_Single_Scattering_Albedo_Land"]["mean"][2]
            )
            out["MODIS_DB_AE"].append(
                stats_full["Deep_Blue_Angstrom_Exponent_Land"]["mean"]
            )

    for key in out:
        out[key] = np.asarray(out[key], dtype=float)
    return out


# AERONET dust-fraction physics helpers
def dust_fraction_from_pldr(
    pldr: np.ndarray, pldr_nd: float = 0.02, pldr_d: float = 0.30
) -> np.ndarray:
    pldr = np.asarray(pldr, dtype=float)
    rd = np.full_like(pldr, np.nan, dtype=float)
    valid = np.isfinite(pldr)

    denom = pldr_d - pldr_nd
    rd[valid] = ((pldr[valid] - pldr_nd) / denom) * (
        (1.0 + pldr_d) / (1.0 + pldr[valid])
    )
    rd = np.where(pldr < pldr_nd, 0.0, rd)
    rd = np.where(pldr > pldr_d, 1.0, rd)
    return np.clip(rd, 0.0, 1.0)


def compute_aeronet_dust_fraction(
    aeronet_dpr: np.ndarray, aeronet_lr: np.ndarray, lr_dust: float
) -> tuple[np.ndarray, np.ndarray]:
    pldr = np.asarray(aeronet_dpr, dtype=float)
    lr = np.asarray(aeronet_lr, dtype=float)
    rd = dust_fraction_from_pldr(pldr)

    dust_fraction = np.full_like(rd, np.nan, dtype=float)
    valid = np.isfinite(rd) & np.isfinite(lr) & (lr > 0)
    dust_fraction[valid] = rd[valid] * float(lr_dust) / lr[valid]
    return np.clip(dust_fraction, 0.0, 1.0), rd


def log1p_transform(y: np.ndarray, eps: float = 0.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)
    return np.log1p(y + eps)


def inv_log1p_transform(y_log: np.ndarray, eps: float = 0.0) -> np.ndarray:
    y_log = np.asarray(y_log, dtype=float)
    y_lin = np.expm1(y_log) - eps
    return np.clip(y_lin, 0.0, None)


def logit_transform(y: np.ndarray, eps: float = 1.0e-4) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def inv_logit_transform(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def compute_sample_weights_by_bins(
    y_lin: np.ndarray, bin_edges: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, object]]:
    if bin_edges is None:
        bin_edges = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=float)

    y_lin = np.asarray(y_lin, dtype=float)
    bin_ids = np.digitize(y_lin, bin_edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=len(bin_edges) - 1).astype(float)
    inv_counts = np.zeros_like(counts)
    nonzero = counts > 0
    inv_counts[nonzero] = 1.0 / counts[nonzero]
    sample_weights = inv_counts[bin_ids]
    sample_weights *= len(sample_weights) / sample_weights.sum()

    summary = {
        "bin_edges": bin_edges.tolist(),
        "bin_counts": counts.astype(int).tolist(),
        "bin_weights": inv_counts.tolist(),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "sample_weight_mean": float(sample_weights.mean()),
    }
    return sample_weights, summary


# Diagnostic plotting helpers
def sahp_parity_plot_log(
    y_true_lin: np.ndarray,
    y_pred_lin: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    lo: float = 1.0e-4,
    hi: float | None = None,
    gridsize: int = 70,
    mincnt: int = 1,
) -> dict[str, float]:
    y_true = np.asarray(y_true_lin, dtype=float)
    y_pred = np.asarray(y_pred_lin, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        raise ValueError("No valid positive samples to plot on log axes.")

    if hi is None:
        hi = float(np.nanmax([yt.max(), yp.max()]) * 1.05)

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))
    metrics = {"N": int(yt.size), "bias": bias, "MAE": mae, "RMSE": rmse, "R2": r2}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    text = (
        f"N = {metrics['N']}\n"
        f"Bias (pred-obs) = {bias:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"R2 = {r2:.4f}"
    )
    xx = np.logspace(np.log10(lo), np.log10(hi), 200)

    axes[0].scatter(yt, yp, s=10, alpha=0.35, linewidths=0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} (scatter)")
    axes[0].plot(xx, xx, linestyle="--", linewidth=1)
    axes[0].text(
        0.02,
        0.98,
        text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    hb = axes[1].hexbin(
        yt, yp, gridsize=gridsize, mincnt=mincnt, xscale="log", yscale="log"
    )
    plt.colorbar(hb, ax=axes[1], label="count")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"{title} (density)")
    axes[1].plot(xx, xx, linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics


# Input loading and dataframe assembly
def combine_arrays(mod04_path: Path, myd04_path: Path) -> dict[str, np.ndarray]:
    with mod04_path.open("rb") as handle:
        dic_mod04 = pickle.load(handle)
    with myd04_path.open("rb") as handle:
        dic_myd04 = pickle.load(handle)

    arr_mod04 = extract_arrays_from_collocation_dict(dic_mod04)
    arr_myd04 = extract_arrays_from_collocation_dict(dic_myd04)

    combined = {}
    for key in arr_mod04:
        combined[key] = np.concatenate([arr_mod04[key], arr_myd04[key]])
    return combined


def build_training_dataframe(combined: dict[str, np.ndarray], lr_dust_675: float) -> pd.DataFrame:
    aeronet_cmf440 = combined["AERONET_EXT_COARSE440"] / combined["AERONET_EXT_TOTAL440"]
    aeronet_cm_aod440 = combined["AERONET_AOD440"] * aeronet_cmf440
    aeronet_cmf675 = combined["AERONET_EXT_COARSE675"] / combined["AERONET_EXT_TOTAL675"]
    aeronet_cm_aod675 = combined["AERONET_AOD675"] * aeronet_cmf675
    aeronet_cmf1020 = (
        combined["AERONET_EXT_COARSE1020"] / combined["AERONET_EXT_TOTAL1020"]
    )
    aeronet_cm_aod1020 = combined["AERONET_AOD1020"] * aeronet_cmf1020

    dust_fraction_675, _ = compute_aeronet_dust_fraction(
        combined["AERONET_DPR675"],
        combined["AERONET_LR675"],
        lr_dust=lr_dust_675,
    )

    arrays = [
        dust_fraction_675,
        combined["MODIS_DB_AOD550"],
        combined["MODIS_DB_AOD412"],
        combined["MODIS_DB_AOD470"],
        combined["MODIS_DB_AOD660"],
        combined["MODIS_DB_AE"],
        combined["MODIS_DB_SSA412"],
        combined["MODIS_DB_SSA470"],
        combined["MODIS_DB_SSA660"],
    ]
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Arrays do not have the same length: {lengths}")

    df = pd.DataFrame(
        {
            "dust_fraction_675": dust_fraction_675,
            "MODIS_DB_AOD550": combined["MODIS_DB_AOD550"],
            "MODIS_DB_AOD412": combined["MODIS_DB_AOD412"],
            "MODIS_DB_AOD470": combined["MODIS_DB_AOD470"],
            "MODIS_DB_AOD660": combined["MODIS_DB_AOD660"],
            "MODIS_DB_AE": combined["MODIS_DB_AE"],
            "MODIS_DB_SSA412": combined["MODIS_DB_SSA412"],
            "MODIS_DB_SSA470": combined["MODIS_DB_SSA470"],
            "MODIS_DB_SSA660": combined["MODIS_DB_SSA660"],
            "AERONET_CM_AOD440": aeronet_cm_aod440,
            "AERONET_CM_AOD675": aeronet_cm_aod675,
            "AERONET_CM_AOD1020": aeronet_cm_aod1020,
        }
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["dust_fraction_675"] >= 0.0]
    for column in ["MODIS_DB_AOD550", "MODIS_DB_AOD412", "MODIS_DB_AOD470", "MODIS_DB_AOD660"]:
        df = df[df[column] > 0]
    for column in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        df[column] = df[column].clip(0.0, 1.0)

    return df


# Model training and explainability
def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    y_lin = df["dust_fraction_675"].astype(float).values
    X = df[
        [
            "MODIS_DB_AOD550",
            "MODIS_DB_AOD412",
            "MODIS_DB_AOD470",
            "MODIS_DB_AOD660",
            "MODIS_DB_AE",
            "MODIS_DB_SSA412",
            "MODIS_DB_SSA470",
            "MODIS_DB_SSA660",
        ]
    ].astype(float)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y_lin)
    X = X.loc[mask]
    y_lin = y_lin[mask]
    return X, y_lin


def train_model_with_transform(
    X: pd.DataFrame,
    y_lin: np.ndarray,
    random_state: int,
    transform_name: str,
    sample_weights: np.ndarray | None = None,
    weighting_name: str = "none",
    weighting_summary: dict[str, object] | None = None,
) -> tuple[XGBRegressor, dict[str, object], pd.Series]:
    if transform_name == "log1p":
        transform = lambda y: log1p_transform(y, eps=0.0)
        inverse_transform = inv_log1p_transform
        transform_meta = {"target_transform": "log1p", "eps": 0.0}
    elif transform_name == "logit":
        transform = lambda y: logit_transform(y, eps=1.0e-4)
        inverse_transform = inv_logit_transform
        transform_meta = {"target_transform": "logit", "eps": 1.0e-4}
    else:
        raise ValueError(f"Unsupported transform: {transform_name}")

    y_trans = transform(y_lin)
    if sample_weights is None:
        sample_weights = np.ones_like(y_lin, dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape != y_lin.shape:
            raise ValueError("sample_weights must have the same shape as y_lin")

    (
        X_train,
        X_tmp,
        y_train_trans,
        y_tmp_trans,
        y_train_lin,
        y_tmp_lin,
        w_train,
        w_tmp,
    ) = train_test_split(
        X,
        y_trans,
        y_lin,
        sample_weights,
        test_size=0.30,
        random_state=random_state,
    )
    (
        X_val,
        X_test,
        y_val_trans,
        y_test_trans,
        y_val_lin,
        y_test_lin,
        w_val,
        w_test,
    ) = train_test_split(
        X_tmp,
        y_tmp_trans,
        y_tmp_lin,
        w_tmp,
        test_size=0.50,
        random_state=random_state,
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
        random_state=random_state,
    )
    fit_kwargs = {
        "eval_set": [(X_val, y_val_trans)],
        "verbose": 200,
    }
    if weighting_name != "none":
        fit_kwargs["sample_weight"] = w_train
    model.fit(X_train, y_train_trans, **fit_kwargs)

    y_pred_trans = model.predict(X_test)
    y_pred_lin = inverse_transform(y_pred_trans)

    metrics = {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "RMSE_linear": float(np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))),
        "MAE_linear": float(mean_absolute_error(y_test_lin, y_pred_lin)),
        "R2_linear": float(r2_score(y_test_lin, y_pred_lin)),
        "mean_bias_linear": float(np.mean(y_pred_lin - y_test_lin)),
        "feature_names": X.columns.tolist(),
        "target_transform": transform_meta["target_transform"],
        "eps": transform_meta["eps"],
        "weighting_name": weighting_name,
        "weighting_summary": weighting_summary,
        "X_test": X_test,
        "y_test_lin": y_test_lin,
        "y_pred_lin": y_pred_lin,
        "test_sample_weights": w_test,
    }
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    return model, metrics, importance


def maybe_run_shap(
    model: XGBRegressor, X_test: pd.DataFrame, output_dir: Path
) -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP plotting requested but the 'shap' package is missing.") from exc

    nsamp = min(5000, len(X_test))
    X_explain = (
        X_test.sample(n=nsamp, random_state=42) if len(X_test) > nsamp else X_test.copy()
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_explain, plot_type="dot", show=False)
    plt.title("XGBoost SHAP Summary (dust_fraction_675 prediction)")
    plt.tight_layout()
    plt.savefig(output_dir / "dust_fraction_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.title("Global feature importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(output_dir / "dust_fraction_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


# End-to-end pipeline entry point
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_arrays(args.mod04, args.myd04)
    df = build_training_dataframe(combined, lr_dust_675=args.lr_dust_675)
    df.to_csv(args.output_dir / "dust_fraction_training_data.csv", index=False)
    X, y_lin = prepare_features_and_target(df)
    weighted_sample_weights, weighted_summary = compute_sample_weights_by_bins(y_lin)

    results = {}
    experiments = [
        {
            "name": "log1p",
            "transform_name": "log1p",
            "sample_weights": None,
            "weighting_name": "none",
            "weighting_summary": None,
        },
        {
            "name": "logit",
            "transform_name": "logit",
            "sample_weights": None,
            "weighting_name": "none",
            "weighting_summary": None,
        },
        {
            "name": "log1p_weighted_bins",
            "transform_name": "log1p",
            "sample_weights": weighted_sample_weights,
            "weighting_name": "inverse_frequency_bins",
            "weighting_summary": weighted_summary,
        },
    ]
    for experiment in experiments:
        model, train_info, importance = train_model_with_transform(
            X,
            y_lin,
            random_state=args.random_state,
            transform_name=experiment["transform_name"],
            sample_weights=experiment["sample_weights"],
            weighting_name=experiment["weighting_name"],
            weighting_summary=experiment["weighting_summary"],
        )

        model_stem = f"dust_aod_xgb_{experiment['name']}"
        model_path = args.output_dir / f"{model_stem}.json"
        meta_path = args.output_dir / f"{model_stem}_meta.json"
        metrics_path = args.output_dir / f"{model_stem}_metrics.json"
        importance_path = args.output_dir / f"{model_stem}_feature_importance.csv"

        model.save_model(model_path)
        meta = {
            "feature_names": train_info["feature_names"],
            "target_transform": train_info["target_transform"],
            "eps": train_info["eps"],
            "weighting_name": train_info["weighting_name"],
            "weighting_summary": train_info["weighting_summary"],
            "target_name": "dust_fraction_675",
            "lr_dust_675": args.lr_dust_675,
            "input_pickles": {
                "mod04": str(args.mod04.resolve()),
                "myd04": str(args.myd04.resolve()),
            },
        }
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2)

        metrics = {
            "final_sample_size": int(len(df)),
            "train_samples": train_info["train_samples"],
            "val_samples": train_info["val_samples"],
            "test_samples": train_info["test_samples"],
            "RMSE_linear": train_info["RMSE_linear"],
            "MAE_linear": train_info["MAE_linear"],
            "R2_linear": train_info["R2_linear"],
            "mean_bias_linear": train_info["mean_bias_linear"],
            "target_transform": train_info["target_transform"],
            "eps": train_info["eps"],
            "weighting_name": train_info["weighting_name"],
            "weighting_summary": train_info["weighting_summary"],
        }

        if args.make_plots:
            parity_metrics = sahp_parity_plot_log(
                y_true_lin=train_info["y_test_lin"],
                y_pred_lin=train_info["y_pred_lin"],
                output_path=args.output_dir / f"{model_stem}_parity_log.png",
                title=f"Dust Fraction: Observed vs Predicted ({experiment['name']})",
                xlabel="Dust fraction from AERONET",
                ylabel="Predicted dust fraction from MODIS DB",
                lo=1.0e-4,
            )
            metrics["parity_plot"] = parity_metrics

        if args.run_shap:
            shap_dir = args.output_dir / f"shap_{experiment['name']}"
            shap_dir.mkdir(exist_ok=True)
            maybe_run_shap(model, train_info["X_test"], shap_dir)

        with metrics_path.open("w") as handle:
            json.dump(metrics, handle, indent=2)
        importance.rename("importance").to_csv(importance_path, header=True)

        results[experiment["name"]] = {
            "metrics": metrics,
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "metrics_path": str(metrics_path),
            "importance_path": str(importance_path),
        }

    comparison_df = pd.DataFrame(
        [
            {
                "transform": name,
                "RMSE_linear": result["metrics"]["RMSE_linear"],
                "MAE_linear": result["metrics"]["MAE_linear"],
                "R2_linear": result["metrics"]["R2_linear"],
                "mean_bias_linear": result["metrics"]["mean_bias_linear"],
                "target_transform": result["metrics"]["target_transform"],
                "weighting_name": result["metrics"]["weighting_name"],
            }
            for name, result in results.items()
        ]
    ).sort_values(by=["RMSE_linear", "MAE_linear"], ascending=[True, True])
    comparison_path = args.output_dir / "dust_aod_xgb_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_transform = comparison_df.iloc[0]["transform"]
    best_result = results[best_transform]
    legacy_model_path = args.output_dir / "dust_aod_xgb.json"
    legacy_meta_path = args.output_dir / "dust_aod_xgb_meta.json"
    legacy_metrics_path = args.output_dir / "dust_aod_xgb_metrics.json"
    legacy_importance_path = args.output_dir / "dust_aod_xgb_feature_importance.csv"

    shutil.copyfile(args.output_dir / f"dust_aod_xgb_{best_transform}.json", legacy_model_path)
    shutil.copyfile(
        args.output_dir / f"dust_aod_xgb_{best_transform}_meta.json", legacy_meta_path
    )
    shutil.copyfile(
        args.output_dir / f"dust_aod_xgb_{best_transform}_metrics.json", legacy_metrics_path
    )
    shutil.copyfile(
        args.output_dir / f"dust_aod_xgb_{best_transform}_feature_importance.csv",
        legacy_importance_path,
    )

    print(f"Training dataframe rows: {len(df)}")
    for experiment_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{experiment_name}: RMSE={metrics['RMSE_linear']:.4f}, "
            f"MAE={metrics['MAE_linear']:.4f}, "
            f"R2={metrics['R2_linear']:.4f}, "
            f"bias={metrics['mean_bias_linear']:.4f}"
        )
    print(f"Best transform by RMSE/MAE: {best_transform}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Legacy outputs now point to: {best_transform}")
    print(f"Primary model: {best_result['model_path']}")


if __name__ == "__main__":
    main()
