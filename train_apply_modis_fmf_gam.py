#!/usr/bin/env python3
"""Train a MODIS-compatible GAM for AERONET FMF and apply it to a test granule."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.genmod.families import Gaussian

from apply_dust_model_to_modis_db import (
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    save_comparison_panel,
    save_scatter_map,
)
from extract_modis_db_dust_model import extract_arrays_from_collocation_dict


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parent = here.parent
    parser = argparse.ArgumentParser(
        description="Train logit(FMF) GAM from collocated MODIS DB / AERONET data and apply it to a granule."
    )
    parser.add_argument(
        "--mod04",
        type=Path,
        default=parent / "AERONET_MOD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the MOD04 collocation pickle.",
    )
    parser.add_argument(
        "--myd04",
        type=Path,
        default=parent / "AERONET_MYD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the MYD04 collocation pickle.",
    )
    parser.add_argument(
        "--granule",
        type=Path,
        default=here
        / "dust_model_application_2025-03-14"
        / "MOD04_L2.A2025073.1705.061.2025074013638.hdf",
        help="Local MODIS DB granule to apply the trained GAM to.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "fmf_gam_results",
        help="Directory for model outputs and case-study figures.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation/test splitting.",
    )
    return parser.parse_args()


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


def load_combined_arrays(mod04_path: Path, myd04_path: Path) -> dict[str, np.ndarray]:
    with mod04_path.open("rb") as handle:
        mod04 = pickle.load(handle)
    with myd04_path.open("rb") as handle:
        myd04 = pickle.load(handle)

    arr_mod = extract_arrays_from_collocation_dict(mod04)
    arr_myd = extract_arrays_from_collocation_dict(myd04)
    return {key: np.concatenate([arr_mod[key], arr_myd[key]]) for key in arr_mod}


def build_fmf_training_dataframe(combined: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "AERONET_SDA_FMF550": combined["AERONET_SDA_FMF550"],
            "MODIS_DB_AOD550": combined["MODIS_DB_AOD550"],
            "MODIS_DB_AOD412": combined["MODIS_DB_AOD412"],
            "MODIS_DB_AOD470": combined["MODIS_DB_AOD470"],
            "MODIS_DB_AOD660": combined["MODIS_DB_AOD660"],
            "MODIS_DB_AE": combined["MODIS_DB_AE"],
            "MODIS_DB_SSA412": combined["MODIS_DB_SSA412"],
            "MODIS_DB_SSA470": combined["MODIS_DB_SSA470"],
            "MODIS_DB_SSA660": combined["MODIS_DB_SSA660"],
        }
    )
    df["MODIS_DB_AOD412_over_470"] = df["MODIS_DB_AOD412"] / df["MODIS_DB_AOD470"]
    df["MODIS_DB_AOD550_over_660"] = df["MODIS_DB_AOD550"] / df["MODIS_DB_AOD660"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["AERONET_SDA_FMF550"] >= 0.0) & (df["AERONET_SDA_FMF550"] <= 1.0)]
    for col in [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "MODIS_DB_AOD412_over_470",
        "MODIS_DB_AOD550_over_660",
    ]:
        df = df[df[col] > 0.0]
    for col in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        df[col] = df[col].clip(0.0, 1.0)
    df = df[df["MODIS_DB_AE"] >= 0.0]
    return df.reset_index(drop=True)


def feature_names() -> list[str]:
    return [
        "MODIS_DB_AOD550",
        "MODIS_DB_AOD412",
        "MODIS_DB_AOD470",
        "MODIS_DB_AOD660",
        "MODIS_DB_AE",
        "MODIS_DB_SSA412",
        "MODIS_DB_SSA470",
        "MODIS_DB_SSA660",
        "MODIS_DB_AOD412_over_470",
        "MODIS_DB_AOD550_over_660",
    ]


def clip_to_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float | int | str]:
    if len(y_true) < 3:
        return {"label": label, "N": int(len(y_true))}
    out = metrics_dict(y_true, y_pred)
    out["label"] = label
    out["N"] = int(len(y_true))
    return out


def save_generic_scatter_comparison(
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> dict[str, float]:
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise ValueError("Not enough paired finite samples for scatter comparison.")

    diff = y - x
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r = float(np.corrcoef(x, y)[0, 1])

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.03 * (hi - lo) if hi > lo else 0.1
    lo -= pad
    hi += pad

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    hb = ax.hexbin(x, y, gridsize=60, mincnt=1, cmap="viridis")
    plt.colorbar(hb, ax=ax, label="count")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.98,
        f"N = {x.size}\n"
        f"r = {r:.3f}\n"
        f"Bias (y-x) = {bias:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"RMSE = {rmse:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"N": int(x.size), "Pearson_r": r, "bias": bias, "MAE": mae, "RMSE": rmse}


def prepare_basis(df: pd.DataFrame, cols: list[str]) -> BSplines:
    return BSplines(df[cols].to_numpy(dtype=float), df=[6] * len(cols), degree=[3] * len(cols))


def fit_best_gam(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
) -> tuple[object, dict[str, tuple[float, float]], float, dict[str, dict[str, float]]]:
    bounds = {
        col: (float(train_df[col].min()), float(train_df[col].max()))
        for col in cols
    }
    train_df_c = clip_to_bounds(train_df, bounds)
    val_df_c = clip_to_bounds(val_df, bounds)

    y_train = logit_transform(train_df_c["AERONET_SDA_FMF550"].to_numpy(dtype=float))
    y_val_lin = val_df_c["AERONET_SDA_FMF550"].to_numpy(dtype=float)

    alpha_grid = [0.01, 0.1, 0.5, 1.0, 5.0]
    basis = prepare_basis(train_df_c, cols)
    fit_log = {}
    best = None
    best_alpha = None
    best_rmse = np.inf

    for alpha in alpha_grid:
        model = GLMGam(
            endog=y_train,
            exog=np.ones((len(train_df_c), 1)),
            smoother=basis,
            alpha=np.full(len(cols), alpha, dtype=float),
            family=Gaussian(),
        )
        result = model.fit()
        val_pred = inv_logit_transform(
            result.predict(
                exog=np.ones((len(val_df_c), 1)),
                exog_smooth=val_df_c[cols].to_numpy(dtype=float),
            )
        )
        fit_log[str(alpha)] = metrics_dict(y_val_lin, val_pred)
        if fit_log[str(alpha)]["RMSE"] < best_rmse:
            best = result
            best_alpha = alpha
            best_rmse = fit_log[str(alpha)]["RMSE"]

    return best, bounds, float(best_alpha), fit_log


def refit_gam(
    trainval_df: pd.DataFrame,
    cols: list[str],
    alpha: float,
) -> tuple[object, dict[str, tuple[float, float]]]:
    bounds = {
        col: (float(trainval_df[col].min()), float(trainval_df[col].max()))
        for col in cols
    }
    trainval_df_c = clip_to_bounds(trainval_df, bounds)
    y_trainval = logit_transform(trainval_df_c["AERONET_SDA_FMF550"].to_numpy(dtype=float))
    basis = prepare_basis(trainval_df_c, cols)
    model = GLMGam(
        endog=y_trainval,
        exog=np.ones((len(trainval_df_c), 1)),
        smoother=basis,
        alpha=np.full(len(cols), alpha, dtype=float),
        family=Gaussian(),
    )
    result = model.fit()
    return result, bounds


def predict_fmf_gam(result: object, X_df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return inv_logit_transform(
        result.predict(
            exog=np.ones((len(X_df), 1)),
            exog_smooth=X_df[cols].to_numpy(dtype=float),
        )
    )


def li_ginoux_fmean(ae: np.ndarray) -> np.ndarray:
    ae = np.asarray(ae, dtype=float)
    return np.clip(0.085 * ae**2 + 0.336 * ae + 0.051, 0.0, 1.0)


def build_inference_feature_frame(features_2d: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "MODIS_DB_AOD550": np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float).ravel(),
            "MODIS_DB_AOD412": np.asarray(features_2d["MODIS_DB_AOD412"], dtype=float).ravel(),
            "MODIS_DB_AOD470": np.asarray(features_2d["MODIS_DB_AOD470"], dtype=float).ravel(),
            "MODIS_DB_AOD660": np.asarray(features_2d["MODIS_DB_AOD660"], dtype=float).ravel(),
            "MODIS_DB_AE": np.asarray(features_2d["MODIS_DB_AE"], dtype=float).ravel(),
            "MODIS_DB_SSA412": np.asarray(features_2d["MODIS_DB_SSA412"], dtype=float).ravel(),
            "MODIS_DB_SSA470": np.asarray(features_2d["MODIS_DB_SSA470"], dtype=float).ravel(),
            "MODIS_DB_SSA660": np.asarray(features_2d["MODIS_DB_SSA660"], dtype=float).ravel(),
        }
    )
    frame["MODIS_DB_AOD412_over_470"] = frame["MODIS_DB_AOD412"] / frame["MODIS_DB_AOD470"]
    frame["MODIS_DB_AOD550_over_660"] = frame["MODIS_DB_AOD550"] / frame["MODIS_DB_AOD660"]
    return frame


def apply_gam_to_granule(
    result: object,
    bounds: dict[str, tuple[float, float]],
    granule_path: Path,
    output_dir: Path,
    prefix: str,
) -> dict[str, object]:
    features_2d, lat2d, lon2d = read_granule_features(granule_path)
    cols = feature_names()
    h, w = np.asarray(features_2d["MODIS_DB_AOD550"]).shape
    frame = build_inference_feature_frame(features_2d)
    valid = np.isfinite(frame[cols]).all(axis=1)
    pred_flat = np.full(len(frame), np.nan, dtype=float)
    if valid.any():
        X_valid = clip_to_bounds(frame.loc[valid, cols], bounds)
        pred_flat[valid] = predict_fmf_gam(result, X_valid, cols)
    pred_fmf = pred_flat.reshape(h, w)

    ssa_constraint = (
        np.isfinite(features_2d["MODIS_DB_SSA412"])
        & np.isfinite(features_2d["MODIS_DB_SSA470"])
        & (np.asarray(features_2d["MODIS_DB_SSA412"], dtype=float) < np.asarray(features_2d["MODIS_DB_SSA470"], dtype=float))
    )
    coarse_fraction = 1.0 - pred_fmf
    gam_dust_aod = coarse_fraction * np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
    gam_dust_aod = np.where(ssa_constraint, gam_dust_aod, np.nan)
    pred_fmf_ssa = np.where(ssa_constraint, pred_fmf, np.nan)

    li_fmf, _, li_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)

    fmf_png = output_dir / f"{granule_path.stem}_{prefix}_fmf_ssa_constraint.png"
    dust_png = output_dir / f"{granule_path.stem}_{prefix}_dust_aod_ssa_constraint.png"
    scatter_png = output_dir / f"{granule_path.stem}_{prefix}_vs_li_ginoux_dust_aod_scatter.png"
    panel_png = output_dir / f"{granule_path.stem}_{prefix}_vs_li_ginoux_panel.png"

    save_scatter_map(
        lat2d,
        lon2d,
        pred_fmf_ssa,
        fmf_png,
        title=f"{granule_path.name}: {prefix} FMF with SSA constraint",
        cbar_label=f"{prefix} FMF",
        is_fraction=True,
    )
    save_scatter_map(
        lat2d,
        lon2d,
        gam_dust_aod,
        dust_png,
        title=f"{granule_path.name}: {prefix} dust AOD proxy with SSA constraint",
        cbar_label=f"{prefix} dust AOD proxy at 550 nm",
        is_fraction=False,
    )
    scatter_stats = save_generic_scatter_comparison(
        x_data=gam_dust_aod,
        y_data=li_dust_aod,
        output_path=scatter_png,
        title=f"{granule_path.name}: {prefix} vs Li-Ginoux dust AOD",
        xlabel=f"{prefix} dust AOD proxy",
        ylabel="Li-Ginoux dust AOD proxy",
    )
    save_comparison_panel(
        lat2d=lat2d,
        lon2d=lon2d,
        panels=[
            {"data": features_2d["MODIS_DB_AOD550"], "title": "MODIS DB AOD550", "label": "AOD550", "cmap": "plasma"},
            {"data": features_2d["MODIS_DB_AE"], "title": "MODIS DB AE", "label": "AE", "cmap": "viridis"},
            {"data": gam_dust_aod, "title": f"{prefix} Dust AOD", "label": f"{prefix} dust AOD proxy", "cmap": "magma"},
            {"data": li_dust_aod, "title": "Li-Ginoux Dust AOD", "label": "Li-Ginoux dust AOD proxy", "cmap": "magma"},
        ],
        output_path=panel_png,
        figure_title=f"{granule_path.name}: {prefix} FMF vs Li-Ginoux dust AOD comparison",
    )
    return {
        "fmf_png": str(fmf_png),
        "dust_aod_png": str(dust_png),
        "scatter_png": str(scatter_png),
        "panel_png": str(panel_png),
        "scatter_stats": scatter_stats,
        "valid_pixels_all_features": int(valid.sum()),
        "valid_pixels_ssa_constraint": int(np.isfinite(gam_dust_aod).sum()),
        "median_fmf_ssa_constraint": float(np.nanmedian(pred_fmf_ssa)),
        "median_dust_aod_ssa_constraint": float(np.nanmedian(gam_dust_aod)),
        "median_li_ginoux_fmf_ssa_constraint": float(np.nanmedian(li_fmf)),
        "median_li_ginoux_dust_aod_ssa_constraint": float(np.nanmedian(li_dust_aod)),
    }


def run_experiment(
    experiment_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_test_df: pd.DataFrame,
    cols: list[str],
    granule_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    best_result, _, best_alpha, val_log = fit_best_gam(train_df, val_df, cols)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_result, bounds = refit_gam(trainval_df, cols, best_alpha)

    test_clipped = clip_to_bounds(test_df[cols], bounds)
    y_test = test_df["AERONET_SDA_FMF550"].to_numpy(dtype=float)
    y_pred = predict_fmf_gam(final_result, test_clipped, cols)
    gam_metrics = metrics_dict(y_test, y_pred)
    li_metrics = metrics_dict(y_test, li_ginoux_fmean(test_df["MODIS_DB_AE"].to_numpy(dtype=float)))

    diagnostics = []
    full_eval_clipped = clip_to_bounds(full_test_df[cols], bounds)
    full_y_true = full_test_df["AERONET_SDA_FMF550"].to_numpy(dtype=float)
    full_y_pred = predict_fmf_gam(final_result, full_eval_clipped, cols)
    full_li_pred = li_ginoux_fmean(full_test_df["MODIS_DB_AE"].to_numpy(dtype=float))
    subsets = {
        "all_test_rows": np.ones(len(full_test_df), dtype=bool),
        "fmf_lt_0p7": full_test_df["AERONET_SDA_FMF550"].to_numpy(dtype=float) < 0.7,
        "fmf_lt_0p5": full_test_df["AERONET_SDA_FMF550"].to_numpy(dtype=float) < 0.5,
        "fmf_lt_0p3": full_test_df["AERONET_SDA_FMF550"].to_numpy(dtype=float) < 0.3,
        "ae_lt_0p8": full_test_df["MODIS_DB_AE"].to_numpy(dtype=float) < 0.8,
        "ae_lt_0p5": full_test_df["MODIS_DB_AE"].to_numpy(dtype=float) < 0.5,
    }
    for label, mask in subsets.items():
        mask = np.asarray(mask, dtype=bool)
        diagnostics.append(
            {
                "subset": label,
                "gam": subset_metrics(full_y_true[mask], full_y_pred[mask], label),
                "li_ginoux": subset_metrics(full_y_true[mask], full_li_pred[mask], label),
                "truth_median_fmf": float(np.nanmedian(full_y_true[mask])) if np.any(mask) else np.nan,
                "gam_median_fmf": float(np.nanmedian(full_y_pred[mask])) if np.any(mask) else np.nan,
                "li_ginoux_median_fmf": float(np.nanmedian(full_li_pred[mask])) if np.any(mask) else np.nan,
            }
        )

    model_bundle_path = output_dir / f"{experiment_name}_modis_fmf_gam.pkl"
    with model_bundle_path.open("wb") as handle:
        pickle.dump(
            {
                "result": final_result,
                "feature_names": cols,
                "bounds": bounds,
                "best_alpha": best_alpha,
                "target": "AERONET_SDA_FMF550",
                "transform": "logit",
                "experiment_name": experiment_name,
            },
            handle,
        )

    case_dir = output_dir / "case_2025-03-14"
    case_dir.mkdir(exist_ok=True)
    case_summary = apply_gam_to_granule(final_result, bounds, granule_path, case_dir, prefix=experiment_name)
    return {
        "name": experiment_name,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_alpha": best_alpha,
        "alpha_grid_validation": val_log,
        "gam_test_metrics": gam_metrics,
        "li_ginoux_test_metrics": li_metrics,
        "diagnostics": diagnostics,
        "model_bundle": str(model_bundle_path),
        "case_application": case_summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined_arrays(args.mod04, args.myd04)
    df = build_fmf_training_dataframe(combined)
    cols = feature_names()

    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=args.random_state)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=args.random_state)

    truncated_df = df[df["AERONET_SDA_FMF550"] < 0.7].reset_index(drop=True)
    train_df_t, tmp_df_t = train_test_split(truncated_df, test_size=0.30, random_state=args.random_state)
    val_df_t, test_df_t = train_test_split(tmp_df_t, test_size=0.50, random_state=args.random_state)

    full_range = run_experiment(
        experiment_name="gam_full_range",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        full_test_df=test_df,
        cols=cols,
        granule_path=args.granule,
        output_dir=args.output_dir,
    )
    truncated = run_experiment(
        experiment_name="gam_fmf_lt_0p7",
        train_df=train_df_t,
        val_df=val_df_t,
        test_df=test_df_t,
        full_test_df=test_df,
        cols=cols,
        granule_path=args.granule,
        output_dir=args.output_dir,
    )

    summary = {
        "training_rows": int(len(df)),
        "feature_names": cols,
        "full_range_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "truncated_rows": {
            "train": int(len(train_df_t)),
            "val": int(len(val_df_t)),
            "test": int(len(test_df_t)),
            "total_available": int(len(truncated_df)),
        },
        "experiments": {
            "gam_full_range": full_range,
            "gam_fmf_lt_0p7": truncated,
        },
        "evaluation_note": "Truncated FMF<0.7 model is evaluated both on its own truncated holdout and diagnostically on the full-range test set subsets.",
    }
    summary_path = args.output_dir / "modis_fmf_gam_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Training rows: {len(df)}")
    for key in ["gam_full_range", "gam_fmf_lt_0p7"]:
        exp = summary["experiments"][key]
        gm = exp["gam_test_metrics"]
        lm = exp["li_ginoux_test_metrics"]
        print(
            f"{key} GAM test metrics: RMSE={gm['RMSE']:.4f}, "
            f"MAE={gm['MAE']:.4f}, R2={gm['R2']:.4f}, bias={gm['bias']:.4f}"
        )
        print(
            f"{key} Li-Ginoux baseline on same test: RMSE={lm['RMSE']:.4f}, "
            f"MAE={lm['MAE']:.4f}, R2={lm['R2']:.4f}, bias={lm['bias']:.4f}"
        )
        print(f"{key} model bundle: {exp['model_bundle']}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved case outputs in: {args.output_dir / 'case_2025-03-14'}")


if __name__ == "__main__":
    main()
