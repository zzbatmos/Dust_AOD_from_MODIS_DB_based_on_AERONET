#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier, XGBRegressor


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
        description="Build QA-tiered annual two-stage DAOD maps and compare them with Li-Ginoux and direct XGBoost."
    )
    parser.add_argument("--two-stage-dir", type=Path, required=True)
    parser.add_argument("--li-annual", type=Path, required=True)
    parser.add_argument("--xgb-annual", type=Path, required=True)
    parser.add_argument("--xgb-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("two_stage_tuning/best_model"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--short-name", default="MOD08_D3")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def inv_logit_transform(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def nanmean_no_warning(stack: np.ndarray, axis: int) -> np.ndarray:
    valid_count = np.sum(np.isfinite(stack), axis=axis)
    total = np.nansum(stack, axis=axis)
    out = np.full_like(total, np.nan, dtype=float)
    np.divide(total, valid_count, out=out, where=valid_count > 0)
    return out


def load_models(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    with (model_dir / "two_stage_metadata.json").open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(model_dir / meta["regressor_model"])
    return classifier, regressor, meta


def files_for_year(directory: Path, prefix: str, year: int) -> list[Path]:
    return sorted(directory.glob(f"{prefix}_{year:04d}-*.nc"))


def compute_raw_two_stage(ds: xr.Dataset, classifier: XGBClassifier, regressor: XGBRegressor, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_map = {
        "MODIS_DB_AOD550": np.asarray(ds["db_aod_550"].values, dtype=float),
        "MODIS_DB_AOD412": np.asarray(ds["db_aod_412"].values, dtype=float),
        "MODIS_DB_AOD470": np.asarray(ds["db_aod_470"].values, dtype=float),
        "MODIS_DB_AOD660": np.asarray(ds["db_aod_660"].values, dtype=float),
        "MODIS_DB_AE": np.asarray(ds["angstrom_exponent"].values, dtype=float),
        "MODIS_DB_SSA412": np.asarray(ds["ssa_412"].values, dtype=float),
        "MODIS_DB_SSA470": np.asarray(ds["ssa_470"].values, dtype=float),
        "MODIS_DB_SSA660": np.asarray(ds["ssa_660"].values, dtype=float),
    }
    h, w = feature_map["MODIS_DB_AOD550"].shape
    x_flat = np.column_stack([feature_map[col].ravel() for col in FEATURE_COLUMNS])
    valid = np.isfinite(x_flat).all(axis=1)

    prob_flat = np.full(h * w, np.nan, dtype=np.float32)
    frac_flat = np.full(h * w, np.nan, dtype=np.float32)
    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=FEATURE_COLUMNS)
        prob = classifier.predict_proba(x_valid)[:, 1]
        frac = inv_logit_transform(regressor.predict(x_valid))
        prob_flat[valid] = prob.astype(np.float32)
        frac_flat[valid] = frac.astype(np.float32)

    return (
        feature_map["MODIS_DB_AOD550"].astype(np.float32),
        prob_flat.reshape(h, w),
        frac_flat.reshape(h, w),
    )


def qa_class_from_probability(prob: np.ndarray, total_aod: np.ndarray, valid_input: np.ndarray) -> np.ndarray:
    qa = np.full(prob.shape, -1, dtype=np.int8)
    total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
    qa[total_valid] = 0

    valid = total_valid & valid_input & np.isfinite(prob)
    qa[valid & (prob >= 0.4) & (total_aod >= 0.03)] = 1
    qa[valid & (prob >= 0.6) & (total_aod >= 0.08)] = 2
    qa[valid & (prob >= 0.7) & (total_aod >= 0.15)] = 3
    return qa


def annual_mean_for_threshold(
    qa_stack: np.ndarray,
    total_aod_stack: np.ndarray,
    raw_fraction_stack: np.ndarray,
    threshold: int,
) -> np.ndarray:
    total_valid = np.isfinite(total_aod_stack) & (total_aod_stack > 0.0)
    dust_aod = raw_fraction_stack * total_aod_stack
    # Missing total AOD stays NaN. Valid retrievals below QA threshold are forced to zero.
    dust_aod = np.where(total_valid, dust_aod, np.nan)
    dust_aod = np.where(total_valid & (qa_stack < threshold), 0.0, dust_aod)
    return nanmean_no_warning(dust_aod, axis=0)


def add_geo(ax) -> None:
    ax.set_global()
    ax.coastlines(linewidth=0.7)
    ax.gridlines(
        draw_labels=False,
        xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-90, 91, 30),
        linewidth=0.35,
        color="0.5",
        alpha=0.45,
        linestyle="--",
    )


def plot_map(ax, lon: np.ndarray, lat: np.ndarray, data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    mesh = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    add_geo(ax)
    ax.set_title(title)
    return mesh


def save_annual_netcdf(path: Path, data: np.ndarray, lat: np.ndarray, lon: np.ndarray, attrs: dict) -> None:
    xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), data.astype("float32")),
            lat=(("y", "x"), lat.astype("float32")),
            lon=(("y", "x"), lon.astype("float32")),
        ),
        attrs=attrs,
    ).to_netcdf(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classifier, regressor, meta = load_models(args.model_dir)

    files = sorted(args.two_stage_dir.glob(f"{args.short_name}_DustAOD_TwoStage_{args.year:04d}-*.nc"))
    if not files:
        raise FileNotFoundError(f"No two-stage daily files found for {args.year} in {args.two_stage_dir}")
    xgb_daily_files = files_for_year(args.xgb_dir, f"{args.short_name}_DustAOD_XGB", args.year)
    if len(xgb_daily_files) != len(files):
        raise ValueError(
            f"Expected matching counts between two-stage and XGBoost daily files for {args.year}, "
            f"got {len(files)} and {len(xgb_daily_files)}."
        )

    total_aod_stack = []
    raw_fraction_stack = []
    qa_stack = []
    xgb_daily_stack = []
    lat = lon = None
    monthly_rows = []
    annual_qa_freq = {1: [], 2: [], 3: []}

    for path, xgb_path in zip(files, xgb_daily_files, strict=True):
        ds = xr.open_dataset(path)
        xgb_ds = xr.open_dataset(xgb_path)
        if lat is None:
            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
        total_aod, prob, raw_fraction = compute_raw_two_stage(ds, classifier, regressor, meta)
        valid_input = np.asarray(ds["model_valid_input"].values, dtype=bool)
        qa = qa_class_from_probability(prob, total_aod, valid_input)
        xgb_daily = np.asarray(xgb_ds["dust_aod"].values, dtype=np.float32)

        total_aod_stack.append(total_aod.astype(np.float32))
        raw_fraction_stack.append(raw_fraction.astype(np.float32))
        qa_stack.append(qa.astype(np.int8))
        xgb_daily_stack.append(xgb_daily)

        month = int(path.stem.split("_")[-1].split("-")[1])
        total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
        day_record = {"month": month, "qa1_mean": float(np.nanmean(xgb_daily))}
        annual_qa_freq[1].append(np.where(total_valid & np.isfinite(xgb_daily), 1.0, np.where(total_valid, 0.0, np.nan)))
        for threshold in (2, 3):
            dust_aod = raw_fraction * total_aod
            dust_aod = np.where(total_valid, dust_aod, np.nan)
            dust_aod = np.where(total_valid & (qa < threshold), 0.0, dust_aod)
            day_record[f"qa{threshold}_mean"] = float(np.nanmean(dust_aod))
            annual_qa_freq[threshold].append(np.where(total_valid, (qa >= threshold).astype(float), np.nan))
        monthly_rows.append(day_record)
        ds.close()
        xgb_ds.close()

    total_aod_stack = np.stack(total_aod_stack, axis=0)
    raw_fraction_stack = np.stack(raw_fraction_stack, axis=0)
    qa_stack = np.stack(qa_stack, axis=0)
    xgb_daily_stack = np.stack(xgb_daily_stack, axis=0)

    qa_annual = {}
    qa_freq_annual = {}
    summary_rows = []
    for threshold in (1, 2, 3):
        if threshold == 1:
            annual = nanmean_no_warning(xgb_daily_stack, axis=0)
        else:
            annual = annual_mean_for_threshold(qa_stack, total_aod_stack, raw_fraction_stack, threshold)
        qa_annual[threshold] = annual
        qa_freq_annual[threshold] = nanmean_no_warning(np.stack(annual_qa_freq[threshold], axis=0), axis=0)
        save_annual_netcdf(
            args.output_dir / f"{args.short_name}_DustAOD_TwoStage_QA{threshold}_annual_mean_{args.year}.nc",
            annual,
            lat,
            lon,
            {"year": args.year, "method": "TwoStage", "qa_threshold": threshold, "daily_files": len(files)},
        )
        summary_rows.append(
            {
                "method": f"Two-stage QA>={threshold}",
                "annual_global_mean": float(np.nanmean(annual)),
                "annual_detection_frequency_mean": float(np.nanmean(qa_freq_annual[threshold])),
            }
        )

    li_ds = xr.open_dataset(args.li_annual)
    xgb_ds = xr.open_dataset(args.xgb_annual)
    li = np.asarray(li_ds["dust_aod"].values, dtype=float)
    xgb = np.asarray(xgb_ds["dust_aod"].values, dtype=float)
    li_ds.close()
    xgb_ds.close()

    monthly = pd.DataFrame(monthly_rows)
    monthly_summary = monthly.groupby("month", as_index=False).mean(numeric_only=True)
    monthly_summary.to_csv(args.output_dir / f"{args.short_name}_TwoStage_QA_monthly_global_mean_{args.year}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / f"{args.short_name}_TwoStage_QA_annual_summary_{args.year}.csv", index=False)

    combined = np.concatenate(
        [
            li[np.isfinite(li)],
            xgb[np.isfinite(xgb)],
            qa_annual[1][np.isfinite(qa_annual[1])],
            qa_annual[2][np.isfinite(qa_annual[2])],
            qa_annual[3][np.isfinite(qa_annual[3])],
        ]
    )
    vmax = float(np.nanpercentile(combined, 99))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    for row, threshold in enumerate((1, 2, 3)):
        mesh = plot_map(axes[row, 0], lon, lat, qa_annual[threshold], f"Two-stage QA>={threshold}", "YlOrBr", 0.0, vmax)
        diff_li = qa_annual[threshold] - li
        diff_xgb = qa_annual[threshold] - xgb
        dv = float(
            np.nanpercentile(
                np.concatenate(
                    [
                        np.abs(diff_li[np.isfinite(diff_li)]),
                        np.abs(diff_xgb[np.isfinite(diff_xgb)]),
                    ]
                ),
                99,
            )
        )
        plot_map(axes[row, 1], lon, lat, diff_li, f"QA>={threshold} minus Li-Ginoux", "RdBu_r", -dv, dv)
        plot_map(axes[row, 2], lon, lat, diff_xgb, f"QA>={threshold} minus direct XGBoost", "RdBu_r", -dv, dv)
    cbar1 = fig.colorbar(mesh, ax=axes[:, 0], location="right", pad=0.02, shrink=0.72, fraction=0.03)
    cbar1.set_label("Annual mean dust AOD at 550 nm")
    fig.savefig(args.output_dir / f"{args.short_name}_TwoStage_QA_compare_maps_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9.5),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    mesh = plot_map(axes[0, 0], lon, lat, li, f"Li-Ginoux {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[0, 1], lon, lat, xgb, f"Direct XGBoost {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[0, 2], lon, lat, qa_annual[1], f"Two-stage QA>=1 {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[1, 0], lon, lat, qa_annual[2], f"Two-stage QA>=2 {args.year}", "YlOrBr", 0.0, vmax)
    plot_map(axes[1, 1], lon, lat, qa_annual[3], f"Two-stage QA>=3 {args.year}", "YlOrBr", 0.0, vmax)
    axes[1, 2].remove()
    cbar = fig.colorbar(mesh, ax=[axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]], location="right", pad=0.02, shrink=0.74, fraction=0.03)
    cbar.set_label("Annual mean dust AOD at 550 nm")
    fig.savefig(args.output_dir / f"{args.short_name}_TwoStage_QA_climatology_maps_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 9.0),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    diff_fields = [
        (qa_annual[2] - li, "QA>=2 minus Li-Ginoux"),
        (qa_annual[2] - xgb, "QA>=2 minus direct XGBoost"),
        (qa_annual[3] - li, "QA>=3 minus Li-Ginoux"),
        (qa_annual[3] - xgb, "QA>=3 minus direct XGBoost"),
    ]
    diff_abs = np.concatenate([np.abs(field[np.isfinite(field)]) for field, _ in diff_fields])
    diff_v = float(np.nanpercentile(diff_abs, 99))
    mesh = None
    for ax, (field, title) in zip(axes.flat, diff_fields, strict=False):
        mesh = plot_map(ax, lon, lat, field, f"{title} {args.year}", "RdBu_r", -diff_v, diff_v)
    cbar = fig.colorbar(mesh, ax=axes, location="right", pad=0.02, shrink=0.74, fraction=0.03)
    cbar.set_label("Annual mean DAOD difference")
    fig.savefig(args.output_dir / f"{args.short_name}_TwoStage_QA_difference_maps_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    mesh = None
    for ax, threshold in zip(axes, (1, 2, 3), strict=False):
        mesh = plot_map(
            ax,
            lon,
            lat,
            qa_freq_annual[threshold],
            f"Fraction of days with QA>={threshold}",
            "viridis",
            0.0,
            1.0,
        )
    cbar = fig.colorbar(mesh, ax=axes, location="right", pad=0.02, shrink=0.72, fraction=0.03)
    cbar.set_label("Detection frequency")
    fig.savefig(args.output_dir / f"{args.short_name}_TwoStage_QA_detection_frequency_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.plot(monthly_summary["month"], monthly_summary["qa1_mean"], marker="o", linewidth=2.0, label="Two-stage QA>=1")
    ax.plot(monthly_summary["month"], monthly_summary["qa2_mean"], marker="o", linewidth=2.0, label="Two-stage QA>=2")
    ax.plot(monthly_summary["month"], monthly_summary["qa3_mean"], marker="o", linewidth=2.0, label="Two-stage QA>=3")
    ax.set_xticks(range(1, 13))
    ax.set_ylabel("Global mean dust AOD at 550 nm")
    ax.set_title(f"{args.short_name} two-stage QA-filtered monthly global mean dust AOD in {args.year}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / f"{args.short_name}_TwoStage_QA_monthly_global_mean_{args.year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
