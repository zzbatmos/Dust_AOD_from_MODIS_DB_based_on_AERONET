#!/usr/bin/env python3
"""Compare MODIS DB aerosol-type QA flag to the tuned two-stage dust detector."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from pyhdf.SD import SD, SDC
from xgboost import XGBClassifier, XGBRegressor

from apply_dust_model_to_modis_db import read_granule_features
from apply_two_stage_dust_model_to_modis_db import predict_two_stage


TYPE_LABELS = {0: "Mixed", 1: "Dust", 2: "Smoke", 3: "Sulfate"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=here / "two_stage_tuning" / "best_model",
    )
    parser.add_argument(
        "--dust-file",
        type=Path,
        default=here / "dust_model_application_2025-03-14" / "MOD04_L2.A2025073.1705.061.2025074013638.hdf",
    )
    parser.add_argument(
        "--smoke-file",
        type=Path,
        default=here / "dust_model_application_2024-09-22_amazon" / "MOD04_L2.A2024266.1345.061.2024268021127.hdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "db_flag_vs_two_stage_comparison",
    )
    return parser.parse_args()


def decode_aerosol_type(granule_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = SD(str(granule_path), SDC.READ)
    qa = np.asarray(f.select("Quality_Assurance_Land").get(), dtype=np.int16)
    f.end()
    byte4 = qa[:, :, 4].astype(np.uint8)
    usefulness = (byte4 & 0b1) == 1
    confidence = ((byte4 >> 1) & 0b11).astype(np.uint8)
    aerosol_type = ((byte4 >> 3) & 0b11).astype(np.uint8)
    return usefulness, confidence, aerosol_type


def load_two_stage_model(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    with (model_dir / "two_stage_metadata.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(model_dir / meta["regressor_model"])
    return classifier, regressor, meta


def compare_masks(
    granule_path: Path,
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    meta: dict,
) -> dict[str, object]:
    features, lat2d, lon2d = read_granule_features(granule_path)
    probability, dust_fraction, model_valid = predict_two_stage(classifier, regressor, meta, features)
    usefulness, confidence, aerosol_type = decode_aerosol_type(granule_path)

    intersection = model_valid & usefulness
    qa_dust = intersection & (aerosol_type == 1)
    qa_smoke = intersection & (aerosol_type == 2)
    two_stage_dust = intersection & (dust_fraction > 0.0)

    agree_dust = qa_dust & two_stage_dust
    agree_nondust = intersection & (~qa_dust) & (~two_stage_dust)
    qa_only_dust = qa_dust & (~two_stage_dust)
    ml_only_dust = two_stage_dust & (~qa_dust)

    summary = {
        "granule": granule_path.name,
        "intersection_pixels": int(intersection.sum()),
        "qa_dust_pixels": int(qa_dust.sum()),
        "qa_smoke_pixels": int(qa_smoke.sum()),
        "two_stage_dust_pixels": int(two_stage_dust.sum()),
        "qa_dust_fraction_of_intersection": float(qa_dust.sum() / intersection.sum()) if intersection.any() else np.nan,
        "qa_smoke_fraction_of_intersection": float(qa_smoke.sum() / intersection.sum()) if intersection.any() else np.nan,
        "two_stage_dust_fraction_of_intersection": float(two_stage_dust.sum() / intersection.sum()) if intersection.any() else np.nan,
        "agree_dust_pixels": int(agree_dust.sum()),
        "agree_nondust_pixels": int(agree_nondust.sum()),
        "qa_only_dust_pixels": int(qa_only_dust.sum()),
        "ml_only_dust_pixels": int(ml_only_dust.sum()),
        "dust_recall_if_qa_is_reference": float(agree_dust.sum() / qa_dust.sum()) if qa_dust.any() else np.nan,
        "dust_precision_if_qa_is_reference": float(agree_dust.sum() / two_stage_dust.sum()) if two_stage_dust.any() else np.nan,
        "mean_probability_on_qa_dust": float(np.nanmean(probability[qa_dust])) if qa_dust.any() else np.nan,
        "mean_probability_on_qa_smoke": float(np.nanmean(probability[qa_smoke])) if qa_smoke.any() else np.nan,
        "mean_probability_on_intersection": float(np.nanmean(probability[intersection])) if intersection.any() else np.nan,
        "mean_qa_confidence_on_qa_dust": float(confidence[qa_dust].mean()) if qa_dust.any() else np.nan,
        "mean_qa_confidence_on_qa_smoke": float(confidence[qa_smoke].mean()) if qa_smoke.any() else np.nan,
    }
    return {
        "lat": lat2d,
        "lon": lon2d,
        "probability": probability,
        "dust_fraction": dust_fraction,
        "intersection": intersection,
        "qa_dust": qa_dust,
        "qa_smoke": qa_smoke,
        "two_stage_dust": two_stage_dust,
        "agreement": np.where(~intersection, np.nan, np.where(agree_dust, 3, np.where(qa_only_dust, 1, np.where(ml_only_dust, 2, 0)))).astype(float),
        "summary": summary,
    }


def save_case_figure(case_name: str, result: dict[str, object], output_path: Path) -> None:
    lat = np.asarray(result["lat"], dtype=float)
    lon = ((np.asarray(result["lon"], dtype=float) + 180.0) % 360.0) - 180.0
    qa_mask = np.where(result["intersection"], np.where(result["qa_dust"], 1.0, 0.0), np.nan)
    ml_mask = np.where(result["intersection"], np.where(result["two_stage_dust"], 1.0, 0.0), np.nan)
    agreement = np.asarray(result["agreement"], dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6.3),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    for ax in axes:
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle=":")

    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(agreement)
    if np.any(finite):
        extent = [
            float(np.nanmin(lon[finite])) - 0.8,
            float(np.nanmax(lon[finite])) + 0.8,
            float(np.nanmin(lat[finite])) - 0.8,
            float(np.nanmax(lat[finite])) + 0.8,
        ]
        for ax in axes:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    binary_cmap = ListedColormap(["#d9d9d9", "#c28f0e"])
    binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)
    mesh1 = axes[0].pcolormesh(lon, lat, np.ma.masked_invalid(qa_mask), transform=ccrs.PlateCarree(), shading="nearest", cmap=binary_cmap, norm=binary_norm)
    mesh2 = axes[1].pcolormesh(lon, lat, np.ma.masked_invalid(ml_mask), transform=ccrs.PlateCarree(), shading="nearest", cmap=binary_cmap, norm=binary_norm)

    agreement_cmap = ListedColormap(["#d9d9d9", "#f4a261", "#4c78a8", "#2a9d8f"])
    agreement_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], agreement_cmap.N)
    mesh3 = axes[2].pcolormesh(lon, lat, np.ma.masked_invalid(agreement), transform=ccrs.PlateCarree(), shading="nearest", cmap=agreement_cmap, norm=agreement_norm)

    axes[0].set_title(f"{case_name}: QA flag dust mask")
    axes[1].set_title(f"{case_name}: two-stage dust mask")
    axes[2].set_title(f"{case_name}: disagreement map")

    cb1 = fig.colorbar(mesh1, ax=axes[:2], location="right", pad=0.02, shrink=0.82, ticks=[0, 1])
    cb1.ax.set_yticklabels(["non-dust", "dust"])
    cb1.set_label("Dust mask")

    cb2 = fig.colorbar(mesh3, ax=axes[2], location="right", pad=0.02, shrink=0.82, ticks=[0, 1, 2, 3])
    cb2.ax.set_yticklabels(["agree non-dust", "QA only dust", "ML only dust", "agree dust"])
    cb2.set_label("Mask agreement")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier, regressor, meta = load_two_stage_model(args.model_dir)

    dust = compare_masks(args.dust_file, classifier, regressor, meta)
    smoke = compare_masks(args.smoke_file, classifier, regressor, meta)

    save_case_figure("Heavy dust 2025-03-14", dust, args.output_dir / f"{args.dust_file.stem}_qa_flag_vs_two_stage.png")
    save_case_figure("Heavy smoke 2024-09-22", smoke, args.output_dir / f"{args.smoke_file.stem}_qa_flag_vs_two_stage.png")

    rows = [dust["summary"], smoke["summary"]]
    csv_path = args.output_dir / "qa_flag_vs_two_stage_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "qa_flag_vs_two_stage_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
