#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from apply_dust_model_to_modis_db import (
    clip_fraction_grid,
    enforce_daod_output_semantics,
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    save_aod_scatter_comparison,
    save_comparison_panel,
    save_reference_feature_maps,
    save_swath_map,
    summarize_grid,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Apply the two-stage dust model to a MODIS L2 granule.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=here / "modis_db_two_stage_xgb",
    )
    return parser.parse_args()


def inv_logit_transform(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def predict_two_stage(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    meta: dict,
    features_2d: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_order = meta["feature_names"]
    h, w = np.asarray(next(iter(features_2d.values()))).shape
    x_flat = np.column_stack([np.asarray(features_2d[col], dtype=float).ravel() for col in feature_order])
    valid = np.isfinite(x_flat).all(axis=1)

    prob_flat = np.full(h * w, np.nan, dtype=np.float32)
    frac_flat = np.full(h * w, np.nan, dtype=np.float32)
    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=feature_order)
        prob = classifier.predict_proba(x_valid)[:, 1]
        reg = inv_logit_transform(regressor.predict(x_valid))
        prob_flat[valid] = prob.astype(np.float32)
        frac = np.where(prob >= float(meta["probability_threshold"]), reg, 0.0)
        frac_flat[valid] = frac.astype(np.float32)

    return prob_flat.reshape(h, w), clip_fraction_grid(frac_flat.reshape(h, w)), valid.reshape(h, w)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.model_dir / "two_stage_metadata.json").open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(args.model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(args.model_dir / meta["regressor_model"])

    granule_path = args.input_file.resolve()
    features_2d, lat2d, lon2d = read_granule_features(granule_path)
    reference_rows = save_reference_feature_maps(
        granule_path=granule_path,
        output_dir=args.output_dir,
        lat2d=lat2d,
        lon2d=lon2d,
        features_2d=features_2d,
    )

    dust_probability, dust_fraction, model_valid = predict_two_stage(
        classifier,
        regressor,
        meta,
        features_2d,
    )
    dust_aod = dust_fraction * np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
    dust_aod = enforce_daod_output_semantics(dust_aod, features_2d["MODIS_DB_AOD550"])

    lg_fmf, lg_coarse_fraction, lg_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)

    prob_png = args.output_dir / f"{granule_path.stem}_two_stage_dust_probability.png"
    frac_png = args.output_dir / f"{granule_path.stem}_two_stage_dust_fraction.png"
    daod_png = args.output_dir / f"{granule_path.stem}_two_stage_dust_aod_550.png"
    lg_png = args.output_dir / f"{granule_path.stem}_li_ginoux_dust_aod_ssa_constraint.png"
    panel_png = args.output_dir / f"{granule_path.stem}_two_stage_vs_li_ginoux_panel.png"
    diff_png = args.output_dir / f"{granule_path.stem}_two_stage_minus_li_ginoux.png"
    scatter_png = args.output_dir / f"{granule_path.stem}_two_stage_vs_li_ginoux_scatter.png"

    save_swath_map(
        lat2d,
        lon2d,
        dust_probability,
        prob_png,
        title=f"{granule_path.name}: two-stage dust probability",
        cbar_label="P(dust)",
        is_fraction=True,
    )
    save_swath_map(
        lat2d,
        lon2d,
        dust_fraction,
        frac_png,
        title=f"{granule_path.name}: two-stage dust fraction",
        cbar_label="Dust fraction",
        is_fraction=True,
    )
    save_swath_map(
        lat2d,
        lon2d,
        dust_aod,
        daod_png,
        title=f"{granule_path.name}: two-stage dust AOD (550 nm)",
        cbar_label="Dust AOD at 550 nm",
    )
    save_swath_map(
        lat2d,
        lon2d,
        lg_dust_aod,
        lg_png,
        title=f"{granule_path.name}: Li-Ginoux dust AOD (550 nm)",
        cbar_label="Dust AOD at 550 nm",
    )

    vmax = float(np.nanpercentile(np.concatenate([dust_aod[np.isfinite(dust_aod)], lg_dust_aod[np.isfinite(lg_dust_aod)]]), 98))
    save_comparison_panel(
        lat2d=lat2d,
        lon2d=lon2d,
        panels=[
            {"data": dust_aod, "title": "Two-stage ML dust AOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": lg_dust_aod, "title": "Li-Ginoux dust AOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
        ],
        output_path=panel_png,
        figure_title=f"{granule_path.name}: two-stage ML vs Li-Ginoux",
    )

    diff = dust_aod - lg_dust_aod
    save_swath_map(
        lat2d,
        lon2d,
        diff,
        diff_png,
        title=f"{granule_path.name}: two-stage ML minus Li-Ginoux",
        cbar_label="Dust AOD difference",
        cmap="RdBu_r",
        vmin=-float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)),
        vmax=float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)),
    )
    scatter_stats = save_aod_scatter_comparison(
        ml_dust_aod=dust_aod,
        li_ginoux_dust_aod=lg_dust_aod,
        output_path=scatter_png,
        title=f"{granule_path.name}: two-stage ML vs Li-Ginoux",
    )

    pd.DataFrame(reference_rows).to_csv(
        args.output_dir / f"{granule_path.stem}_reference_products_summary.csv",
        index=False,
    )
    summary = {
        "granule": granule_path.name,
        "model_dir": str(args.model_dir),
        "dust_probability_summary": summarize_grid("dust_probability", dust_probability),
        "dust_fraction_summary": summarize_grid("dust_fraction", dust_fraction),
        "dust_aod_summary": summarize_grid("dust_aod", dust_aod),
        "li_ginoux_summary": summarize_grid("li_ginoux_dust_aod", lg_dust_aod),
        "scatter_stats": scatter_stats,
        "model_valid_pixels": int(np.sum(model_valid)),
    }
    with (args.output_dir / f"{granule_path.stem}_two_stage_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Granule: {granule_path}")
    print(f"Two-stage dust AOD median: {summary['dust_aod_summary'].get('median')}")
    print(f"Li-Ginoux dust AOD median: {summary['li_ginoux_summary'].get('median')}")


if __name__ == "__main__":
    main()
