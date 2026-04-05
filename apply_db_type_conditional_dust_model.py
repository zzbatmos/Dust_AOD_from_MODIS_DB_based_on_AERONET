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
from pyhdf.SD import SD, SDC
from xgboost import XGBClassifier, XGBRegressor

from apply_dust_model_to_modis_db import (
    enforce_daod_output_semantics,
    li_ginoux_dust_aod_with_ssa_constraint,
    read_granule_features,
    save_aod_scatter_comparison,
    save_comparison_panel,
    save_swath_map,
    summarize_grid,
)
from train_modis_db_type_conditional_from_qa_xgb import (
    AEROSOL_GROUP_MAP,
    FEATURES_BY_GROUP,
)


TYPE_LABELS = {
    0: "Mixed",
    1: "Dust",
    2: "Smoke",
    3: "Sulfate",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Apply DB aerosol-type-conditional dust models to MODIS L2 granules.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=here / "modis_db_type_conditional_detection_from_qa",
    )
    return parser.parse_args()


def decode_db_qa(granule_path: Path) -> dict[str, np.ndarray]:
    f = SD(str(granule_path), SDC.READ)
    qa = np.asarray(f.select("Quality_Assurance_Land").get(), dtype=np.int16)
    f.end()
    byte4 = qa[:, :, 4].astype(np.uint8)
    usefulness = (byte4 & 0b1).astype(np.uint8)
    confidence = ((byte4 >> 1) & 0b11).astype(np.uint8)
    aerosol_type_code = ((byte4 >> 3) & 0b11).astype(np.uint8)
    return {
        "usefulness": usefulness,
        "confidence": confidence,
        "aerosol_type_code": aerosol_type_code,
    }


def build_pixel_features(features_2d: dict[str, np.ndarray], qa: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=float).copy() for k, v in features_2d.items()}
    out["has_ssa412"] = np.isfinite(out["MODIS_DB_SSA412"]).astype(float)
    out["has_ssa470"] = np.isfinite(out["MODIS_DB_SSA470"]).astype(float)
    out["has_ssa660"] = np.isfinite(out["MODIS_DB_SSA660"]).astype(float)
    out["has_ae"] = np.isfinite(out["MODIS_DB_AE"]).astype(float)
    for key in ["MODIS_DB_SSA412", "MODIS_DB_SSA470", "MODIS_DB_SSA660"]:
        arr = out[key]
        arr = np.clip(arr, 0.0, 1.0)
        arr[~np.isfinite(arr)] = -0.05
        out[key] = arr
    ae = out["MODIS_DB_AE"]
    ae = ae.copy()
    ae[~np.isfinite(ae)] = -0.05
    out["MODIS_DB_AE"] = ae
    out["AOD412_over_470"] = out["MODIS_DB_AOD412"] / out["MODIS_DB_AOD470"]
    out["AOD470_over_660"] = out["MODIS_DB_AOD470"] / out["MODIS_DB_AOD660"]
    out["AOD550_over_660"] = out["MODIS_DB_AOD550"] / out["MODIS_DB_AOD660"]
    out["SSA470_minus_412"] = out["MODIS_DB_SSA470"] - out["MODIS_DB_SSA412"]
    out["SSA660_minus_470"] = out["MODIS_DB_SSA660"] - out["MODIS_DB_SSA470"]

    usefulness = np.asarray(qa["usefulness"], dtype=float)
    confidence = np.asarray(qa["confidence"], dtype=int)
    aerosol_code = np.asarray(qa["aerosol_type_code"], dtype=int)
    out["qa_useful_fraction"] = usefulness
    for i in range(4):
        out[f"qa_conf{i}_fraction_all"] = (confidence == i).astype(float)
    out["qa_aerosol_mixed_fraction_all"] = (aerosol_code == 0).astype(float)
    out["qa_aerosol_dust_fraction_all"] = (aerosol_code == 1).astype(float)
    out["qa_aerosol_smoke_fraction_all"] = (aerosol_code == 2).astype(float)
    out["qa_aerosol_sulfate_fraction_all"] = (aerosol_code == 3).astype(float)
    return out


def load_models(model_dir: Path) -> tuple[dict[str, XGBClassifier], dict[str, XGBRegressor | None]]:
    classifiers: dict[str, XGBClassifier] = {}
    regressors: dict[str, XGBRegressor | None] = {}
    for group in ["dust", "mixed", "smoke_sulfate"]:
        clf_path = model_dir / f"{group}_classifier.json"
        reg_path = model_dir / f"{group}_fraction_regressor.json"
        if clf_path.exists():
            clf = XGBClassifier()
            clf.load_model(clf_path)
            classifiers[group] = clf
        if reg_path.exists():
            reg = XGBRegressor()
            reg.load_model(reg_path)
            regressors[group] = reg
        else:
            regressors[group] = None
    return classifiers, regressors


def compute_qa_class(db_type_label: str, prob: np.ndarray) -> np.ndarray:
    qa = np.zeros_like(prob, dtype=np.float32)
    valid = np.isfinite(prob)
    if db_type_label == "Dust":
        qa[valid & (prob >= 0.75)] = 3
        qa[valid & (prob >= 0.45) & (prob < 0.75)] = 2
        qa[valid & (prob < 0.45)] = 1
    elif db_type_label == "Mixed":
        qa[valid & (prob >= 0.75)] = 2
        qa[valid & (prob >= 0.40) & (prob < 0.75)] = 1
        qa[valid & (prob < 0.40)] = 0
    else:  # Smoke or Sulfate
        qa[valid & (prob >= 0.80)] = 1
        qa[valid & (prob < 0.80)] = 0
    qa[~valid] = np.nan
    return qa


def predict_routed(
    classifiers: dict[str, XGBClassifier],
    regressors: dict[str, XGBRegressor | None],
    pixel_features: dict[str, np.ndarray],
    qa: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = np.asarray(next(iter(pixel_features.values()))).shape
    aerosol_code = np.asarray(qa["aerosol_type_code"], dtype=int)
    probability = np.full((h, w), np.nan, dtype=np.float32)
    fraction = np.full((h, w), np.nan, dtype=np.float32)
    qa_class = np.full((h, w), np.nan, dtype=np.float32)

    valid_geo = np.isfinite(pixel_features["MODIS_DB_AOD550"]) & (pixel_features["MODIS_DB_AOD550"] > 0.0)
    for code, label in TYPE_LABELS.items():
        group = AEROSOL_GROUP_MAP.get(label)
        if group is None or group not in classifiers:
            continue
        mask = valid_geo & (aerosol_code == code)
        if not np.any(mask):
            continue
        features = FEATURES_BY_GROUP[group]
        x = np.column_stack([np.asarray(pixel_features[f], dtype=float)[mask] for f in features])
        finite = np.isfinite(x).all(axis=1)
        if not np.any(finite):
            continue
        rows = pd.DataFrame(x[finite], columns=features)
        prob = classifiers[group].predict_proba(rows)[:, 1].astype(np.float32)
        pred_frac = np.zeros_like(prob, dtype=np.float32)
        reg = regressors.get(group)
        if reg is not None:
            pred_frac = np.clip(reg.predict(rows), 0.0, 1.0).astype(np.float32)
        mask_idx = np.flatnonzero(mask)
        valid_idx = mask_idx[finite]
        probability.ravel()[valid_idx] = prob
        fraction.ravel()[valid_idx] = pred_frac
        qa_vals = compute_qa_class(label, prob)
        qa_class.ravel()[valid_idx] = qa_vals[np.isfinite(qa_vals)]

    return probability, fraction, qa_class


def qa_to_daod(fraction: np.ndarray, qa_class: np.ndarray, aod550: np.ndarray, minimum_qa: int) -> np.ndarray:
    out = np.asarray(fraction, dtype=float).copy() * np.asarray(aod550, dtype=float)
    out[np.asarray(qa_class, dtype=float) < float(minimum_qa)] = np.nan
    return enforce_daod_output_semantics(out, aod550)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_2d, lat2d, lon2d = read_granule_features(args.input_file)
    qa = decode_db_qa(args.input_file)
    pixel_features = build_pixel_features(features_2d, qa)
    classifiers, regressors = load_models(args.model_dir)

    dust_probability, dust_fraction, dust_qa = predict_routed(classifiers, regressors, pixel_features, qa)
    raw_dust_aod = enforce_daod_output_semantics(
        np.asarray(dust_fraction, dtype=float) * np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float),
        features_2d["MODIS_DB_AOD550"],
    )
    qa2_dust_aod = qa_to_daod(dust_fraction, dust_qa, features_2d["MODIS_DB_AOD550"], minimum_qa=2)
    qa1_dust_aod = qa_to_daod(dust_fraction, dust_qa, features_2d["MODIS_DB_AOD550"], minimum_qa=1)

    _, _, li_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(features_2d)

    stem = args.input_file.stem
    save_swath_map(
        lat2d, lon2d, dust_probability,
        args.output_dir / f"{stem}_db_type_conditional_probability.png",
        title=f"{args.input_file.name}: DB-type-conditional dust probability",
        cbar_label="Dust probability",
        is_fraction=True,
    )
    save_swath_map(
        lat2d, lon2d, dust_qa,
        args.output_dir / f"{stem}_db_type_conditional_qa.png",
        title=f"{args.input_file.name}: DB-type-conditional dust QA",
        cbar_label="Dust QA class",
        vmin=0.0,
        vmax=3.0,
        cmap="viridis",
    )
    vmax = float(np.nanpercentile(np.concatenate([raw_dust_aod[np.isfinite(raw_dust_aod)], li_dust_aod[np.isfinite(li_dust_aod)]]), 98))
    save_swath_map(
        lat2d, lon2d, raw_dust_aod,
        args.output_dir / f"{stem}_db_type_conditional_dust_aod_raw.png",
        title=f"{args.input_file.name}: DB-type-conditional dust AOD (raw)",
        cbar_label="Dust AOD at 550 nm",
        vmin=0.0,
        vmax=vmax,
        cmap="YlOrBr",
    )
    save_swath_map(
        lat2d, lon2d, qa2_dust_aod,
        args.output_dir / f"{stem}_db_type_conditional_dust_aod_qa2.png",
        title=f"{args.input_file.name}: DB-type-conditional dust AOD (QA>=2)",
        cbar_label="Dust AOD at 550 nm",
        vmin=0.0,
        vmax=vmax,
        cmap="YlOrBr",
    )

    save_comparison_panel(
        lat2d=lat2d,
        lon2d=lon2d,
        panels=[
            {"data": raw_dust_aod, "title": "DB-type-conditional ML DAOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": qa2_dust_aod, "title": "DB-type-conditional ML DAOD (QA>=2)", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": li_dust_aod, "title": "Li-Ginoux DAOD", "label": "Dust AOD (550 nm)", "cmap": "YlOrBr", "vmin": 0.0, "vmax": vmax},
            {"data": raw_dust_aod - li_dust_aod, "title": "ML raw minus Li-Ginoux", "label": "DAOD difference", "cmap": "RdBu_r"},
        ],
        output_path=args.output_dir / f"{stem}_db_type_conditional_comparison_panel.png",
        figure_title=f"{args.input_file.name}: DB-type-conditional ML vs Li-Ginoux",
    )
    scatter_stats = save_aod_scatter_comparison(
        ml_dust_aod=raw_dust_aod,
        li_ginoux_dust_aod=li_dust_aod,
        output_path=args.output_dir / f"{stem}_db_type_conditional_vs_li_ginoux_scatter.png",
        title=f"{args.input_file.name}: DB-type-conditional ML vs Li-Ginoux",
    )

    summary = {
        "granule": args.input_file.name,
        "model_dir": str(args.model_dir),
        "dust_probability_summary": summarize_grid("dust_probability", dust_probability),
        "dust_qa_summary": summarize_grid("dust_qa", dust_qa),
        "raw_dust_aod_summary": summarize_grid("raw_dust_aod", raw_dust_aod),
        "qa1_dust_aod_summary": summarize_grid("qa1_dust_aod", qa1_dust_aod),
        "qa2_dust_aod_summary": summarize_grid("qa2_dust_aod", qa2_dust_aod),
        "li_ginoux_summary": summarize_grid("li_ginoux_dust_aod", li_dust_aod),
        "scatter_stats": scatter_stats,
        "db_aerosol_type_counts": {
            TYPE_LABELS[i]: int(np.count_nonzero(qa["aerosol_type_code"] == i)) for i in range(4)
        },
    }
    with (args.output_dir / f"{stem}_db_type_conditional_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
