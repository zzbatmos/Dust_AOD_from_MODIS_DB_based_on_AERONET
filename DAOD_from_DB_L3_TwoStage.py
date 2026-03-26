#!/usr/bin/env python3

"""Apply the tuned two-stage dust model to MODIS Deep Blue Level-3 daily files."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import boto3
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier, XGBRegressor

REPO_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = REPO_DIR.parent
sys.path.insert(0, str(RESEARCH_DIR))

import MODIS_Lib  # noqa: E402
from AWS_Utils import get_NASA_creds  # noqa: E402
from DAOD_from_DB_L3_XGBoost import (  # noqa: E402
    build_mask_for_bbox,
    boto_session_from_earthdata,
    download_to_temp,
    extract_band,
    find_db_attr,
    month_chunks,
    parse_yd_filename,
    safe_subset,
    s3_exists,
    upload_nc,
)

DEFAULT_MODEL_DIR = REPO_DIR / "two_stage_tuning" / "best_model"
OUT_LOCAL_DIR = REPO_DIR / "MODIS_DB_L3_DustAOD_TwoStage"
S3_BUCKET = "zhibo-zhang-bucket"
S3_PREFIX = "Data/MODIS_DB_Dust_AOD_Level3_TwoStage"

ENC = {
    "dust_aod": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "dust_fraction": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "dust_probability": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "db_aod_550": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "db_aod_412": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "db_aod_470": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "db_aod_660": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "angstrom_exponent": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "ssa_412": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "ssa_470": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "ssa_660": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "model_valid_input": {"zlib": True, "complevel": 4, "dtype": "int8", "_FillValue": np.int8(-1)},
    "lat": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "lon": {"zlib": True, "complevel": 4, "dtype": "float32"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the tuned two-stage dust model to MODIS Deep Blue L3 files."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--short-name", choices=["MOD08_D3", "MYD08_D3"], default="MOD08_D3")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Optional output subset bbox: lon_min lon_max lat_min lat_max",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--sub-sample", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT_LOCAL_DIR)
    parser.add_argument("--s3-bucket", default=S3_BUCKET)
    parser.add_argument("--s3-prefix", default=S3_PREFIX)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def inv_logit_transform(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def enforce_daod_output_semantics(dust_aod: np.ndarray, total_aod: np.ndarray) -> np.ndarray:
    total_aod = np.asarray(total_aod, dtype=float)
    dust_aod = np.asarray(dust_aod, dtype=float).copy()
    total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
    dust_aod[~np.isfinite(total_aod)] = np.nan
    dust_aod[total_valid & ~np.isfinite(dust_aod)] = 0.0
    return dust_aod.astype(np.float32)


def read_l3_features(l3_obj) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    aod550_attr = find_db_attr(
        l3_obj,
        ["aerosol", "optical", "depth", "550"],
        preferred_substrings=["land", "mean"],
    )
    ae_attr = find_db_attr(
        l3_obj,
        ["angstrom", "exponent"],
        preferred_substrings=["land", "mean"],
    )
    spec_aod_attr = find_db_attr(
        l3_obj,
        ["aerosol", "optical", "depth", "land", "mean"],
        preferred_substrings=["deep_blue"],
    )
    spec_ssa_attr = find_db_attr(
        l3_obj,
        ["single", "scattering", "albedo"],
        preferred_substrings=["land", "mean"],
    )

    spec_aod = np.asarray(getattr(l3_obj, spec_aod_attr), dtype=np.float32)
    spec_ssa = np.asarray(getattr(l3_obj, spec_ssa_attr), dtype=np.float32)
    features = {
        "MODIS_DB_AOD550": np.asarray(getattr(l3_obj, aod550_attr), dtype=np.float32),
        "MODIS_DB_AOD412": extract_band(spec_aod, 0),
        "MODIS_DB_AOD470": extract_band(spec_aod, 1),
        "MODIS_DB_AOD660": extract_band(spec_aod, 2),
        "MODIS_DB_AE": np.asarray(getattr(l3_obj, ae_attr), dtype=np.float32),
        "MODIS_DB_SSA412": extract_band(spec_ssa, 0),
        "MODIS_DB_SSA470": extract_band(spec_ssa, 1),
        "MODIS_DB_SSA660": extract_band(spec_ssa, 2),
    }
    return features, np.asarray(l3_obj.lat, dtype=np.float32), np.asarray(l3_obj.lon, dtype=np.float32)


def load_models(model_dir: Path) -> tuple[XGBClassifier, XGBRegressor, dict]:
    meta_path = model_dir / "two_stage_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Two-stage metadata not found: {meta_path}")
    with meta_path.open("r") as handle:
        meta = json.load(handle)
    classifier = XGBClassifier()
    classifier.load_model(model_dir / meta["classifier_model"])
    regressor = XGBRegressor()
    regressor.load_model(model_dir / meta["regressor_model"])
    return classifier, regressor, meta


def predict_two_stage_grid(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    meta: dict,
    features_2d: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_order = meta["feature_names"]
    h, w = np.asarray(next(iter(features_2d.values()))).shape
    x_flat = np.column_stack([np.asarray(features_2d[name], dtype=float).ravel() for name in feature_order])
    valid = np.isfinite(x_flat).all(axis=1)

    prob_flat = np.full(h * w, np.nan, dtype=np.float32)
    frac_flat = np.full(h * w, np.nan, dtype=np.float32)
    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=feature_order)
        prob = classifier.predict_proba(x_valid)[:, 1]
        frac = inv_logit_transform(regressor.predict(x_valid))
        frac = np.where(prob >= float(meta["probability_threshold"]), frac, 0.0)
        prob_flat[valid] = prob.astype(np.float32)
        frac_flat[valid] = np.clip(frac, 0.0, 1.0).astype(np.float32)

    return prob_flat.reshape(h, w), frac_flat.reshape(h, w), valid.reshape(h, w)


def build_dataset(
    lat: np.ndarray,
    lon: np.ndarray,
    dust_aod: np.ndarray,
    dust_fraction: np.ndarray,
    dust_probability: np.ndarray,
    valid_mask: np.ndarray,
    features: dict[str, np.ndarray],
    attrs: dict,
) -> xr.Dataset:
    ny, nx = dust_aod.shape
    return xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), dust_aod.astype("float32"), {"long_name": "Dust AOD at 550 nm from two-stage MODIS DB model", "units": "1"}),
            dust_fraction=(("y", "x"), dust_fraction.astype("float32"), {"long_name": "Dust fraction from two-stage MODIS DB model", "units": "1"}),
            dust_probability=(("y", "x"), dust_probability.astype("float32"), {"long_name": "Dust probability from stage-1 classifier", "units": "1"}),
            db_aod_550=(("y", "x"), features["MODIS_DB_AOD550"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 550 nm", "units": "1"}),
            db_aod_412=(("y", "x"), features["MODIS_DB_AOD412"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 412 nm", "units": "1"}),
            db_aod_470=(("y", "x"), features["MODIS_DB_AOD470"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 470 nm", "units": "1"}),
            db_aod_660=(("y", "x"), features["MODIS_DB_AOD660"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 660 nm", "units": "1"}),
            angstrom_exponent=(("y", "x"), features["MODIS_DB_AE"].astype("float32"), {"long_name": "Deep Blue daily-mean Angstrom exponent", "units": "1"}),
            ssa_412=(("y", "x"), features["MODIS_DB_SSA412"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 412 nm", "units": "1"}),
            ssa_470=(("y", "x"), features["MODIS_DB_SSA470"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 470 nm", "units": "1"}),
            ssa_660=(("y", "x"), features["MODIS_DB_SSA660"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 660 nm", "units": "1"}),
            model_valid_input=(("y", "x"), valid_mask.astype("int8"), {"long_name": "1 where all two-stage model inputs are finite", "units": "1"}),
            lat=(("y", "x"), lat.astype("float32"), {"standard_name": "latitude", "units": "degrees_north"}),
            lon=(("y", "x"), lon.astype("float32"), {"standard_name": "longitude", "units": "degrees_east"}),
        ),
        coords=dict(y=np.arange(ny), x=np.arange(nx)),
        attrs=attrs,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier, regressor, meta = load_models(args.model_dir)

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    bbox = tuple(args.bbox) if args.bbox else None

    for mstart, mend in month_chunks(start_date, end_date):
        laads_creds = get_NASA_creds("laadsdaac")
        earthaccess.login(persist=True)
        s3_session = boto_session_from_earthdata(laads_creds)

        temporal = (mstart.strftime("%Y-%m-%d"), mend.strftime("%Y-%m-%d"))
        print(f"[SEARCH] {args.short_name} {temporal}")
        items = earthaccess.search_data(short_name=args.short_name, temporal=temporal)
        if not items:
            print("[INFO] no files found in this chunk")
            continue

        objs = earthaccess.open(items)
        for obj in objs:
            try:
                gid = obj.details["name"]
                s3_uri = f"s3://{gid}"
                fname = os.path.basename(gid)
                fdate = parse_yd_filename(fname)
                out_name = f"{args.short_name}_DustAOD_TwoStage_{fdate.isoformat()}.nc"
                local_nc = args.output_dir / out_name
                s3_key = f"{args.s3_prefix}/{fdate.year}/{out_name}"

                if local_nc.exists():
                    print(f"[SKIP exists] {local_nc}")
                    continue
                if args.upload and s3_exists(s3_session, args.s3_bucket, s3_key):
                    print(f"[SKIP exists] s3://{args.s3_bucket}/{s3_key}")
                    continue

                local_hdf = download_to_temp(s3_uri, s3_session)
                l3 = MODIS_Lib.MODIS_DeepBlue_Level3(local_hdf, sub_sample=args.sub_sample)
                features, lat, lon = read_l3_features(l3)
                dust_probability, dust_fraction, valid_mask = predict_two_stage_grid(
                    classifier,
                    regressor,
                    meta,
                    features,
                )
                dust_aod = dust_fraction * np.asarray(features["MODIS_DB_AOD550"], dtype=float)
                dust_aod = enforce_daod_output_semantics(dust_aod, features["MODIS_DB_AOD550"])

                arrays_to_subset = [
                    dust_aod,
                    dust_fraction,
                    dust_probability,
                    valid_mask.astype(np.float32),
                    features["MODIS_DB_AOD550"],
                    features["MODIS_DB_AOD412"],
                    features["MODIS_DB_AOD470"],
                    features["MODIS_DB_AOD660"],
                    features["MODIS_DB_AE"],
                    features["MODIS_DB_SSA412"],
                    features["MODIS_DB_SSA470"],
                    features["MODIS_DB_SSA660"],
                ]
                lat, lon, arrays = safe_subset(lat, lon, arrays_to_subset, bbox)
                if lat.size == 0:
                    print("[SKIP] no pixels in bbox for this daily file.")
                    os.remove(local_hdf)
                    continue

                (
                    dust_aod,
                    dust_fraction,
                    dust_probability,
                    valid_mask_float,
                    aod550,
                    aod412,
                    aod470,
                    aod660,
                    ae,
                    ssa412,
                    ssa470,
                    ssa660,
                ) = arrays
                subset_features = {
                    "MODIS_DB_AOD550": aod550,
                    "MODIS_DB_AOD412": aod412,
                    "MODIS_DB_AOD470": aod470,
                    "MODIS_DB_AOD660": aod660,
                    "MODIS_DB_AE": ae,
                    "MODIS_DB_SSA412": ssa412,
                    "MODIS_DB_SSA470": ssa470,
                    "MODIS_DB_SSA660": ssa660,
                }
                valid_mask = valid_mask_float.astype(bool)

                attrs = {
                    "title": f"Daily dust AOD from MODIS Deep Blue {args.short_name} by tuned two-stage model",
                    "source": f"MODIS Deep Blue L3 daily product {args.short_name}",
                    "processing": "Stage-1 XGBoost dust classifier plus stage-2 XGBoost dust-fraction regressor",
                    "bbox": str(bbox) if bbox else "global",
                    "model_dir": str(args.model_dir),
                    "dust_label_definition": meta.get("dust_label_definition", ""),
                    "probability_threshold": str(meta.get("probability_threshold", "")),
                    "history": f"Generated on {dt.datetime.now(dt.timezone.utc).isoformat()}",
                    "caveat": "Model was trained on L2 collocations and is being applied here to L3 daily means.",
                }
                ds = build_dataset(
                    lat,
                    lon,
                    dust_aod,
                    dust_fraction,
                    dust_probability,
                    valid_mask,
                    subset_features,
                    attrs,
                )
                ds.to_netcdf(local_nc, format="NETCDF4", encoding=ENC)
                print(f"[WROTE] {local_nc}")

                if args.upload:
                    upload_nc(s3_session, str(local_nc), args.s3_bucket, s3_key)
                    print(f"[UPLOADED] s3://{args.s3_bucket}/{s3_key}")
                    try:
                        os.remove(local_nc)
                    except OSError:
                        pass

                try:
                    os.remove(local_hdf)
                except OSError:
                    pass

            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {exc}")
                continue


if __name__ == "__main__":
    main()
