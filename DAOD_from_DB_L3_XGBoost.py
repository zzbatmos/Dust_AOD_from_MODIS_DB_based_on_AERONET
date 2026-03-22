#!/usr/bin/env python3

"""Apply the trained XGBoost dust-AOD model to MODIS Deep Blue Level-3 files.

This mirrors the batch-processing pattern of `DAOD_from_DB_L3-v4.py`, but
predicts dust AOD directly from L3 daily-mean Deep Blue predictor fields using
the collocation-trained 550 nm XGBoost model.
"""

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
from xgboost import XGBRegressor

REPO_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = REPO_DIR.parent
sys.path.insert(0, str(RESEARCH_DIR))

import MODIS_Lib  # noqa: E402
from AWS_Utils import get_NASA_creds  # noqa: E402

DEFAULT_MODEL_JSON = REPO_DIR / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_xgb.json"
DEFAULT_MODEL_META = REPO_DIR / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_metadata.json"
OUT_LOCAL_DIR = REPO_DIR / "MODIS_DB_L3_DustAOD_XGBoost"
S3_BUCKET = "zhibo-zhang-bucket"
S3_PREFIX = "Data/MODIS_DB_Dust_AOD_Level3_XGBoost"

ENC = {
    "dust_aod": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
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
        description="Apply the trained XGBoost dust-AOD model to MODIS Deep Blue L3 files."
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
    parser.add_argument("--upload", action="store_true", help="Upload NetCDFs to S3 and delete local files on success.")
    parser.add_argument("--sub-sample", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT_LOCAL_DIR)
    parser.add_argument("--s3-bucket", default=S3_BUCKET)
    parser.add_argument("--s3-prefix", default=S3_PREFIX)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--model-meta", type=Path, default=DEFAULT_MODEL_META)
    return parser.parse_args()


def month_chunks(start: dt.date, end: dt.date) -> Iterable[tuple[dt.date, dt.date]]:
    cursor = start.replace(day=1)
    while cursor <= end:
        if cursor.month == 12:
            nxt = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            nxt = cursor.replace(month=cursor.month + 1, day=1)
        chunk_end = min(end, nxt - dt.timedelta(days=1))
        yield cursor, chunk_end
        cursor = nxt


def parse_yd_filename(hdf_name: str) -> dt.date:
    match = re.search(r"\.A(\d{4})(\d{3})", hdf_name)
    if not match:
        raise ValueError(f"Cannot parse date from filename: {hdf_name}")
    year = int(match.group(1))
    jday = int(match.group(2))
    return dt.date.fromordinal(dt.date(year, 1, 1).toordinal() + jday - 1)


def boto_session_from_earthdata(creds: dict) -> boto3.Session:
    return boto3.Session(
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"],
    )


def s3_exists(session: boto3.Session, bucket: str, key: str) -> bool:
    s3 = session.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def download_to_temp(s3_uri: str, session: boto3.Session) -> str:
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3 = session.client("s3")
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf", delete=False)
    with open(tmp.name, "wb") as handle:
        s3.download_fileobj(bucket, key, handle)
    return tmp.name


def upload_nc(session: boto3.Session, local_path: str, bucket: str, key: str) -> None:
    session.client("s3").upload_file(local_path, bucket, key)


def load_model_and_meta(model_json: Path, model_meta: Path) -> tuple[XGBRegressor, dict]:
    if not model_json.exists():
        raise FileNotFoundError(f"Model JSON not found: {model_json}")
    if not model_meta.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_meta}")
    model = XGBRegressor()
    model.load_model(model_json)
    with model_meta.open("r") as handle:
        meta = json.load(handle)
    return model, meta


def inv_log1p_transform(values: np.ndarray, eps: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.clip(np.expm1(values) - eps, 0.0, None)


def predict_grid(model: XGBRegressor, meta: dict, features_2d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feature_order = meta.get("feature_names", meta.get("feature_columns"))
    if feature_order is None:
        raise KeyError("Model metadata is missing feature_names/feature_columns.")
    target_transform = meta.get("target_transform", "log1p")
    eps = float(meta.get("eps", 0.0))

    h, w = np.asarray(next(iter(features_2d.values()))).shape
    x_flat = np.column_stack([np.asarray(features_2d[name], dtype=float).ravel() for name in feature_order])
    valid = np.isfinite(x_flat).all(axis=1)
    pred_flat = np.full(h * w, np.nan, dtype=np.float32)

    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=feature_order)
        y_pred = model.predict(x_valid)
        if target_transform != "log1p":
            raise ValueError(f"Unsupported target transform for this script: {target_transform}")
        pred_flat[valid] = inv_log1p_transform(y_pred, eps=eps).astype(np.float32)

    return pred_flat.reshape(h, w), valid.reshape(h, w)


def enforce_daod_output_semantics(dust_aod: np.ndarray, total_aod: np.ndarray) -> np.ndarray:
    """Match the Li-Ginoux DAOD semantics.

    1. If DB total AOD is missing: dust AOD = NaN.
    2. If DB total AOD is finite and > 0, but the model has no valid prediction: dust AOD = 0.
    3. If DB total AOD is finite and the model predicts a value: keep the predicted dust AOD.
    """
    total_aod = np.asarray(total_aod, dtype=float)
    dust_aod = np.asarray(dust_aod, dtype=float).copy()

    total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
    no_retrieval = ~np.isfinite(total_aod)

    dust_aod[no_retrieval] = np.nan
    dust_aod[total_valid & ~np.isfinite(dust_aod)] = 0.0
    return dust_aod.astype(np.float32)


def find_db_attr(
    db_obj,
    required_substrings: list[str],
    preferred_substrings: Optional[list[str]] = None,
) -> str:
    preferred_substrings = preferred_substrings or []
    names = [name for name in dir(db_obj) if name.lower().startswith("deep_blue")]
    candidates = []
    for name in names:
        lname = name.lower()
        if all(token in lname for token in required_substrings):
            score = sum(token in lname for token in preferred_substrings)
            candidates.append((score, name))
    if not candidates:
        raise AttributeError(f"Could not find Deep Blue field matching {required_substrings}")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def extract_band(arr: np.ndarray, index: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    if arr.shape[0] <= 16:
        return arr[index, ...]
    if arr.shape[-1] <= 16:
        return arr[..., index]
    raise ValueError(f"Cannot infer band axis for array with shape {arr.shape}")


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


def build_mask_for_bbox(lat: np.ndarray, lon: np.ndarray, bbox: Optional[tuple[float, float, float, float]]) -> np.ndarray:
    if bbox is None:
        return np.ones_like(lat, dtype=bool)
    lon_min, lon_max, lat_min, lat_max = bbox
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def safe_subset(
    lat: np.ndarray,
    lon: np.ndarray,
    arrays: list[np.ndarray],
    bbox: Optional[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if bbox is None:
        return lat, lon, arrays
    mask = build_mask_for_bbox(lat, lon, bbox)
    if not np.any(mask):
        empty = [np.full((0, 0), np.nan, dtype=np.float32) for _ in arrays]
        return lat[mask], lon[mask], empty
    yy, xx = np.where(mask)
    y0, y1 = yy.min(), yy.max()
    x0, x1 = xx.min(), xx.max()
    lat2 = lat[y0 : y1 + 1, x0 : x1 + 1]
    lon2 = lon[y0 : y1 + 1, x0 : x1 + 1]
    out = [np.asarray(arr)[y0 : y1 + 1, x0 : x1 + 1] for arr in arrays]
    return lat2, lon2, out


def build_dataset(
    lat: np.ndarray,
    lon: np.ndarray,
    dust_aod: np.ndarray,
    valid_mask: np.ndarray,
    features: dict[str, np.ndarray],
    attrs: dict,
) -> xr.Dataset:
    ny, nx = dust_aod.shape
    ds = xr.Dataset(
        data_vars=dict(
            dust_aod=(
                ("y", "x"),
                dust_aod.astype("float32"),
                {
                    "long_name": "Dust AOD at 550 nm predicted from MODIS Deep Blue daily means by XGBoost",
                    "units": "1",
                    "note": "NaN where required model inputs are unavailable.",
                },
            ),
            db_aod_550=(("y", "x"), features["MODIS_DB_AOD550"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 550 nm", "units": "1"}),
            db_aod_412=(("y", "x"), features["MODIS_DB_AOD412"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 412 nm", "units": "1"}),
            db_aod_470=(("y", "x"), features["MODIS_DB_AOD470"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 470 nm", "units": "1"}),
            db_aod_660=(("y", "x"), features["MODIS_DB_AOD660"].astype("float32"), {"long_name": "Deep Blue daily-mean AOD at 660 nm", "units": "1"}),
            angstrom_exponent=(("y", "x"), features["MODIS_DB_AE"].astype("float32"), {"long_name": "Deep Blue daily-mean Angstrom exponent", "units": "1"}),
            ssa_412=(("y", "x"), features["MODIS_DB_SSA412"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 412 nm", "units": "1"}),
            ssa_470=(("y", "x"), features["MODIS_DB_SSA470"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 470 nm", "units": "1"}),
            ssa_660=(("y", "x"), features["MODIS_DB_SSA660"].astype("float32"), {"long_name": "Deep Blue daily-mean SSA at 660 nm", "units": "1"}),
            model_valid_input=(("y", "x"), valid_mask.astype("int8"), {"long_name": "1 where all XGBoost inputs are finite", "units": "1"}),
            lat=(("y", "x"), lat.astype("float32"), {"standard_name": "latitude", "units": "degrees_north"}),
            lon=(("y", "x"), lon.astype("float32"), {"standard_name": "longitude", "units": "degrees_east"}),
        ),
        coords=dict(y=np.arange(ny), x=np.arange(nx)),
        attrs=attrs,
    )
    return ds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, meta = load_model_and_meta(args.model_json, args.model_meta)

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
                out_name = f"{args.short_name}_DustAOD_XGB_{fdate.isoformat()}.nc"
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
                dust_aod, valid_mask = predict_grid(model, meta, features)
                dust_aod = enforce_daod_output_semantics(dust_aod, features["MODIS_DB_AOD550"])

                arrays_to_subset = [
                    dust_aod,
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
                    "title": f"Daily dust AOD from MODIS Deep Blue {args.short_name} by XGBoost",
                    "source": f"MODIS Deep Blue L3 daily product {args.short_name}",
                    "processing": "Direct XGBoost DAOD550 prediction from daily-mean MODIS DB predictors",
                    "bbox": str(bbox) if bbox else "global",
                    "model_json": str(args.model_json),
                    "model_meta": str(args.model_meta),
                    "target_wavelength_nm": str(meta.get('target_wavelength_nm', 550)),
                    "feature_columns": ",".join(meta.get("feature_columns", [])),
                    "history": f"Generated on {dt.datetime.utcnow().isoformat()}Z",
                    "caveat": "Model was trained on L2 collocations and is being applied here to L3 daily means.",
                }
                ds = build_dataset(lat, lon, dust_aod, valid_mask, subset_features, attrs)
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
