#!/usr/bin/env python3

"""Apply the trained XGBoost dust-AOD model to MODIS Deep Blue Level-2 granules.

This mirrors the batch-search/download/write workflow of `DAOD_from_DB_L2-v4.py`,
but replaces the Li-Ginoux AE-to-FMF parameterization with the direct
550 nm XGBoost dust-AOD model trained from collocated AERONET/MODIS DB data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import boto3
import botocore
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

OUT_LOCAL_DIR = REPO_DIR / "MODIS_DB_L2_DustAOD_XGBoost"
S3_BUCKET = "zhibo-zhang-bucket"
S3_PREFIX = "Data/MODIS_DB_Dust_AOD_Level2_XGBoost"

DEFAULT_MODEL_JSON = REPO_DIR / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_xgb.json"
DEFAULT_MODEL_META = REPO_DIR / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_metadata.json"

ANGLE_NAME_CANDIDATES = {
    "sza": ["Solar_Zenith", "Deep_Blue_Solar_Zenith_Land", "Solar_Zenith_Angle_Land", "SolarZenith"],
    "vza": ["Sensor_Zenith", "View_Zenith", "Sensor_Zenith_Angle_Land", "SensorZenith"],
    "saa": ["Solar_Azimuth", "Deep_Blue_Solar_Azimuth_Land", "Solar_Azimuth_Angle_Land", "SolarAzimuth"],
    "vaa": ["Sensor_Azimuth", "View_Azimuth", "Sensor_Azimuth_Angle_Land", "SensorAzimuth"],
}

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
    "sza": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "vza": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "raa": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "lat": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "lon": {"zlib": True, "complevel": 4, "dtype": "float32"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the trained XGBoost dust-AOD model to MODIS Deep Blue L2 granules."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--satellite", choices=["MYD", "MOD"], default="MYD", help="MYD=Aqua, MOD=Terra")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Optional output subset bbox: lon_min lon_max lat_min lat_max",
    )
    parser.add_argument("--upload", action="store_true", help="Upload NetCDFs to S3 and delete local files on success.")
    parser.add_argument("--sub-sample", type=int, default=1, help="Spatial sub-sampling step for the reader.")
    parser.add_argument("--output-dir", type=Path, default=OUT_LOCAL_DIR)
    parser.add_argument("--s3-bucket", default=S3_BUCKET)
    parser.add_argument("--s3-prefix", default=S3_PREFIX)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--model-meta", type=Path, default=DEFAULT_MODEL_META)
    return parser.parse_args()


def temporal_tuple(start_str: str, end_str: str) -> tuple[str, str]:
    return start_str, end_str


def earthaccess_bbox(bbox: Optional[tuple[float, float, float, float]]) -> Optional[tuple[float, float, float, float]]:
    if bbox is None:
        return None
    lon_min, lon_max, lat_min, lat_max = bbox
    return lon_min, lat_min, lon_max, lat_max


def laads_session_from_creds(creds: dict) -> boto3.Session:
    return boto3.Session(
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"],
    )


def refresh_laads_creds() -> dict:
    creds = get_NASA_creds("laadsdaac")
    earthaccess.login(persist=True)
    return creds


def s3_download_to_temp(s3_uri: str, session: boto3.Session) -> str:
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3 = session.client("s3")
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf", delete=False)
    with open(tmp.name, "wb") as handle:
        s3.download_fileobj(bucket, key, handle)
    return tmp.name


def s3_download_to_temp_robust(s3_uri: str, max_retries: int = 4, base_sleep: float = 1.5) -> str:
    attempt = 0
    while True:
        try:
            creds = refresh_laads_creds()
            session = laads_session_from_creds(creds)
            return s3_download_to_temp(s3_uri, session)
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            transient = code in ("ExpiredToken", "InvalidToken", "403", "400") or "Bad Request" in str(exc)
        except Exception as exc:  # noqa: BLE001
            code = ""
            transient = "Bad Request" in str(exc) or "ExpiredToken" in str(exc)
        if attempt >= max_retries or not transient:
            raise
        sleep = base_sleep * (2**attempt)
        print(f"[RETRY download] {code or 'Exception'} sleeping {sleep:.1f}s")
        time.sleep(sleep)
        attempt += 1


def ec2_s3_client():
    return boto3.client("s3")


def upload_with_verify(local_path: str, bucket: str, key: str, max_retries: int = 3, base_sleep: float = 1.5) -> None:
    attempt = 0
    while True:
        try:
            s3 = ec2_s3_client()
            s3.upload_file(local_path, bucket, key)
            s3.head_object(Bucket=bucket, Key=key)
            return
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            transient = code in ("SlowDown", "RequestTimeout", "Throttling", "InternalError", "ExpiredToken")
        except Exception as exc:  # noqa: BLE001
            code = ""
            transient = "timed out" in str(exc).lower()
        if attempt >= max_retries or not transient:
            raise
        sleep = base_sleep * (2**attempt)
        print(f"[RETRY upload] {code or 'Exception'} sleeping {sleep:.1f}s")
        time.sleep(sleep)
        attempt += 1


def parse_l2_filename(fname: str) -> str:
    match = re.search(r"\.A(\d{4})(\d{3})\.", fname)
    if not match:
        return "1970-01-01"
    year = int(match.group(1))
    jday = int(match.group(2))
    day = dt.date.fromordinal(dt.date(year, 1, 1).toordinal() + jday - 1)
    return day.isoformat()


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


def try_get_angle(db_obj, keys: list[str], fallback_shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        if hasattr(db_obj, key):
            return np.asarray(getattr(db_obj, key), dtype=np.float32)
    return np.full(fallback_shape, np.nan, dtype=np.float32)


def angle_normalize180(angle_deg: np.ndarray) -> np.ndarray:
    return ((angle_deg + 180.0) % 360.0) - 180.0


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


def read_l2_features(l2_obj) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    features = {
        "MODIS_DB_AOD550": np.asarray(l2_obj.Deep_Blue_Aerosol_Optical_Depth_550_Land, dtype=np.float32),
        "MODIS_DB_AOD412": np.asarray(l2_obj.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[0], dtype=np.float32),
        "MODIS_DB_AOD470": np.asarray(l2_obj.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[1], dtype=np.float32),
        "MODIS_DB_AOD660": np.asarray(l2_obj.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[2], dtype=np.float32),
        "MODIS_DB_AE": np.asarray(l2_obj.Deep_Blue_Angstrom_Exponent_Land, dtype=np.float32),
        "MODIS_DB_SSA412": np.asarray(l2_obj.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[0], dtype=np.float32),
        "MODIS_DB_SSA470": np.asarray(l2_obj.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[1], dtype=np.float32),
        "MODIS_DB_SSA660": np.asarray(l2_obj.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[2], dtype=np.float32),
    }
    return features, np.asarray(l2_obj.lat, dtype=np.float32), np.asarray(l2_obj.lon, dtype=np.float32)


def build_dataset_l2(
    lat: np.ndarray,
    lon: np.ndarray,
    dust_aod: np.ndarray,
    valid_mask: np.ndarray,
    features: dict[str, np.ndarray],
    sza: np.ndarray,
    vza: np.ndarray,
    raa: np.ndarray,
    global_attrs: dict,
) -> xr.Dataset:
    ny, nx = dust_aod.shape
    ds = xr.Dataset(
        data_vars=dict(
            dust_aod=(
                ("y", "x"),
                dust_aod.astype("float32"),
                {
                    "long_name": "Dust AOD at 550 nm predicted from MODIS Deep Blue by XGBoost",
                    "units": "1",
                    "note": "NaN where required model inputs are unavailable.",
                },
            ),
            db_aod_550=(("y", "x"), features["MODIS_DB_AOD550"].astype("float32"), {"long_name": "Deep Blue AOD at 550 nm", "units": "1"}),
            db_aod_412=(("y", "x"), features["MODIS_DB_AOD412"].astype("float32"), {"long_name": "Deep Blue AOD at 412 nm", "units": "1"}),
            db_aod_470=(("y", "x"), features["MODIS_DB_AOD470"].astype("float32"), {"long_name": "Deep Blue AOD at 470 nm", "units": "1"}),
            db_aod_660=(("y", "x"), features["MODIS_DB_AOD660"].astype("float32"), {"long_name": "Deep Blue AOD at 660 nm", "units": "1"}),
            angstrom_exponent=(
                ("y", "x"),
                features["MODIS_DB_AE"].astype("float32"),
                {"long_name": "Deep Blue Angstrom exponent", "units": "1"},
            ),
            ssa_412=(("y", "x"), features["MODIS_DB_SSA412"].astype("float32"), {"long_name": "Deep Blue SSA at 412 nm", "units": "1"}),
            ssa_470=(("y", "x"), features["MODIS_DB_SSA470"].astype("float32"), {"long_name": "Deep Blue SSA at 470 nm", "units": "1"}),
            ssa_660=(("y", "x"), features["MODIS_DB_SSA660"].astype("float32"), {"long_name": "Deep Blue SSA at 660 nm", "units": "1"}),
            model_valid_input=(("y", "x"), valid_mask.astype("int8"), {"long_name": "1 where all XGBoost inputs are finite", "units": "1"}),
            sza=(("y", "x"), sza.astype("float32"), {"long_name": "Solar zenith angle", "units": "degree"}),
            vza=(("y", "x"), vza.astype("float32"), {"long_name": "Sensor zenith angle", "units": "degree"}),
            raa=(("y", "x"), raa.astype("float32"), {"long_name": "Relative azimuth angle", "units": "degree"}),
            lat=(("y", "x"), lat.astype("float32"), {"standard_name": "latitude", "units": "degrees_north"}),
            lon=(("y", "x"), lon.astype("float32"), {"standard_name": "longitude", "units": "degrees_east"}),
        ),
        coords=dict(y=np.arange(ny, dtype="int32"), x=np.arange(nx, dtype="int32")),
        attrs=global_attrs,
    )
    return ds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_model_and_meta(args.model_json, args.model_meta)

    short_name = f"{args.satellite}04_L2"
    bbox = tuple(args.bbox) if args.bbox else None
    query_bbox = earthaccess_bbox(bbox)
    t0, t1 = temporal_tuple(args.start, args.end)

    print(f"[SEARCH] {short_name} {t0}..{t1} bbox={bbox if bbox else 'global'}")
    items = earthaccess.search_data(short_name=short_name, temporal=(t0, t1), bounding_box=query_bbox)
    if not items:
        print("[INFO] no items found for this query.")
        return

    objs = earthaccess.open(items)

    for obj in objs:
        try:
            gid = obj.details["name"]
            s3_uri = f"s3://{gid}"
            base = os.path.basename(gid)
            iso = parse_l2_filename(base)
            out_nc = args.output_dir / f"{base[:-4]}_DAOD_XGB.nc"

            if out_nc.exists():
                print(f"[SKIP exists] {out_nc}")
                continue

            print(f"[GRANULE] {base}")
            local_hdf = s3_download_to_temp_robust(s3_uri)

            l2 = MODIS_Lib.MODIS_DeepBlue_Level2(local_hdf, sub_sample=args.sub_sample)
            if not hasattr(l2, "Deep_Blue_Aerosol_Optical_Depth_550_Land"):
                print("[SKIP] no Deep Blue AOD field in this granule.")
                os.remove(local_hdf)
                continue

            features, lat, lon = read_l2_features(l2)
            if not np.isfinite(features["MODIS_DB_AOD550"]).any():
                print("[SKIP] no valid AOD in this granule.")
                os.remove(local_hdf)
                continue

            dust_aod, valid_mask = predict_grid(model, meta, features)
            dust_aod = enforce_daod_output_semantics(dust_aod, features["MODIS_DB_AOD550"])

            sza = try_get_angle(l2, ANGLE_NAME_CANDIDATES["sza"], lat.shape)
            vza = try_get_angle(l2, ANGLE_NAME_CANDIDATES["vza"], lat.shape)
            saa = try_get_angle(l2, ANGLE_NAME_CANDIDATES["saa"], lat.shape)
            vaa = try_get_angle(l2, ANGLE_NAME_CANDIDATES["vaa"], lat.shape)
            raa = angle_normalize180(vaa - saa)

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
                sza,
                vza,
                raa,
            ]
            lat, lon, arrays = safe_subset(lat, lon, arrays_to_subset, bbox)
            if lat.size == 0:
                print("[SKIP] no pixels in bbox for this granule.")
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
                sza,
                vza,
                raa,
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

            gattrs = {
                "title": "Dust AOD derived from MODIS Deep Blue Level-2 by XGBoost",
                "platform": "Aqua (MYD)" if args.satellite == "MYD" else "Terra (MOD)",
                "source": short_name,
                "date": iso,
                "bbox": str(bbox) if bbox else "global",
                "model_json": str(args.model_json),
                "model_meta": str(args.model_meta),
                "target_wavelength_nm": str(meta.get("target_wavelength_nm", 550)),
                "feature_columns": ",".join(meta.get("feature_columns", [])),
                "history": f"generated on {dt.datetime.utcnow().isoformat()}Z",
            }
            ds = build_dataset_l2(lat, lon, dust_aod, valid_mask, subset_features, sza, vza, raa, gattrs)
            ds.to_netcdf(out_nc, format="NETCDF4", encoding=ENC)
            print(f"[WROTE] {out_nc}")

            if args.upload:
                key = f"{args.s3_prefix}/{out_nc.name}"
                try:
                    upload_with_verify(str(out_nc), args.s3_bucket, key)
                    print(f"[UPLOADED] s3://{args.s3_bucket}/{key}")
                    os.remove(out_nc)
                    print(f"[CLEANED]  {out_nc}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] upload failed; keeping local file. {exc}")

            try:
                os.remove(local_hdf)
            except OSError:
                pass

        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
