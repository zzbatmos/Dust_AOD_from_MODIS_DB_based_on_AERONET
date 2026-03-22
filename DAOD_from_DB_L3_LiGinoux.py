#!/usr/bin/env python3

"""Apply the Li-Ginoux L3 Deep Blue dust-AOD parameterization over an arbitrary date range."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import boto3
import earthaccess
import numpy as np
import xarray as xr

REPO_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = REPO_DIR.parent
sys.path.insert(0, str(RESEARCH_DIR))

import MODIS_Lib  # noqa: E402
from AWS_Utils import get_NASA_creds  # noqa: E402

OUT_LOCAL_DIR = REPO_DIR / "MODIS_DB_L3_DustAOD_LiGinoux"
S3_BUCKET = "zhibo-zhang-bucket"
S3_PREFIX = "Data/MODIS_DB_L3_DustAOD_LiGinoux"

ENC = {
    "dust_aod": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "db_aod_550": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "angstrom_exponent": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "fmf": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
    "lat": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "lon": {"zlib": True, "complevel": 4, "dtype": "float32"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Li-Ginoux DAOD to MODIS DB L3 files.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--short-name", choices=["MOD08_D3", "MYD08_D3"], default="MOD08_D3")
    parser.add_argument("--platform", choices=["MOD", "MYD", "MEAN"], default="MOD")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--sub-sample", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT_LOCAL_DIR)
    parser.add_argument("--s3-bucket", default=S3_BUCKET)
    parser.add_argument("--s3-prefix", default=S3_PREFIX)
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


def parse_l3_filename(hdf_name: str) -> dt.date:
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


def pick_l3_db_fields(db_obj) -> tuple[str, str]:
    names = [name for name in dir(db_obj) if name.lower().startswith("deep_blue")]
    def find_one(key: str) -> str:
        candidates = [name for name in names if key.lower() in name.lower() and "_Land_" in name and name.endswith("_Mean")]
        if not candidates:
            raise AttributeError(f"Could not find L3 field for {key}")
        return candidates[0]
    return find_one("Aerosol_Optical_Depth_550"), find_one("Angstrom_Exponent")


def enforce_daod_output_semantics(dust_aod: np.ndarray, total_aod: np.ndarray) -> np.ndarray:
    total_aod = np.asarray(total_aod, dtype=float)
    dust_aod = np.asarray(dust_aod, dtype=float).copy()
    total_valid = np.isfinite(total_aod) & (total_aod > 0.0)
    dust_aod[~np.isfinite(total_aod)] = np.nan
    dust_aod[total_valid & ~np.isfinite(dust_aod)] = 0.0
    return dust_aod


def derive_dust_aod_from_db(
    db_obj,
    platform: str,
    min_aod: float = 0.05,
    ae_clip: tuple[float, float] = (0.0, 1.8),
    cap_fmf_at_07: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aod_attr, ae_attr = pick_l3_db_fields(db_obj)
    aod_raw = np.asarray(getattr(db_obj, aod_attr), dtype=np.float64)
    alpha = np.asarray(getattr(db_obj, ae_attr), dtype=np.float64)
    alpha = np.clip(alpha, ae_clip[0], ae_clip[1])

    plat = platform.upper()
    if plat == "MYD":
        a, b, c = 0.052, 0.333, 0.082
    elif plat == "MOD":
        a, b, c = 0.051, 0.338, 0.087
    elif plat == "MEAN":
        a, b, c = 0.051, 0.336, 0.085
    else:
        raise ValueError("platform must be MOD, MYD, or MEAN")

    fmf = np.clip(a + b * alpha + c * alpha**2, 0.0, 1.0)
    if cap_fmf_at_07:
        fmf = np.minimum(fmf, 0.7)

    ssa_arr = np.asarray(getattr(db_obj, "Deep_Blue_Single_Scattering_Albedo_Land_Mean"), dtype=np.float64)
    ssa412 = ssa_arr[0, ...]
    ssa670 = ssa_arr[2, ...]
    ssa_ok = (
        np.isfinite(ssa412)
        & np.isfinite(ssa670)
        & (ssa412 >= 0.0)
        & (ssa412 <= 1.0)
        & (ssa670 >= 0.0)
        & (ssa670 <= 1.0)
        & (ssa670 > ssa412)
    )
    cond_ok = np.isfinite(alpha) & (aod_raw >= min_aod) & ssa_ok
    dust_out = np.full_like(aod_raw, np.nan, dtype=np.float64)
    dust_out[cond_ok] = ((1.0 - fmf) * aod_raw)[cond_ok]
    dust_out = enforce_daod_output_semantics(dust_out, aod_raw)

    aod_out = np.where(np.isfinite(aod_raw), aod_raw, np.nan)
    fmf_out = np.where(np.isfinite(alpha), fmf, np.nan)
    alpha_out = np.where(np.isfinite(alpha), alpha, np.nan)
    return dust_out, fmf_out, aod_out, alpha_out


def build_dataset(lat: np.ndarray, lon: np.ndarray, dust_aod: np.ndarray, aod: np.ndarray, alpha: np.ndarray, fmf: np.ndarray, attrs: dict) -> xr.Dataset:
    ny, nx = dust_aod.shape
    return xr.Dataset(
        data_vars=dict(
            dust_aod=(("y", "x"), dust_aod.astype("float32"), {"long_name": "Dust AOD at 550 nm", "units": "1"}),
            db_aod_550=(("y", "x"), aod.astype("float32"), {"long_name": "Deep Blue AOD at 550 nm", "units": "1"}),
            angstrom_exponent=(("y", "x"), alpha.astype("float32"), {"long_name": "Deep Blue Angstrom exponent", "units": "1"}),
            fmf=(("y", "x"), fmf.astype("float32"), {"long_name": "Fine mode fraction", "units": "1"}),
            lat=(("y", "x"), lat.astype("float32"), {"standard_name": "latitude", "units": "degrees_north"}),
            lon=(("y", "x"), lon.astype("float32"), {"standard_name": "longitude", "units": "degrees_east"}),
        ),
        coords=dict(y=np.arange(ny), x=np.arange(nx)),
        attrs=attrs,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

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
                fdate = parse_l3_filename(fname)
                out_name = f"{args.short_name}_DustAOD_LiGinoux_{fdate.isoformat()}.nc"
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
                dust_aod, fmf, aod, alpha = derive_dust_aod_from_db(l3, platform=args.platform)
                attrs = {
                    "title": f"Daily dust AOD from MODIS Deep Blue {args.short_name} by Li-Ginoux",
                    "source": args.short_name,
                    "processing": "FMF(AE) parameterization plus SSA670 > SSA412 screen",
                    "platform": args.platform,
                    "history": f"Generated on {dt.datetime.utcnow().isoformat()}Z",
                }
                ds = build_dataset(l3.lat, l3.lon, dust_aod, aod, alpha, fmf, attrs)
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
