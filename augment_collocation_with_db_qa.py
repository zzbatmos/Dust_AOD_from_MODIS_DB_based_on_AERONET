from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
from pyhdf.SD import SD, SDC

sys.path.insert(0, str(Path("..").resolve()))
import earthaccess  # type: ignore
import MODIS_Lib  # type: ignore
from AERONET_MODIS_Collocation_v3 import extract_deep_blue_within_25km  # type: ignore


AEROSOL_TYPE_LABELS = {0: "Mixed", 1: "Dust", 2: "Smoke", 3: "Sulfate"}
ALGORITHM_FLAG_LABELS = {0: "DeepBlue", 1: "Vegetated", 2: "Mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment an existing per-site MODIS DB collocation pickle with raw and "
            "decoded DB QA variables while preserving the original DB collocation stats."
        )
    )
    parser.add_argument(
        "--site-file",
        type=Path,
        default=Path("../AERONET_MODIS_DB_collocation_files/AERONET_MOD04_L2_collocation_Capo_Verde.pkl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("AERONET_MOD04_L2_collocation_Capo_Verde_with_QA.pkl"),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("./temp_downloads"),
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded granules instead of deleting them after processing.",
    )
    return parser.parse_args()


def histogram_dict(values: np.ndarray, labels: dict[int, str]) -> dict[str, dict[str, float | int]]:
    values = np.asarray(values)
    out: dict[str, dict[str, float | int]] = {}
    total = int(values.size)
    for code, label in labels.items():
        count = int(np.count_nonzero(values == code))
        out[label] = {
            "code": int(code),
            "count": count,
            "fraction": float(count / total) if total > 0 else float("nan"),
        }
    return out


def dominant_label(values: np.ndarray, labels: dict[int, str]) -> str | None:
    values = np.asarray(values)
    if values.size == 0:
        return None
    uniq, counts = np.unique(values, return_counts=True)
    if uniq.size == 0:
        return None
    code = int(uniq[np.argmax(counts)])
    return labels.get(code, str(code))


def decode_quality_assurance_land(byte4: np.ndarray) -> dict[str, np.ndarray]:
    byte4 = np.asarray(byte4, dtype=np.uint8)
    usefulness = (byte4 & 0b1).astype(np.uint8)
    confidence = ((byte4 >> 1) & 0b11).astype(np.uint8)
    aerosol_type = ((byte4 >> 3) & 0b11).astype(np.uint8)
    return {
        "usefulness": usefulness,
        "confidence": confidence,
        "aerosol_type": aerosol_type,
    }


def load_qa_fields(granule_path: str) -> tuple[np.ndarray, np.ndarray]:
    sd = SD(granule_path, SDC.READ)
    qa_land = np.asarray(sd.select("Quality_Assurance_Land").get(), dtype=np.int16)
    algorithm_flag = np.asarray(sd.select("Deep_Blue_Algorithm_Flag_Land").get(), dtype=np.int16)
    sd.end()
    return qa_land, algorithm_flag


def search_and_download_granule(
    granule: str,
    download_dir: Path,
    max_attempts: int = 4,
    sleep_seconds: float = 3.0,
) -> tuple[str | None, str | None]:
    short_name = granule.split(".")[0]
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            results = earthaccess.search_data(short_name=short_name, granule_name=granule)
            if not results:
                return None, "Granule not found"
            files = earthaccess.download(results[0], local_path=str(download_dir))
            if not files:
                last_error = "Download failed"
            else:
                return str(files[0]), None
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
        if attempt < max_attempts:
            time.sleep(sleep_seconds * attempt)
            earthaccess.login(strategy="netrc")
    return None, last_error or "Unknown Earthdata error"


def summarize_masked_qa(
    qa_land_masked: np.ndarray,
    algorithm_flag_masked: np.ndarray,
    lat_masked: np.ndarray,
    lon_masked: np.ndarray,
    dist_km_masked: np.ndarray,
) -> dict[str, object]:
    byte4 = qa_land_masked[:, 4].astype(np.uint8)
    decoded = decode_quality_assurance_land(byte4)
    useful_mask = decoded["usefulness"] == 1
    useful_types = decoded["aerosol_type"][useful_mask]
    useful_conf = decoded["confidence"][useful_mask]
    useful_alg = algorithm_flag_masked[useful_mask]

    summary: dict[str, object] = {
        "pixel_count_masked": int(qa_land_masked.shape[0]),
        "qa_raw_masked_shape": [int(x) for x in qa_land_masked.shape],
        "usefulness_histogram": histogram_dict(decoded["usefulness"], {0: "not_useful", 1: "useful"}),
        "confidence_histogram_all": histogram_dict(decoded["confidence"], {0: "conf_0", 1: "conf_1", 2: "conf_2", 3: "conf_3"}),
        "confidence_histogram_useful": histogram_dict(useful_conf, {0: "conf_0", 1: "conf_1", 2: "conf_2", 3: "conf_3"}),
        "aerosol_type_histogram_all": histogram_dict(decoded["aerosol_type"], AEROSOL_TYPE_LABELS),
        "aerosol_type_histogram_useful": histogram_dict(useful_types, AEROSOL_TYPE_LABELS),
        "algorithm_flag_histogram_all": histogram_dict(algorithm_flag_masked, ALGORITHM_FLAG_LABELS),
        "algorithm_flag_histogram_useful": histogram_dict(useful_alg, ALGORITHM_FLAG_LABELS),
        "dominant_aerosol_type_all": dominant_label(decoded["aerosol_type"], AEROSOL_TYPE_LABELS),
        "dominant_aerosol_type_useful": dominant_label(useful_types, AEROSOL_TYPE_LABELS),
        "dominant_algorithm_flag_all": dominant_label(algorithm_flag_masked, ALGORITHM_FLAG_LABELS),
        "dominant_algorithm_flag_useful": dominant_label(useful_alg, ALGORITHM_FLAG_LABELS),
        "mean_distance_km": float(np.nanmean(dist_km_masked)) if dist_km_masked.size else float("nan"),
        "mean_lat": float(np.nanmean(lat_masked)) if lat_masked.size else float("nan"),
        "mean_lon": float(np.nanmean(lon_masked)) if lon_masked.size else float("nan"),
    }
    return {
        "summary": summary,
        "decoded_arrays": {
            "usefulness": decoded["usefulness"],
            "confidence": decoded["confidence"],
            "aerosol_type": decoded["aerosol_type"],
            "algorithm_flag": algorithm_flag_masked.astype(np.int16),
            "lat": np.asarray(lat_masked, dtype=np.float32),
            "lon": np.asarray(lon_masked, dtype=np.float32),
            "dist_km": np.asarray(dist_km_masked, dtype=np.float32),
        },
    }


def augment_one_granule(entry: dict, granule_name: str, local_file: str, radius_km: float) -> dict:
    mod = MODIS_Lib.MODIS_DeepBlue_Level2(local_file, sub_sample=1)
    stats = extract_deep_blue_within_25km(
        entry["Latitude(Degrees)"],
        entry["Longitude(Degrees)"],
        mod,
        radius_km=radius_km,
        return_mask=True,
    )
    qa_land, algorithm_flag = load_qa_fields(local_file)
    mask = np.asarray(stats["mask"], dtype=bool)
    lat = np.asarray(mod.lat, dtype=float)
    lon = np.asarray(mod.lon, dtype=float)
    dist_km = np.asarray(stats["dist_km"], dtype=float)
    qa_masked = qa_land[mask]
    alg_masked = algorithm_flag[mask]
    qa_bundle = summarize_masked_qa(
        qa_land_masked=qa_masked,
        algorithm_flag_masked=alg_masked,
        lat_masked=lat[mask],
        lon_masked=lon[mask],
        dist_km_masked=dist_km[mask],
    )

    base = {
        "granule_name": granule_name,
        "pixel_count": int(stats.get("pixel_count", 0)),
        "aod550": stats.get("Deep_Blue_Aerosol_Optical_Depth_550_Land"),
        "stats_full": {
            k: v for k, v in stats.items() if k not in {"mask", "dist_km"}
        },
    }
    base["qa"] = {
        "summary": qa_bundle["summary"],
        "raw_quality_assurance_land_masked": qa_masked.astype(np.int16),
        "raw_algorithm_flag_masked": alg_masked.astype(np.int16),
        "decoded_masked": qa_bundle["decoded_arrays"],
    }
    return base


def augment_site_file(
    site_file: Path,
    output_file: Path,
    download_dir: Path,
    radius_km: float = 25.0,
    keep_downloads: bool = False,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with site_file.open("rb") as fh:
        colloc = pickle.load(fh)

    output: dict[str, dict] = {}
    for idx, (granule, entry) in enumerate(colloc.items(), start=1):
        print(f"[{idx}/{len(colloc)}] {granule}")
        local_file, error = search_and_download_granule(granule, download_dir)
        if local_file is None:
            out_entry = dict(entry)
            out_entry["Deep_Blue_data_with_QA"] = {"granule_name": granule, "error": error}
            output[granule] = out_entry
            continue
        try:
            out_entry = dict(entry)
            out_entry["Deep_Blue_data_with_QA"] = augment_one_granule(entry, granule, local_file, radius_km)
            output[granule] = out_entry
        finally:
            if os.path.exists(local_file) and not keep_downloads:
                os.remove(local_file)

    with output_file.open("wb") as fh:
        pickle.dump(output, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output_file}")


def main() -> None:
    args = parse_args()
    earthaccess.login(strategy="netrc")
    augment_site_file(
        site_file=args.site_file,
        output_file=args.output_file,
        download_dir=args.download_dir,
        radius_km=args.radius_km,
        keep_downloads=args.keep_downloads,
    )


if __name__ == "__main__":
    main()
