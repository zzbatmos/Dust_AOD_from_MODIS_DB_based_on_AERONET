from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INV_DIR = Path("/home/ec2-user/Research/AERONET_INV_Level2_v3.0_allsites")
SHIN_2019_SITES = [
    "Beijing",
    "XiangHe",
    "Seoul_SNU",
    "Yonsei_University",
    "Gosan_SNU",
    "Osaka",
    "Shirahama",
    "Capo_Verde",
    "Banizoumbou",
    "Dakar",
    "Abracos_Hill",
    "Mongu",
    "Alta_Floresta",
    "GSFC",
    "Ispra",
    "Mexico_City",
]


def read_aeronet_to_dict(file_path: str | Path) -> dict[str, np.ndarray] | None:
    """Read one AERONET inversion file and keep the fields used in the notebook."""
    target_columns = [
        "Date(dd:mm:yyyy)",
        "Time(hh:mm:ss)",
        "Day_of_Year",
        "Day_of_Year(Fraction)",
        "AOD_Coincident_Input[440nm]",
        "AOD_Coincident_Input[675nm]",
        "AOD_Coincident_Input[870nm]",
        "AOD_Coincident_Input[1020nm]",
        "Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD",
        "AOD_Extinction-Total[440nm]",
        "AOD_Extinction-Total[675nm]",
        "AOD_Extinction-Total[870nm]",
        "AOD_Extinction-Total[1020nm]",
        "AOD_Extinction-Fine[440nm]",
        "AOD_Extinction-Fine[675nm]",
        "AOD_Extinction-Fine[870nm]",
        "AOD_Extinction-Fine[1020nm]",
        "AOD_Extinction-Coarse[440nm]",
        "AOD_Extinction-Coarse[675nm]",
        "AOD_Extinction-Coarse[870nm]",
        "AOD_Extinction-Coarse[1020nm]",
        "Extinction_Angstrom_Exponent_440-870nm-Total",
        "Single_Scattering_Albedo[440nm]",
        "Single_Scattering_Albedo[675nm]",
        "Single_Scattering_Albedo[870nm]",
        "Single_Scattering_Albedo[1020nm]",
        "Latitude(Degrees)",
        "Longitude(Degrees)",
        "Lidar_Ratio[440nm]",
        "Lidar_Ratio[675nm]",
        "Lidar_Ratio[870nm]",
        "Lidar_Ratio[1020nm]",
        "Depolarization_Ratio[440nm]",
        "Depolarization_Ratio[675nm]",
        "Depolarization_Ratio[870nm]",
        "Depolarization_Ratio[1020nm]",
    ]
    try:
        df = pd.read_csv(
            file_path,
            skiprows=6,
            na_values=[-999, -999.0],
            encoding="latin-1",
        )
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        return None
    return {col: df[col].to_numpy() for col in target_columns}


def load_aeronet_inversion_data(
    inv_dir: str | Path = DEFAULT_INV_DIR,
    requested_sites: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    inv_dir = Path(inv_dir)
    requested = set(requested_sites) if requested_sites else None
    site_data: dict[str, dict[str, np.ndarray]] = {}
    for file_path in sorted(inv_dir.glob("*.all")):
        site_name = file_path.name.split("_", 2)[2].replace(".all", "")
        if requested is not None and site_name not in requested:
            continue
        data = read_aeronet_to_dict(file_path)
        if data is not None:
            site_data[site_name] = data
    return site_data


def resolve_site_list(all_sites: Iterable[str], site_mode: str, explicit_sites: Iterable[str] | None) -> list[str]:
    all_sites = list(all_sites)
    if explicit_sites:
        selected = [site for site in explicit_sites if site in all_sites]
    elif site_mode == "shin2019":
        selected = [site for site in SHIN_2019_SITES if site in all_sites]
    else:
        selected = all_sites
    return selected


def dust_fraction_from_pldr(pldr: np.ndarray, pldr_nd: float = 0.02, pldr_d: float = 0.30) -> np.ndarray:
    """Shin et al. (2019) Eq. 3 with notebook defaults."""
    pldr = np.asarray(pldr, dtype=float)
    rd = np.full_like(pldr, np.nan, dtype=float)
    valid = np.isfinite(pldr)
    denom = pldr_d - pldr_nd
    rd[valid] = ((pldr[valid] - pldr_nd) / denom) * ((1.0 + pldr_d) / (1.0 + pldr[valid]))
    rd = np.where(pldr < pldr_nd, 0.0, rd)
    rd = np.where(pldr > pldr_d, 1.0, rd)
    return np.clip(rd, 0.0, 1.0)


def wavelength_columns(wavelength_nm: int) -> dict[str, str]:
    wl = int(wavelength_nm)
    return {
        "pldr": f"Depolarization_Ratio[{wl}nm]",
        "lr": f"Lidar_Ratio[{wl}nm]",
        "aod": f"AOD_Coincident_Input[{wl}nm]",
        "ext_coarse": f"AOD_Extinction-Coarse[{wl}nm]",
        "ext_total": f"AOD_Extinction-Total[{wl}nm]",
    }


@dataclass
class DustDominantStats:
    wavelength_nm: int
    variable: str
    count: int
    mean: float
    std: float
    median: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "wavelength_nm": self.wavelength_nm,
            "variable": self.variable,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
        }


def summarize_array(values: np.ndarray, wavelength_nm: int, variable: str) -> DustDominantStats:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return DustDominantStats(
        wavelength_nm=int(wavelength_nm),
        variable=variable,
        count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
    )


def collect_dust_dominant_samples(
    aeronet_data: dict[str, dict[str, np.ndarray]],
    site_list: Iterable[str],
    wavelengths: Iterable[int],
    dust_threshold: float = 0.89,
    pldr_nd: float = 0.02,
    pldr_d: float = 0.30,
) -> pd.DataFrame:
    """Use 1020 nm PLDR to identify dust-dominant samples, matching the notebook."""
    records: list[dict[str, float | int | str]] = []
    wavelengths = [int(wl) for wl in wavelengths]
    for site in site_list:
        data = aeronet_data[site]
        pldr_1020 = np.asarray(data["Depolarization_Ratio[1020nm]"], dtype=float)
        valid = np.isfinite(pldr_1020)
        for wl in wavelengths:
            cols = wavelength_columns(wl)
            valid &= np.isfinite(np.asarray(data[cols["pldr"]], dtype=float))
            valid &= np.isfinite(np.asarray(data[cols["lr"]], dtype=float))
        if not np.any(valid):
            continue
        rd = dust_fraction_from_pldr(pldr_1020[valid], pldr_nd=pldr_nd, pldr_d=pldr_d)
        dust_mask = rd > dust_threshold
        if not np.any(dust_mask):
            continue
        original_index = np.where(valid)[0][dust_mask]
        for idx_pos, original_idx in enumerate(original_index):
            rec: dict[str, float | int | str] = {
                "Site": site,
                "orig_index": int(original_idx),
                "Rd_1020": float(rd[dust_mask][idx_pos]),
            }
            for wl in wavelengths:
                cols = wavelength_columns(wl)
                rec[f"DPR_{wl}"] = float(data[cols["pldr"]][original_idx])
                rec[f"LR_{wl}"] = float(data[cols["lr"]][original_idx])
            records.append(rec)
    return pd.DataFrame(records)


def derive_dust_aod_dataframe(
    aeronet_data: dict[str, dict[str, np.ndarray]],
    site_list: Iterable[str],
    wavelengths: Iterable[int],
    dust_lidar_ratio_by_wavelength: dict[int, float],
    pldr_nd: float = 0.02,
    pldr_d: float = 0.30,
) -> pd.DataFrame:
    """Replicate notebook dust-AOD derivation for one or more wavelengths."""
    per_site_frames: list[pd.DataFrame] = []
    for wl in wavelengths:
        cols = wavelength_columns(wl)
        dust_lr = float(dust_lidar_ratio_by_wavelength[int(wl)])
        wavelength_records: list[pd.DataFrame] = []
        for site in site_list:
            data = aeronet_data[site]
            pldr = np.asarray(data[cols["pldr"]], dtype=float)
            lr = np.asarray(data[cols["lr"]], dtype=float)
            aod = np.asarray(data[cols["aod"]], dtype=float)
            ext_coarse = np.asarray(data[cols["ext_coarse"]], dtype=float)
            ext_total = np.asarray(data[cols["ext_total"]], dtype=float)
            mask = (
                np.isfinite(pldr)
                & np.isfinite(lr)
                & np.isfinite(aod)
                & np.isfinite(ext_coarse)
                & np.isfinite(ext_total)
                & (lr > 0.0)
                & (aod >= 0.0)
                & (ext_total > 0.0)
            )
            if not np.any(mask):
                continue
            rd = dust_fraction_from_pldr(pldr[mask], pldr_nd=pldr_nd, pldr_d=pldr_d)
            dust_fraction = np.clip(rd * dust_lr / lr[mask], 0.0, 1.0)
            coarse_mode_fraction = np.clip(ext_coarse[mask] / ext_total[mask], 0.0, 1.0)
            site_df = pd.DataFrame(
                {
                    "Site": site,
                    "wavelength_nm": int(wl),
                    "orig_index": np.where(mask)[0],
                    "PLDR": pldr[mask],
                    "LR": lr[mask],
                    "AOD": aod[mask],
                    "dust_fraction": dust_fraction,
                    "dust_AOD": aod[mask] * dust_fraction,
                    "coarse_mode_fraction": coarse_mode_fraction,
                    "coarse_AOD": aod[mask] * coarse_mode_fraction,
                    "dust_lidar_ratio": dust_lr,
                }
            )
            wavelength_records.append(site_df)
        if wavelength_records:
            per_site_frames.append(pd.concat(wavelength_records, ignore_index=True))
    if not per_site_frames:
        return pd.DataFrame()
    return pd.concat(per_site_frames, ignore_index=True)


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2))
