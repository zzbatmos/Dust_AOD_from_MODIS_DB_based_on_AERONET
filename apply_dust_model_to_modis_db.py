#!/usr/bin/env python3
"""Apply trained MODIS Deep Blue dust-fraction models to a MODIS granule.

This script extracts and generalizes the notebook section
"Apply Trained XGBoost model to DB". It can:

1. Search/download a MOD04/MYD04 granule through the Earthdata access API.
2. Read the Deep Blue retrieval fields from the granule.
3. Apply one or more trained XGBoost models to predict dust fraction.
4. Derive dust AOD from the predicted fraction and a selected MODIS AOD band.
5. Save map PNGs for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import earthaccess

sys.path.insert(0, str((Path(__file__).resolve().parent.parent)))
import MODIS_Lib  # noqa: E402


@dataclass
class ModelSpec:
    name: str
    model_path: Path
    meta_path: Path
    output_kind: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply trained dust-fraction XGBoost models to a MODIS DB granule."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to a local MOD04_L2/MYD04_L2 HDF granule. If omitted, Earthaccess search/download is used.",
    )
    parser.add_argument(
        "--short-name",
        default="MOD04_L2",
        help="Earthdata short name to search, e.g. MOD04_L2 or MYD04_L2.",
    )
    parser.add_argument(
        "--start",
        default="2025-03-14 17:00:00",
        help="Search start time in UTC, format 'YYYY-MM-DD HH:MM:SS'.",
    )
    parser.add_argument(
        "--end",
        default="2025-03-14 17:10:00",
        help="Search end time in UTC, format 'YYYY-MM-DD HH:MM:SS'.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-101.112,
        help="Longitude for Earthaccess point search.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=33.98,
        help="Latitude for Earthaccess point search.",
    )
    parser.add_argument(
        "--result-index",
        type=int,
        default=0,
        help="Which Earthaccess search result to download.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=here / "temp_downloads",
        help="Directory for downloaded granules.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "dust_model_application_2025-03-14",
        help="Directory for output PNGs and summary files.",
    )
    parser.add_argument(
        "--daod-source-band",
        choices=["550", "660"],
        default="660",
        help="MODIS DB AOD band used as the total-AOD proxy when deriving dust AOD from predicted dust fraction.",
    )
    return parser.parse_args()


def inv_log1p_transform(y_log: np.ndarray, eps: float = 0.0) -> np.ndarray:
    y_log = np.asarray(y_log, dtype=float)
    y_lin = np.expm1(y_log) - eps
    return np.clip(y_lin, 0.0, None)


def inv_logit_transform(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    positive = z >= 0
    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return np.clip(out, 0.0, 1.0)


def load_model_spec(model_path: Path, meta_path: Path) -> ModelSpec:
    return ModelSpec(
        name=model_path.stem.replace("dust_aod_xgb_", ""),
        model_path=model_path,
        meta_path=meta_path,
        output_kind="dust_fraction",
    )


def default_model_specs(here: Path) -> list[ModelSpec]:
    pairs = [
        ("log1p", here / "dust_aod_xgb_log1p.json", here / "dust_aod_xgb_log1p_meta.json", "dust_fraction"),
        ("logit", here / "dust_aod_xgb_logit.json", here / "dust_aod_xgb_logit_meta.json", "dust_fraction"),
        (
            "log1p_weighted_bins",
            here / "dust_aod_xgb_log1p_weighted_bins.json",
            here / "dust_aod_xgb_log1p_weighted_bins_meta.json",
            "dust_fraction",
        ),
        (
            "dust_aod550_log1p",
            here / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_xgb.json",
            here / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_metadata.json",
            "dust_aod",
        ),
    ]
    specs = []
    for name, model_path, meta_path, output_kind in pairs:
        if model_path.exists() and meta_path.exists():
            specs.append(
                ModelSpec(
                    name=name,
                    model_path=model_path,
                    meta_path=meta_path,
                    output_kind=output_kind,
                )
            )
    if not specs:
        raise FileNotFoundError("No trained model/metadata pairs found in the current workspace.")
    return specs


def search_and_download_granule(args: argparse.Namespace) -> Path:
    args.download_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="netrc")
    granules = earthaccess.search_data(
        short_name=args.short_name,
        temporal=(args.start, args.end),
        point=(args.lon, args.lat),
    )
    if not granules:
        raise RuntimeError("Earthaccess search returned no granules for the requested query.")
    if args.result_index >= len(granules):
        raise IndexError(
            f"Requested result index {args.result_index} but only {len(granules)} granule(s) were found."
        )
    target = granules[args.result_index]
    downloaded = earthaccess.download(target, local_path=str(args.download_dir))
    if not downloaded:
        raise RuntimeError("Earthaccess download returned no local file path.")
    return Path(downloaded[0]).resolve()


def read_granule_features(granule_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    granule = MODIS_Lib.MODIS_DeepBlue_Level2(str(granule_path))
    features_2d = {
        "MODIS_DB_AOD550": np.asarray(granule.Deep_Blue_Aerosol_Optical_Depth_550_Land, dtype=float),
        "MODIS_DB_AOD412": np.asarray(granule.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[0], dtype=float),
        "MODIS_DB_AOD470": np.asarray(granule.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[1], dtype=float),
        "MODIS_DB_AOD660": np.asarray(granule.Deep_Blue_Spectral_Aerosol_Optical_Depth_Land[2], dtype=float),
        "MODIS_DB_AE": np.asarray(granule.Deep_Blue_Angstrom_Exponent_Land, dtype=float),
        "MODIS_DB_SSA412": np.asarray(granule.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[0], dtype=float),
        "MODIS_DB_SSA470": np.asarray(granule.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[1], dtype=float),
        "MODIS_DB_SSA660": np.asarray(granule.Deep_Blue_Spectral_Single_Scattering_Albedo_Land[2], dtype=float),
    }
    lat = np.asarray(granule.lat, dtype=float)
    lon = np.asarray(granule.lon, dtype=float)
    return features_2d, lat, lon


def load_model_and_meta(spec: ModelSpec) -> tuple[XGBRegressor, dict]:
    model = XGBRegressor()
    model.load_model(spec.model_path)
    with spec.meta_path.open("r") as handle:
        meta = json.load(handle)
    return model, meta


def predict_model_grid(
    model: XGBRegressor,
    meta: dict,
    features_2d: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    feature_order = meta.get("feature_names", meta.get("feature_columns"))
    if feature_order is None:
        raise KeyError("Model metadata is missing 'feature_names'/'feature_columns'.")
    target_transform = meta.get("target_transform", "log1p")
    eps = float(meta.get("eps", 0.0))

    h, w = np.asarray(next(iter(features_2d.values()))).shape
    for name, arr in features_2d.items():
        if np.asarray(arr).shape != (h, w):
            raise ValueError(f"{name} has shape {np.asarray(arr).shape}, expected {(h, w)}")

    missing = [col for col in feature_order if col not in features_2d]
    if missing:
        raise KeyError(f"Missing required features for model: {missing}")

    x_flat = np.column_stack([np.asarray(features_2d[col], dtype=float).ravel() for col in feature_order])
    valid = np.isfinite(x_flat).all(axis=1)
    pred_flat = np.full(h * w, np.nan, dtype=np.float32)

    if np.any(valid):
        x_valid = pd.DataFrame(x_flat[valid], columns=feature_order)
        y_pred = model.predict(x_valid)
        if target_transform == "log1p":
            pred = inv_log1p_transform(y_pred, eps=eps)
        elif target_transform == "logit":
            pred = inv_logit_transform(y_pred)
        else:
            raise ValueError(f"Unsupported target transform in metadata: {target_transform}")
        pred_flat[valid] = pred.astype(np.float32)

    return pred_flat.reshape(h, w), valid.reshape(h, w)


def derive_dust_aod(
    dust_fraction: np.ndarray,
    features_2d: dict[str, np.ndarray],
    source_band: str,
) -> np.ndarray:
    total_aod = np.asarray(features_2d[f"MODIS_DB_AOD{source_band}"], dtype=float)
    dust_aod = np.asarray(dust_fraction, dtype=float) * total_aod
    dust_aod[~np.isfinite(total_aod)] = np.nan
    return dust_aod


def clip_fraction_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[np.isfinite(arr)] = np.clip(arr[np.isfinite(arr)], 0.0, 1.0)
    return arr


def li_ginoux_fmf_from_ae(ae: np.ndarray) -> np.ndarray:
    """Eq. 7 in Li and Ginoux (2025): FMF = 0.085*AE^2 + 0.336*AE + 0.051."""
    ae = np.asarray(ae, dtype=float)
    fmf = 0.085 * ae**2 + 0.336 * ae + 0.051
    return np.clip(fmf, 0.0, 1.0)


def li_ginoux_dust_aod_with_ssa_constraint(
    features_2d: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate coarse/dust AOD from MODIS DB AE following Li and Ginoux.

    FMF is derived from MODIS DB AE using Eq. 7. Dust AOD is approximated as:
      (1 - FMF) * AOD550
    and retained only where SSA412 < SSA470.
    """
    ae = np.asarray(features_2d["MODIS_DB_AE"], dtype=float)
    aod550 = np.asarray(features_2d["MODIS_DB_AOD550"], dtype=float)
    ssa412 = np.asarray(features_2d["MODIS_DB_SSA412"], dtype=float)
    ssa470 = np.asarray(features_2d["MODIS_DB_SSA470"], dtype=float)

    fmf = li_ginoux_fmf_from_ae(ae)
    coarse_fraction = 1.0 - fmf
    dust_aod = coarse_fraction * aod550

    valid = (
        np.isfinite(ae)
        & np.isfinite(aod550)
        & np.isfinite(ssa412)
        & np.isfinite(ssa470)
        & (ssa412 < ssa470)
    )
    fmf_out = np.where(valid, fmf, np.nan)
    coarse_fraction_out = np.where(valid, coarse_fraction, np.nan)
    dust_aod_out = np.where(valid, dust_aod, np.nan)
    return fmf_out, coarse_fraction_out, dust_aod_out


def save_scatter_map(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    z2d: np.ndarray,
    output_path: Path,
    title: str,
    cbar_label: str,
    is_fraction: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    extent_pad_deg: float = 0.8,
) -> dict[str, float]:
    lat = np.asarray(lat2d, dtype=float)
    lon = np.asarray(lon2d, dtype=float)
    z = np.asarray(z2d, dtype=float)
    lon = ((lon + 180.0) % 360.0) - 180.0

    lat_f = lat.ravel()
    lon_f = lon.ravel()
    z_f = z.ravel()
    mask = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(z_f)
    lat_f = lat_f[mask]
    lon_f = lon_f[mask]
    z_f = z_f[mask]
    if lat_f.size == 0:
        raise ValueError(f"No finite samples found for plot {output_path.name}.")

    if is_fraction:
        vvmin, vvmax = 0.0, 1.0
        cmap = "inferno"
    else:
        vvmin = float(np.nanpercentile(z_f, 2)) if vmin is None else vmin
        vvmax = float(np.nanpercentile(z_f, 98)) if vmax is None else vmax
        if not np.isfinite(vvmin) or not np.isfinite(vvmax) or vvmin == vvmax:
            vvmin = float(np.nanmin(z_f))
            vvmax = float(np.nanmax(z_f))
        cmap = "plasma"

    proj_data = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 6))
    ax = plt.axes(projection=proj_data)
    ax.coastlines(linewidth=0.8)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle=":")
    ax.set_extent(
        [lon_f.min() - extent_pad_deg, lon_f.max() + extent_pad_deg, lat_f.min() - extent_pad_deg, lat_f.max() + extent_pad_deg],
        crs=proj_data,
    )
    sc = ax.scatter(
        lon_f,
        lat_f,
        c=z_f,
        s=5,
        alpha=0.85,
        transform=proj_data,
        vmin=vvmin,
        vmax=vvmax,
        cmap=cmap,
        linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"N": int(lat_f.size), "vmin": vvmin, "vmax": vvmax}


def save_comparison_panel(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    panels: list[dict[str, object]],
    output_path: Path,
    figure_title: str,
    extent_pad_deg: float = 0.8,
) -> None:
    lat = np.asarray(lat2d, dtype=float)
    lon = ((np.asarray(lon2d, dtype=float) + 180.0) % 360.0) - 180.0
    finite_geo = np.isfinite(lat) & np.isfinite(lon)
    lat_f = lat[finite_geo]
    lon_f = lon[finite_geo]
    if lat_f.size == 0:
        raise ValueError("No finite geolocation points for comparison panel.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        z = np.asarray(panel["data"], dtype=float)
        z_f = z.ravel()
        mask = finite_geo.ravel() & np.isfinite(z_f)
        if not np.any(mask):
            ax.set_title(f"{panel['title']} (no valid data)")
            ax.coastlines(linewidth=0.8)
            continue

        lon_plot = lon.ravel()[mask]
        lat_plot = lat.ravel()[mask]
        z_plot = z_f[mask]
        is_fraction = bool(panel.get("is_fraction", False))
        if is_fraction:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = panel.get("vmin")
            vmax = panel.get("vmax")
            if vmin is None:
                vmin = float(np.nanpercentile(z_plot, 2))
            if vmax is None:
                vmax = float(np.nanpercentile(z_plot, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(z_plot))
                vmax = float(np.nanmax(z_plot))

        sc = ax.scatter(
            lon_plot,
            lat_plot,
            c=z_plot,
            s=5,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            cmap=panel.get("cmap", "plasma"),
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.coastlines(linewidth=0.8)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle=":")
        ax.set_extent(
            [
                lon_f.min() - extent_pad_deg,
                lon_f.max() + extent_pad_deg,
                lat_f.min() - extent_pad_deg,
                lat_f.max() + extent_pad_deg,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_title(str(panel["title"]))
        cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.82)
        cb.set_label(str(panel["label"]))

    fig.suptitle(figure_title, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_aod_scatter_comparison(
    ml_dust_aod: np.ndarray,
    li_ginoux_dust_aod: np.ndarray,
    output_path: Path,
    title: str,
) -> dict[str, float]:
    x = np.asarray(ml_dust_aod, dtype=float).ravel()
    y = np.asarray(li_ginoux_dust_aod, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise ValueError("Not enough paired finite samples for dust-AOD scatter comparison.")

    diff = y - x
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r = float(np.corrcoef(x, y)[0, 1])

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.03 * (hi - lo) if hi > lo else 0.1
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    hb = ax.hexbin(x, y, gridsize=60, mincnt=1, cmap="viridis")
    plt.colorbar(hb, ax=ax, label="count")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ML dust AOD")
    ax.set_ylabel("Li-Ginoux dust AOD proxy")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.98,
        f"N = {x.size}\n"
        f"r = {r:.3f}\n"
        f"Bias (LG-ML) = {bias:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"RMSE = {rmse:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"N": int(x.size), "Pearson_r": r, "bias": bias, "MAE": mae, "RMSE": rmse}


def summarize_grid(name: str, grid: np.ndarray) -> dict[str, float | int | str]:
    arr = np.asarray(grid, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return {"name": name, "valid_pixels": 0}
    vals = arr[valid]
    return {
        "name": name,
        "valid_pixels": int(vals.size),
        "min": float(np.nanmin(vals)),
        "p02": float(np.nanpercentile(vals, 2)),
        "median": float(np.nanmedian(vals)),
        "p98": float(np.nanpercentile(vals, 98)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
    }


def save_reference_feature_maps(
    granule_path: Path,
    output_dir: Path,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    features_2d: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    reference_specs = [
        ("MODIS_DB_AOD550", "MODIS Deep Blue AOD 550 nm", False),
        ("MODIS_DB_AOD412", "MODIS Deep Blue Spectral AOD 412 nm", False),
        ("MODIS_DB_AOD470", "MODIS Deep Blue Spectral AOD 470 nm", False),
        ("MODIS_DB_AOD660", "MODIS Deep Blue Spectral AOD 660 nm", False),
        ("MODIS_DB_AE", "MODIS Deep Blue Angstrom Exponent", False),
        ("MODIS_DB_SSA412", "MODIS Deep Blue SSA 412 nm", True),
        ("MODIS_DB_SSA470", "MODIS Deep Blue SSA 470 nm", True),
        ("MODIS_DB_SSA660", "MODIS Deep Blue SSA 660 nm", True),
    ]

    rows = []
    for feature_name, title, is_fraction in reference_specs:
        output_path = output_dir / f"{granule_path.stem}_{feature_name}.png"
        plot_stats = save_scatter_map(
            lat2d,
            lon2d,
            features_2d[feature_name],
            output_path,
            title=f"{granule_path.name}: {title}",
            cbar_label=feature_name,
            is_fraction=is_fraction,
        )
        rows.append(
            {
                "product_name": feature_name,
                "png_path": str(output_path),
                **{k: v for k, v in summarize_grid(feature_name, features_2d[feature_name]).items() if k != "name"},
                **{f"plot_{k}": v for k, v in plot_stats.items()},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    granule_path = args.input_file.resolve() if args.input_file else search_and_download_granule(args)
    if granule_path.parent != args.output_dir:
        try:
            shutil.copy2(granule_path, args.output_dir / granule_path.name)
        except Exception:
            pass

    features_2d, lat2d, lon2d = read_granule_features(granule_path)
    specs = default_model_specs(here)
    reference_rows = save_reference_feature_maps(
        granule_path=granule_path,
        output_dir=args.output_dir,
        lat2d=lat2d,
        lon2d=lon2d,
        features_2d=features_2d,
    )

    lg_fmf, lg_coarse_fraction, lg_dust_aod = li_ginoux_dust_aod_with_ssa_constraint(
        features_2d
    )
    lg_fmf_png = args.output_dir / f"{granule_path.stem}_li_ginoux_fmf_ssa_constraint.png"
    lg_dust_aod_png = (
        args.output_dir / f"{granule_path.stem}_li_ginoux_dust_aod_ssa_constraint.png"
    )
    lg_fmf_stats = save_scatter_map(
        lat2d,
        lon2d,
        lg_fmf,
        lg_fmf_png,
        title=f"{granule_path.name}: Li-Ginoux FMF with SSA412 < SSA470 constraint",
        cbar_label="Li-Ginoux FMF",
        is_fraction=True,
    )
    lg_dust_aod_stats = save_scatter_map(
        lat2d,
        lon2d,
        lg_dust_aod,
        lg_dust_aod_png,
        title=f"{granule_path.name}: Li-Ginoux coarse-mode dust AOD with SSA constraint",
        cbar_label="Li-Ginoux dust AOD proxy at 550 nm",
        is_fraction=False,
    )

    summary_rows = []
    for spec in specs:
        model, meta = load_model_and_meta(spec)
        pred_grid = predict_model_grid(model, meta, features_2d)[0]

        if spec.output_kind == "dust_fraction":
            dust_fraction = clip_fraction_grid(pred_grid)
            dust_aod = derive_dust_aod(dust_fraction, features_2d, source_band=args.daod_source_band)
            frac_png = args.output_dir / f"{granule_path.stem}_{spec.name}_dust_fraction.png"
            frac_stats = save_scatter_map(
                lat2d,
                lon2d,
                dust_fraction,
                frac_png,
                title=f"{granule_path.name} {spec.name}: predicted dust fraction",
                cbar_label="Predicted dust fraction",
                is_fraction=True,
            )
            dust_fraction_summary = summarize_grid("dust_fraction", dust_fraction)
            daod_suffix = args.daod_source_band
            daod_title = f"{granule_path.name} {spec.name}: derived dust AOD ({args.daod_source_band} nm proxy)"
            daod_label = f"Derived dust AOD ({args.daod_source_band} nm proxy)"
        elif spec.output_kind == "dust_aod":
            dust_fraction = None
            dust_aod = np.asarray(pred_grid, dtype=float)
            frac_png = None
            frac_stats = None
            dust_fraction_summary = {}
            daod_suffix = "550"
            daod_title = f"{granule_path.name} {spec.name}: predicted dust AOD (550 nm)"
            daod_label = "Predicted dust AOD (550 nm)"
        else:
            raise ValueError(f"Unsupported model output kind: {spec.output_kind}")

        daod_png = args.output_dir / f"{granule_path.stem}_{spec.name}_dust_aod_{daod_suffix}.png"
        daod_stats = save_scatter_map(
            lat2d,
            lon2d,
            dust_aod,
            daod_png,
            title=daod_title,
            cbar_label=daod_label,
            is_fraction=False,
        )

        summary_rows.append(
            {
                "model_name": spec.name,
                "output_kind": spec.output_kind,
                "granule": granule_path.name,
                "target_transform": meta.get("target_transform"),
                "weighting_name": meta.get("weighting_name"),
                "dust_fraction_png": str(frac_png) if frac_png is not None else None,
                "dust_aod_png": str(daod_png),
                **{f"dust_fraction_{k}": v for k, v in dust_fraction_summary.items() if k != "name"},
                **{f"dust_aod_{k}": v for k, v in summarize_grid("dust_aod", dust_aod).items() if k != "name"},
                **({f"dust_fraction_plot_{k}": v for k, v in frac_stats.items()} if frac_stats is not None else {}),
                **{f"dust_aod_plot_{k}": v for k, v in daod_stats.items()},
            }
        )

    summary_path = args.output_dir / f"{granule_path.stem}_model_application_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    reference_summary_path = args.output_dir / f"{granule_path.stem}_reference_products_summary.csv"
    pd.DataFrame(reference_rows).to_csv(reference_summary_path, index=False)
    li_ginoux_summary_path = args.output_dir / f"{granule_path.stem}_li_ginoux_summary.json"
    with li_ginoux_summary_path.open("w") as handle:
        json.dump(
            {
                "parameterization": "FMF = 0.085*AE^2 + 0.336*AE + 0.051",
                "reference": "Li and Ginoux (2025), Eq. 7",
                "aod_proxy": "AOD550",
                "dust_aod_proxy": "(1 - FMF) * AOD550",
                "ssa_constraint": "SSA412 < SSA470; otherwise NaN",
                "fmf_png": str(lg_fmf_png),
                "dust_aod_png": str(lg_dust_aod_png),
                "fmf_summary": summarize_grid("li_ginoux_fmf", lg_fmf),
                "coarse_fraction_summary": summarize_grid(
                    "li_ginoux_coarse_fraction", lg_coarse_fraction
                ),
                "dust_aod_summary": summarize_grid("li_ginoux_dust_aod", lg_dust_aod),
                "fmf_plot": lg_fmf_stats,
                "dust_aod_plot": lg_dust_aod_stats,
            },
            handle,
            indent=2,
        )

    if summary_rows:
        ml_dust_aod550_row = next((row for row in summary_rows if row["model_name"] == "dust_aod550_log1p"), None)
        if ml_dust_aod550_row is not None:
            ml_model = XGBRegressor()
            ml_model.load_model(here / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_xgb.json")
            with (here / "modis_db_dust_aod550_xgb" / "modis_db_dust_aod550_metadata.json").open("r") as handle:
                ml_meta = json.load(handle)
            ml_dust_aod = predict_model_grid(ml_model, ml_meta, features_2d)[0]
            ml_label = "Dust AOD (550 nm)"
        else:
            ml_log1p_row = next((row for row in summary_rows if row["model_name"] == "log1p"), None)
            if ml_log1p_row is not None:
                ml_log1p_model = XGBRegressor()
                ml_log1p_model.load_model(here / "dust_aod_xgb_log1p.json")
                with (here / "dust_aod_xgb_log1p_meta.json").open("r") as handle:
                    ml_log1p_meta = json.load(handle)
                ml_log1p_fraction = clip_fraction_grid(predict_model_grid(ml_log1p_model, ml_log1p_meta, features_2d)[0])
                ml_dust_aod = derive_dust_aod(ml_log1p_fraction, features_2d, source_band=args.daod_source_band)
                ml_label = f"Dust AOD ({args.daod_source_band} nm proxy)"
            else:
                ml_dust_aod = None
                ml_label = None
        if ml_dust_aod is not None:
            comparison_panel_path = args.output_dir / f"{granule_path.stem}_comparison_panel.png"
            save_comparison_panel(
                lat2d=lat2d,
                lon2d=lon2d,
                panels=[
                    {
                        "data": features_2d["MODIS_DB_AOD550"],
                        "title": "MODIS DB AOD550",
                        "label": "AOD550",
                        "cmap": "plasma",
                    },
                    {
                        "data": features_2d["MODIS_DB_AE"],
                        "title": "MODIS DB AE",
                        "label": "AE",
                        "cmap": "viridis",
                    },
                    {
                        "data": ml_dust_aod,
                        "title": "ML Dust AOD",
                        "label": ml_label,
                        "cmap": "magma",
                    },
                    {
                        "data": lg_dust_aod,
                        "title": "Li-Ginoux Dust AOD",
                        "label": "Coarse-mode AOD proxy at 550 nm",
                        "cmap": "magma",
                    },
                ],
                output_path=comparison_panel_path,
                figure_title=f"{granule_path.name}: MODIS DB and dust-AOD comparison",
            )
            scatter_comparison_path = (
                args.output_dir / f"{granule_path.stem}_ml_vs_li_ginoux_dust_aod_scatter.png"
            )
            scatter_stats = save_aod_scatter_comparison(
                ml_dust_aod=ml_dust_aod,
                li_ginoux_dust_aod=lg_dust_aod,
                output_path=scatter_comparison_path,
                title=f"{granule_path.name}: ML vs Li-Ginoux dust AOD",
            )
        else:
            scatter_comparison_path = None
            scatter_stats = None
    else:
        scatter_comparison_path = None
        scatter_stats = None

    run_meta = {
        "granule_path": str(granule_path),
        "short_name": args.short_name,
        "start": args.start,
        "end": args.end,
        "point": {"lon": args.lon, "lat": args.lat},
        "daod_source_band": args.daod_source_band,
        "summary_csv": str(summary_path),
        "reference_summary_csv": str(reference_summary_path),
        "li_ginoux_summary_json": str(li_ginoux_summary_path),
        "comparison_panel_png": str(
            args.output_dir / f"{granule_path.stem}_comparison_panel.png"
        ),
        "ml_vs_li_ginoux_scatter_png": (
            str(scatter_comparison_path) if scatter_comparison_path is not None else None
        ),
        "ml_vs_li_ginoux_scatter_stats": scatter_stats,
    }
    with (args.output_dir / f"{granule_path.stem}_run_metadata.json").open("w") as handle:
        json.dump(run_meta, handle, indent=2)

    print(f"Granule: {granule_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved reference summary: {reference_summary_path}")
    print(f"Saved Li-Ginoux summary: {li_ginoux_summary_path}")
    for row in summary_rows:
        if row.get("output_kind") == "dust_aod":
            print(f"{row['model_name']}: dust_aod_median={row['dust_aod_median']:.4f}")
        else:
            print(
                f"{row['model_name']}: "
                f"dust_fraction_median={row['dust_fraction_median']:.4f}, "
                f"dust_aod_median={row['dust_aod_median']:.4f}"
            )
    print(
        "li_ginoux: "
        f"fmf_median={summarize_grid('li_ginoux_fmf', lg_fmf).get('median', float('nan')):.4f}, "
        f"dust_aod_median={summarize_grid('li_ginoux_dust_aod', lg_dust_aod).get('median', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
