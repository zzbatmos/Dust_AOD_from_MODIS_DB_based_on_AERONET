#!/usr/bin/env python3
"""Plot MODIS Deep Blue aerosol-type diagnostics for selected L2 granules."""

from __future__ import annotations

import argparse
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

import sys

sys.path.insert(0, str((Path(__file__).resolve().parent.parent)))
import MODIS_Lib  # noqa: E402


TYPE_LABELS = {
    0: "Mixed",
    1: "Dust",
    2: "Smoke",
    3: "Sulfate",
}

TYPE_COLORS = {
    0: "#8e8e8e",
    1: "#c28f0e",
    2: "#7d3cff",
    3: "#2f77b4",
}

WAVELENGTHS = [412, 470, 660]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--dust-output-dir",
        type=Path,
        default=here / "dust_model_application_2025-03-14",
    )
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        default=here / "dust_model_application_2024-09-22_amazon",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=here / "modis_db_aerosol_type_diagnostics_summary.json",
    )
    return parser.parse_args()


def scale_dataset(ds) -> np.ndarray:
    arr = np.asarray(ds.get(), dtype=float)
    attrs = ds.attributes()
    fill = attrs.get("_FillValue", -9999)
    arr[arr == fill] = np.nan
    scale = attrs.get("scale_factor", 1.0)
    offset = attrs.get("add_offset", 0.0)
    return (arr - offset) * scale


def load_case(granule_path: Path) -> dict[str, np.ndarray]:
    granule = MODIS_Lib.MODIS_DeepBlue_Level2(str(granule_path))
    lat = np.asarray(granule.lat, dtype=float)
    lon = np.asarray(granule.lon, dtype=float)

    f = SD(str(granule_path), SDC.READ)
    qa = np.asarray(f.select("Quality_Assurance_Land").get(), dtype=np.int16)
    ae = scale_dataset(f.select("Deep_Blue_Angstrom_Exponent_Land"))
    ssa = scale_dataset(f.select("Deep_Blue_Spectral_Single_Scattering_Albedo_Land"))
    aod = scale_dataset(f.select("Deep_Blue_Spectral_Aerosol_Optical_Depth_Land"))
    f.end()

    byte4 = qa[:, :, 4].astype(np.uint8)
    usefulness = (byte4 & 0b1) == 1
    confidence = ((byte4 >> 1) & 0b11).astype(np.uint8)
    aerosol_type = ((byte4 >> 3) & 0b11).astype(np.uint8)

    aerosol_type_masked = aerosol_type.astype(float)
    aerosol_type_masked[~usefulness] = np.nan

    return {
        "lat": lat,
        "lon": lon,
        "usefulness": usefulness,
        "confidence": confidence,
        "aerosol_type": aerosol_type,
        "aerosol_type_masked": aerosol_type_masked,
        "ae": ae,
        "ssa": ssa,
        "aod": aod,
    }


def save_aerosol_type_map(case: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lon = ((np.asarray(case["lon"], dtype=float) + 180.0) % 360.0) - 180.0
    lat = np.asarray(case["lat"], dtype=float)
    z = np.asarray(case["aerosol_type_masked"], dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
    if not np.any(finite):
        raise ValueError(f"No useful aerosol-type pixels for {output_path.name}.")

    cmap = ListedColormap([TYPE_COLORS[i] for i in range(4)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig = plt.figure(figsize=(11, 6.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle=":")
    ax.set_extent(
        [
            float(np.nanmin(lon[finite])) - 0.8,
            float(np.nanmax(lon[finite])) + 0.8,
            float(np.nanmin(lat[finite])) - 0.8,
            float(np.nanmax(lat[finite])) + 0.8,
        ],
        crs=ccrs.PlateCarree(),
    )
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(z),
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    cb = plt.colorbar(mesh, ax=ax, pad=0.03, shrink=0.82, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels([TYPE_LABELS[i] for i in range(4)])
    cb.set_label("Deep Blue Aerosol Type")
    ax.set_title(title)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_hist_panel(ax, values_by_type: dict[int, np.ndarray], bins: np.ndarray, title: str, xlabel: str) -> None:
    plotted = False
    for code in range(4):
        vals = values_by_type.get(code)
        if vals is None or vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=TYPE_COLORS[code],
            label=f"{TYPE_LABELS[code]} (n={vals.size})",
        )
        plotted = True
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.grid(alpha=0.25, linestyle=":")
    if plotted:
        ax.legend(fontsize=8)


def save_spectral_histograms(case: dict[str, np.ndarray], output_path: Path, field: str, title_prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(case[field], dtype=float)
    useful = np.asarray(case["usefulness"], dtype=bool)
    aerosol_type = np.asarray(case["aerosol_type"], dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for idx, wl in enumerate(WAVELENGTHS):
        values_by_type = {}
        band = data[idx]
        valid = useful & np.isfinite(band)
        band_vals = band[valid]
        if band_vals.size == 0:
            bins = np.linspace(0.0, 1.0, 40)
        else:
            lo = float(np.nanmin(band_vals))
            hi = float(np.nanmax(band_vals))
            if hi <= lo:
                hi = lo + 1e-3
            bins = np.linspace(lo, hi, 40)
        for code in range(4):
            mask = valid & (aerosol_type == code)
            values_by_type[code] = band[mask]
        xlabel = f"{field.upper()} at {wl} nm" if field == "aod" else f"{field.upper()} at {wl} nm"
        if field == "aod":
            xlabel = f"AOD at {wl} nm"
        elif field == "ssa":
            xlabel = f"SSA at {wl} nm"
        _plot_hist_panel(axes[idx], values_by_type, bins, f"{title_prefix}: {wl} nm", xlabel)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ae_histogram(case: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    useful = np.asarray(case["usefulness"], dtype=bool)
    aerosol_type = np.asarray(case["aerosol_type"], dtype=np.uint8)
    ae = np.asarray(case["ae"], dtype=float)
    valid = useful & np.isfinite(ae)
    vals = ae[valid]
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    if hi <= lo:
        hi = lo + 1e-3
    bins = np.linspace(lo, hi, 50)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    values_by_type = {}
    for code in range(4):
        mask = valid & (aerosol_type == code)
        values_by_type[code] = ae[mask]
    _plot_hist_panel(ax, values_by_type, bins, title, "Angstrom exponent")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(case_name: str, case: dict[str, np.ndarray]) -> dict[str, object]:
    useful = np.asarray(case["usefulness"], dtype=bool)
    aerosol_type = np.asarray(case["aerosol_type"], dtype=np.uint8)
    ae = np.asarray(case["ae"], dtype=float)
    ssa = np.asarray(case["ssa"], dtype=float)
    aod = np.asarray(case["aod"], dtype=float)

    out: dict[str, object] = {"case": case_name, "n_useful": int(useful.sum()), "types": {}}
    for code in range(4):
        mask = useful & (aerosol_type == code)
        if not np.any(mask):
            continue
        entry: dict[str, object] = {"count": int(mask.sum())}
        ae_vals = ae[mask]
        entry["ae_valid"] = int(np.isfinite(ae_vals).sum())
        entry["ae_median"] = float(np.nanmedian(ae_vals)) if np.isfinite(ae_vals).any() else np.nan
        for idx, wl in enumerate(WAVELENGTHS):
            aod_vals = aod[idx][mask]
            ssa_vals = ssa[idx][mask]
            entry[f"aod{wl}_valid"] = int(np.isfinite(aod_vals).sum())
            entry[f"ssa{wl}_valid"] = int(np.isfinite(ssa_vals).sum())
        out["types"][TYPE_LABELS[code]] = entry
    return out


def process_case(case_name: str, granule_path: Path, output_dir: Path) -> dict[str, object]:
    case = load_case(granule_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = granule_path.stem

    save_aerosol_type_map(case, output_dir / f"{stem}_db_aerosol_type_map.png", f"{case_name}: Deep Blue aerosol type")
    save_spectral_histograms(case, output_dir / f"{stem}_db_aerosol_type_aod_histograms.png", "aod", f"{case_name} AOD")
    save_spectral_histograms(case, output_dir / f"{stem}_db_aerosol_type_ssa_histograms.png", "ssa", f"{case_name} SSA")
    save_ae_histogram(case, output_dir / f"{stem}_db_aerosol_type_ae_histogram.png", f"{case_name}: AE by aerosol type")
    return summarize(case_name, case)


def main() -> None:
    args = parse_args()
    summaries = []
    summaries.append(process_case("Heavy dust 2025-03-14", args.dust_file, args.dust_output_dir))
    summaries.append(process_case("Heavy smoke 2024-09-22", args.smoke_file, args.smoke_output_dir))
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
