#!/usr/bin/env python3
"""Plot quick-look MOD08_M3/MYD08_M3 Dark Target and Deep Blue AOD diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "modis_m3_dt_db_quicklook" / "full_2003_2025"
DEFAULT_OUTPUT = ROOT / "modis_m3_dt_db_quicklook" / "figures"
PRODUCTS = ("MOD08_M3", "MYD08_M3")
FIELDS = {
    "dt_land_ocean_aod550": ("DT land+ocean AOD", "YlOrBr", 0.0, 0.55),
    "dt_land_qa_aod550": ("DT land AOD, QA-weighted", "YlOrBr", 0.0, 0.65),
    "db_land_aod550": ("DB land AOD", "YlOrBr", 0.0, 0.55),
    "dt_db_combined_aod550": ("Operational DT+DB combined AOD", "YlOrBr", 0.0, 0.55),
}
REGIONS = {
    "Global": (-90.0, 90.0, -180.0, 180.0),
    "East China": (20.0, 45.0, 100.0, 125.0),
    "East Asia": (20.0, 50.0, 95.0, 135.0),
    "India": (5.0, 35.0, 65.0, 90.0),
    "North Africa/Sahel": (0.0, 35.0, -20.0, 35.0),
    "Middle East": (10.0, 35.0, 35.0, 65.0),
    "Tropical S. America": (-20.0, 10.0, -80.0, -45.0),
    "Europe": (35.0, 60.0, -10.0, 40.0),
    "CONUS": (25.0, 50.0, -125.0, -65.0),
}
REFERENCE_YEARS = tuple(range(2008, 2018))
RECENT_YEARS = (2024, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def as_float_array(var) -> np.ndarray:
    values = var[:]
    if np.ma.isMaskedArray(values):
        return np.asarray(values.filled(np.nan), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def load_product(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with Dataset(path) as ds:
        years = np.asarray(ds.variables["year"][:], dtype=np.int16)
        months = np.asarray(ds.variables["month"][:], dtype=np.int16)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        fields = {name: as_float_array(ds.variables[name]) for name in FIELDS}
    return years, months, lat, lon, fields


def load_combined(input_dir: Path, start_year: int, end_year: int):
    products = []
    lat_ref = lon_ref = years_ref = months_ref = None
    for product in PRODUCTS:
        path = input_dir / f"{product}_AOD_monthly_{start_year}_{end_year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        years, months, lat, lon, fields = load_product(path)
        years_ref = years if years_ref is None else years_ref
        months_ref = months if months_ref is None else months_ref
        lat_ref = lat if lat_ref is None else lat_ref
        lon_ref = lon if lon_ref is None else lon_ref
        products.append(fields)

    combined = {}
    for name in FIELDS:
        stack = np.stack([item[name] for item in products], axis=0)
        with np.errstate(invalid="ignore"):
            combined[name] = np.nanmean(stack, axis=0)
    assert years_ref is not None and months_ref is not None and lat_ref is not None and lon_ref is not None
    return years_ref, months_ref, lat_ref, lon_ref, combined


def area_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (lat.size, lon.size))


def region_mask(lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = bounds
    return (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max) & (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)


def weighted_mean(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(field) & mask
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(field[valid] * weights[valid]) / np.nansum(weights[valid]))


def annual_maps(years: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    out = {name: {} for name in FIELDS}
    for year in sorted(set(int(y) for y in years)):
        idx = years == year
        for name, arr in fields.items():
            with np.errstate(invalid="ignore"):
                out[name][year] = np.nanmean(arr[idx], axis=0)
    return out


def period_mean(annual: dict[int, np.ndarray], target_years: tuple[int, ...]) -> np.ndarray:
    arrays = [annual[year] for year in target_years if year in annual]
    if not arrays:
        raise ValueError(f"No fields available for years {target_years}")
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(arrays, axis=0), axis=0)


def write_regional_annual_csv(
    output: Path,
    years: np.ndarray,
    fields: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
) -> Path:
    weights = area_weights(lat, lon)
    masks = {name: region_mask(lat, lon, bounds) for name, bounds in REGIONS.items()}
    unique_years = sorted(set(int(year) for year in years))
    rows = []
    for year in unique_years:
        year_idx = np.where(years == year)[0]
        for region, mask in masks.items():
            row: dict[str, object] = {"year": year, "region": region}
            for field_name in FIELDS:
                monthly_means = [weighted_mean(fields[field_name][i], weights, mask) for i in year_idx]
                row[field_name] = float(np.nanmean(monthly_means))
            rows.append(row)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "region", *FIELDS.keys()])
        writer.writeheader()
        writer.writerows(rows)
    return output


def draw_map(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, extend: str = "max"):
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(field),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_global()
    ax.coastlines(linewidth=0.45)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="0.6", alpha=0.5)
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-60, 91, 30))
    return mesh


def plot_annual_global_timeseries(csv_path: Path, output: Path) -> Path:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    global_rows = data[data["region"] == "Global"]
    years = global_rows["year"]
    colors = {
        "dt_land_ocean_aod550": "#7f3b08",
        "dt_land_qa_aod550": "#b35806",
        "db_land_aod550": "#2166ac",
        "dt_db_combined_aod550": "#4d9221",
    }

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=220, sharex=True)
    ax, ax_anom = axes
    all_values = []
    for field, (label, _cmap, _vmin, _vmax) in FIELDS.items():
        values = np.asarray(global_rows[field], dtype=float)
        all_values.extend(values[np.isfinite(values)].tolist())
        reference = np.nanmean(values[np.isin(years, REFERENCE_YEARS)])
        ax.plot(years, values, marker="o", linewidth=1.8, markersize=3.5, label=label, color=colors[field])
        ax_anom.plot(
            years,
            values - reference,
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
            color=colors[field],
        )
    ymin, ymax = min(all_values), max(all_values)
    pad = max(0.008, 0.10 * (ymax - ymin))
    ax.axvspan(2008, 2017, color="0.88", alpha=0.7, zorder=0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title("MOD08/MYD08 monthly product global annual mean AOD", loc="left", fontweight="bold")
    ax.set_ylabel("AOD at 550 nm")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper right")

    ax_anom.axhline(0.0, color="0.25", linewidth=0.8)
    ax_anom.axvspan(2008, 2017, color="0.88", alpha=0.7, zorder=0, label="2008-2017 reference")
    ax_anom.set_title("Anomaly relative to each product's 2008-2017 mean", loc="left", fontsize=10.5)
    ax_anom.set_xlabel("Year")
    ax_anom.set_ylabel("AOD anomaly")
    ax_anom.set_ylim(-0.04, 0.025)
    ax_anom.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_regional_recent_anomaly(csv_path: Path, output: Path) -> Path:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    regions = [r for r in REGIONS if r != "Global"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), dpi=220, sharex=True)
    for ax, field in zip(axes.ravel(), FIELDS):
        values = []
        for region in regions:
            rows = data[data["region"] == region]
            ref = np.nanmean(rows[np.isin(rows["year"], REFERENCE_YEARS)][field])
            recent = np.nanmean(rows[np.isin(rows["year"], RECENT_YEARS)][field])
            values.append(recent - ref)
        colors = ["#b2182b" if v > 0 else "#2166ac" for v in values]
        ax.barh(regions, values, color=colors)
        ax.axvline(0, color="0.2", linewidth=0.8)
        ax.set_title(FIELDS[field][0], loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Regional AOD anomaly: 2024-2025 minus 2008-2017", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_climatology_maps(lat: np.ndarray, lon: np.ndarray, annual: dict[str, dict[int, np.ndarray]], output: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.8), dpi=220, subplot_kw={"projection": ccrs.Robinson()})
    for ax, field in zip(axes.ravel(), FIELDS):
        label, cmap, vmin, vmax = FIELDS[field]
        clim = period_mean(annual[field], tuple(sorted(annual[field].keys())))
        mesh = draw_map(ax, lon, lat, clim, label, cmap, vmin, vmax)
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, extend="max")
        cbar.set_label("AOD at 550 nm", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("MOD08/MYD08 AOD climatology from monthly products", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_recent_difference_maps(lat: np.ndarray, lon: np.ndarray, annual: dict[str, dict[int, np.ndarray]], output: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.8), dpi=220, subplot_kw={"projection": ccrs.Robinson()})
    for ax, field in zip(axes.ravel(), FIELDS):
        ref = period_mean(annual[field], REFERENCE_YEARS)
        recent = period_mean(annual[field], RECENT_YEARS)
        diff = recent - ref
        mesh = draw_map(ax, lon, lat, diff, FIELDS[field][0], "RdBu_r", -0.18, 0.18, extend="both")
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, extend="both")
        cbar.set_label("AOD difference", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("AOD difference: 2024-2025 minus 2008-2017", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    years, _months, lat, lon, fields = load_combined(args.input_dir, args.start_year, args.end_year)
    annual = annual_maps(years, fields)
    csv_path = write_regional_annual_csv(args.output_dir / "modis_m3_dt_db_annual_region_means.csv", years, fields, lat, lon)
    outputs = [
        plot_annual_global_timeseries(csv_path, args.output_dir / "modis_m3_dt_db_global_annual_aod.png"),
        plot_regional_recent_anomaly(csv_path, args.output_dir / "modis_m3_dt_db_regional_recent_anomaly.png"),
        plot_climatology_maps(lat, lon, annual, args.output_dir / "modis_m3_dt_db_climatology_maps.png"),
        plot_recent_difference_maps(lat, lon, annual, args.output_dir / "modis_m3_dt_db_recent_minus_reference_maps.png"),
        csv_path,
    ]
    for output in outputs:
        print(f"[WROTE] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
