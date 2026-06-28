#!/usr/bin/env python3
"""Diagnose the 2025 single-year total-AOD anomaly in the production-freeze MODIS DB record."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "production_freeze_v1_evaluation_figures" / "aod_2025_single_year_anomaly"
YEARS = tuple(range(2003, 2026))
BASELINE_YEARS = tuple(range(2003, 2020))
REFERENCE_YEARS = tuple(range(2008, 2018))
TARGET_YEAR = 2025
PRODUCTS = ("MOD04_L2", "MYD04_L2")
ALPHA = 0.05
MIN_BASELINE_YEARS = 10
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Non-overlapping masks are applied in this order. Bounds: lat_min, lat_max, lon_min, lon_max.
REGION_BOUNDS = {
    "East China": (20.0, 45.0, 100.0, 125.0),
    "India": (5.0, 35.0, 65.0, 90.0),
    "North Africa/Sahel": (0.0, 35.0, -20.0, 35.0),
    "Middle East": (10.0, 35.0, 35.0, 65.0),
    "Tropical S. America": (-20.0, 10.0, -80.0, -45.0),
    "Europe": (35.0, 60.0, -10.0, 40.0),
    "CONUS": (25.0, 50.0, -125.0, -65.0),
}


def as_float_array(var) -> np.ndarray:
    values = var[:]
    if np.ma.isMaskedArray(values):
        return np.asarray(values.filled(np.nan), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def read_monthly_total_aod(year: int, product: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / f"modis_l3_production_freeze_v1_{year}_0p5" / f"{product}_monthly"
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        total_aod = as_float_array(ds.variables["total_aod"])
    return lat, lon, total_aod


def combined_monthly_total_aod(year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_ref = lon_ref = None
    product_fields = []
    for product in PRODUCTS:
        lat, lon, monthly = read_monthly_total_aod(year, product)
        lat_ref = lat if lat_ref is None else lat_ref
        lon_ref = lon if lon_ref is None else lon_ref
        product_fields.append(monthly)
    with np.errstate(invalid="ignore"):
        combined = np.nanmean(np.stack(product_fields, axis=0), axis=0)
    assert lat_ref is not None and lon_ref is not None
    return lat_ref, lon_ref, combined


def build_monthly_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = []
    domain_count = None
    lat_ref = lon_ref = None
    for year in YEARS:
        lat, lon, monthly = combined_monthly_total_aod(year)
        lat_ref = lat if lat_ref is None else lat_ref
        lon_ref = lon if lon_ref is None else lon_ref
        fields.append(monthly)
        finite_any_year = np.any(np.isfinite(monthly), axis=0)
        domain_count = finite_any_year.astype(np.int16) if domain_count is None else domain_count + finite_any_year
    assert lat_ref is not None and lon_ref is not None and domain_count is not None
    return lat_ref, lon_ref, np.stack(fields, axis=0), domain_count >= MIN_BASELINE_YEARS


def area_weights(lat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape)


def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(field) & mask
    if not np.any(valid):
        return float("nan")
    weights = area_weights(lat, field.shape)
    return float(np.nansum(field[valid] * weights[valid]) / np.nansum(weights[valid]))


def weighted_contribution(anomaly: np.ndarray, lat: np.ndarray, region_mask: np.ndarray, domain_mask: np.ndarray) -> float:
    weights = area_weights(lat, anomaly.shape)
    valid_global = np.isfinite(anomaly) & domain_mask
    valid_region = np.isfinite(anomaly) & region_mask
    if not np.any(valid_region) or not np.any(valid_global):
        return float("nan")
    return float(np.nansum(anomaly[valid_region] * weights[valid_region]) / np.nansum(weights[valid_global]))


def weighted_fraction(mask: np.ndarray, lat: np.ndarray, domain_mask: np.ndarray) -> float:
    weights = area_weights(lat, domain_mask.shape)
    valid = np.isfinite(weights) & domain_mask
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(weights[mask & valid]) / np.nansum(weights[valid]))


def box_mask(lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = bounds
    return (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max) & (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)


def nonoverlap_region_masks(lat: np.ndarray, lon: np.ndarray, domain_mask: np.ndarray) -> dict[str, np.ndarray]:
    assigned = np.zeros(domain_mask.shape, dtype=bool)
    masks = {}
    for name, bounds in REGION_BOUNDS.items():
        mask = box_mask(lat, lon, bounds) & domain_mask & ~assigned
        masks[name] = mask
        assigned |= mask
    masks["Other land DB domain"] = domain_mask & ~assigned
    return masks


def annual_fields(monthly_stack: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(monthly_stack, axis=1)


def monthly_global_means(monthly_stack: np.ndarray, lat: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    out = np.full(monthly_stack.shape[:2], np.nan, dtype=np.float64)
    for iy in range(monthly_stack.shape[0]):
        for im in range(monthly_stack.shape[1]):
            out[iy, im] = weighted_mean(monthly_stack[iy, im], lat, domain_mask)
    return out


def single_year_significance(baseline_fields: np.ndarray, target_field: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore"):
        baseline_mean = np.nanmean(baseline_fields, axis=0)
        baseline_std = np.nanstd(baseline_fields, axis=0, ddof=1)
    n = np.sum(np.isfinite(baseline_fields), axis=0).astype(np.float64)
    difference = target_field - baseline_mean
    valid = np.isfinite(difference) & np.isfinite(baseline_std) & (baseline_std > 0) & (n >= MIN_BASELINE_YEARS)
    t_stat = np.full_like(difference, np.nan, dtype=np.float64)
    p_value = np.full_like(difference, np.nan, dtype=np.float64)
    # One target year compared with the distribution of baseline annual means.
    se = baseline_std * np.sqrt(1.0 + 1.0 / n)
    t_stat[valid] = difference[valid] / se[valid]
    p_value[valid] = 2.0 * stats.t.sf(np.abs(t_stat[valid]), df=np.maximum(n[valid] - 1, 1))
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "difference": difference,
        "t_stat": t_stat,
        "p_value": p_value,
        "valid": valid,
        "significant": valid & (p_value < ALPHA),
    }


def draw_map(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, label: str):
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
    ax.gridlines(draw_labels=False, linewidth=0.25, color="0.65", alpha=0.45)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    cbar = ax.figure.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, fraction=0.055, extend="both")
    cbar.set_label(label, fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)


def add_stippling(ax, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, stride: int = 4) -> None:
    yy, xx = np.where(mask[::stride, ::stride])
    if yy.size == 0:
        return
    ax.scatter(lon[::stride][xx], lat[::stride][yy], s=0.25, c="k", alpha=0.35, linewidths=0, transform=ccrs.PlateCarree())


def plot_2025_maps(lat: np.ndarray, lon: np.ndarray, sig: dict[str, np.ndarray]) -> Path:
    output = OUT_DIR / "total_aod_2025_minus_2003_2019_maps_significance.png"
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12.2), dpi=240, subplot_kw={"projection": ccrs.Robinson()})
    draw_map(axes[0], lon, lat, sig["baseline_mean"], "A) 2003-2019 Baseline Mean", "YlOrBr", 0.0, 0.65, "Total AOD at 550 nm")
    draw_map(axes[1], lon, lat, sig["difference"], "B) 2025 Minus 2003-2019 Baseline", "RdBu_r", -0.18, 0.18, "AOD difference")
    add_stippling(axes[1], lon, lat, sig["significant"])
    draw_map(axes[2], lon, lat, sig["t_stat"], "C) Baseline-Variance t Statistic", "RdBu_r", -6.0, 6.0, "t statistic")
    add_stippling(axes[2], lon, lat, sig["significant"])
    fig.text(
        0.025,
        0.018,
        "Stippling marks two-sided p<0.05. Baseline variance is from 2003-2019 annual means; target year is 2025 only.",
        fontsize=8.8,
        color="0.30",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.995])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_monthly_anomaly(monthly_global: np.ndarray, annual_global: np.ndarray) -> Path:
    output = OUT_DIR / "total_aod_2025_monthly_and_annual_strength.png"
    years = np.asarray(YEARS)
    baseline_idx = np.isin(years, BASELINE_YEARS)
    ref_idx = np.isin(years, REFERENCE_YEARS)
    target_idx = YEARS.index(TARGET_YEAR)

    baseline_monthly_mean = np.nanmean(monthly_global[baseline_idx], axis=0)
    baseline_monthly_std = np.nanstd(monthly_global[baseline_idx], axis=0, ddof=1)
    ref_mean = np.nanmean(annual_global[ref_idx])

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), dpi=220)
    x = np.arange(12)
    axes[0].fill_between(
        x,
        baseline_monthly_mean - baseline_monthly_std,
        baseline_monthly_mean + baseline_monthly_std,
        color="0.82",
        label="2003-2019 ±1σ",
    )
    axes[0].plot(x, baseline_monthly_mean, color="black", linewidth=2.0, label="2003-2019 mean")
    axes[0].plot(x, monthly_global[YEARS.index(2024)], marker="o", color="#7570b3", linewidth=1.9, label="2024")
    axes[0].plot(x, monthly_global[target_idx], marker="o", color="#d95f02", linewidth=2.2, label="2025")
    axes[0].set_xticks(x, MONTH_LABELS)
    axes[0].set_ylabel("Total AOD at 550 nm")
    axes[0].set_title("A) Monthly Global Mean Total AOD", loc="left", fontweight="bold")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    annual_anom = annual_global - ref_mean
    axes[1].bar(years, annual_anom, color=np.where(annual_anom < 0, "#2166ac", "#b2182b"), width=0.75)
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set_ylabel("Annual anomaly")
    axes[1].set_xlabel("Year")
    axes[1].set_title("B) Annual Anomaly Relative to 2008-2017", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.01,
        0.05,
        f"2025 anomaly={annual_anom[target_idx]:+.3f}; rank=lowest of {len(YEARS)} years.",
        transform=axes[1].transAxes,
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_regional_contributions(
    lat: np.ndarray,
    masks: dict[str, np.ndarray],
    domain_mask: np.ndarray,
    anomaly_2025: np.ndarray,
    regional_rows: list[dict[str, object]],
) -> Path:
    output = OUT_DIR / "total_aod_2025_regional_contribution_decomposition.png"
    region_order = [name for name in REGION_BOUNDS] + ["Other land DB domain"]
    contributions = np.array([float(next(row["global_anomaly_contribution"] for row in regional_rows if row["region"] == name)) for name in region_order])
    regional_anoms = np.array([float(next(row["regional_anomaly"] for row in regional_rows if row["region"] == name)) for name in region_order])
    area_fracs = np.array([weighted_fraction(masks[name], lat, domain_mask) for name in region_order])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9), dpi=220)
    colors = np.where(contributions < 0, "#2166ac", "#b2182b")
    order = np.argsort(contributions)
    axes[0].barh(np.array(region_order)[order], contributions[order], color=colors[order])
    axes[0].axvline(0.0, color="0.2", linewidth=0.9)
    axes[0].set_xlabel("Contribution to global AOD anomaly")
    axes[0].set_title("A) Area-Weighted Contribution", loc="left", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)

    width = 0.38
    x = np.arange(len(region_order))
    axes[1].bar(x - width / 2, regional_anoms, width=width, color="#d95f02", label="regional AOD anomaly")
    axes[1].bar(x + width / 2, area_fracs, width=width, color="0.55", label="area fraction")
    axes[1].axhline(0.0, color="0.2", linewidth=0.9)
    axes[1].set_xticks(x, region_order, rotation=45, ha="right")
    axes[1].set_title("B) Regional Anomaly and Area Leverage", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8.5)
    net = weighted_mean(anomaly_2025, lat, domain_mask)
    fig.suptitle(f"Why the 2025 Global Mean Is Low: Regional Contributions Sum to {net:+.3f}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def write_tables(
    lat: np.ndarray,
    domain_mask: np.ndarray,
    masks: dict[str, np.ndarray],
    annual: np.ndarray,
    monthly_global: np.ndarray,
    sig: dict[str, np.ndarray],
) -> tuple[Path, Path, Path]:
    summary_path = OUT_DIR / "total_aod_2025_single_year_summary.csv"
    regional_path = OUT_DIR / "total_aod_2025_regional_contributions.csv"
    m3_path = OUT_DIR / "total_aod_2025_m3_corroboration.csv"

    baseline_idx = np.isin(YEARS, BASELINE_YEARS)
    ref_idx = np.isin(YEARS, REFERENCE_YEARS)
    target_idx = YEARS.index(TARGET_YEAR)
    annual_global = np.array([weighted_mean(field, lat, domain_mask) for field in annual])
    baseline_mean = float(np.nanmean(annual_global[baseline_idx]))
    baseline_std = float(np.nanstd(annual_global[baseline_idx], ddof=1))
    ref_mean = float(np.nanmean(annual_global[ref_idx]))
    target = float(annual_global[target_idx])
    target_diff_baseline = target - baseline_mean
    target_diff_ref = target - ref_mean
    target_z = target_diff_baseline / baseline_std
    target_t = target_diff_baseline / (baseline_std * np.sqrt(1.0 + 1.0 / np.sum(baseline_idx)))
    target_p = float(2.0 * stats.t.sf(abs(target_t), df=np.sum(baseline_idx) - 1))
    rank_low = int(np.argsort(annual_global).tolist().index(target_idx) + 1)
    sig_valid = sig["valid"]
    sig_mask = sig["significant"]
    sig_negative = sig_mask & (sig["difference"] < 0)
    sig_positive = sig_mask & (sig["difference"] > 0)

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["target_year", TARGET_YEAR])
        writer.writerow(["baseline_years_for_spatial_t_test", "2003-2019"])
        writer.writerow(["reference_years_for_anomaly_plot", "2008-2017"])
        writer.writerow(["global_baseline_2003_2019_mean_aod", baseline_mean])
        writer.writerow(["global_baseline_2003_2019_std_aod", baseline_std])
        writer.writerow(["global_reference_2008_2017_mean_aod", ref_mean])
        writer.writerow(["global_2025_aod", target])
        writer.writerow(["global_2025_minus_2003_2019", target_diff_baseline])
        writer.writerow(["global_2025_minus_2008_2017", target_diff_ref])
        writer.writerow(["global_2025_percent_difference_from_2003_2019", 100.0 * target_diff_baseline / baseline_mean])
        writer.writerow(["global_2025_z_vs_2003_2019", target_z])
        writer.writerow(["global_2025_t_vs_2003_2019", target_t])
        writer.writerow(["global_2025_two_sided_p_vs_2003_2019", target_p])
        writer.writerow(["global_2025_rank_from_lowest_2003_2025", rank_low])
        writer.writerow(["area_fraction_significant_p_lt_0p05", weighted_fraction(sig_mask, lat, sig_valid)])
        writer.writerow(["area_fraction_significant_negative", weighted_fraction(sig_negative, lat, sig_valid)])
        writer.writerow(["area_fraction_significant_positive", weighted_fraction(sig_positive, lat, sig_valid)])

    anomaly_2025 = sig["difference"]
    regional_rows: list[dict[str, object]] = []
    with regional_path.open("w", newline="") as f:
        fieldnames = ["region", "area_fraction", "regional_baseline_aod", "regional_2025_aod", "regional_anomaly", "global_anomaly_contribution"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for region, mask in masks.items():
            baseline_field = sig["baseline_mean"]
            target_field = annual[target_idx]
            row = {
                "region": region,
                "area_fraction": weighted_fraction(mask, lat, domain_mask),
                "regional_baseline_aod": weighted_mean(baseline_field, lat, mask),
                "regional_2025_aod": weighted_mean(target_field, lat, mask),
                "regional_anomaly": weighted_mean(anomaly_2025, lat, mask),
                "global_anomaly_contribution": weighted_contribution(anomaly_2025, lat, mask, domain_mask),
            }
            regional_rows.append(row)
            writer.writerow(row)

    m3_csv = ROOT / "modis_m3_dt_db_quicklook" / "figures" / "modis_m3_dt_db_annual_region_means.csv"
    if m3_csv.exists():
        fields = ("dt_land_ocean_aod550", "dt_land_qa_aod550", "db_land_aod550", "dt_db_combined_aod550")
        records: dict[str, dict[int, float]] = {field: {} for field in fields}
        with m3_csv.open() as f:
            for row in csv.DictReader(f):
                if row["region"] != "Global":
                    continue
                year = int(row["year"])
                for field in fields:
                    records[field][year] = float(row[field])
        with m3_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["m3_field", "reference_2008_2017", "aod_2025", "anomaly_2025", "rank_from_lowest_2003_2025"])
            for field in fields:
                years = sorted(records[field])
                values = np.array([records[field][year] for year in years], dtype=float)
                ref = float(np.nanmean([records[field][year] for year in REFERENCE_YEARS if year in records[field]]))
                target_value = records[field][TARGET_YEAR]
                rank = int(np.argsort(values).tolist().index(years.index(TARGET_YEAR)) + 1)
                writer.writerow([field, ref, target_value, target_value - ref, rank])
    else:
        m3_path.write_text("M3 quick-look CSV not found.\n")

    return summary_path, regional_path, m3_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lat, lon, monthly_stack, domain_mask = build_monthly_stack()
    annual = annual_fields(monthly_stack)
    baseline = annual[np.isin(YEARS, BASELINE_YEARS)]
    target = annual[YEARS.index(TARGET_YEAR)]
    sig = single_year_significance(baseline, target)
    monthly_global = monthly_global_means(monthly_stack, lat, domain_mask)
    annual_global = np.array([weighted_mean(field, lat, domain_mask) for field in annual])
    masks = nonoverlap_region_masks(lat, lon, domain_mask)

    map_path = plot_2025_maps(lat, lon, sig)
    monthly_path = plot_monthly_anomaly(monthly_global, annual_global)
    summary_path, regional_path, m3_path = write_tables(lat, domain_mask, masks, annual, monthly_global, sig)
    with regional_path.open() as f:
        regional_rows = list(csv.DictReader(f))
    contribution_path = plot_regional_contributions(lat, masks, domain_mask, sig["difference"], regional_rows)

    for path in (map_path, monthly_path, contribution_path, summary_path, regional_path, m3_path):
        print(f"[WROTE] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
