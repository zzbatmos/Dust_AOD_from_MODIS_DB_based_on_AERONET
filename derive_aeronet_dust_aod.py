from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from aeronet_dpr_utils import (
    DEFAULT_INV_DIR,
    SHIN_2019_SITES,
    collect_dust_dominant_samples,
    derive_dust_aod_dataframe,
    load_aeronet_inversion_data,
    resolve_site_list,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive AERONET dust AOD at one or more wavelengths using notebook DPR/LR logic."
    )
    parser.add_argument("--inv-dir", type=Path, default=DEFAULT_INV_DIR)
    parser.add_argument("--site-mode", choices=["all", "shin2019"], default="all")
    parser.add_argument("--sites", nargs="+", help="Explicit site names. If provided, overrides --site-mode.")
    parser.add_argument("--wavelengths", nargs="+", type=int, default=[440, 675, 1020])
    parser.add_argument("--dust-threshold", type=float, default=0.89, help="Threshold on Rd_1020 for dust-dominant samples.")
    parser.add_argument("--pldr-nd", type=float, default=0.02)
    parser.add_argument("--pldr-d", type=float, default=0.30)
    parser.add_argument(
        "--lr-stat",
        choices=["mean", "median"],
        default="mean",
        help="Which dust-dominant lidar-ratio statistic to use as LR_DUST.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("aeronet_dust_aod"))
    parser.add_argument("--prefix", default="aeronet_dust_aod")
    return parser.parse_args()


def compute_dust_lidar_ratios(
    aeronet_data: dict,
    site_list: list[str],
    wavelengths: list[int],
    dust_threshold: float,
    pldr_nd: float,
    pldr_d: float,
    lr_stat: str,
) -> tuple[dict[int, float], pd.DataFrame]:
    dust_samples = collect_dust_dominant_samples(
        aeronet_data=aeronet_data,
        site_list=site_list,
        wavelengths=wavelengths,
        dust_threshold=dust_threshold,
        pldr_nd=pldr_nd,
        pldr_d=pldr_d,
    )
    if dust_samples.empty:
        raise SystemExit("No dust-dominant samples were found. Dust AOD cannot be derived.")
    stats_records = []
    dust_lidar_ratio_by_wavelength: dict[int, float] = {}
    for wl in wavelengths:
        values = dust_samples[f"LR_{wl}"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        dust_lidar_ratio_by_wavelength[int(wl)] = float(np.mean(values) if lr_stat == "mean" else np.median(values))
        stats_records.append(
            {
                "wavelength_nm": int(wl),
                "dust_lidar_ratio_mean": float(np.mean(values)),
                "dust_lidar_ratio_median": float(np.median(values)),
                "dust_lidar_ratio_std": float(np.std(values)),
                "dust_dominant_count": int(values.size),
            }
        )
    return dust_lidar_ratio_by_wavelength, pd.DataFrame(stats_records)


def plot_scatter(df_wl: pd.DataFrame, wl: int, output_path: Path) -> dict[str, float | int]:
    x = df_wl["coarse_AOD"].to_numpy(dtype=float)
    y = df_wl["dust_AOD"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    r_p, _ = pearsonr(x, y)
    r_s, _ = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(x, y, gridsize=60, mincnt=1, bins="log", cmap="viridis")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(N)")
    xfit = np.linspace(0.0, float(np.nanmax(x)), 300)
    ax.plot(xfit, slope * xfit + intercept, color="red", linewidth=2, label="Linear fit")
    ax.plot(xfit, xfit, linestyle="--", color="black", linewidth=1.5, label="1:1 line")
    ax.set_xlabel(f"Coarse-mode AOD at {wl} nm")
    ax.set_ylabel(f"Dust AOD at {wl} nm")
    ax.set_title(f"Dust AOD vs Coarse-mode AOD ({wl} nm)")
    text = (
        f"N = {len(x)}\n"
        f"Pearson r = {r_p:.3f}\n"
        f"Spearman rho = {r_s:.3f}\n"
        f"Slope = {slope:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return {
        "wavelength_nm": int(wl),
        "count": int(len(x)),
        "pearson_r": float(r_p),
        "spearman_rho": float(r_s),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "dust_aod_mean": float(np.mean(y)),
        "dust_aod_median": float(np.median(y)),
        "coarse_aod_mean": float(np.mean(x)),
        "coarse_aod_median": float(np.median(x)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_sites = args.sites if args.sites else (SHIN_2019_SITES if args.site_mode == "shin2019" else None)
    aeronet_data = load_aeronet_inversion_data(args.inv_dir, requested_sites=requested_sites)
    site_list = resolve_site_list(aeronet_data.keys(), args.site_mode, args.sites)
    wavelengths = [int(wl) for wl in args.wavelengths]
    if not site_list:
        raise SystemExit("No matching AERONET sites were found.")

    dust_lr_by_wl, dust_lr_stats_df = compute_dust_lidar_ratios(
        aeronet_data=aeronet_data,
        site_list=site_list,
        wavelengths=wavelengths,
        dust_threshold=args.dust_threshold,
        pldr_nd=args.pldr_nd,
        pldr_d=args.pldr_d,
        lr_stat=args.lr_stat,
    )
    dust_lr_stats_df.to_csv(args.output_dir / f"{args.prefix}_dust_lidar_ratio_stats.csv", index=False)

    dust_aod_df = derive_dust_aod_dataframe(
        aeronet_data=aeronet_data,
        site_list=site_list,
        wavelengths=wavelengths,
        dust_lidar_ratio_by_wavelength=dust_lr_by_wl,
        pldr_nd=args.pldr_nd,
        pldr_d=args.pldr_d,
    )
    if dust_aod_df.empty:
        raise SystemExit("No valid dust-AOD samples were produced.")

    dust_aod_df.to_csv(args.output_dir / f"{args.prefix}_samples.csv", index=False)

    summary_records = []
    for wl in wavelengths:
        df_wl = dust_aod_df[dust_aod_df["wavelength_nm"] == wl].copy()
        if df_wl.empty:
            continue
        summary_records.append(
            plot_scatter(
                df_wl=df_wl,
                wl=wl,
                output_path=args.output_dir / f"{args.prefix}_{wl}nm_scatter.png",
            )
        )
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)

    write_json(
        args.output_dir / f"{args.prefix}_summary.json",
        {
            "site_count": len(site_list),
            "wavelengths_nm": wavelengths,
            "dust_threshold": args.dust_threshold,
            "pldr_nd": args.pldr_nd,
            "pldr_d": args.pldr_d,
            "lr_stat": args.lr_stat,
            "dust_lidar_ratio_by_wavelength": dust_lr_by_wl,
            "per_wavelength_summary": summary_records,
        },
    )

    print(f"Selected sites: {len(site_list)}")
    print(f"Total dust-AOD rows: {len(dust_aod_df)}")
    for wl in wavelengths:
        if wl in dust_lr_by_wl:
            print(f"{wl} nm LR_DUST ({args.lr_stat}): {dust_lr_by_wl[wl]:.3f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
