from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from aeronet_dpr_utils import (
    DEFAULT_INV_DIR,
    SHIN_2019_SITES,
    collect_dust_dominant_samples,
    load_aeronet_inversion_data,
    resolve_site_list,
    summarize_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive dust-dominant DPR and lidar-ratio statistics from AERONET inversion data."
    )
    parser.add_argument("--inv-dir", type=Path, default=DEFAULT_INV_DIR)
    parser.add_argument("--site-mode", choices=["all", "shin2019"], default="all")
    parser.add_argument("--sites", nargs="+", help="Explicit site names. If provided, overrides --site-mode.")
    parser.add_argument("--dust-threshold", type=float, default=0.89, help="Dust-dominant threshold on Rd_1020.")
    parser.add_argument("--pldr-nd", type=float, default=0.02)
    parser.add_argument("--pldr-d", type=float, default=0.30)
    parser.add_argument("--wavelengths", nargs="+", type=int, default=[440, 675, 1020])
    parser.add_argument("--output-dir", type=Path, default=Path("aeronet_dust_dominant_stats"))
    parser.add_argument("--prefix", default="dust_dominant")
    return parser.parse_args()


def plot_histograms(samples: pd.DataFrame, variable_prefix: str, output_path: Path, title_prefix: str) -> None:
    wavelengths = [int(col.split("_")[1]) for col in samples.columns if col.startswith(variable_prefix)]
    wavelengths = sorted(wavelengths)
    fig, axes = plt.subplots(1, len(wavelengths), figsize=(4 * len(wavelengths), 5), sharey=True)
    if len(wavelengths) == 1:
        axes = [axes]
    for ax, wl in zip(axes, wavelengths):
        values = samples[f"{variable_prefix}_{wl}"].to_numpy(dtype=float)
        values = values[pd.notna(values)]
        stats = summarize_array(values, wl, variable_prefix)
        ax.hist(values, bins=40, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(stats.mean, linestyle="-", linewidth=2, label="Mean")
        ax.axvline(stats.median, linestyle="--", linewidth=2, label="Median")
        ax.set_xlabel(f"{variable_prefix} at {wl} nm")
        ax.set_title(f"{title_prefix}\nRd > threshold")
        ax.grid(True, alpha=0.3)
        text = (
            f"N = {stats.count}\n"
            f"Mean = {stats.mean:.3f}\n"
            f"Std = {stats.std:.3f}\n"
            f"Median = {stats.median:.3f}"
        )
        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax.legend()
    axes[0].set_ylabel("Probability Density")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_sites = args.sites if args.sites else (SHIN_2019_SITES if args.site_mode == "shin2019" else None)
    aeronet_data = load_aeronet_inversion_data(args.inv_dir, requested_sites=requested_sites)
    site_list = resolve_site_list(aeronet_data.keys(), args.site_mode, args.sites)
    if not site_list:
        raise SystemExit("No matching AERONET sites were found.")

    samples = collect_dust_dominant_samples(
        aeronet_data=aeronet_data,
        site_list=site_list,
        wavelengths=args.wavelengths,
        dust_threshold=args.dust_threshold,
        pldr_nd=args.pldr_nd,
        pldr_d=args.pldr_d,
    )
    if samples.empty:
        raise SystemExit("No dust-dominant samples were found with the requested threshold.")

    samples.to_csv(args.output_dir / f"{args.prefix}_samples.csv", index=False)

    summary_records = []
    for wl in args.wavelengths:
        summary_records.append(summarize_array(samples[f"DPR_{wl}"], wl, "DPR").as_dict())
        summary_records.append(summarize_array(samples[f"LR_{wl}"], wl, "LR").as_dict())
    summary_df = pd.DataFrame(summary_records).sort_values(["variable", "wavelength_nm"])
    summary_df.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)

    payload = {
        "site_count": len(site_list),
        "dust_threshold": args.dust_threshold,
        "pldr_nd": args.pldr_nd,
        "pldr_d": args.pldr_d,
        "sample_count": int(len(samples)),
        "stats": summary_records,
    }
    write_json(args.output_dir / f"{args.prefix}_summary.json", payload)

    plot_histograms(
        samples,
        variable_prefix="DPR",
        output_path=args.output_dir / f"{args.prefix}_dpr_histograms.png",
        title_prefix="Dust-dominant DPR",
    )
    plot_histograms(
        samples,
        variable_prefix="LR",
        output_path=args.output_dir / f"{args.prefix}_lr_histograms.png",
        title_prefix="Dust-dominant lidar ratio",
    )

    print(f"Selected sites: {len(site_list)}")
    print(f"Dust-dominant samples: {len(samples)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
