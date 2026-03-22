from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aeronet_dpr_utils import DEFAULT_INV_DIR, load_aeronet_inversion_data, resolve_site_list
from aeronet_dpr_utils import SHIN_2019_SITES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read AERONET inversion files and plot per-site DPR distributions."
    )
    parser.add_argument("--inv-dir", type=Path, default=DEFAULT_INV_DIR)
    parser.add_argument("--site-mode", choices=["all", "shin2019"], default="all")
    parser.add_argument(
        "--sites",
        nargs="+",
        help="Explicit site names. If provided, overrides --site-mode.",
    )
    parser.add_argument("--wavelength", type=int, choices=[440, 675, 870, 1020], default=1020)
    parser.add_argument("--output-dir", type=Path, default=Path("aeronet_dpr_site_analysis"))
    parser.add_argument("--prefix", default="aeronet_dpr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_sites = args.sites if args.sites else (SHIN_2019_SITES if args.site_mode == "shin2019" else None)
    aeronet_data = load_aeronet_inversion_data(args.inv_dir, requested_sites=requested_sites)
    site_list = resolve_site_list(aeronet_data.keys(), args.site_mode, args.sites)
    if not site_list:
        raise SystemExit("No matching AERONET sites were found.")

    dpr_col = f"Depolarization_Ratio[{args.wavelength}nm]"
    records: list[dict[str, float | str]] = []
    aggregate_values: list[np.ndarray] = []

    for site in site_list:
        values = np.asarray(aeronet_data[site][dpr_col], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        aggregate_values.append(values)
        records.extend({"Site": site, "DPR": float(v)} for v in values)

    if not records:
        raise SystemExit(f"No finite DPR samples were found for {dpr_col}.")

    df = pd.DataFrame(records)
    dpr_all = np.concatenate(aggregate_values)
    summary = {
        "site_count": int(df["Site"].nunique()),
        "sample_count": int(dpr_all.size),
        "mean": float(np.mean(dpr_all)),
        "std": float(np.std(dpr_all)),
        "median": float(np.median(dpr_all)),
        "wavelength_nm": int(args.wavelength),
    }
    summary_df = (
        df.groupby("Site")["DPR"]
        .agg(count="count", mean="mean", std="std", median="median")
        .reset_index()
        .sort_values("Site")
    )
    summary_df.to_csv(args.output_dir / f"{args.prefix}_{args.wavelength}nm_site_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="Site",
        y="DPR",
        inner="quartile",
        cut=0,
        density_norm="width",
        ax=ax,
    )
    ax.set_ylabel(f"Depolarization Ratio at {args.wavelength} nm")
    ax.set_xlabel("AERONET Site")
    ax.set_title(f"AERONET Lidar Depolarization Ratio ({args.wavelength} nm) by Site")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(args.output_dir / f"{args.prefix}_{args.wavelength}nm_violin.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dpr_all, bins=50, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(summary["mean"], linestyle="-", linewidth=2, label=f"Mean = {summary['mean']:.3f}")
    ax.axvline(summary["median"], linestyle="--", linewidth=2, label=f"Median = {summary['median']:.3f}")
    ax.axvline(summary["mean"] + summary["std"], linestyle=":", linewidth=1.5)
    ax.axvline(summary["mean"] - summary["std"], linestyle=":", linewidth=1.5, label=f"Std = {summary['std']:.3f}")
    ax.set_xlabel(f"Depolarization Ratio at {args.wavelength} nm")
    ax.set_ylabel("Probability Density")
    ax.set_title("Histogram of AERONET DPR")
    text = (
        f"N = {summary['sample_count']}\n"
        f"Mean = {summary['mean']:.3f}\n"
        f"Std = {summary['std']:.3f}\n"
        f"Median = {summary['median']:.3f}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output_dir / f"{args.prefix}_{args.wavelength}nm_histogram.png", dpi=200)
    plt.close(fig)

    print(f"Selected sites: {len(site_list)}")
    print(f"Valid DPR samples: {summary['sample_count']}")
    print(f"Mean DPR: {summary['mean']:.4f}")
    print(f"Median DPR: {summary['median']:.4f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
