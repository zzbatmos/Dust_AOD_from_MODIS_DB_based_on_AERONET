#!/usr/bin/env python3
"""Replicate Li & Ginoux Figure 2b/2c style panels from collocated AERONET data.

This script uses the collocated AERONET SDA data embedded in the MOD04 and
MYD04 collocation pickle files to reproduce the global FMF-versus-AE panels
analogous to Figure 2b and 2c in:

Li, X., & Ginoux, P. (2025), GRL, 52, e2024GL114397.
https://doi.org/10.1029/2024GL114397

Data used here:
- AERONET SDA FineModeFraction_500nm[eta]
- AERONET SDA Angstrom_Exponent(AE)-Total_500nm[alpha]

The script fits:
- g(FMF): cubic AE = p + q*FMF + r*FMF^2 + s*FMF^3 using all valid points
- f(AE): quadratic FMF = a + b*AE + c*AE^2 using only FMF < 0.7

It also overlays the Anderson et al. (2005) quadratic relation commonly used in
later dust studies:
- FMF ~= SMF = 0.02 + 0.5089*AE - 0.0512*AE^2
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parent = here.parent
    parser = argparse.ArgumentParser(
        description="Replicate Li & Ginoux Figure 2b/2c from AERONET collocation pickles."
    )
    parser.add_argument(
        "--mod04",
        type=Path,
        default=parent / "AERONET_MOD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the MOD04 collocation pickle.",
    )
    parser.add_argument(
        "--myd04",
        type=Path,
        default=parent / "AERONET_MYD04_L2_collocation_with_SDA_data.pkl",
        help="Path to the MYD04 collocation pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "li_ginoux_figure2_replication",
        help="Directory for figures and fit summaries.",
    )
    return parser.parse_args()


def extract_aeronet_sda_arrays(pickle_path: Path) -> pd.DataFrame:
    with pickle_path.open("rb") as handle:
        colloc = pickle.load(handle)

    records = []
    for site_name, site_data in colloc.items():
        for granule_name, record in site_data.items():
            meta = record.get("nearest_AERONET_SDA_L1.5_data_meta", {})
            if not meta.get("found", False):
                continue
            sda = record.get("nearest_AERONET_SDA_L1.5_data", {})
            fmf = sda.get("FineModeFraction_500nm[eta]", np.nan)
            ae = sda.get("Angstrom_Exponent(AE)-Total_500nm[alpha]", np.nan)
            if not (np.isfinite(fmf) and np.isfinite(ae)):
                continue
            records.append(
                {
                    "site_name": site_name,
                    "granule_name": granule_name,
                    "fmf": float(fmf),
                    "ae": float(ae),
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No finite AE/FMF SDA records found in {pickle_path}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fmf", "ae"])
    df = df[(df["fmf"] >= 0.0) & (df["fmf"] <= 1.0)]
    return df.reset_index(drop=True)


def fit_global_relationships(df: pd.DataFrame) -> dict[str, np.ndarray]:
    fmf = df["fmf"].to_numpy(dtype=float)
    ae = df["ae"].to_numpy(dtype=float)

    g_coeffs = np.polyfit(fmf, ae, deg=3)

    fit_mask = fmf < 0.7
    if fit_mask.sum() < 10:
        raise ValueError("Not enough FMF < 0.7 samples to fit f(AE).")
    f_coeffs = np.polyfit(ae[fit_mask], fmf[fit_mask], deg=2)

    return {"g_coeffs": g_coeffs, "f_coeffs": f_coeffs}


def evaluate_g(coeffs: np.ndarray, fmf: np.ndarray) -> np.ndarray:
    return np.polyval(coeffs, fmf)


def evaluate_f(coeffs: np.ndarray, ae: np.ndarray) -> np.ndarray:
    return np.clip(np.polyval(coeffs, ae), 0.0, 1.0)


def evaluate_anderson_2005(ae: np.ndarray) -> np.ndarray:
    return np.clip(0.02 + 0.5089 * ae - 0.0512 * ae**2, 0.0, 1.0)


def published_li_ginoux_f(ae: np.ndarray, dataset_key: str) -> np.ndarray:
    ae = np.asarray(ae, dtype=float)
    if dataset_key == "MOD":
        out = 0.087 * ae**2 + 0.338 * ae + 0.051
    elif dataset_key == "MYD":
        out = 0.082 * ae**2 + 0.333 * ae + 0.052
    elif dataset_key == "MEAN":
        out = 0.085 * ae**2 + 0.336 * ae + 0.051
    else:
        raise ValueError(f"Unsupported dataset_key: {dataset_key}")
    return np.clip(out, 0.0, 1.0)


def published_li_ginoux_g(fmf: np.ndarray, dataset_key: str) -> np.ndarray:
    fmf = np.asarray(fmf, dtype=float)
    if dataset_key == "MOD":
        return -3.205 * fmf**3 + 2.706 * fmf**2 + 1.913 * fmf - 0.151
    if dataset_key == "MYD":
        return -3.310 * fmf**3 + 2.836 * fmf**2 + 1.905 * fmf - 0.151
    raise ValueError(f"Unsupported dataset_key: {dataset_key}")


def compute_binned_stats(df: pd.DataFrame, bin_width: float = 0.05) -> pd.DataFrame:
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(df["fmf"].to_numpy(dtype=float), edges) - 1

    rows = []
    for i, center in enumerate(centers):
        sel = idx == i
        if np.sum(sel) == 0:
            rows.append({"fmf_center": center, "ae_mean": np.nan, "ae_std": np.nan, "count": 0})
            continue
        ae_bin = df["ae"].to_numpy(dtype=float)[sel]
        rows.append(
            {
                "fmf_center": center,
                "ae_mean": float(np.nanmean(ae_bin)),
                "ae_std": float(np.nanstd(ae_bin, ddof=1)) if ae_bin.size > 1 else 0.0,
                "count": int(ae_bin.size),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_fit_bands(
    df: pd.DataFrame,
    n_boot: int = 150,
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    fmf = df["fmf"].to_numpy(dtype=float)
    ae = df["ae"].to_numpy(dtype=float)
    fit_mask = fmf < 0.7

    fmf_grid = np.linspace(0.0, 1.0, 300)
    ae_grid = np.linspace(0.0, max(2.0, float(np.nanpercentile(ae, 99))), 300)

    g_curves = []
    f_curves = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fmf_s = fmf[idx]
        ae_s = ae[idx]
        try:
            g_coeffs = np.polyfit(fmf_s, ae_s, deg=3)
            g_curves.append(evaluate_g(g_coeffs, fmf_grid))
        except Exception:
            pass

        f_mask_s = fmf_s < 0.7
        if np.sum(f_mask_s) < 10:
            continue
        try:
            f_coeffs = np.polyfit(ae_s[f_mask_s], fmf_s[f_mask_s], deg=2)
            f_curves.append(evaluate_f(f_coeffs, ae_grid))
        except Exception:
            pass

    g_curves = np.asarray(g_curves, dtype=float)
    f_curves = np.asarray(f_curves, dtype=float)
    if g_curves.size == 0 or f_curves.size == 0:
        raise ValueError("Bootstrap fit bands failed; no successful fits.")

    return {
        "g": (
            fmf_grid,
            np.nanpercentile(g_curves, 2.5, axis=0),
            np.nanpercentile(g_curves, 97.5, axis=0),
            np.nanmedian(g_curves, axis=0),
        ),
        "f": (
            ae_grid,
            np.nanpercentile(f_curves, 2.5, axis=0),
            np.nanpercentile(f_curves, 97.5, axis=0),
            np.nanmedian(f_curves, axis=0),
        ),
    }


def save_figure2_style_panel(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_key: str,
    output_path: Path,
    fit_summary_path: Path,
) -> None:
    fits = fit_global_relationships(df)
    binned = compute_binned_stats(df)
    bands = bootstrap_fit_bands(df)

    fmf = df["fmf"].to_numpy(dtype=float)
    ae = df["ae"].to_numpy(dtype=float)
    ae_max = max(2.0, float(np.nanpercentile(ae, 99)))
    ae_grid = np.linspace(0.0, ae_max, 400)
    fmf_grid = np.linspace(0.0, 1.0, 400)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(ae, fmf, s=1, alpha=0.08, color="0.55", rasterized=True)

    valid_bins = binned["count"] > 0
    ax.plot(
        binned.loc[valid_bins, "ae_mean"],
        binned.loc[valid_bins, "fmf_center"],
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="Binned average (ΔFMF=0.05)",
    )
    ax.fill_betweenx(
        binned.loc[valid_bins, "fmf_center"],
        binned.loc[valid_bins, "ae_mean"] - binned.loc[valid_bins, "ae_std"],
        binned.loc[valid_bins, "ae_mean"] + binned.loc[valid_bins, "ae_std"],
        color="0.7",
        alpha=0.25,
        linewidth=0,
        label="Binned average ±1σ",
    )

    g_grid, g_lo, g_hi, g_med = bands["g"]
    ax.plot(g_med, g_grid, color="#1b9e77", linewidth=2.4, label="Global g(FMF) fit")
    ax.fill_betweenx(g_grid, g_lo, g_hi, color="#1b9e77", alpha=0.18, linewidth=0)

    f_grid, f_lo, f_hi, f_med = bands["f"]
    ax.plot(f_grid, f_med, color="#d95f02", linewidth=2.4, label="Global f(AE) fit")
    ax.fill_between(f_grid, f_lo, f_hi, color="#d95f02", alpha=0.18, linewidth=0)
    ax.plot(
        ae_grid,
        published_li_ginoux_f(ae_grid, dataset_key),
        color="#e6ab02",
        linewidth=2.0,
        linestyle="-.",
        label="Published Li-Ginoux f(AE)",
    )

    ax.plot(
        ae_grid,
        evaluate_anderson_2005(ae_grid),
        color="#377eb8",
        linewidth=2.0,
        label="Anderson et al. (2005)",
    )

    ax.set_xlim(0.0, ae_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("AE (500 nm SDA)")
    ax.set_ylabel("FMF (500 nm SDA)")
    ax.set_title(f"Figure 2-style global FMF-AE panel ({dataset_name})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.98,
        f"N = {len(df):,}\n"
        f"f(AE) using FMF < 0.7\n"
        f"g(FMF) cubic fit",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fit_summary = {
        "dataset_name": dataset_name,
        "n_samples": int(len(df)),
        "f_coeffs_quadratic": {
            "c2": float(fits["f_coeffs"][0]),
            "c1": float(fits["f_coeffs"][1]),
            "c0": float(fits["f_coeffs"][2]),
        },
        "g_coeffs_cubic": {
            "c3": float(fits["g_coeffs"][0]),
            "c2": float(fits["g_coeffs"][1]),
            "c1": float(fits["g_coeffs"][2]),
            "c0": float(fits["g_coeffs"][3]),
        },
        "references": {
            "paper": "Li & Ginoux (2025), https://doi.org/10.1029/2024GL114397",
            "figure_2_caption": "Figure 2b/2c global FMF versus AE panels",
            "fitted_to_fmf_lt_0p7": True,
        },
        "published_li_ginoux": {
            "dataset_key": dataset_key,
        },
    }
    with fit_summary_path.open("w") as handle:
        json.dump(fit_summary, handle, indent=2)


def save_overlay_comparison_figure(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_key: str,
    output_path: Path,
) -> None:
    fits = fit_global_relationships(df)
    ae = df["ae"].to_numpy(dtype=float)
    fmf = df["fmf"].to_numpy(dtype=float)

    ae_max = max(2.0, float(np.nanpercentile(ae, 99)))
    ae_grid = np.linspace(0.0, ae_max, 400)
    fmf_grid = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    ax = axes[0]
    ax.scatter(ae, fmf, s=1, alpha=0.07, color="0.55", rasterized=True)
    ax.plot(
        ae_grid,
        evaluate_f(fits["f_coeffs"], ae_grid),
        color="#d95f02",
        linewidth=2.4,
        label="Fit from saved collocations",
    )
    ax.plot(
        ae_grid,
        published_li_ginoux_f(ae_grid, dataset_key),
        color="#1b9e77",
        linewidth=2.4,
        linestyle="--",
        label="Published Li-Ginoux",
    )
    ax.plot(
        ae_grid,
        published_li_ginoux_f(ae_grid, "MEAN"),
        color="#7570b3",
        linewidth=2.0,
        linestyle=":",
        label="Published Li-Ginoux mean",
    )
    ax.set_xlim(0.0, ae_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("AE (500 nm SDA)")
    ax.set_ylabel("FMF (500 nm SDA)")
    ax.set_title(f"{dataset_name}: FMF = f(AE)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.scatter(fmf, ae, s=1, alpha=0.07, color="0.55", rasterized=True)
    ax.plot(
        fmf_grid,
        evaluate_g(fits["g_coeffs"], fmf_grid),
        color="#d95f02",
        linewidth=2.4,
        label="Fit from saved collocations",
    )
    ax.plot(
        fmf_grid,
        published_li_ginoux_g(fmf_grid, dataset_key),
        color="#1b9e77",
        linewidth=2.4,
        linestyle="--",
        label="Published Li-Ginoux",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(2.0, ae_max))
    ax.set_xlabel("FMF (500 nm SDA)")
    ax.set_ylabel("AE (500 nm SDA)")
    ax.set_title(f"{dataset_name}: AE = g(FMF)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle(f"{dataset_name}: saved-data fits vs published Li-Ginoux", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mod_df = extract_aeronet_sda_arrays(args.mod04)
    myd_df = extract_aeronet_sda_arrays(args.myd04)

    mod_csv = args.output_dir / "aeronet_mod_fmf_ae_samples.csv"
    myd_csv = args.output_dir / "aeronet_myd_fmf_ae_samples.csv"
    mod_df.to_csv(mod_csv, index=False)
    myd_df.to_csv(myd_csv, index=False)

    save_figure2_style_panel(
        df=mod_df,
        dataset_name="MOD-collocated AERONET",
        dataset_key="MOD",
        output_path=args.output_dir / "li_ginoux_figure2b_mod_style.png",
        fit_summary_path=args.output_dir / "li_ginoux_figure2b_mod_fit_summary.json",
    )
    save_figure2_style_panel(
        df=myd_df,
        dataset_name="MYD-collocated AERONET",
        dataset_key="MYD",
        output_path=args.output_dir / "li_ginoux_figure2c_myd_style.png",
        fit_summary_path=args.output_dir / "li_ginoux_figure2c_myd_fit_summary.json",
    )
    save_overlay_comparison_figure(
        df=mod_df,
        dataset_name="MOD-collocated AERONET",
        dataset_key="MOD",
        output_path=args.output_dir / "li_ginoux_overlay_mod_saved_vs_published.png",
    )
    save_overlay_comparison_figure(
        df=myd_df,
        dataset_name="MYD-collocated AERONET",
        dataset_key="MYD",
        output_path=args.output_dir / "li_ginoux_overlay_myd_saved_vs_published.png",
    )

    print(f"Saved MOD samples: {mod_csv}")
    print(f"Saved MYD samples: {myd_csv}")
    print(f"Saved MOD figure: {args.output_dir / 'li_ginoux_figure2b_mod_style.png'}")
    print(f"Saved MYD figure: {args.output_dir / 'li_ginoux_figure2c_myd_style.png'}")
    print(
        f"Saved MOD overlay: {args.output_dir / 'li_ginoux_overlay_mod_saved_vs_published.png'}"
    )
    print(
        f"Saved MYD overlay: {args.output_dir / 'li_ginoux_overlay_myd_saved_vs_published.png'}"
    )


if __name__ == "__main__":
    main()
