from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from extract_modis_db_dust_model import extract_arrays_from_collocation_dict


PRESET_COMPARISONS = {
    "basic": [
        {
            "name": "aod440_vs_db550",
            "x": "AERONET_AOD440",
            "y": "MODIS_DB_AOD550",
            "xlabel": "AERONET AOD (440 nm)",
            "ylabel": "MODIS DB AOD (550 nm)",
            "title": "AERONET vs MODIS DB AOD",
            "scale": "log",
        },
        {
            "name": "aod675_vs_db660",
            "x": "AERONET_AOD675",
            "y": "MODIS_DB_AOD660",
            "xlabel": "AERONET AOD (675 nm)",
            "ylabel": "MODIS DB AOD (660 nm)",
            "title": "AERONET vs MODIS DB AOD",
            "scale": "log",
        },
        {
            "name": "sda_ae550_vs_db_ae",
            "x": "AERONET_SDA_AE550",
            "y": "MODIS_DB_AE",
            "xlabel": "AERONET SDA AE (500 nm)",
            "ylabel": "MODIS DB AE",
            "title": "AERONET SDA AE vs MODIS DB AE",
            "scale": "linear",
        },
    ],
    "extended": [
        {
            "name": "aod440_vs_db550",
            "x": "AERONET_AOD440",
            "y": "MODIS_DB_AOD550",
            "xlabel": "AERONET AOD (440 nm)",
            "ylabel": "MODIS DB AOD (550 nm)",
            "title": "AERONET vs MODIS DB AOD",
            "scale": "log",
        },
        {
            "name": "aod675_vs_db660",
            "x": "AERONET_AOD675",
            "y": "MODIS_DB_AOD660",
            "xlabel": "AERONET AOD (675 nm)",
            "ylabel": "MODIS DB AOD (660 nm)",
            "title": "AERONET vs MODIS DB AOD",
            "scale": "log",
        },
        {
            "name": "sda_aod550_vs_db550",
            "x": "AERONET_SDA_AOD550",
            "y": "MODIS_DB_AOD550",
            "xlabel": "AERONET SDA AOD (500 nm)",
            "ylabel": "MODIS DB AOD (550 nm)",
            "title": "AERONET SDA AOD vs MODIS DB AOD",
            "scale": "log",
        },
        {
            "name": "sda_ae550_vs_db_ae",
            "x": "AERONET_SDA_AE550",
            "y": "MODIS_DB_AE",
            "xlabel": "AERONET SDA AE (500 nm)",
            "ylabel": "MODIS DB AE",
            "title": "AERONET SDA AE vs MODIS DB AE",
            "scale": "linear",
        },
        {
            "name": "ssa440_vs_db412",
            "x": "AERONET_SSA440",
            "y": "MODIS_DB_SSA412",
            "xlabel": "AERONET SSA (440 nm)",
            "ylabel": "MODIS DB SSA (412 nm)",
            "title": "AERONET vs MODIS DB SSA",
            "scale": "linear",
        },
        {
            "name": "ssa675_vs_db660",
            "x": "AERONET_SSA675",
            "y": "MODIS_DB_SSA660",
            "xlabel": "AERONET SSA (675 nm)",
            "ylabel": "MODIS DB SSA (660 nm)",
            "title": "AERONET vs MODIS DB SSA",
            "scale": "linear",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare collocated AERONET and MODIS Deep Blue products with flexible density plots."
    )
    parser.add_argument("--mod04", type=Path, default=Path("../AERONET_MOD04_L2_collocation_with_SDA_data.pkl"))
    parser.add_argument("--myd04", type=Path, default=Path("../AERONET_MYD04_L2_collocation_with_SDA_data.pkl"))
    parser.add_argument(
        "--dataset",
        choices=["mod04", "myd04", "both"],
        default="both",
        help="Use Terra only, Aqua only, or the combined collocation set.",
    )
    parser.add_argument(
        "--comparison-set",
        choices=sorted(PRESET_COMPARISONS),
        default="basic",
        help="Preset set of comparisons to generate.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Custom pair in the form name:x:y:scale where scale is linear or log. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("aeronet_modis_db_comparisons"))
    parser.add_argument("--gridsize", type=int, default=60)
    parser.add_argument("--mincnt", type=int, default=1)
    parser.add_argument("--log-lo", type=float, default=1e-3)
    parser.add_argument("--log-hi", type=float, default=5.0)
    return parser.parse_args()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        colloc = pickle.load(handle)
    return extract_arrays_from_collocation_dict(colloc)


def combine_arrays(arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = arrays[0].keys()
    return {key: np.concatenate([arr[key] for arr in arrays]) for key in keys}


def build_comparison_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs = list(PRESET_COMPARISONS[args.comparison_set])
    for pair in args.pair:
        parts = pair.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid --pair '{pair}'. Expected name:x:y:scale")
        name, x_name, y_name, scale = parts
        if scale not in {"linear", "log"}:
            raise ValueError(f"Invalid scale '{scale}' in --pair '{pair}'")
        specs.append(
            {
                "name": name,
                "x": x_name,
                "y": y_name,
                "xlabel": x_name,
                "ylabel": y_name,
                "title": f"{x_name} vs {y_name}",
                "scale": scale,
            }
        )
    return specs


def paired_xy(data: dict[str, np.ndarray], x_key: str, y_key: str, scale: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data[x_key], dtype=float).ravel()
    y = np.asarray(data[y_key], dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if scale == "log":
        mask &= (x > 0.0) & (y > 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError(f"Not enough valid samples for {x_key} vs {y_key}")
    return x, y


def compute_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    slope, intercept, r, _, _ = linregress(x, y)
    diff = y - x
    return {
        "N": int(x.size),
        "mean_bias_modis_minus_aeronet": float(np.mean(diff)),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(mean_squared_error(x, y))),
        "Pearson_r": float(r),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def decorate_axes(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    metrics: dict[str, float | int],
    spec: dict[str, str],
    scale: str,
    log_lo: float,
    log_hi: float,
) -> None:
    if scale == "log":
        xx = np.logspace(np.log10(log_lo), np.log10(log_hi), 200)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(log_lo, log_hi)
        ax.set_ylim(log_lo, log_hi)
    else:
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        pad = 0.03 * (hi - lo) if hi > lo else 0.1
        lo = lo - pad
        hi = hi + pad
        xx = np.linspace(lo, hi, 200)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.plot(xx, xx, linestyle="--", linewidth=1.0, color="black")
    ax.plot(xx, metrics["slope"] * xx + metrics["intercept"], linewidth=1.2, color="crimson")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    ax.set_aspect("equal", adjustable="box")
    txt = (
        f"N = {metrics['N']}\n"
        f"r = {metrics['Pearson_r']:.3f}\n"
        f"Bias = {metrics['mean_bias_modis_minus_aeronet']:.3f}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"Fit: y = {metrics['slope']:.3f}x + {metrics['intercept']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    spec: dict[str, str],
    metrics: dict[str, float | int],
    output_path: Path,
    gridsize: int,
    mincnt: int,
    log_lo: float,
    log_hi: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    axes[0].scatter(x, y, s=8, alpha=0.18, linewidths=0, color="0.4", rasterized=True)
    decorate_axes(axes[0], x, y, metrics, spec, spec["scale"], log_lo, log_hi)
    axes[0].set_title(f"{spec['title']} (scatter)")

    hb = axes[1].hexbin(
        x,
        y,
        gridsize=gridsize,
        mincnt=mincnt,
        bins="log",
        xscale=spec["scale"],
        yscale=spec["scale"],
        cmap="viridis",
    )
    fig.colorbar(hb, ax=axes[1], label="log10(N)")
    decorate_axes(axes[1], x, y, metrics, spec, spec["scale"], log_lo, log_hi)
    axes[1].set_title(f"{spec['title']} (density)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    if args.dataset in {"mod04", "both"}:
        loaded["mod04"] = load_arrays(args.mod04)
    if args.dataset in {"myd04", "both"}:
        loaded["myd04"] = load_arrays(args.myd04)

    if args.dataset == "mod04":
        combined = loaded["mod04"]
    elif args.dataset == "myd04":
        combined = loaded["myd04"]
    else:
        combined = combine_arrays([loaded["mod04"], loaded["myd04"]])

    summary_records = []
    for spec in build_comparison_specs(args):
        x, y = paired_xy(combined, spec["x"], spec["y"], spec["scale"])
        metrics = compute_metrics(x, y)
        summary_records.append(
            {
                "comparison": spec["name"],
                "x_key": spec["x"],
                "y_key": spec["y"],
                "scale": spec["scale"],
                **metrics,
            }
        )
        plot_comparison(
            x=x,
            y=y,
            spec=spec,
            metrics=metrics,
            output_path=args.output_dir / f"{spec['name']}.png",
            gridsize=args.gridsize,
            mincnt=args.mincnt,
            log_lo=args.log_lo,
            log_hi=args.log_hi,
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(args.output_dir / "comparison_summary.csv", index=False)

    print(f"Dataset: {args.dataset}")
    print(f"Comparisons: {len(summary_df)}")
    print(summary_df[["comparison", "N", "Pearson_r", "RMSE", "mean_bias_modis_minus_aeronet"]].to_string(index=False))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
