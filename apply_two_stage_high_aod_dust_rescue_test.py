#!/usr/bin/env python3
"""Test a high-AOD dust-rescue layer on saved 2017 Aqua L2 NPZ products.

This diagnostic script does not modify the standard two-stage product. It
reconstructs the standard QA2 dust AOD from saved per-granule NPZ fields and
adds a separate rescue layer for intense dust pixels that the two-stage model
suppresses.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - cartopy is optional.
    ccrs = None


ROOT = (
    Path(__file__).resolve().parent
    / "aqua_l2_2017_three_methods_noplot_by_day"
    / "MYD04_L2_2017-01-01_2017-12-31"
)
OUT_DIR = (
    Path(__file__).resolve().parent
    / "two_stage_high_aod_dust_rescue_test"
)

TYPE_LABELS = {0: "Mixed", 1: "Dust", 2: "Smoke", 3: "Sulfate"}


@dataclass(frozen=True)
class Case:
    name: str
    date: str
    label: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    min_pixels: int
    rois: tuple[tuple[str, float, float, float, float], ...]


CASES = {
    "dec7_north_africa": Case(
        name="dec7_north_africa",
        date="2017-12-07",
        label="North Africa / Sahara-Sahel",
        lat_min=0.0,
        lat_max=40.0,
        lon_min=-20.0,
        lon_max=60.0,
        min_pixels=100,
        rois=(
            ("full_domain", -20.0, 60.0, 0.0, 40.0),
            ("high_li_plume_core", 8.0, 17.0, 8.0, 16.0),
            ("western_plume", 0.0, 18.0, 8.0, 18.0),
            ("west_reference", -10.0, 0.0, 15.0, 25.0),
        ),
    ),
    "aug29_north_america": Case(
        name="aug29_north_america",
        date="2017-08-29",
        label="North America smoke / false-dust test",
        lat_min=32.0,
        lat_max=60.0,
        lon_min=-132.0,
        lon_max=-85.0,
        min_pixels=50,
        rois=(
            ("full_domain", -132.0, -85.0, 32.0, 60.0),
            ("upwind_smoke_west", -128.0, -112.0, 40.0, 57.0),
            ("central_false_dust_candidate", -112.0, -94.0, 40.0, 56.0),
            ("eastern_downwind", -96.0, -85.0, 40.0, 55.0),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="Case to run. May be repeated. Defaults to both initial test cases.",
    )
    parser.add_argument("--qa2-threshold", type=float, default=0.6)
    parser.add_argument("--fmf-max", type=float, default=0.7)
    parser.add_argument("--high-aod-threshold", type=float, default=1.5)
    parser.add_argument("--li-dust-threshold", type=float, default=1.0)
    parser.add_argument("--min-two-stage-probability", type=float, default=0.35)
    parser.add_argument("--rescue-cap-fraction", type=float, default=0.95)
    parser.add_argument(
        "--allowed-db-types",
        default="Dust",
        help="Comma-separated DB aerosol type labels allowed for rescue. Default: Dust.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def infer_li_ae_screen(li: np.ndarray, aod: np.ndarray, fmf_max: float) -> np.ndarray:
    coarse_fraction = np.full(aod.shape, np.nan, dtype=float)
    valid = np.isfinite(li) & np.isfinite(aod) & (aod > 0.0)
    np.divide(li, aod, out=coarse_fraction, where=valid)
    coarse_fraction = np.clip(coarse_fraction, 0.0, 1.0)
    fmf = 1.0 - coarse_fraction
    keep = valid & np.isfinite(fmf) & (fmf <= fmf_max)
    out = np.where(keep, li, np.nan)
    total_valid = np.isfinite(aod) & (aod > 0.0)
    out[total_valid & ~np.isfinite(out)] = 0.0
    out[~np.isfinite(aod)] = np.nan
    return out


def type_labels_to_codes(labels: str) -> set[int]:
    inverse = {v.lower(): k for k, v in TYPE_LABELS.items()}
    codes: set[int] = set()
    for raw in labels.split(","):
        label = raw.strip().lower()
        if not label:
            continue
        if label not in inverse:
            raise ValueError(f"Unknown DB aerosol type label: {raw!r}")
        codes.add(inverse[label])
    if not codes:
        raise ValueError("No allowed DB aerosol types were provided.")
    return codes


def compute_two_stage_qa2(
    aod: np.ndarray,
    probability: np.ndarray,
    fraction: np.ndarray,
    qa2_threshold: float,
) -> np.ndarray:
    out = np.zeros(aod.shape, dtype=float)
    keep = (
        np.isfinite(aod)
        & np.isfinite(probability)
        & np.isfinite(fraction)
        & (probability >= qa2_threshold)
    )
    out[keep] = np.clip(fraction[keep], 0.0, 1.0) * aod[keep]
    out[~np.isfinite(aod)] = np.nan
    return out


def apply_high_aod_rescue(
    *,
    aod: np.ndarray,
    li_ae_screened: np.ndarray,
    two_stage_qa2: np.ndarray,
    probability: np.ndarray,
    aerosol_type: np.ndarray,
    high_aod_threshold: float,
    li_dust_threshold: float,
    min_probability: float,
    cap_fraction: float,
    allowed_db_type_codes: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    allowed_type = np.isin(aerosol_type.astype(float), list(allowed_db_type_codes))
    candidate = (
        np.isfinite(aod)
        & np.isfinite(li_ae_screened)
        & np.isfinite(two_stage_qa2)
        & np.isfinite(probability)
        & (aod >= high_aod_threshold)
        & (li_ae_screened >= li_dust_threshold)
        & (probability >= min_probability)
        & allowed_type
    )
    capped_li = np.minimum(li_ae_screened, cap_fraction * aod)
    candidate_daod = np.where(candidate, capped_li, np.nan)
    rescued = two_stage_qa2.copy()
    improve = candidate & np.isfinite(candidate_daod) & (candidate_daod > two_stage_qa2)
    rescued[improve] = candidate_daod[improve]
    increment = rescued - two_stage_qa2
    increment[~np.isfinite(aod)] = np.nan
    return rescued, improve.astype(np.uint8), candidate.astype(np.uint8), increment


def region_mask(data: dict[str, np.ndarray], lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> np.ndarray:
    return (
        (data["lon"] >= lon_min)
        & (data["lon"] <= lon_max)
        & (data["lat"] >= lat_min)
        & (data["lat"] <= lat_max)
        & np.isfinite(data["aod550"])
    )


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def finite_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def count_types(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for code, label in TYPE_LABELS.items():
        out[label] = int(np.count_nonzero(values == code))
    return out


def summarize_subset(name: str, mask: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, object]:
    n = int(np.count_nonzero(mask))
    rescue_pixels = int(np.count_nonzero(mask & (data["rescue_flag"] > 0)))
    candidate_pixels = int(np.count_nonzero(mask & (data["rescue_candidate"] > 0)))
    return {
        "region": name,
        "n_valid_aod_pixels": n,
        "rescue_candidate_pixels": candidate_pixels,
        "rescued_pixels": rescue_pixels,
        "rescued_pixel_fraction": float(rescue_pixels / n) if n else float("nan"),
        "mean_aod550": finite_mean(data["aod550"][mask]),
        "mean_li_ginoux_ae142": finite_mean(data["li_ae142"][mask]),
        "mean_two_stage_qa2": finite_mean(data["two_stage_qa2"][mask]),
        "mean_rescued_qa2": finite_mean(data["two_stage_qa2_rescue"][mask]),
        "mean_rescue_increment": finite_mean(data["rescue_increment"][mask]),
        "p90_rescue_increment": finite_percentile(data["rescue_increment"][mask], 90.0),
        "mean_two_stage_probability": finite_mean(data["two_stage_probability"][mask]),
        "mean_two_stage_fraction": finite_mean(data["two_stage_fraction"][mask]),
        "aerosol_type_counts": count_types(data["aerosol_type"][mask]),
    }


def robust_max(arrays: Iterable[np.ndarray], percentile: float, minimum: float) -> float:
    finite_parts = [arr[np.isfinite(arr)].ravel() for arr in arrays if np.any(np.isfinite(arr))]
    if not finite_parts:
        return minimum
    merged = np.concatenate(finite_parts)
    if merged.size == 0:
        return minimum
    return float(max(np.percentile(merged, percentile), minimum))


def add_map_context(ax, case: Case) -> None:
    if ccrs:
        ax.set_extent([case.lon_min, case.lon_max, case.lat_min, case.lat_max], crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="110m", linewidth=0.7)
        except Exception:
            pass
        ax.gridlines(draw_labels=True, linewidth=0.3, color="0.4", alpha=0.5)
    else:
        ax.set_xlim(case.lon_min, case.lon_max)
        ax.set_ylim(case.lat_min, case.lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)


def collect_case(case: Case, args: argparse.Namespace, allowed_db_type_codes: set[int]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    granule_dir = args.input_root / case.date / "granules"
    if not granule_dir.exists():
        raise FileNotFoundError(f"Missing granule directory: {granule_dir}")

    rows: list[dict[str, object]] = []
    pieces: dict[str, list[np.ndarray]] = {
        "lat": [],
        "lon": [],
        "aod550": [],
        "li_ae142": [],
        "two_stage_probability": [],
        "two_stage_fraction": [],
        "two_stage_qa2": [],
        "two_stage_qa2_rescue": [],
        "rescue_flag": [],
        "rescue_candidate": [],
        "rescue_increment": [],
        "aerosol_type": [],
    }

    for npz_path in sorted(granule_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            lat = np.asarray(data["lat"], dtype=float)
            lon = wrap_lon(np.asarray(data["lon"], dtype=float))
            aod = np.asarray(data["aod550"], dtype=float)
            li_original = np.asarray(data["li_dust_aod"], dtype=float)
            probability = np.asarray(data["two_stage_probability"], dtype=float)
            fraction = np.asarray(data["two_stage_fraction"], dtype=float)
            aerosol_type = np.asarray(data["aerosol_type_code"], dtype=float)

        mask = (
            (lat >= case.lat_min)
            & (lat <= case.lat_max)
            & (lon >= case.lon_min)
            & (lon <= case.lon_max)
            & np.isfinite(aod)
        )
        n_pixels = int(np.count_nonzero(mask))
        if n_pixels < case.min_pixels:
            continue

        li_ae142 = infer_li_ae_screen(li_original, aod, fmf_max=args.fmf_max)
        two_stage_qa2 = compute_two_stage_qa2(aod, probability, fraction, args.qa2_threshold)
        rescue, flag, candidate, increment = apply_high_aod_rescue(
            aod=aod,
            li_ae_screened=li_ae142,
            two_stage_qa2=two_stage_qa2,
            probability=probability,
            aerosol_type=aerosol_type,
            high_aod_threshold=args.high_aod_threshold,
            li_dust_threshold=args.li_dust_threshold,
            min_probability=args.min_two_stage_probability,
            cap_fraction=args.rescue_cap_fraction,
            allowed_db_type_codes=allowed_db_type_codes,
        )

        rows.append(
            {
                "date": case.date,
                "granule": npz_path.stem,
                "path": str(npz_path),
                "n_pixels": n_pixels,
                "mean_aod550": finite_mean(aod[mask]),
                "mean_li_ae142": finite_mean(li_ae142[mask]),
                "mean_two_stage_qa2": finite_mean(two_stage_qa2[mask]),
                "mean_rescued_qa2": finite_mean(rescue[mask]),
                "rescued_pixels": int(np.count_nonzero(flag[mask] > 0)),
            }
        )

        arrays = {
            "lat": lat,
            "lon": lon,
            "aod550": aod,
            "li_ae142": li_ae142,
            "two_stage_probability": probability,
            "two_stage_fraction": fraction,
            "two_stage_qa2": two_stage_qa2,
            "two_stage_qa2_rescue": rescue,
            "rescue_flag": flag,
            "rescue_candidate": candidate,
            "rescue_increment": increment,
            "aerosol_type": aerosol_type,
        }
        for key, arr in arrays.items():
            pieces[key].append(arr[mask])

    if not rows:
        return rows, {}
    return rows, {key: np.concatenate(parts) for key, parts in pieces.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(case: Case, data: dict[str, np.ndarray], args: argparse.Namespace, out_dir: Path) -> Path:
    out = out_dir / f"MYD04_L2_{case.date}_{case.name}_two_stage_high_aod_rescue.png"
    aod_vmax = robust_max([data["aod550"]], 98.0, 0.2)
    daod_vmax = robust_max(
        [data["li_ae142"], data["two_stage_qa2"], data["two_stage_qa2_rescue"]],
        98.0,
        0.05,
    )
    inc_vmax = robust_max([data["rescue_increment"]], 99.0, 0.02)

    panels = [
        ("aod550", "DB AOD550", "viridis", 0.0, aod_vmax),
        ("li_ae142", "Li-Ginoux DAOD, AE<1.42", "plasma", 0.0, daod_vmax),
        ("two_stage_qa2", "Two-stage QA2 DAOD", "plasma", 0.0, daod_vmax),
        ("two_stage_qa2_rescue", "Two-stage QA2 + high-AOD rescue", "plasma", 0.0, daod_vmax),
        ("rescue_increment", "Rescue increment", "magma", 0.0, inc_vmax),
        ("rescue_flag", "Rescued pixels", "gray_r", 0.0, 1.0),
        ("two_stage_probability", "Two-stage dust probability", "viridis", 0.0, 1.0),
        ("aerosol_type", "DB aerosol type code", "tab10", -0.5, 3.5),
    ]

    projection = ccrs.PlateCarree() if ccrs else None
    subplot_kw = {"projection": projection} if projection else {}
    transform = ccrs.PlateCarree() if ccrs else None
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), subplot_kw=subplot_kw, constrained_layout=True)

    for ax, (key, title, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        add_map_context(ax, case)
        scatter_kwargs = {
            "c": data[key],
            "s": 3,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "linewidths": 0,
            "alpha": 0.9,
        }
        if transform:
            scatter_kwargs["transform"] = transform
        mesh = ax.scatter(data["lon"], data["lat"], **scatter_kwargs)
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
        if key == "aerosol_type":
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(["Mixed", "Dust", "Smoke", "Sulfate"])
        elif key in {"rescue_flag"}:
            cbar.set_ticks([0, 1])

    fig.suptitle(
        f"Aqua MYD04_L2 high-AOD dust-rescue v0 | {case.label} | {case.date}\n"
        f"AOD>={args.high_aod_threshold:g}, Li DAOD>={args.li_dust_threshold:g}, "
        f"Pdust>={args.min_two_stage_probability:g}, cap={args.rescue_cap_fraction:g}*AOD, "
        f"DB types={args.allowed_db_types}",
        fontsize=15,
    )
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def run_case(case: Case, args: argparse.Namespace, allowed_db_type_codes: set[int]) -> dict[str, object]:
    rows, data = collect_case(case, args, allowed_db_type_codes)
    if not rows or not data:
        raise RuntimeError(f"No valid granules selected for {case.name}")

    out_dir = args.output_dir / case.name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / f"MYD04_L2_{case.date}_{case.name}_selected_granules.csv", rows)

    npz_path = out_dir / f"MYD04_L2_{case.date}_{case.name}_two_stage_high_aod_rescue_arrays.npz"
    np.savez_compressed(npz_path, **data)

    summary_rows = []
    for name, lon_min, lon_max, lat_min, lat_max in case.rois:
        mask = region_mask(data, lon_min, lon_max, lat_min, lat_max)
        summary_rows.append(summarize_subset(name, mask, data))

    summary_csv_rows = []
    for row in summary_rows:
        flat = {key: value for key, value in row.items() if key != "aerosol_type_counts"}
        for label, count in row["aerosol_type_counts"].items():
            flat[f"aerosol_type_{label.lower()}_pixels"] = count
        summary_csv_rows.append(flat)
    write_csv(out_dir / f"MYD04_L2_{case.date}_{case.name}_rescue_summary.csv", summary_csv_rows)

    figure_path = plot_case(case, data, args, out_dir)

    payload = {
        "case": case.name,
        "date": case.date,
        "label": case.label,
        "thresholds": {
            "qa2_threshold": args.qa2_threshold,
            "fmf_max": args.fmf_max,
            "high_aod_threshold": args.high_aod_threshold,
            "li_dust_threshold": args.li_dust_threshold,
            "min_two_stage_probability": args.min_two_stage_probability,
            "rescue_cap_fraction": args.rescue_cap_fraction,
            "allowed_db_types": args.allowed_db_types,
        },
        "n_granules": len(rows),
        "n_valid_pixels": int(data["aod550"].size),
        "outputs": {
            "arrays_npz": str(npz_path),
            "figure": str(figure_path),
            "summary_csv": str(out_dir / f"MYD04_L2_{case.date}_{case.name}_rescue_summary.csv"),
        },
        "roi_summaries": summary_rows,
    }
    summary_json = out_dir / f"MYD04_L2_{case.date}_{case.name}_rescue_summary.json"
    with summary_json.open("w") as handle:
        json.dump(payload, handle, indent=2)
    payload["outputs"]["summary_json"] = str(summary_json)
    return payload


def main() -> None:
    args = parse_args()
    allowed_db_type_codes = type_labels_to_codes(args.allowed_db_types)
    case_names = args.case or ["dec7_north_africa", "aug29_north_america"]
    outputs = []
    for case_name in case_names:
        payload = run_case(CASES[case_name], args, allowed_db_type_codes)
        outputs.append(payload)
        print(f"{case_name}: figure={payload['outputs']['figure']}")
        print(f"{case_name}: summary={payload['outputs']['summary_json']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_summary.json").open("w") as handle:
        json.dump(outputs, handle, indent=2)


if __name__ == "__main__":
    main()
