#!/usr/bin/env python3
"""Plot MODIS Deep Blue Aerosol Type QA flag for selected L2 granules."""

from __future__ import annotations

import argparse
import csv
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
        "--comparison-output",
        type=Path,
        default=here / "modis_db_aerosol_type_flag_two_case_comparison.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=here / "modis_db_aerosol_type_flag_two_case_summary.csv",
    )
    return parser.parse_args()


def decode_db_aerosol_type(granule_path: Path) -> dict[str, np.ndarray]:
    granule = MODIS_Lib.MODIS_DeepBlue_Level2(str(granule_path))
    lat = np.asarray(granule.lat, dtype=float)
    lon = np.asarray(granule.lon, dtype=float)

    f = SD(str(granule_path), SDC.READ)
    qa = np.asarray(f.select("Quality_Assurance_Land").get(), dtype=np.int16)
    f.end()

    # Per the C6 QA plan, the Deep Blue QA block is in the land QA SDS.
    # In these files, byte index 4 contains:
    #   bit 0   : DB usefulness
    #   bits 1-2: DB confidence
    #   bits 3-4: DB aerosol type
    byte4 = qa[:, :, 4].astype(np.uint8)
    usefulness = (byte4 & 0b1).astype(np.uint8)
    confidence = ((byte4 >> 1) & 0b11).astype(np.uint8)
    aerosol_type = ((byte4 >> 3) & 0b11).astype(np.uint8)

    aerosol_type_masked = aerosol_type.astype(float)
    aerosol_type_masked[usefulness == 0] = np.nan

    return {
        "lat": lat,
        "lon": lon,
        "usefulness": usefulness,
        "confidence": confidence,
        "aerosol_type": aerosol_type,
        "aerosol_type_masked": aerosol_type_masked,
    }


def summarize_case(case_name: str, decoded: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    useful = decoded["usefulness"] == 1
    confidence = decoded["confidence"]
    aerosol_type = decoded["aerosol_type"]
    rows.append(
        {
            "case": case_name,
            "category": "All useful",
            "count": int(useful.sum()),
            "fraction_of_useful": 1.0 if useful.any() else np.nan,
            "mean_confidence": float(confidence[useful].mean()) if useful.any() else np.nan,
        }
    )
    for code, label in TYPE_LABELS.items():
        mask = useful & (aerosol_type == code)
        rows.append(
            {
                "case": case_name,
                "category": label,
                "count": int(mask.sum()),
                "fraction_of_useful": float(mask.sum() / useful.sum()) if useful.any() else np.nan,
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            }
        )
    return rows


def save_case_map(
    output_path: Path,
    title: str,
    lat: np.ndarray,
    lon: np.ndarray,
    aerosol_type_masked: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    lat = np.asarray(lat, dtype=float)
    z = np.asarray(aerosol_type_masked, dtype=float)

    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
    if not np.any(finite):
        raise ValueError(f"No useful aerosol-type pixels found for {output_path.name}.")

    cmap = ListedColormap(["#9e9e9e", "#c28f0e", "#7d3cff", "#2f77b4"])
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


def save_two_case_comparison(
    output_path: Path,
    dust_case: dict[str, np.ndarray],
    smoke_case: dict[str, np.ndarray],
    dust_title: str,
    smoke_title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#9e9e9e", "#c28f0e", "#7d3cff", "#2f77b4"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    for ax, case, title in zip(axes, [dust_case, smoke_case], [dust_title, smoke_title]):
        lat = np.asarray(case["lat"], dtype=float)
        lon = ((np.asarray(case["lon"], dtype=float) + 180.0) % 360.0) - 180.0
        z = np.asarray(case["aerosol_type_masked"], dtype=float)
        finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle=":")
        if np.any(finite):
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
            ax.set_extent(
                [
                    float(np.nanmin(lon[finite])) - 0.8,
                    float(np.nanmax(lon[finite])) + 0.8,
                    float(np.nanmin(lat[finite])) - 0.8,
                    float(np.nanmax(lat[finite])) + 0.8,
                ],
                crs=ccrs.PlateCarree(),
            )
        ax.set_title(title)

    cb = fig.colorbar(mesh, ax=axes, location="right", pad=0.03, shrink=0.82, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels([TYPE_LABELS[i] for i in range(4)])
    cb.set_label("Deep Blue Aerosol Type")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    dust_case = decode_db_aerosol_type(args.dust_file)
    smoke_case = decode_db_aerosol_type(args.smoke_file)

    dust_png = args.dust_output_dir / f"{args.dust_file.stem}_db_aerosol_type_flag.png"
    smoke_png = args.smoke_output_dir / f"{args.smoke_file.stem}_db_aerosol_type_flag.png"

    save_case_map(dust_png, "Deep Blue Aerosol Type QA Flag: heavy dust case", **{
        "lat": dust_case["lat"], "lon": dust_case["lon"], "aerosol_type_masked": dust_case["aerosol_type_masked"]
    })
    save_case_map(smoke_png, "Deep Blue Aerosol Type QA Flag: heavy smoke case", **{
        "lat": smoke_case["lat"], "lon": smoke_case["lon"], "aerosol_type_masked": smoke_case["aerosol_type_masked"]
    })
    save_two_case_comparison(
        args.comparison_output,
        dust_case,
        smoke_case,
        "Heavy dust case: 2025-03-14 Terra",
        "Heavy smoke case: 2024-09-22 Terra",
    )

    rows = summarize_case("heavy_dust_2025-03-14", dust_case) + summarize_case("heavy_smoke_2024-09-22", smoke_case)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", "category", "count", "fraction_of_useful", "mean_confidence"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.summary_csv.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print(f"Wrote {dust_png}")
    print(f"Wrote {smoke_png}")
    print(f"Wrote {args.comparison_output}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
