#!/usr/bin/env python3
"""Plot Niamey yearly duststorm counts against Sahel Point 1 DAOD anomalies."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


NIAMEY_CSV = Path("/home/ec2-user/Research/Codex/niamey_duststorm_dates/niamey_duststorm_dates_2002_2024.csv")
MONTHLY_FILE = Path("/home/ec2-user/Research/Codex/modis_l3_monthly_products/MYD08_D3_TwoStageQA_monthly_2002_2024.nc")
OUTPUT_PNG = Path("/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/niamey_duststorm_vs_sahel_point1_yearly.png")
OUTPUT_CSV = Path("/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/niamey_duststorm_vs_sahel_point1_yearly.csv")


def load_duststorm_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    with NIAMEY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = int(row["year"])
            counts.setdefault(year, 0)
            counts[year] += int(row["n_duststorm_dates"])
    return counts


def load_sahel_point1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0, lon0 = 10.5, 5.5
    with Dataset(MONTHLY_FILE) as ds:
        years = np.asarray(ds["year"][:], dtype=int)
        months = np.asarray(ds["month"][:], dtype=int)
        lat = np.asarray(ds["lat"][:], dtype=float)
        lon = np.asarray(ds["lon"][:], dtype=float)
        arr = np.asarray(ds["dust_aod_qa2"][:], dtype=float)
    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        arr = arr[:, :, ::-1, :].copy()
    lat_idx = int(np.argmin(np.abs(lat - lat0)))
    lon_idx = int(np.argmin(np.abs(lon - lon0)))
    series = arr[:, :, lat_idx, lon_idx]
    climatology = np.nanmean(series, axis=0)
    anomaly = series - climatology[None, :]
    annual_mean = np.nanmean(anomaly, axis=1)
    annual_max = np.nanmax(anomaly, axis=1)
    return years, annual_mean, annual_max


def main() -> None:
    duststorm_counts = load_duststorm_counts()
    years, annual_mean, annual_max = load_sahel_point1()

    common_years = np.array([y for y in years if y in duststorm_counts], dtype=int)
    counts = np.array([duststorm_counts[y] for y in common_years], dtype=float)
    mean_vals = np.array([annual_mean[np.where(years == y)[0][0]] for y in common_years], dtype=float)
    max_vals = np.array([annual_max[np.where(years == y)[0][0]] for y in common_years], dtype=float)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "niamey_duststorm_dates", "sahel_point1_annual_mean_anomaly_qa2", "sahel_point1_annual_max_anomaly_qa2"])
        for y, c, m, mx in zip(common_years, counts, mean_vals, max_vals):
            writer.writerow([int(y), int(c), float(m), float(mx)])

    fig, ax1 = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax1.bar(common_years, counts, color="#c0841a", alpha=0.6, width=0.8, label="Niamey duststorm dates")
    ax1.set_ylabel("Niamey duststorm dates per year", color="#8a5b00")
    ax1.tick_params(axis="y", labelcolor="#8a5b00")
    ax1.set_xlabel("Year")
    ax1.grid(True, axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(common_years, mean_vals, color="#c81e1e", marker="o", linewidth=2.0, label="Sahel point #1 annual mean anomaly (QA2)")
    ax2.plot(common_years, max_vals, color="#2563eb", marker="^", linewidth=1.6, linestyle="--", label="Sahel point #1 annual max anomaly (QA2)")
    ax2.set_ylabel("Sahel point #1 deseasonalized DAOD anomaly", color="black")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.set_title("Niamey duststorm-date counts vs Sahel point #1 QA2 DAOD anomalies")
    ax1.set_xticks(common_years)
    ax1.set_xticklabels([str(y) for y in common_years], rotation=45, ha="right")

    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
