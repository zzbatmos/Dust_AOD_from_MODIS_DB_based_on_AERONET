#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cartopy.crs as ccrs
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render daily Aqua MODIS L3 dust AOD maps and assemble GIF animations."
    )
    parser.add_argument(
        "--base-input-dir",
        type=Path,
        default=here / "modis_l3_daod_backfill",
        help="Directory containing MYD08_D3_LiGinoux and MYD08_D3_XGBoost daily NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "aqua_l3_daod_movies",
        help="Output directory for frames and GIFs.",
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--fps", type=float, default=12.0, help="GIF frame rate.")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


def parse_date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def list_files(path: Path, prefix: str, start_date: dt.date, end_date: dt.date) -> list[Path]:
    files = []
    for day in sorted(path.glob(f"{prefix}_*.nc")):
        date_text = day.stem.rsplit("_", 1)[-1]
        day_date = parse_date(date_text)
        if start_date <= day_date <= end_date:
            files.append(day)
    return files


def ensure_mplconfig() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def render_frame(
    input_path: str,
    output_path: str,
    title: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> str:
    ensure_mplconfig()
    ds = xr.open_dataset(input_path)
    dust = np.asarray(ds["dust_aod"].values, dtype=float)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)

    fig = plt.figure(figsize=(12, 6.2))
    ax = plt.axes(projection=ccrs.Robinson())
    mesh = ax.pcolormesh(
        lon,
        lat,
        dust,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="YlOrBr",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines(linewidth=0.5)
    ax.set_global()
    ax.set_title(title, fontsize=13)
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.03, shrink=0.7, fraction=0.035)
    cbar.set_label("Dust AOD at 550 nm")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    return output_path


def render_frame_job(args: tuple[str, str, str, float, float, int]) -> str:
    return render_frame(*args)


def build_frames(
    files: list[Path],
    frame_dir: Path,
    method_label: str,
    vmin: float,
    vmax: float,
    dpi: int,
    workers: int,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    frame_paths = []
    for path in files:
        date_text = path.stem.rsplit("_", 1)[-1]
        out = frame_dir / f"{path.stem}.png"
        title = f"{method_label} Aqua Daily Dust AOD ({date_text})"
        frame_paths.append(out)
        if not out.exists():
            jobs.append((str(path), str(out), title, vmin, vmax, dpi))

    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(render_frame_job, jobs))
    return frame_paths


def save_gif(frame_paths: list[Path], output_gif: Path, fps: float) -> None:
    duration = 1.0 / fps
    with imageio.get_writer(output_gif, mode="I", duration=duration, loop=0) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))


def render_side_by_side_frame(
    li_path: str,
    xgb_path: str,
    output_path: str,
    date_text: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> str:
    ensure_mplconfig()
    with xr.open_dataset(li_path) as li_ds, xr.open_dataset(xgb_path) as xgb_ds:
        li_dust = np.asarray(li_ds["dust_aod"].values, dtype=float)
        xgb_dust = np.asarray(xgb_ds["dust_aod"].values, dtype=float)
        lat = np.asarray(li_ds["lat"].values, dtype=float)
        lon = np.asarray(li_ds["lon"].values, dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.0),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    for ax, data, title in [
        (axes[0], li_dust, "Li-Ginoux"),
        (axes[1], xgb_dust, "XGBoost"),
    ]:
        mesh = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="YlOrBr",
            vmin=vmin,
            vmax=vmax,
        )
        ax.coastlines(linewidth=0.5)
        ax.set_global()
        ax.set_title(title, fontsize=12)
    fig.suptitle(f"Aqua Daily Dust AOD Comparison ({date_text})", fontsize=14)
    cbar = fig.colorbar(mesh, ax=axes, orientation="vertical", shrink=0.7, fraction=0.03, pad=0.03)
    cbar.set_label("Dust AOD at 550 nm")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_side_by_side_frame_job(
    args: tuple[str, str, str, str, float, float, int]
) -> str:
    return render_side_by_side_frame(*args)


def build_comparison_frames(
    li_files: list[Path],
    xgb_files: list[Path],
    frame_dir: Path,
    vmin: float,
    vmax: float,
    dpi: int,
    workers: int,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    xgb_by_date = {path.stem.rsplit("_", 1)[-1]: path for path in xgb_files}
    jobs = []
    frame_paths = []
    for li_path in li_files:
        date_text = li_path.stem.rsplit("_", 1)[-1]
        xgb_path = xgb_by_date.get(date_text)
        if xgb_path is None:
            continue
        out = frame_dir / f"aqua_compare_{date_text}.png"
        frame_paths.append(out)
        if not out.exists():
            jobs.append((str(li_path), str(xgb_path), str(out), date_text, vmin, vmax, dpi))
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(render_side_by_side_frame_job, jobs))
    return frame_paths


def main() -> None:
    args = parse_args()
    ensure_mplconfig()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    li_dir = args.base_input_dir / "MYD08_D3_LiGinoux"
    xgb_dir = args.base_input_dir / "MYD08_D3_XGBoost"
    li_files = list_files(li_dir, "MYD08_D3_DustAOD_LiGinoux", start_date, end_date)
    xgb_files = list_files(xgb_dir, "MYD08_D3_DustAOD_XGB", start_date, end_date)
    if not li_files or not xgb_files:
        raise SystemExit("No Aqua daily NetCDF files found in the requested date range.")

    li_frames = build_frames(
        li_files,
        args.output_dir / "frames_li_ginoux",
        "Li-Ginoux",
        args.vmin,
        args.vmax,
        args.dpi,
        args.workers,
    )
    xgb_frames = build_frames(
        xgb_files,
        args.output_dir / "frames_xgboost",
        "XGBoost",
        args.vmin,
        args.vmax,
        args.dpi,
        args.workers,
    )
    compare_frames = build_comparison_frames(
        li_files,
        xgb_files,
        args.output_dir / "frames_comparison",
        args.vmin,
        args.vmax,
        args.dpi,
        args.workers,
    )

    save_gif(li_frames, args.output_dir / "aqua_li_ginoux_daily_daod_2022_2024.gif", args.fps)
    save_gif(xgb_frames, args.output_dir / "aqua_xgboost_daily_daod_2022_2024.gif", args.fps)
    save_gif(compare_frames, args.output_dir / "aqua_li_vs_xgboost_daily_daod_2022_2024.gif", args.fps)

    print(f"[WROTE] {args.output_dir / 'aqua_li_ginoux_daily_daod_2022_2024.gif'}")
    print(f"[WROTE] {args.output_dir / 'aqua_xgboost_daily_daod_2022_2024.gif'}")
    print(f"[WROTE] {args.output_dir / 'aqua_li_vs_xgboost_daily_daod_2022_2024.gif'}")


if __name__ == "__main__":
    main()
