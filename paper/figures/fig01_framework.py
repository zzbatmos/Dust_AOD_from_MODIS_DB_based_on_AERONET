"""Frozen workflow schematic and unique-collocation site distribution."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import numpy as np

from dbdust.config import CONFIG
from paper.figures import figstyle as style
from paper.numbers import Registry


def box(ax, x, y, width, height, text, face="#F3F3F3"):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.003,rounding_size=.012",
                              facecolor=face, edgecolor="#505050", linewidth=.6))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7)


def arrow(ax, start, end, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                               linewidth=.8, color="#505050", **kwargs))


def make(registry_path, output):
    registry = Registry.read(registry_path)
    data = registry.table("derived/site_overview")
    evaluated = data.loc[data.role.eq("evaluation")]
    calibration = data.loc[data.role.eq("calibration")]
    fig = plt.figure(figsize=(style.DOUBLE, 100 * style.MM))
    flow = fig.add_axes([.025, .58, .95, .36])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis("off")
    style.panel_label(flow, "a", x=0, y=1.04)
    top_x = [.005, .25, .495, .74]
    top_text = ["AERONET V3 Level 2\ninversion retrievals",
                "1020-nm PLDR-LR\nDust fraction; $S_d$ = 47.3 sr",
                "Spectral transfer\nto 550 nm", "Reference DAOD\nHeld-out evaluation"]
    for x, text in zip(top_x, top_text):
        box(flow, x, .70, .22, .28, text, face="#FFF1DE")
    for x, nx in zip(top_x[:-1], top_x[1:]):
        arrow(flow, (x + .223, .84), (nx - .004, .84))
    lower_x = [.005, .205, .405, .605, .805]
    lower_text = ["MODIS Deep Blue\nLevel 2 retrievals",
                  "Valid inputs\nfull8; otherwise noSSA5",
                  "Two-stage XGBoost\nDust probability P\nConditional fraction",
                  "P and QA tiers: events\nQA2 DAOD: detection\nExpected value: averages",
                  "Monthly grids\nCompare with CALIOP\nand held-out AERONET"]
    for x, text in zip(lower_x, lower_text):
        box(flow, x, .07, .18, .43, text, face="#EAF3FA")
    for x, nx in zip(lower_x[:-1], lower_x[1:]):
        arrow(flow, (x + .183, .28), (nx - .004, .28))
    flow.plot([.85, .85, .495], [.696, .59, .59], color="#505050", linewidth=.8)
    arrow(flow, (.495, .59), (.495, .505))
    flow.text(.67, .63, "Training reference", ha="center", va="center", fontsize=6.5)

    ax = fig.add_axes([.045, .075, .61, .45], projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=.4, color="#4D4D4D")
    grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, xlocs=np.arange(-180, 181, 30),
                       ylocs=np.arange(-60, 91, 30), linewidth=.25, color="#BDBDBD")
    grid.top_labels = grid.right_labels = False
    grid.geo_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 6.5}
    grid.rotate_labels = False
    style.panel_label(ax, "b", x=-.04, y=1.02)
    regions = sorted(evaluated.region.unique())
    markers = ["o", "s", "D", "^", "v", "p", "h", "X", "P", "<", ">", "d", "H", "*", "8", "x", "+"]
    if len(regions) > len(markers):
        raise ValueError("Add distinguishable region markers rather than silently recycling them")
    colors = cmc.batlowS(np.arange(len(regions)))
    handles = []
    for index, region in enumerate(regions):
        q = evaluated.loc[evaluated.region.eq(region)]
        color, marker = colors[index], markers[index]
        ax.scatter(q.lon, q.lat, s=q.collocations * .18, marker=marker, color=color,
                   edgecolors="#505050", linewidths=.2, alpha=.8, transform=ccrs.PlateCarree(), zorder=3)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=3.5, ls="none",
                              label=region.replace("_", " ")))
    ax.scatter(calibration.lon, calibration.lat, s=15, marker="^", facecolors="none", edgecolors="black",
               linewidths=.6, transform=ccrs.PlateCarree(), zorder=4)
    fig.text(.66, .49, f"{len(evaluated)} evaluation sites\n{int(evaluated.collocations.sum()):,} collocations",
             fontsize=7, va="top")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.66, .42), ncol=2,
               columnspacing=.4, handletextpad=.3, borderaxespad=0, fontsize=6.5)
    sizes = [10, 100, 300]
    size_handles = [Line2D([], [], marker="o", color="#7F7F7F", ls="none",
                           markersize=np.sqrt(x * .18), label=f"{x}") for x in sizes]
    fig.legend(handles=size_handles, title="Collocations (marker area)", loc="lower left",
               bbox_to_anchor=(.65, .01), ncol=3, handletextpad=.3, columnspacing=.8, title_fontsize=6.5)
    fig.legend(handles=[Line2D([], [], marker="^", color="black", markerfacecolor="none", ls="none",
                               label=f"{len(calibration)} calibration sites")],
               loc="lower left", bbox_to_anchor=(.04, .005))
    entry = registry.payload["tables"]["derived/site_overview"]
    evidence = [{"table": "derived/site_overview", "source_file": entry["source_file"],
                 "source_sha256": entry["source_sha256"], "derivation": entry["derivation"]},
                {"workflow_config": CONFIG}]
    style.export(fig, output, "Fig01_framework", data, registry_path, evidence, __file__, raster=True)
    (Path(output) / "Fig01_caption.md").write_text(
        "Framework and data. (a) The nonsphericity-based AERONET reference trains a two-stage "
        "Deep Blue estimator, with distinct event-oriented and average-oriented outputs. "
        "(b) Evaluation sites colored and shaped by the fixed geographic holdout partition; "
        "marker area counts unique hierarchy-eligible collocations, not repeated predictions. "
        "Open black triangles mark the separate lidar-ratio calibration sites. All reserved "
        "calibration sites are excluded from evaluation, including reserved sites that contribute "
        "no selected calibration records.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
