"""Display the frozen lidar-ratio calibration, without recalibrating it."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter
import numpy as np
import pandas as pd

from paper.figures import figstyle as style
from paper.numbers import Registry


def make(registry_path, output):
    registry = Registry.read(registry_path)
    records = registry.table("derived/calibration_records")
    sites = registry.table("derived/site_overview", role="calibration")
    selectors = dict(reference="experimental1020", support="pre2010_reserved_dust_calibration", scope="calibration")
    numbers = {key: registry.number("derived/calibration_summary", key, **selectors)
               for key in ("selected_rows", "selected_sites", "Sd_sr")}
    sd = numbers["Sd_sr"]["value"]
    fig = plt.figure(figsize=(style.DOUBLE, 75 * style.MM))
    ax = fig.add_axes([.08, .22, .32, .66])
    counts, edges, _ = ax.hist(records["Lidar_Ratio[1020nm]"], bins=np.arange(10, 151, 5),
                                color="#BDBDBD", edgecolor="#505050", linewidth=.4)
    ax.axvline(sd, color="black", ls="--", lw=1)
    ax.text(.98, .96, f"N = {int(numbers['selected_rows']['value']):,}\n"
                     f"{int(numbers['selected_sites']['value'])} calibration sites\n"
                     f"Pooled median = {sd:.3f} sr", ha="right", va="top", transform=ax.transAxes)
    ax.set(xlim=(10, 150), xlabel="Lidar ratio at 1020 nm (sr)", ylabel="Selected record count")
    style.panel_label(ax, "a", x=-.14)
    geo = fig.add_axes([.48, .22, .50, .66], projection=ccrs.Robinson())
    geo.set_global()
    geo.coastlines(resolution="110m", linewidth=.4, color="#4D4D4D")
    geo.scatter(sites.lon, sites.lat, s=20, marker="^", facecolors="none", edgecolors="black",
                linewidths=.6, transform=ccrs.PlateCarree(), zorder=3)
    grid = geo.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 61, 30),
                        linewidth=.25, color="#BDBDBD")
    grid.top_labels = grid.right_labels = grid.geo_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 6.5}
    grid.rotate_labels = False
    longitude = LongitudeFormatter()
    grid.xformatter = FuncFormatter(lambda value, position: longitude(value) if value % 60 == 0 else "")
    style.panel_label(geo, "b", x=-.06)
    data = pd.concat([records.assign(panel="a", record_type="selected_record"),
                      sites.assign(panel="b", record_type="calibration_site"),
                      pd.DataFrame({"panel": "a", "record_type": "histogram_bin", "bin_low": edges[:-1],
                                    "bin_high": edges[1:], "count": counts})], ignore_index=True)
    evidence = list(numbers.values()) + [{"table": name,
                  "source_file": registry.payload["tables"][name]["source_file"],
                  "source_sha256": registry.payload["tables"][name]["source_sha256"],
                  "derivation": registry.payload["tables"][name]["derivation"]}
                 for name in ("derived/calibration_records", "derived/site_overview")]
    style.export(fig, output, "FigS02_calibration", data, registry_path, evidence, __file__, raster=True)
    (Path(output) / "FigS02_caption.md").write_text(
        "Frozen dust lidar-ratio calibration. (a) Distribution of the saved selected 1020-nm "
        "lidar-ratio records; the dashed line marks the pooled-record median, not the median "
        "of site medians. (b) Calibration-site locations. Selection is restricted to reserved "
        "sites before 2010, clipped dust backscatter fraction above 0.9, and lidar ratio "
        "10-150 sr. No CALIOP data enter this calibration; calibration sites are excluded "
        "from model evaluation. No selection or fitting was repeated for this plot.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
