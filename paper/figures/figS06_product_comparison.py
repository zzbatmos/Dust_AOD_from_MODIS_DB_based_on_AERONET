"""Compare paired DB/AERONET products on the broad saved collocation cohort."""
import argparse
from pathlib import Path

import numpy as np

from paper.derive_collocation_overview import PAIR_FIELDS, VIEWS
from paper.numbers import Registry


def select(registry):
    pairs = registry.table("derived/collocation_product_pairs")
    statistics = registry.table("derived/collocation_product_statistics")
    if pairs.row_id.duplicated().any() or len(statistics) != 2 or set(statistics.panel) != set(PAIR_FIELDS):
        raise ValueError("Ambiguous product comparison")
    for panel, columns in PAIR_FIELDS.items():
        q = pairs.loc[pairs[f"{panel}_pair_valid"]]
        row = statistics.loc[statistics.panel.eq(panel)].iloc[0]
        if len(q) != row.rows or q.site_key.nunique() != row.sites:
            raise ValueError("Product comparison counts disagree")
        values = q[list(columns)].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Invalid comparison pair")
        low, high = VIEWS[panel]
        inside = ((values >= low) & (values <= high)).all(axis=1)
        if (row.view_min != low or row.view_max != high or row.n_inside_view != inside.sum()
                or row.n_outside_view != (~inside).sum()):
            raise ValueError("Display-window counts disagree")
    return pairs, statistics


def make(registry_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import pandas as pd
    from paper.figures import figstyle as style

    registry = Registry.read(registry_path)
    pairs, statistics = select(registry)
    fig = plt.figure(figsize=(style.DOUBLE, 100 * style.MM))
    artists = []
    for column, (panel, fields) in enumerate(PAIR_FIELDS.items()):
        ax = fig.add_axes([.10 + column * .43, .23, .34, .65])
        q = pairs.loc[pairs[f"{panel}_pair_valid"]]
        row = statistics.loc[statistics.panel.eq(panel)].iloc[0]
        low, high = VIEWS[panel]
        xy = q[list(fields)].to_numpy(float)
        inside = ((xy >= low) & (xy <= high)).all(axis=1)
        mesh = ax.hexbin(xy[inside, 0], xy[inside, 1], gridsize=70, mincnt=1,
                        extent=(low, high, low, high), norm=LogNorm(vmin=1),
                        cmap=style.CMAP_DENSITY, linewidths=0, rasterized=True)
        artists.append(mesh)
        ax.plot([low, high], [low, high], color="#888888", ls="--", lw=.6)
        ax.set(xlim=(low, high), ylim=(low, high), aspect="equal",
               xlabel="AERONET interpolated AOD550" if panel == "aod" else "AERONET AE (440-675 nm)",
               ylabel="MODIS DB AOD550" if panel == "aod" else "MODIS DB reported AE")
        ax.text(.035, .965, f"N = {int(row.rows):,} ({int(row.sites)} sites)\n"
                f"r = {row.r:.3f}; RMSE = {row.rmse:.3f}\nBias = {row.bias:+.3f}\n"
                f"Outside axes: {int(row.n_outside_view):,}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=.92))
        style.panel_label(ax, "ab"[column], x=-.14, y=1.04)
    maximum = max(2, max(float(h.get_array().max()) for h in artists))
    for mesh in artists:
        mesh.set_norm(LogNorm(vmin=1, vmax=maximum))
    colorbar = fig.colorbar(artists[0], cax=fig.add_axes([.91, .25, .015, .59]))
    colorbar.set_label("Collocations per hexagon")
    fig.text(.49, .055, "Central display windows; all valid pairs are retained in the statistics.",
             ha="center", fontsize=6.5)
    names = ["collocation_product_pairs", "collocation_product_statistics"]
    evidence = [dict(table="derived/" + name, **{k: registry.payload["tables"]["derived/" + name][k]
                    for k in ("source_file", "source_sha256", "derivation")}) for name in names]
    exported = pd.concat([pairs.assign(record_type="paired_collocation"),
                          statistics.assign(record_type="all_pair_statistics")], ignore_index=True, sort=False)
    style.export(fig, output, "FigS06_product_comparison", exported, registry_path, evidence, __file__, raster=True)
    (Path(output) / "FigS06_caption.md").write_text(
        "MODIS Deep Blue versus AERONET on the broad archived collocation cohort, not the smaller "
        "dust-reference training set. AOD550 uses the saved log-spectral interpolation of AERONET "
        "440- and 675-nm AOD; AERONET AE uses the same wavelength pair. DB AE is the reported "
        "product value. Each panel applies its own paired-valid mask; no dust-fraction, SSA, AOD-loading "
        "or calibration-site exclusion is imposed. Hexagon colors show counts on a shared logarithmic "
        "scale, and dashed lines show equality. Display windows are 0-2 for AOD and -0.5-3 for AE; "
        "out-of-window counts are stated. All valid pairs, including tails, contribute to Pearson r, "
        "RMSE and mean DB-minus-AERONET bias. Repeated site/granule sampling is not treated as "
        "independent for significance testing; no inferential test or regression fit is shown.\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
