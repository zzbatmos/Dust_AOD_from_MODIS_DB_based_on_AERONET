"""Registry-only S1 archive diagnostic, qualified presentation approved 2026-10-14."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from paper.numbers import Registry, digest
from paper.figures import figstyle as style
from paper.figures.figS13_all_seasonal_maps import check_layout

TABLES = dict(points="coarse_fraction_display", tail="coarse_fraction_tail", checks="coarse_fraction_formula_checks")
WAVES = (675, 1020)
AUTHOR_ACCEPTANCE = "approved_2026-10-14"
HEX_GRID = 40  # Display resolution only; no selection or scientific binning.


def select(registry):
    frames = {key: registry.table("derived/" + name) for key, name in TABLES.items()}
    points, tail, checks = (frames[k] for k in TABLES)
    for key, frame in frames.items():
        metadata = registry.payload["tables"]["derived/" + TABLES[key]]["derivation"]
        if (not frame.author_acceptance.eq(AUTHOR_ACCEPTANCE).all()
                or metadata.get("author_acceptance") != AUTHOR_ACCEPTANCE):
            raise ValueError("Require the author-approved 2026-10-14 qualified presentation")
    if (len(tail) != len(WAVES) or set(tail.wavelength) != set(WAVES)
            or tail.wavelength.duplicated().any() or not tail.subset.eq("reference_valid").all()):
        raise ValueError("Wrong wavelength-specific reference_valid tails")
    if not points.cohort.eq("reference_valid").all() or set(points.wavelength) != set(WAVES):
        raise ValueError("Archive cohort changed or silently pooled")
    if (points.duplicated(["wavelength", "source_row"]).any() or (points.source_row < 0).any()
            or not np.equal(points.source_row, np.floor(points.source_row)).all()):
        raise ValueError("Invalid archive row identity")
    if (len(checks) != 3 or set(checks.saved_column) != {"old675LR56", "experimental1020_native_raw", "experimental1020_native"}
            or not checks.cohort.eq("grouped_rows_formula_check_only").all()
            or not (checks.n_finite.gt(0) & checks.n.ge(checks.n_finite)).all()
            or not checks.max_abs_error.between(0, 1e-12).all()):
        raise ValueError("Missing/failed saved grouped formula checks")
    metadata = registry.payload["tables"]["derived/coarse_fraction_display"]["derivation"]
    recipe = metadata["fixed_recipe"]
    reference = CONFIG["reference"]
    if (recipe["primary_lr_sr"] != reference["dust_lidar_ratio_sr"]
            or recipe["nondust_pldr"] != reference["nondust_pldr"] or recipe["dust_pldr"] != reference["dust_pldr"]):
        raise ValueError("Registry recipe differs from frozen reference")
    for wave in WAVES:
        frame = points.loc[points.wavelength.eq(wave)]
        row = tail.loc[tail.wavelength.eq(wave)].iloc[0]
        primary = wave == CONFIG["reference"]["wavelength_nm"]
        expected = dict(reference="experimental1020_native" if primary else "old675LR56",
                        role="primary_native" if primary else "sensitivity",
                        fixed_lr_sr=recipe["primary_lr_sr"] if primary else recipe["old_lr_sr"])
        if any(not frame[key].eq(value).all() for key, value in expected.items()):
            raise ValueError("Native primary/sensitivity labels or constants changed")
        values = frame[["inversion_cmf", "pldr", "observed_lr_sr", "backscatter_raw", "backscatter_fraction",
                        "native_extinction_raw", "native_extinction_fraction"]]
        if (not np.isfinite(values).all().all() or not frame.inversion_cmf.between(recipe["cmf_lower"], row.cmf_upper).all()
                or not frame.pldr.between(recipe["pldr_lower"], recipe["pldr_upper"]).all()
                or not frame.observed_lr_sr.between(recipe["lr_lower"], recipe["lr_upper"]).all()):
            raise ValueError("Invalid values on saved reference_valid support")
        if (not np.allclose(frame.backscatter_fraction, np.clip(frame.backscatter_raw, 0, 1), atol=1e-12, rtol=0)
                or not np.allclose(frame.native_extinction_raw, frame.backscatter_fraction * frame.fixed_lr_sr / frame.observed_lr_sr, atol=1e-12, rtol=0)
                or not np.allclose(frame.native_extinction_fraction, np.clip(frame.native_extinction_raw, 0, 1), atol=1e-12, rtol=0)):
            raise ValueError("Registered fraction violates frozen clip order")
        tail_count = int((frame.inversion_cmf > row.tail_threshold).sum())
        if (len(frame) != row.n or tail_count != row.n_gt_0p85
                or not np.isclose(tail_count / len(frame), row.fraction_gt_0p85, rtol=0, atol=1e-12)
                or not np.isclose(frame.inversion_cmf.max(), row.maximum_before_clip, rtol=0, atol=1e-12)
                or row.tail_threshold != recipe["tail_threshold"] or row.cmf_upper != recipe["cmf_upper"]):
            raise ValueError("Registry tail counts/extent differ from display cohort")
    evidence = [dict(table="derived/" + name, source_file=registry.payload["tables"]["derived/" + name]["source_file"],
        source_sha256=registry.payload["tables"]["derived/" + name]["source_sha256"],
        derivation=registry.payload["tables"]["derived/" + name]["derivation"]) for name in TABLES.values()]
    return frames, evidence


def build_figure(frames):
    fig = plt.figure(figsize=(style.DOUBLE, 110 * style.MM))
    fig.text(.02, .98, "Inversion coarse-mode extinction fraction, not SDA", fontsize=7, va="top")
    points, tail = frames["points"], frames["tail"]
    upper = tail.cmf_upper.iloc[0]
    collections = []
    for index, wave in enumerate(WAVES):
        frame = points.loc[points.wavelength.eq(wave)]
        ax = fig.add_axes([.08 + index * .325, .34, .245, .46])
        collection = ax.hexbin(frame.inversion_cmf, frame.native_extinction_fraction, gridsize=HEX_GRID,
            extent=(0, upper, 0, 1), mincnt=1, cmap=style.CMAP_DENSITY, linewidths=0, rasterized=True)
        collections.append(collection)
        ax.set(xlim=(0, upper), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1],
               xlabel=f"Inversion CMF ({wave} nm)")
        if index == 0:
            ax.set_ylabel("PLDR-LR native dust\nextinction fraction")
        role = "old56 sensitivity" if index == 0 else "primary native, LR47.308"
        ax.set_title(f"{wave} nm: {role}\nn = {len(frame):,}", fontsize=7, pad=10)
        style.panel_label(ax, chr(ord("a") + index), x=-.22, y=1.15)
    maximum = max(float(c.get_array().max()) for c in collections)
    norm = LogNorm(vmin=1, vmax=max(2, maximum))
    for collection in collections:
        collection.set_norm(norm)
    cax = fig.add_axes([.15, .18, .42, .022])
    bar = fig.colorbar(collections[0], cax=cax, orientation="horizontal")
    ticks = np.unique([1, max(1, int(np.sqrt(maximum))), max(1, int(maximum))])
    bar.set_ticks(ticks)
    bar.set_ticklabels([str(x) for x in ticks])
    bar.minorticks_off()
    bar.set_label("Archive records per display hexagon")
    ax = fig.add_axes([.745, .34, .235, .46])
    for wave, color, linestyle in zip(WAVES, ("#4D4D4D", "#D55E00"), ("--", "-")):
        values = np.sort(points.loc[points.wavelength.eq(wave), "inversion_cmf"].to_numpy())
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), drawstyle="steps-post", color=color,
                linestyle=linestyle, label=f"{wave} nm", rasterized=True)
    ax.axvline(tail.tail_threshold.iloc[0], color="#7F7F7F", lw=.7, linestyle=":")
    ax.set(xlim=(0, upper), ylim=(0, 1.03), xticks=[0, .5, 1], yticks=[0, .5, 1],
           xlabel="Inversion CMF", ylabel="Cumulative proportion")
    ax.set_title("Distinct archive cohorts\nNo CMF clipping", fontsize=7, pad=10)
    style.panel_label(ax, "c", x=-.22, y=1.15)
    ax.legend(loc="lower right", fontsize=6.5)
    lines = []
    for wave in WAVES:
        row = tail.loc[tail.wavelength.eq(wave)].iloc[0]
        lines.append(f"{wave} nm > {row.tail_threshold:g}: {100 * row.fraction_gt_0p85:.2f}%\n"
                     f"{int(row.n_gt_0p85):,} / {int(row.n):,}")
    fig.text(.745, .20, "\n".join(lines), fontsize=6.5, va="top", linespacing=1.5)
    fig.text(.02, .025, "Archive reference_valid cohorts, not held-out validation; native 1020-nm primary and 675-nm old56 sensitivity.", fontsize=6.5)
    return fig


def caption(frames):
    tail = frames["tail"].set_index("wavelength")
    a, b = tail.loc[WAVES[0]], tail.loc[WAVES[1]]
    return ("Figure S1. Fixed-formula inversion archive diagnostics. "
        "The horizontal quantity is the AERONET inversion coarse-mode extinction fraction "
        "(coarse extinction AOD / total extinction AOD), NOT SDA. The saved SDA proxy is at "
        f"{int(a.sda_wavelength_nm)} nm and is not used. (a) Native 675-nm old675LR56 dust-fraction sensitivity, "
        "fixed dust LR 56 sr. (b) Primary-compatible experimental1020 native dust extinction fraction, "
        "fixed dust LR 47.308 sr. Both use frozen nondust/dust PLDR 0.02/0.30: clip the "
        "backscatter fraction to [0,1] first, multiply by fixed/observed LR, then clip the "
        "extinction fraction to [0,1]. No 550-nm transfer or recalibration. Each wavelength "
        "retains its separate saved reference_valid archive cohort: positive total extinction, "
        "nonnegative coarse extinction, CMF in inclusive [0,1.05], LR in inclusive [10,150] sr "
        "and PLDR in inclusive [0,1]. These are not the historical six-site paired cohort or "
        "the ML validation cohort. Hexagons summarize display density only (shared log count "
        "scale, 40 horizontal bins); all original coordinates accompany the figure and no "
        "CMF coordinate is clipped. (c) Empirical cumulative distributions, no histogram "
        f"bin selection. Strict CMF > {a.tail_threshold:g}, before clipping: 675 nm "
        f"{int(a.n_gt_0p85):,}/{int(a.n):,} ({100*a.fraction_gt_0p85:.4f}%); 1020 nm "
        f"{int(b.n_gt_0p85):,}/{int(b.n):,} ({100*b.fraction_gt_0p85:.4f}%). Full saved valid "
        "range is visible; no off-axis CMF exclusions. The formula matches existing grouped "
        "rows, but that arithmetic check does not make the archive diagnostic independent "
        "validation. Tails do not by themselves establish physical attribution to a dust "
        "middle mode. No fitting, new thresholds, bootstrap, SDA conversion or downloads.\n")


def make(registry_path, output):
    before = digest(Path(registry_path))
    frames, evidence = select(Registry.read(registry_path))
    fig = build_figure(frames)
    try:
        check_layout(fig)
        if digest(Path(registry_path)) != before:
            raise RuntimeError("Registry changed during render")
        data = pd.concat([frame.assign(table_kind=key) for key, frame in frames.items()], ignore_index=True)
        style.export(fig, output, "FigS01_coarse_fraction", data, registry_path, evidence, __file__, raster=True)
        (Path(output) / "FigS01_caption.md").write_text(caption(frames))
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make(args.registry, args.output_dir)


if __name__ == "__main__":
    main()
