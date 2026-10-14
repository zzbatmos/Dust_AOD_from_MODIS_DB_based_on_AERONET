"""Frozen-reference sensitivity on saved paired collocations, not model skill."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from paper.derive_reference_display import (
    COUNT_METRICS, FLOAT_METRICS, LABEL_THRESHOLD, POPULATION, SEMANTICS, TABLE,
    guard_output, verify_pairs,
)
from paper.figures import figstyle as style
from paper.numbers import Registry, digest


STEM = "FigS03_reference_sensitivity"
PAIRS_TABLE = "derived/reference_pairs"
CAPTION = (
    "Reference sensitivity on identical non-calibration collocations with both frozen fractions finite. "
    "(a) Saved experimental1020 fractions (transferred to 550 nm) versus frozen old675LR56 fractions; "
    "no new transfer or clipping is applied. Every paired row is plotted; coincident points overlap. "
    "Dashed diagonal: equality. Dotted lines: the existing fraction threshold of 0.2. Mean and RMS "
    "changes are new minus old, not prediction errors. (b) Label agreement, 100*(rows-label_changed)/rows, "
    "overall and by the original first-match collocation region; N counts paired collocations, not "
    "OOF repeats or independent sites. Small regional samples are retained and labeled. Regional "
    "boundaries differ from Stage C climate boxes. No model-feature eligibility filter, uncertainty "
    "interval, bootstrap or reference selection is introduced. Reference sensitivity is not calibrated "
    "uncertainty or evidence that one definition is truth."
)


def load_pairs(registry):
    """Use embedded records only; extraction-era registry hashes are historical."""
    table = registry.payload["tables"][PAIRS_TABLE]
    derivation = table["derivation"]
    if not table.get("source_sha256") or table["source_sha256"] != derivation["output_sha256"]:
        raise ValueError("Paired table source SHA differs from extraction provenance")
    if derivation["label_threshold"] != LABEL_THRESHOLD:
        raise ValueError("Unexpected saved label threshold")
    pairs = registry.table(PAIRS_TABLE)
    if len(pairs) != table["source_rows"] or len(pairs) != derivation["checks"]["paired_rows"]:
        raise ValueError("Paired table row count differs from extraction provenance")
    if not pairs.population.eq(POPULATION).all():
        raise ValueError("Unexpected paired table population")
    positions = pairs.source_record_index.to_numpy(dtype=float)
    if (not np.isfinite(positions).all() or (positions < 0).any()
            or (positions != np.floor(positions)).any() or pairs.source_record_index.duplicated().any()):
        raise ValueError("Invalid paired source record indices")
    evidence = {"table": PAIRS_TABLE, **copy.deepcopy({k: v for k, v in table.items() if k != "records"})}
    return pairs, evidence


def build_display(registry, pairs):
    checks = verify_pairs(pairs, registry)
    source = registry.table(TABLE, population=POPULATION)
    summaries = pd.concat([source[(source.scope == "overall") & (source.group == "all")],
                           source[source.scope == "region"]]).copy()
    summaries["agreement_fraction"] = 1 - summaries.label_changed / summaries.rows
    summaries["agreement_percent"] = 100 * summaries.agreement_fraction
    summaries["source_record_index"] = summaries.index
    summaries["population"] = POPULATION
    evidence = []
    for _, row in summaries.iterrows():
        for metric in [*COUNT_METRICS, *FLOAT_METRICS]:
            evidence.append(registry.number(TABLE, metric, population=POPULATION,
                                            scope=row.scope, group=row.group))
    data = pd.concat([pairs.assign(panel="a", record_type="paired_fraction"),
                      summaries.assign(panel="b", record_type="saved_stratum")], ignore_index=True)
    return summaries.reset_index(drop=True), data, evidence, checks


def draw(pairs, summaries):
    fig = plt.figure(figsize=(style.DOUBLE, 140 * style.MM))
    ax = fig.add_axes([.085, .30, .35, .475])
    regional = fig.add_axes([.745, .15, .235, .75])
    ax.scatter(pairs.old675LR56, pairs.experimental1020, s=12.25, marker="o",
               color="#0072B2", alpha=.20, linewidths=0, rasterized=True)
    ax.plot([0, 1], [0, 1], color="#7F7F7F", lw=.6, ls="--")
    for axis in (ax.axhline, ax.axvline):
        axis(LABEL_THRESHOLD, color="#7F7F7F", lw=.6, ls=":")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect="equal",
           xlabel="Frozen old675LR56 fraction", ylabel="experimental1020 fraction (550 nm)")
    overall = summaries.iloc[0]
    ax.text(.035, .965, f"N = {int(overall.rows):,}\n"
            f"Mean change = {overall.mean_fraction_change:+.3f}\n"
            f"RMS change = {overall.fraction_change_rmse:.3f}",
            ha="left", va="top", transform=ax.transAxes, fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    style.panel_label(ax, "a", x=-.16)
    y = np.arange(len(summaries))
    regional.plot(summaries.agreement_percent, y, "o", color="black", markersize=3.5, linestyle="none")
    regional.plot(summaries.agreement_percent.iloc[0], 0, "s", color="black", markersize=4, linestyle="none")
    labels = [("All regions" if r.scope == "overall" else r.group.replace("_", " "))
              + f" (N={int(r.rows):,})" for _, r in summaries.iterrows()]
    regional.set(xlim=(0, 105), xticks=[0, 25, 50, 75, 100], yticks=y, yticklabels=labels,
                 ylim=(len(y) - .4, -.8), xlabel="Label agreement (%)")
    regional.tick_params(axis="y", length=0)
    regional.axhline(.5, color="#BDBDBD", lw=.5)
    regional.grid(axis="x", color="#E6E6E6", linestyle=":", linewidth=.4)
    regional.set_axisbelow(True)
    style.panel_label(regional, "b", x=-.08)
    return fig


def make(registry_path, output):
    registry_path, output = Path(registry_path).resolve(), Path(output).resolve()
    inputs = [registry_path]
    guard_output(output, inputs)
    hashes = {str(p): digest(p) for p in inputs}
    registry = Registry.read(registry_path)
    pairs, paired_source = load_pairs(registry)
    summaries, data, evidence, checks = build_display(registry, pairs)
    fig = draw(pairs, summaries)
    try:
        fig.canvas.draw()
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        if min(t.get_fontsize() for t in texts) < 6.5:
            raise ValueError("Figure text below 6.5 pt")
        renderer = fig.canvas.get_renderer()
        outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).p0)
                   or not fig.bbox.contains(*t.get_window_extent(renderer).p1)]
        if outside:
            raise ValueError(f"Text outside figure canvas: {outside}")
        if any(digest(p) != hashes[str(p)] for p in inputs):
            raise RuntimeError("Display inputs changed during rendering")
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / f"{STEM}.pdf", dpi=600)
        preview = output / f"{STEM}_preview.png"
        fig.savefig(preview, dpi=300)
        fig.savefig(output / f"{STEM}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        with Image.open(output / f"{STEM}.tif") as im:
            rgb = im.convert("RGB")
        rgb.save(output / f"{STEM}.tif", compression="tiff_lzw", dpi=(600, 600))
        data.to_csv(output / "FigS03_data.csv", index=False)
        (output / "FigS03_caption.md").write_text(CAPTION + "\n\n" + SEMANTICS + "\n")
        from colorspacious import cspace_convert
        image = np.asarray(Image.open(preview).convert("RGB"), dtype=float) / 255
        qa = output / "qa"
        qa.mkdir(exist_ok=True)
        for mode in ("deuteranomaly", "protanomaly", "tritanomaly"):
            simulated = cspace_convert(image, {"name": "sRGB1+CVD", "cvd_type": mode, "severity": 100}, "sRGB1")
            Image.fromarray(np.uint8(np.clip(simulated, 0, 1) * 255)).save(qa / f"{STEM}_{mode}.png")
        Image.open(preview).convert("L").save(qa / f"{STEM}_grayscale.png")
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": hashes,
            "registry_file": str(registry_path), "registry_sha256": hashes[str(registry_path)],
            "registry_code_commit": registry.payload.get("code_commit"),
            "code_commit": None, "code_commit_note": "No git operations; script/style SHA256 identify executed code.",
            "script_sha256": digest(Path(__file__)), "style_sha256": digest(Path(style.__file__)),
            "paired_table": paired_source, "derivation": paired_source["derivation"],
            "extraction_registry_hash_note": "Historical extraction dependency, not the current render registry hash.",
            "source_numbers": evidence, "checks": checks,
            "selectors": {"population": POPULATION, "scope": ["overall (group=all)", "region (all saved groups)"]},
            "agreement_equation": "100 * (1 - label_changed / rows)",
            "counts": {"plotted_pairs": len(pairs), "plotted_strata": len(summaries)},
            "semantics": SEMANTICS, "interval": None,
            "font": style.FONT, "minimum_font_pt": min(t.get_fontsize() for t in texts),
            "width_mm": fig.get_figwidth() / style.MM, "height_mm": fig.get_figheight() / style.MM,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "matplotlib", "cmcrameri", "colorspacious", "Pillow")},
            "visual_qa": {"canvas_bounds_pass": True, "CVD_and_grayscale_previews_generated": True,
                          "interpretation": "No color-dependent class encoding; every paired row shown; overplotting retained."},
            "outputs": {p.relative_to(output).as_posix(): digest(p) for p in
                        [*output.glob("FigS03*"), *qa.glob(f"{STEM}*")] if p.is_file() and p.name != "FigS03_provenance.json"},
        }
        if digest(registry_path) != hashes[str(registry_path)]:
            raise RuntimeError("Registry changed during export; outputs are not a coherent render")
        (output / "FigS03_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        return provenance
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(make(args.registry, args.output_dir)["counts"], indent=2))


if __name__ == "__main__":
    main()
