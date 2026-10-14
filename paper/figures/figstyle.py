"""Shared publication style and auditable exports; no scientific calculations."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from PIL import Image

from paper.numbers import digest

MM = 1 / 25.4
SINGLE, DOUBLE = 95 * MM, 190 * MM
available = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((x for x in ("Arial", "Helvetica", "DejaVu Sans") if x in available), "DejaVu Sans")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": .6,
    "lines.linewidth": 1., "lines.markersize": 4, "xtick.direction": "out",
    "ytick.direction": "out", "xtick.major.width": .6, "ytick.major.width": .6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.unicode_minus": True,
})
EST = {
    "Eselected": ("Expected value (this study)", "#D55E00", "o", "-"),
    "E1": ("QA2 (this study)", "#0072B2", "s", "--"),
    "E5": ("DB aerosol type", "#7F7F7F", "D", ":"),
    "E7": ("LG25", "#E69F00", "^", "-."),
    "E6": ("LG25 screened", "#CC79A7", "v", "-."),
    "E8": ("AE refit (DB space)", "#56B4E9", "p", "--"),
    "Song": ("Song et al. (2021) MODIS", "#009E73", "h", "-"),
    "CALIOP": ("CALIOP V2", "#000000", "*", "-"),
}
ZERO_GREY, MISSING = "#BDBDBD", "white"
CMAP_DAOD = ListedColormap(cmc.lajolla_r(np.linspace(.12, 1, 256)))
CMAP_P = ListedColormap(cmc.batlow(np.linspace(0, .9, 256)))
CMAP_DIFF, CMAP_DENSITY = cmc.vik, cmc.oslo_r


def panel_label(ax, letter, x=-.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def export(fig, output, stem, data, registry_path, sources, script, raster=False):
    """Keep physical dimensions fixed; never crop away the specified canvas."""
    from colorspacious import cspace_convert
    output = Path(output)
    repo = Path(__file__).resolve().parents[2]
    if output.resolve().is_relative_to(repo):
        raise ValueError("Unpublished figures must be outside the public code worktree")
    output.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    small = [(t.get_text(), t.get_fontsize()) for t in texts if t.get_fontsize() < 6.5]
    if small:
        raise ValueError(f"Text below the minimum printed size: {small}")
    fig.savefig(output / f"{stem}.pdf", dpi=600)
    preview = output / f"{stem}_preview.png"
    fig.savefig(preview, dpi=300)
    if raster:
        fig.savefig(output / f"{stem}.tif", dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"})
        with Image.open(output / f"{stem}.tif") as im:
            rgb = im.convert("RGB")
        rgb.save(output / f"{stem}.tif", compression="tiff_lzw", dpi=(600, 600))
    data.to_csv(output / f"{stem.split('_')[0]}_data.csv", index=False)
    image = np.asarray(Image.open(preview).convert("RGB"), dtype=float) / 255
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    for mode in ("deuteranomaly", "protanomaly", "tritanomaly"):
        simulated = cspace_convert(image, {"name": "sRGB1+CVD", "cvd_type": mode, "severity": 100}, "sRGB1")
        Image.fromarray(np.uint8(np.clip(simulated, 0, 1) * 255)).save(qa / f"{stem}_{mode}.png")
    Image.open(preview).convert("L").save(qa / f"{stem}_grayscale.png")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "script_sha256": digest(Path(script)), "style_sha256": digest(Path(__file__)),
        "registry_file": str(Path(registry_path).resolve()), "registry_sha256": digest(Path(registry_path)),
        "source_numbers": sources, "font": FONT, "width_mm": fig.get_figwidth() / MM,
        "height_mm": fig.get_figheight() / MM, "minimum_font_pt": min(t.get_fontsize() for t in texts),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "matplotlib", "cmcrameri", "colorspacious", "Pillow")},
        "visual_qa": "CVD/grayscale previews generated; human inspection still required",
    }
    (output / f"{stem.split('_')[0]}_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance
