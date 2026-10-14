"""Mechanical delivery checks; never a substitute for human figure review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from PIL import Image

from paper.numbers import digest


def font_check(report):
    rows = report.strip().splitlines()[2:]
    if not rows:
        raise ValueError("PDF has no reported embedded text fonts")
    for row in rows:
        fields = row.split()
        if len(fields) < 8 or fields[-5] != "yes" or "TrueType" not in row or "Type 3" in row:
            raise ValueError("PDF contains nonembedded or non-TrueType fonts: " + row)
    return len(rows)


def raster_check(path, width_mm, height_mm, tiff=False):
    with Image.open(path) as im:
        dpi = im.info.get("dpi", ())
        required = 600 if tiff else 300
        if len(dpi) != 2 or any(abs(float(x) - required) > .1 for x in dpi):
            raise ValueError(f"Unexpected raster DPI: {path}")
        dimensions = [pixels / float(d) * 25.4 for pixels, d in zip(im.size, dpi)]
        if any(abs(a - b) > .15 for a, b in zip(dimensions, [width_mm, height_mm])):
            raise ValueError(f"Raster physical dimensions disagree: {path}")
        if tiff and (im.mode != "RGB" or im.tag_v2.get(259) != 5):
            raise ValueError(f"TIFF must be RGB with LZW compression: {path}")
        return dict(pixels=list(im.size), dpi=list(dpi), mode=im.mode, width_mm=dimensions[0], height_mm=dimensions[1])


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "review_manifest.json").read_text())
    records, warnings = [], []
    for figure in manifest["figures"]:
        path = Path(figure["path"])
        if not path.is_relative_to(root):
            raise ValueError("Manifest figure outside requested bundle")
        if digest(path) != figure["sha256"]:
            raise ValueError(f"Figure changed after bundle creation: {path}")
        stem = figure["figure"]
        provenance_path = path.parent / f"{stem}_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if provenance["registry_sha256"] != manifest["registry_sha256"]:
            raise ValueError("Mixed registry hashes")
        if provenance["minimum_font_pt"] < 6.5 or provenance["height_mm"] > 230.01:
            raise ValueError("Invalid printed font size or height")
        if min(abs(provenance["width_mm"] - w) for w in [95, 190]) > .1:
            raise ValueError("Nonstandard printed width")
        if not provenance.get("code_commit"):
            warnings.append(stem + ": source identified by script/style hashes; executed git commit is not recorded")
        font_report = subprocess.check_output(["pdffonts", str(path)], text=True)
        fonts = font_check(font_report)
        pdfinfo = subprocess.check_output(["pdfinfo", str(path)], text=True)
        match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", pdfinfo)
        if not match or not re.search(r"Pages:\s*1\b", pdfinfo):
            raise ValueError("Figure must be a single-page PDF")
        size_mm = [float(v) / 72 * 25.4 for v in match.groups()]
        if any(abs(a - b) > .05 for a, b in zip(size_mm, [provenance["width_mm"], provenance["height_mm"]])):
            raise ValueError("PDF dimensions differ from provenance")
        data = path.parent / f"{stem}_data.csv"
        caption = path.parent / f"{stem}_caption.md"
        if not data.is_file() or not caption.is_file():
            raise ValueError("Missing numerical export or caption")
        if not stem.startswith("FigS") and len(caption.read_text().split()) > 80:
            raise ValueError("Main caption exceeds 80 words")
        rasters = {"preview": raster_check(path.with_name(path.stem + "_preview.png"), *size_mm)}
        tiff = path.with_suffix(".tif")
        if tiff.exists():
            rasters["tiff"] = raster_check(tiff, *size_mm, tiff=True)
        for mode in ("deuteranomaly", "protanomaly", "tritanomaly", "grayscale"):
            if not (path.parent / "qa" / f"{path.stem}_{mode}.png").is_file():
                raise ValueError("Missing accessibility preview")
        records.append(dict(figure=stem, pdf_sha256=digest(path), data_sha256=digest(data),
                            provenance_sha256=digest(provenance_path), fonts=fonts, rasters=rasters))
    if not records:
        raise ValueError("Empty review bundle")
    overfull = []
    for log in (root / "tables").rglob("*.txt"):
        if "Overfull" in log.read_text(errors="replace"):
            overfull.append(str(log))
    if overfull:
        raise ValueError("Table overflow reported: " + ", ".join(overfull))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), figures=records,
                  registry_sha256=manifest["registry_sha256"], warnings=warnings,
                  missing_main_figures=manifest["missing_main_figures"],
                  missing_supplementary_figures=manifest.get("missing_supplementary_figures"),
                  status="Mechanical checks passed; incomplete figure set; human/CVD review remains separate")
    (root / "mechanical_QA.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bundle)
    print(json.dumps(dict(figures_checked=len(result["figures"]), warnings=result["warnings"])))
