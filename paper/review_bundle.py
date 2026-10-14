"""Render a reproducible partial review batch and assemble final-size PDFs.

Missing numbered figures are explicitly listed; this never certifies that the
whole manuscript figure specification has been fulfilled or visually reviewed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from paper.numbers import digest


MODULES = {
    "1": "fig01_framework", "2": "fig02_learning_space", "3": "fig03_information_content", "4": "fig04_pixel_skill",
    "6": "fig06_seasonal_maps", "7": "fig07_regional_comparison",
    "8": "fig08_cases", "S2": "figS02_calibration", "S3": "figS03_reference_sensitivity",
    "S4": "figS04_collocation_ledger", "S5": "figS05_learning_space_sensitivity",
    "S6": "figS06_product_comparison", "S7": "figS07_ablation_coverage",
    "S8": "figS08_probability_calibration", "S9": "figS09_stratified_skill",
    "S10": "figS10_fraction_reliability",
    "S11": "figS11_site_month_sensitivity", "S12": "figS12_retransformation",
    "S13": "figS13_all_seasonal_maps", "S14": "figS14_monthly_cycles",
    "S15": "figS15_low_loading",
    "S16": "figS16_reference_means", "S17": "figS17_case_inputs", "S18": "figS18_temporal",
}


def missing_supplementary(stems):
    """Multi-page supplements require every named page, not just a prefix match."""
    found = set(stems)
    required = {f"FigS{i:02}": {f"FigS{i:02}"} for i in range(1, 19)}
    required["FigS13"] = {"FigS13a", "FigS13b"}
    return [name for name, pages in required.items() if not pages.issubset(found)]


def supplementary_table_modules(table_keys):
    """Retain older snapshots; add the site inventory only when it is registered."""
    modules = ["paper.supplement_tables", "paper.selection_table", "paper.configuration_tables"]
    if "derived/site_inventory_table" in table_keys:
        modules.append("paper.site_inventory_table")
    return modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=list(MODULES), default=["1", "2", "4", "6", "7", "S5"])
    parser.add_argument("--image-dir", type=Path, help="Verified GIBS context cache; required for Figure 8")
    parser.add_argument("--pilot-inventory", type=Path, help="Completed August 29 coverage inventory; required for Figure 3")
    parser.add_argument("--supplement-tables", action="store_true", help="Export and compile available SI tables, including S1 when registered")
    parser.add_argument("--assemble-only", action="store_true", help="Check existing figure provenance and rebuild contacts/manifest without rerendering")
    args = parser.parse_args()
    if args.assemble_only:
        args.figures = []
    if "8" in args.figures and args.image_dir is None:
        parser.error("Figure 8 requires --image-dir")
    if "3" in args.figures and args.pilot_inventory is None:
        parser.error("Figure 3 requires --pilot-inventory")
    root = args.output_dir.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Review bundles must be outside public worktree")
    root.mkdir(parents=True, exist_ok=True)
    registry_sha = digest(args.registry)
    env = dict(os.environ, MPLCONFIGDIR="/tmp/dbdust-mpl", XDG_CACHE_HOME="/tmp/dbdust-cache")
    for number in args.figures:
        module = MODULES[number]
        destination = root / ("si" if number.startswith("S") else "main")
        print(f"Rendering {module}", flush=True)
        command = [sys.executable, "-m", "paper.figures." + module,
                   "--registry", str(args.registry), "--output-dir", str(destination)]
        if number == "8":
            command += ["--image-dir", str(args.image_dir)]
        if number == "3":
            command += ["--pilot-inventory", str(args.pilot_inventory)]
        subprocess.run(command,
                       check=True, env=env)
    if not args.assemble_only:
        subprocess.run([sys.executable, "-m", "paper.tables", "--registry", str(args.registry),
                        "--output-dir", str(root / "tables")], check=True, env=env)
        wrapper = root / "tables/table_preview.tex"
        wrapper.write_text("\\documentclass{article}\n\\usepackage[a4paper,margin=10mm]{geometry}\n"
                           "\\usepackage[T1]{fontenc}\n\\usepackage{booktabs}\n\\begin{document}\n"
                           "\\input{Table01.tex}\n\\clearpage\n\\input{Table02.tex}\n\\end{document}\n")
        compile_result = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=nonstopmode", wrapper.name],
                                        cwd=wrapper.parent, capture_output=True, text=True)
        (wrapper.parent / "compile_output.txt").write_text(compile_result.stdout + compile_result.stderr)
        compile_result.check_returncode()
        if "Overfull" in compile_result.stdout or "Font shape" in compile_result.stdout:
            raise ValueError("Inspect table overflow/font-substitution warning")
    if args.supplement_tables and not args.assemble_only:
        table_keys = json.loads(args.registry.read_text())["tables"].keys()
        for module in supplementary_table_modules(table_keys):
            subprocess.run([sys.executable, "-m", module, "--registry", str(args.registry),
                            "--output-dir", str(root / "tables" / "supplement"), "--compile"],
                           check=True, env=env)
    captions, figures = [], []
    for subdir, contact in (("main", "CONTACT_SHEET.pdf"), ("si", "CONTACT_SHEET_SI.pdf")):
        folder = root / subdir
        pdfs = sorted(folder.glob("Fig*.pdf"))
        if pdfs:
            subprocess.run(["pdfunite", *map(str, pdfs), str(folder / contact)], check=True)
        for path in pdfs:
            stem = path.name.split("_")[0]
            provenance = json.loads((folder / (stem + "_provenance.json")).read_text())
            if provenance["registry_sha256"] != registry_sha:
                raise ValueError(f"Mixed registry versions: {path}")
            caption = (folder / (stem + "_caption.md")).read_text().strip()
            if subdir == "main" and len(caption.split()) > 80:
                raise ValueError(f"Caption exceeds 80 words: {stem}")
            captions.append(f"## {stem}\n\n{caption}")
            fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True, check=True).stdout
            (folder / (stem + "_fonts.txt")).write_text(fonts)
            figures.append(dict(figure=stem, path=str(path), sha256=digest(path),
                                caption_words=len(caption.split()), provenance=provenance))
    if digest(args.registry) != registry_sha:
        raise RuntimeError("Registry changed during rendering")
    (root / "captions.md").write_text("# Draft figure captions\n\n" + "\n\n".join(captions) + "\n")
    missing = [f"Fig{i:02}" for i in range(1, 9) if f"Fig{i:02}" not in {f["figure"] for f in figures}]
    missing_si = missing_supplementary(f["figure"] for f in figures)
    (root / "review_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "registry_sha256": registry_sha,
        "figures": figures, "missing_main_figures": missing,
        "missing_supplementary_figures": missing_si,
        "status": "Partial review batch; generated diagnostics do not replace visual inspection."
    }, indent=2, allow_nan=False) + "\n")
    print(f"Partial batch complete. Missing main figures: {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
