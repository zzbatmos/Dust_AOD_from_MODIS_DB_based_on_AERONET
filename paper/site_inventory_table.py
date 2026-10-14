"""Registry-only Table S1 export with full CSV and readable all-site longtables."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from paper.configuration_tables import escape, longtable
from paper.derive_configuration_tables import guard_output
from paper.derive_site_inventory_table import FILTERS, NAME, TABLE
from paper.numbers import Registry, digest, ingest


CAPTION = "AERONET site inventory for the frozen training/evaluation cohort and independent lidar-ratio calibration."
NOTE = (
    "Every site is shown. Training/evaluation sites participate in grouped validation; this is not "
    "a claim that every site trains every fold. N in that block counts distinct saved MODIS "
    "collocations, not repeated held-out predictions. N in the calibration block counts saved "
    "AERONET inversion records, including repeated site/date records, not MODIS collocations. "
    "The two site roles are disjoint. Years are actual cohort years only; gaps remain explicit. "
    "Coordinates (latitude, longitude; degrees north/east) are displayed to four decimals; "
    "calibration coordinates retain original-cache site medians. Regions are saved validation "
    "regions, not newly assigned geographical labels; NOT RECORDED marks unavailable metadata. "
    "PI names come from existing local AERONET headers: underscores are rendered as spaces only "
    "for readability. Exact names, per-year counts, platforms, source selectors, file/header "
    "hashes and encodings remain in the full CSV/provenance. These names do not verify historical "
    "PI tenure, current contacts, or acknowledgment approval. No emails or network lookup."
)


def load(registry):
    table = registry.payload["tables"][TABLE]
    side = table["derivation"]
    frame = registry.table(TABLE)
    if table["source_sha256"] != side["output_sha256"]:
        raise ValueError("Derived site inventory SHA mismatch")
    if len(frame) != table["source_rows"] or len(frame) != side["rows"] or frame.empty:
        raise ValueError("Site inventory row count mismatch")
    if frame.site_key.isna().any() or frame.site_key.duplicated().any():
        raise ValueError("Ambiguous site inventory")
    if not frame.role.isin(["evaluation", "calibration"]).all():
        raise ValueError("Unknown site role")
    if not np.isfinite(frame[["lat", "lon"]].to_numpy(float)).all():
        raise ValueError("Missing site coordinates")
    if not frame.pi_status.isin(["local_header", "missing_local_PI", "conflicting_local_headers"]).all():
        raise ValueError("Unknown PI status")
    for _, row in frame.iterrows():
        counts = json.loads(row.year_counts_json)
        years = [int(y) for y in str(row.years).split(";")]
        if (sorted(map(int, counts)) != years or len(years) != row.n_years
                or min(years) != row.first_year or max(years) != row.last_year
                or any(not isinstance(v, int) or v <= 0 for v in counts.values())
                or sum(counts.values()) != row.n_records):
            raise ValueError("Per-site year/count ledger does not close")
        count = row.collocations if row.role == "evaluation" else row.calibration_records
        if count != row.n_records:
            raise ValueError("Role-specific site count differs")
        candidates = json.loads(row.pi_evidence_json)
        for item in candidates:
            if side["sources"].get(item["source_file"]) != item["source_sha256"]:
                raise ValueError("PI row/source evidence mismatch")
        names = {item["principal_investigators"] for item in candidates if item["principal_investigators"]}
        if row.pi_status == "local_header" and names != {row.principal_investigators}:
            raise ValueError("PI string differs from saved header evidence")
        if row.pi_status != "local_header" and row.principal_investigators != "NOT RECORDED":
            raise ValueError("Missing/conflicting PI must not be guessed")
    return frame, {"table": TABLE, **copy.deepcopy({k: v for k, v in table.items() if k != "records"})}


def render(frame):
    lines = [r"\section*{Table S1}", r"\renewcommand{\thetable}{S1}",
             r"\begingroup\fontsize{8}{9.5}\selectfont", escape(CAPTION), r"\par\smallskip", escape(NOTE),
             r"\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\linewidth}"]
    for role in ("evaluation", "calibration"):
        part = frame.loc[frame.role.eq(role)].sort_values("site_key", kind="stable")
        if part.empty:
            continue
        title = "Training/evaluation" if role == "evaluation" else "Independent calibration"
        kind = "unique MODIS collocations" if role == "evaluation" else "saved AERONET inversion records"
        heading = f"{title}: {len(part)} sites; N = {kind}. All sites retained."
        rows = [(r.site, f"{r.lat:.4f}, {r.lon:.4f}", r.region, str(int(r.n_records)),
                 ", ".join(str(r.years).split(";")), r.principal_investigators.replace("_", " "))
                for r in part.itertuples(index=False)]
        lines.append(longtable(["Site", "Latitude, longitude", "Saved region", "N", "Observed years", "Principal investigators (local header)"],
                               rows, [.15, .105, .115, .05, .21, .37], heading))
    return "\n".join([*lines, r"\endgroup", ""])


def export(registry_path, output_dir, compile_latex=False):
    registry_path, output_dir = Path(registry_path).resolve(), Path(output_dir).resolve()
    guard_output(output_dir)
    if output_dir.is_relative_to(registry_path.parent):
        raise ValueError("Keep table output separate from registry directory")
    before = digest(registry_path)
    frame, source = load(Registry.read(registry_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "TableS01.csv").write_text(frame.to_csv(index=False))
    (output_dir / "TableS01.tex").write_text(render(frame))
    (output_dir / "TableS01_notes.md").write_text(CAPTION + "\n\n" + NOTE + "\n")
    wrapper = output_dir / "TableS01_preview.tex"
    wrapper.write_text("\n".join([
        r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
        r"\usepackage[a4paper,landscape,margin=16mm]{geometry}",
        r"\usepackage{longtable,booktabs,array}", r"\begin{document}",
        r"\input{TableS01.tex}", r"\end{document}", ""]))
    outputs = ["TableS01.csv", "TableS01.tex", "TableS01_notes.md", wrapper.name]
    latex = {"status": "not_run"}
    if compile_latex:
        for run in (1, 2):
            result = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                                    cwd=output_dir, capture_output=True, text=True, timeout=120)
            (output_dir / f"TableS01_compile_pass{run}.txt").write_text(result.stdout + result.stderr)
            result.check_returncode()
        if "Overfull" in result.stdout or "Font shape" in result.stdout:
            raise ValueError("Table S1 layout/font warning: inspect private preview")
        pages = re.search(r"Output written on .*?\((\d+) pages?", result.stdout, re.DOTALL)
        latex = {"status": "compiled", "passes": 2, "pages": int(pages.group(1)) if pages else None}
        outputs.append("TableS01_preview.pdf")
    if digest(registry_path) != before:
        raise RuntimeError("Registry changed during export")
    provenance = dict(registry_file=str(registry_path), registry_sha256=before,
                      script_sha256=digest(Path(__file__)), source=source, rows=len(frame),
                      caption=CAPTION, notes=NOTE, font_pt=8, latex=latex,
                      typeset_selection="All sites, sorted by site_key within each preserved role; no omissions",
                      outputs={name: digest(output_dir / name) for name in outputs})
    (output_dir / "TableS01_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return {"rows": len(frame), "latex": latex, "coverage": source["derivation"]["coverage"]}


def temporary_registry(derived_dir, path):
    """Private one-table preview only, never overwrite or rebuild a shared registry."""
    path = Path(path).resolve()
    guard_output(path.parent)
    if path.exists():
        raise FileExistsError(path)
    csv = Path(derived_dir) / (NAME + ".csv")
    side = json.loads(csv.with_suffix(".provenance.json").read_text())
    if digest(csv) != side["output_sha256"]:
        raise ValueError("Derived site inventory SHA mismatch")
    table = ingest(csv, TABLE)
    table.update(derivation=side, semantics=side["semantics"], filter_columns=FILTERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, scope="PRIVATE S1 preview only; not canonical", tables={TABLE: table}),
                               allow_nan=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.registry, args.output_dir, args.compile), indent=2))


if __name__ == "__main__":
    main()
