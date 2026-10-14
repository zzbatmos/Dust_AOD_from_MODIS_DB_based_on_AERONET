"""Registry-native private S6/S7 CSV, readable LaTeX and optional PDF exports."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd

from paper.derive_configuration_tables import S6, S7, guard_output
from paper.numbers import Registry, digest, ingest


TABLES = {6: "derived/" + S6, 7: "derived/" + S7}
CAPTIONS = {
    6: "Frozen production configuration, ordered inputs, recorded training settings and persistent model metadata.",
    7: "All configured monthly product variables: definitions, encoded types, denominators and missing-versus-zero conventions.",
}
NOTES = {
    6: "All extracted settings are typeset; full CSV also retains exact JSON values, notes, source paths, "
       "SHA256s and field locators. Feature order is explicit. Validity checks apply only to each "
       "branch's listed inputs; noSSA5 does not require SSA. QA thresholds apply to raw probability, "
       "not calibrated probability. Branch-level training settings are not independently documented "
       "per learner. NOT RECORDED means metadata unavailable, never a library default. Serialized "
       "tree counts and learner fields are persistent inference-model properties, not reconstructed "
       "training configuration. Artifact hashes are identifiers, not scientific parameters. Baseline "
       "YAML settings are diagnostics, not extra production branches. No fitting or prediction.",
    7: "All configured science/count variables are shown; coordinate and bounds metadata accompany the "
       "provenance. Shape is (time, lat, lon), with one monthly interval. Weighting is supported "
       "retrieval count, not equal-day, equal-granule or area weighting. AOD is averaged on ML support. "
       "Raw-probability QA fractions are cumulative thresholds, not disjoint tiers. Low-loading "
       "contributions divide by full n_ml, not subset counts. Counts retain true zeros without fill; "
       "floating statistics have NaN fill. Zero denominators yield missing statistics, not zero dust. "
       "The full CSV retains all writer attributes, encodings and exact source evidence. Writer "
       "snapshots used to extract metadata were empty; no observed monthly product was computed.",
}


def escape(value):
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return "--"
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "|": r"\textbar{}", "`": ""}
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text):
        return r"\allowbreak{}".join(text[i:i+8] for i in range(0, len(text), 8))
    return "".join(replacements.get(c, c + r"\allowbreak{}" if c in ",:/=" else c) for c in text)


def longtable(headers, rows, widths, caption):
    columns = "".join(r">{\raggedright\arraybackslash}p{\dimexpr" + str(w)
                      + r"\linewidth-2\tabcolsep\relax}" for w in widths)
    header = " & ".join(escape(h) for h in headers) + r" \\"
    lines = [r"\begin{longtable}{" + columns + "}", r"\caption{" + escape(caption) + r"}\\",
             r"\toprule", header, r"\midrule\endfirsthead",
             r"\caption[]{" + escape(caption + " (continued)") + r"}\\",
             r"\toprule", header, r"\midrule\endhead", r"\midrule\endfoot", r"\bottomrule\endlastfoot"]
    lines.extend(" & ".join(escape(v) for v in row) + r" \\" for row in rows)
    return "\n".join([*lines, r"\end{longtable}"])


def load_table(registry, index):
    name = TABLES[index]
    table = registry.payload["tables"][name]
    sidecar = table["derivation"]
    if table["source_sha256"] != sidecar["output_sha256"]:
        raise ValueError(f"Derived source SHA mismatch: {name}")
    frame = registry.table(name)
    keys = ["scope", "branch", "component", "metric"] if index == 6 else ["variable"]
    if frame.empty or frame.duplicated(keys).any() or frame.row_order.duplicated().any():
        raise ValueError(f"Empty/ambiguous derived records: {name}")
    if len(frame) != sidecar["rows"] or len(frame) != table["source_rows"]:
        raise ValueError(f"Derived record count mismatch: {name}")
    for source in frame.evidence_json:
        for operand in json.loads(source):
            if sidecar["sources"].get(operand["source_file"]) != operand["source_sha256"]:
                raise ValueError(f"Row/source evidence mismatch: {name}")
    return frame.sort_values("row_order", kind="stable"), {
        "table": name, **copy.deepcopy({k: v for k, v in table.items() if k != "records"})}


def render(index, frame, provenance):
    lines = [r"\section*{Table S" + str(index) + "}", r"\renewcommand{\thetable}{S" + str(index) + "}",
             r"\begingroup\fontsize{8}{9.5}\selectfont",
             escape(CAPTIONS[index]), r"\par\smallskip", escape(NOTES[index]),
             r"\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\linewidth}"
             r"\setlength{\LTpre}{4pt}\setlength{\LTpost}{4pt}"]
    if index == 6:
        for scope, group in frame.groupby("scope", sort=False):
            lines.append(longtable(["Branch", "Component", "Setting", "Saved value", "Availability"],
                                   group[["branch", "component", "metric", "value_text", "availability"]].itertuples(index=False, name=None),
                                   [.10, .14, .25, .40, .11], f"S6: {scope}; exact values, no inferred defaults."))
    else:
        grids = provenance["derivation"]["grid_metadata"]
        lines.append(longtable(["Grid", "Dimensions", "Weighting"],
                               [(g, json.dumps(v["sizes"], sort_keys=True), v["attrs"]["weighting"]) for g, v in grids.items()],
                               [.15, .35, .50], "S7: supported grids; empty snapshot dimensions, not observed counts."))
        rows = []
        for _, row in frame.iterrows():
            encoded = f"{row['dtype']}; units {row.units}; denominator {row.denominator}"
            rows.append((row.variable, encoded, row.definition, row.missing_vs_zero))
        lines.append(longtable(["Variable", "Type / Units / Denominator", "Definition", "Missing Versus Zero"],
                               rows, [.18, .19, .30, .33], "S7: every configured monthly variable, in frozen output order."))
    return "\n".join([*lines, r"\endgroup", ""])


def export(registry_path, output, compile_latex=False):
    registry_path, output = Path(registry_path).resolve(), Path(output).resolve()
    guard_output(output)
    if output == registry_path or output.is_relative_to(registry_path.parent):
        raise ValueError("Keep table output separate from registry directory")
    before = digest(registry_path)
    registry = Registry.read(registry_path)
    loaded = {index: load_table(registry, index) for index in TABLES}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"registry_file": str(registry_path), "registry_sha256": before,
                "script_sha256": digest(Path(__file__)), "tables": {}}
    for index, (frame, source) in loaded.items():
        stem = f"TableS{index:02d}"
        (output / f"{stem}.csv").write_text(frame.to_csv(index=False))
        (output / f"{stem}.tex").write_text(render(index, frame, source))
        (output / f"{stem}_notes.md").write_text(CAPTIONS[index] + "\n\n" + NOTES[index] + "\n")
        wrapper = output / f"{stem}_preview.tex"
        wrapper.write_text("\n".join([r"\documentclass{article}", r"\usepackage[T1]{fontenc}",
                                      r"\usepackage{lmodern}", r"\usepackage[a4paper,landscape,margin=16mm]{geometry}",
                                      r"\usepackage{longtable,booktabs,array}", r"\begin{document}",
                                      r"\input{" + stem + ".tex}", r"\end{document}", ""]))
        compilation = {"status": "not_run"}
        if compile_latex:
            executable = shutil.which("pdflatex")
            if not executable:
                raise RuntimeError("pdflatex unavailable; no successful compile claimed")
            for run in (1, 2):
                result = subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                                        cwd=output, capture_output=True, text=True, timeout=120)
                (output / f"{stem}_compile_pass{run}.txt").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"{stem} compilation failed; see private logs")
            compilation = {"status": "compiled", "passes": 2, "pdf": wrapper.with_suffix(".pdf").name}
            match = re.search(r"Output written on .*?\((\d+) pages?", result.stdout, re.DOTALL)
            if match:
                compilation["pages"] = int(match.group(1))
        names = [f"{stem}.csv", f"{stem}.tex", f"{stem}_notes.md", wrapper.name]
        if compile_latex:
            names.append(wrapper.with_suffix(".pdf").name)
        provenance = {k: v for k, v in manifest.items() if k != "tables"}
        provenance.update(source=source, rows=len(frame), caption=CAPTIONS[index], notes=NOTES[index],
                          font_pt=8, typeset_selection=("All derived rows grouped by scope; saved order within each scope."
                                                       if index == 6 else "All derived rows in saved variable order."),
                          latex=compilation, outputs={n: digest(output / n) for n in names})
        (output / f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        manifest["tables"][stem] = {"rows": len(frame), "latex": compilation}
    if digest(registry_path) != before:
        raise RuntimeError("Registry changed during table export")
    (output / "configuration_tables_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def temporary_registry(derived_dir, path):
    """Explicit private two-table preview registry, never a canonical rebuild."""
    path = Path(path).resolve()
    guard_output(path.parent)
    if path.exists():
        raise ValueError("Refuse to overwrite an existing registry")
    tables = {}
    for name in (S6, S7):
        source = Path(derived_dir) / f"{name}.csv"
        sidecar = json.loads(source.with_suffix(".provenance.json").read_text())
        if digest(source) != sidecar["output_sha256"]:
            raise ValueError("Derived CSV SHA mismatch")
        table = ingest(source, "derived/" + name)
        table.update(derivation=sidecar, semantics=sidecar["semantics"],
                     filter_columns=sidecar["recommended_filter_columns"])
        tables["derived/" + name] = table
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "scope": "PRIVATE S6/S7 preview only; not canonical", "tables": tables},
                               allow_nan=False) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.registry, args.output_dir, args.compile)["tables"], indent=2))


if __name__ == "__main__":
    main()
