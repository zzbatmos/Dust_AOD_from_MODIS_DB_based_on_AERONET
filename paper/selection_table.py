"""Table S8: saved inner expected-value selection, never a new model choice."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from paper.numbers import Registry, digest
from paper.supplement_tables import _guard_output, _longtable, tex


TABLE = "expected_value_test/stage_b_support/inner_selection"
KEYS = ["reference", "population", "branch", "scheme", "split_id", "repeat"]
COUNTS = ["minimum_cell_rows", "qualifying_sites", "qualifying_cells", "input_rows", "paired_rows", "dropped_rows"]
NOTE = (
    "Saved honest inner selection for every outer evaluation fold, under both AERONET references. "
    "E3c is the calibrated, residual-smeared mixture; E4 is direct expected-fraction regression. "
    "The criterion is inner site-month DAOD RMSE with at least three collocations per cell; exact ties "
    "favor E4. These are training-side choices, not choices based on the outer evaluation scores. "
    "Full8 common and own populations remain separate; minimal2 is an exploratory configuration, "
    "not a deployed fallback. Fold-specific choices must not be replaced by the final deployment "
    "forms (full8 E3c and noSSA5 E4). Full-precision scores, support counts and row hashes are retained "
    "in the CSV. Typeset scores have six decimals; selection is copied from the saved record, "
    "not inferred from rounded scores. N is paired inner-validation rows, not outer test rows."
)


def validate(registry):
    d = registry.table(TABLE)
    if d.empty or d.duplicated(KEYS).any() or d[KEYS].isna().any().any():
        raise ValueError("Missing or ambiguous inner selection")
    if not d.selected.isin(["E3c", "E4"]).all() or not d.exact_tie.eq("E4").all():
        raise ValueError("Unknown saved selection or tie rule")
    if not d.criterion.eq("site_month_RMSE").all() or not d.minimum_cell_rows.eq(3).all():
        raise ValueError("Unexpected selection criterion")
    values = d[COUNTS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values == np.floor(values)).all():
        raise ValueError("Invalid inner support counts")
    if not d.input_rows.eq(d.paired_rows + d.dropped_rows).all():
        raise ValueError("Inner support ledger does not close")
    scores = d[["E3c_inner_RMSE", "E4_inner_RMSE"]].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid saved inner RMSE")
    wrong = ((d.selected.eq("E3c") & (d.E3c_inner_RMSE >= d.E4_inner_RMSE))
             | (d.selected.eq("E4") & (d.E4_inner_RMSE > d.E3c_inner_RMSE)))
    if wrong.any():
        raise ValueError("Saved choice contradicts stated criterion; do not repair or reselect")
    return d


def export(registry_path, output_dir, compile_latex=False):
    output_dir = Path(output_dir)
    _guard_output(output_dir)
    registry_hash = digest(Path(registry_path))
    registry = Registry.read(registry_path)
    d = validate(registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = output_dir / "TableS08_inner_selection.csv"
    d.to_csv(csv, index=False)
    lines = [r"\section*{Table S8}", r"\renewcommand{\thetable}{S8}", tex(NOTE),
             r"\par\medskip\begingroup\fontsize{7}{8}\selectfont\setlength{\tabcolsep}{2pt}"]
    for key, part in d.groupby(["reference", "population", "branch", "scheme"], sort=True):
        heading = "; ".join(f"{k}={v}" for k, v in zip(["reference", "population", "branch", "scheme"], key))
        rows = []
        for _, row in part.sort_values(["repeat", "split_id"]).iterrows():
            rows.append([tex(row.split_id), tex(row.selected), f"{row.E3c_inner_RMSE:.6f}",
                         f"{row.E4_inner_RMSE:.6f}", str(int(row.paired_rows)),
                         str(int(row.qualifying_sites)), str(int(row.qualifying_cells)), str(int(row.dropped_rows))])
        lines += [r"\clearpage", _longtable(
            ["Outer split", "Selected", "E3c RMSE", "E4 RMSE", "N paired", "Sites", "Cells", "Dropped"],
            rows, heading, repeat_caption=True)]
    lines.append(r"\endgroup")
    (output_dir / "TableS08.tex").write_text("\n".join(lines) + "\n")
    (output_dir / "TableS08_notes.md").write_text("# Table S8\n\n" + NOTE + "\n")
    wrapper = output_dir / "TableS08_preview.tex"
    wrapper.write_text("\n".join([
        r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
        r"\usepackage[a4paper,landscape,margin=16mm]{geometry}",
        r"\usepackage{longtable,booktabs,array}", r"\begin{document}",
        r"\input{TableS08.tex}", r"\end{document}", ""]))
    if compile_latex:
        for i in (1, 2):
            result = subprocess.run(["pdflatex", "-no-shell-escape", "-halt-on-error",
                                     "-interaction=nonstopmode", wrapper.name], cwd=output_dir,
                                    capture_output=True, text=True, timeout=300)
            (output_dir / f"TableS08_compile_pass{i}.txt").write_text(result.stdout + result.stderr)
            result.check_returncode()
            if "Overfull" in result.stdout or "Font shape" in result.stdout:
                raise ValueError("Table S8 layout/font warning; inspect preview before delivery")
    if digest(Path(registry_path)) != registry_hash:
        raise RuntimeError("Registry changed during table generation")
    table = registry.payload["tables"][TABLE]
    (output_dir / "TableS08_provenance.json").write_text(json.dumps(dict(
        registry_sha256=registry_hash, script_sha256=digest(Path(__file__)),
        source_file=table["source_file"], source_sha256=table["source_sha256"],
        rows=len(d), filters={}, notes=NOTE, csv_sha256=digest(csv),
        selection="Verbatim saved choice; criterion checked but never refitted or reselected",
        compiled=compile_latex), indent=2) + "\n")
    return len(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    print(f"Exported {export(args.registry, args.output_dir, args.compile)} saved inner choices")
