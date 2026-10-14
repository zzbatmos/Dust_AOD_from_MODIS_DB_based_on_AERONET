"""Record plot-range counts before plotting; out-of-range observations stay in scores."""
import argparse
import json
from pathlib import Path

import pandas as pd

from paper.numbers import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.derived_dir
    if root.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Keep plot counts outside the public code worktree")
    rows, sources = [], {}
    for filename, figure, maximum, estimates in (
        ("pixel_ev_scatter.csv", "Fig04", 1.5, {"c": "Eselected"}),
        ("site_months_hierarchy.csv", "Fig05", .8, {"a": "E1", "b": "E5", "c": "Eselected"}),
    ):
        path = root / filename
        sources[str(path.resolve())] = digest(path)
        frame = pd.read_csv(path)
        for (reference, scheme), group in frame.groupby(["reference", "scheme"]):
            for panel, estimator in estimates.items():
                x, y = group.truth_daod, group[estimator + "_daod"]
                outside = (x > maximum) | (y > maximum) | (x < 0) | (y < 0)
                rows.append(dict(figure=figure, panel=panel, reference=reference, scheme=scheme,
                                 estimator=estimator, axis_max=maximum, prediction_events=len(group),
                                 sites=group.site_key.nunique(), n_outside=int(outside.sum()),
                                 n_visible=int((~outside).sum())))
    output = root / "plot_range_counts.csv"
    if output.exists():
        raise FileExistsError("Preserve prior plot-range counts")
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "output_sha256": digest(output), "script_sha256": digest(Path(__file__)), "sources": sources,
        "semantics": "Counts at the figure-guide axis ranges, inclusive endpoints. Both held-out "
        "repeats remain separate prediction events. No rows are removed from statistic calculation "
        "or from the exported source coordinates when outside the displayed axes.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
