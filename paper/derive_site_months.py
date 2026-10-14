"""Recover plotted site-month arrays from cached OOF predictions, without fitting.

Retains both CV repeats separately. Saved annotations are means of repeat-specific
metrics, not scores of predictions averaged across repeats.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import digest


CORE = ["E0", "E1_QA1", "E1", "E1_QA3", "E2", "E3", "E5", "E7", "E3c", "E4", "E8", "Eselected"]


def monthly(frame, minimum=3):
    if frame.scheme.nunique() != 1:
        raise ValueError("Separate schemes before monthly aggregation")
    if frame.duplicated(["row_id", "repeat"]).any():
        raise ValueError("Duplicate held-out row/repeat event")
    columns = ["truth_daod", "truth_fraction", "DB_AOD550"] + CORE
    q = frame.loc[np.isfinite(frame[columns].to_numpy(float)).all(axis=1)].copy()
    for estimator in CORE:
        q[estimator + "_daod"] = q[estimator] * q.DB_AOD550
    keys = ["site_key", "year", "month", "repeat"]
    columns = ["truth_daod", "truth_fraction"] + CORE + [e + "_daod" for e in CORE]
    groups = q.groupby(keys, observed=True)
    result = groups[columns].mean()
    result["unique_rows"] = groups.row_id.nunique()
    return result[result.unique_rows >= minimum].reset_index()


def verify(months, saved, reference, scheme):
    checks = []
    for estimator in ("E1", "E5", "Eselected"):
        row = saved[(saved.reference == reference) & (saved.branch == "hierarchy_own")
                    & (saved.scheme == scheme) & (saved.pairing == "core") & (saved.stratum == "all")
                    & (saved.estimator == estimator) & (saved.minimum == 3)
                    & (saved.repeat_summary == "mean_over_repeats")]
        if len(row) != 1:
            raise ValueError("Missing or ambiguous saved site-month reference")
        observed = []
        for _, group in months.groupby("repeat"):
            x, y = group.truth_daod.to_numpy(), group[estimator + "_daod"].to_numpy()
            observed.append({"bias": np.mean(y - x), "rmse": np.sqrt(np.mean((y - x) ** 2)),
                             "mae": np.mean(abs(y - x)), "r": np.corrcoef(x, y)[0, 1]})
        means = pd.DataFrame(observed).mean()
        for metric, value in means.items():
            expected = row.iloc[0][metric]
            if not np.isclose(value, expected, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"Site-month reconstruction mismatch {reference} {scheme} {estimator} {metric}")
            checks.append({"estimator": estimator, "metric": metric, "absolute_difference": abs(value - expected)})
        actual_counts = {"prediction_events": len(months), "sites": months.site_key.nunique(),
                         "unique_site_months": len(months[["site_key", "year", "month"]].drop_duplicates())}
        for key, value in actual_counts.items():
            if value != row.iloc[0][key]:
                raise ValueError(f"Count mismatch: {key}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Write unpublished arrays outside the public worktree")
    ev = args.source_root / "expected_value_test"
    metrics_path = ev / "stage_b/site_month_metrics.csv"
    saved = pd.read_csv(metrics_path)
    source = {str(metrics_path.resolve()): digest(metrics_path)}
    arrays, checks = [], []
    for reference in ("experimental1020", "old675LR56"):
        path = ev / "predictions" / f"stage_b_{reference}_hierarchy_own.parquet"
        source[str(path.resolve())] = digest(path)
        frame = pd.read_parquet(path)
        for scheme, group in frame.groupby("scheme"):
            result = monthly(group)
            checks += [{"reference": reference, "scheme": scheme, **c}
                       for c in verify(result, saved, reference, scheme)]
            result["reference"], result["scheme"] = reference, scheme
            result["minimum"], result["pairing"] = 3, "core"
            arrays.append(result)
    for path, expected in source.items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "site_months_hierarchy.csv"
    pd.concat(arrays, ignore_index=True).to_csv(path, index=False)
    path.with_suffix(".provenance.json").write_text(json.dumps({
        "sources": source, "script_sha256": digest(Path(__file__)), "output_sha256": digest(path),
        "semantics": "Minimum three unique collocations per site/year/month/repeat, core estimator "
        "intersection; saved OOF fractions multiplied by DB AOD once before averaging. Separate repeats "
        "are retained as separate plotted points, not new independent collocations. Annotation metrics "
        "average repeat-specific scores. QA2/DB-type/EV scores and counts verified against saved tables.",
        "checks": checks,
    }, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {path}; {len(checks)} saved-score checks passed")


if __name__ == "__main__":
    main()
