"""Brier accounting from saved outer-fold scores; no calibration or fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import clean, digest

REFERENCES = ("experimental1020", "old675LR56")
BRANCHES = ("full8_common", "noSSA5_own", "minimal2_own", "hierarchy_own")
ESTIMATORS = ("probability", "probability_calibrated")
BINS = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.000001])
METRICS = ("brier", "reliability", "resolution", "uncertainty", "within_bin_correction", "prevalence")
SEMANTICS = (
    "Arithmetic on saved Stage B held-out predictions only. Labels are truth_fraction >= 0.2. "
    "Raw and nested-calibrated probabilities retain their common finite reference/AOD support. "
    "Brier scores and each decomposition term are computed separately per repeat, then equally "
    "averaged across repeats, never from ensemble-averaged predictions. The bins are exactly "
    "those used by the saved Stage B reliability table. Continuous probabilities require the "
    "within-bin correction E[var(P|bin)-2cov(P,Y|bin)]: BS=REL-RES+UNC+correction. "
    "Without that term REL-RES+UNC is the score of bin-mean predictions, not the actual score. "
    "No new bootstrap intervals, hypothesis tests, fitting or calibration. Reliability means, "
    "counts and bin identities are verified against the saved table for both schemes/references."
)


def decompose(probabilities, labels):
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Empty or mismatched probability/label arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Nonfinite or out-of-range probabilities/labels")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    index = np.searchsorted(BINS, p, side="right") - 1
    prevalence = y.mean()
    rel, res, correction = 0., 0., 0.
    for b in np.unique(index):
        q, z = p[index == b], y[index == b]
        w = len(q) / len(p)
        rel += w * (q.mean() - z.mean()) ** 2
        res += w * (z.mean() - prevalence) ** 2
        correction += w * (np.mean((q - q.mean()) ** 2)
                           - 2 * np.mean((q - q.mean()) * (z - z.mean())))
    result = dict(brier=np.mean((p - y) ** 2), reliability=rel, resolution=res,
                  uncertainty=prevalence * (1 - prevalence), within_bin_correction=correction,
                  prevalence=prevalence)
    if not np.isclose(result["brier"], rel - res + result["uncertainty"] + correction,
                      atol=2e-15, rtol=0):
        raise ValueError("Brier decomposition does not conserve score")
    return result


def extract(events, saved_bins, reference, branch):
    required = ["truth_daod", "truth_fraction", "DB_AOD550", *ESTIMATORS]
    if events.duplicated(["scheme", "row_id", "repeat"]).any():
        raise ValueError("Duplicate held-out prediction identity")
    if set(events.scheme) != {"site_cv", "region_out"}:
        raise ValueError("Missing validation scheme")
    masks = [np.isfinite(events[["truth_daod", "truth_fraction", "DB_AOD550", e]].to_numpy(float)).all(axis=1)
             for e in ESTIMATORS]
    if not np.array_equal(*masks):
        raise ValueError("Raw/calibrated supports differ; do not silently intersect")
    d = events.loc[masks[0]].copy()
    if d.empty or d[required].isna().any().any() or not d.truth_fraction.between(0, 1).all():
        raise ValueError("Missing or invalid reference support")
    # The original sufficient-statistic accumulator promoted saved float32 scores.
    for estimator in ESTIMATORS:
        d[estimator] = d[estimator].astype(float)
    results, bin_checks = [], []
    for scheme, s in d.groupby("scheme"):
        if set(s["repeat"]) != ({0, 1} if scheme == "site_cv" else {0}):
            raise ValueError("Unexpected repeat identities")
        for estimator in ESTIMATORS:
            repeat_results = []
            for repeat, q in s.groupby("repeat"):
                scores = decompose(q[estimator], q.truth_fraction.ge(.2).astype(float))
                repeat_results.append(scores)
                results.append(dict(reference=reference, branch=branch, scheme=scheme, estimator=estimator,
                    repeat_summary=f"repeat_{repeat}", prediction_events=len(q), unique_rows=q.row_id.nunique(),
                    sites=q.site_key.nunique(), repeats=1, **scores))
            means = {k: float(np.mean([r[k] for r in repeat_results])) for k in METRICS}
            results.append(dict(reference=reference, branch=branch, scheme=scheme, estimator=estimator,
                repeat_summary="mean_over_repeats", prediction_events=len(s), unique_rows=s.row_id.nunique(),
                sites=s.site_key.nunique(), repeats=s["repeat"].nunique(), **means))
            q = s.assign(bin=np.searchsorted(BINS, s[estimator], side="right") - 1)
            saved = saved_bins.loc[saved_bins.reference.eq(reference) & saved_bins.branch.eq(branch)
                & saved_bins.scheme.eq(scheme) & saved_bins.estimator.eq(estimator) & saved_bins.stage.eq("b")]
            if len(saved) != q.bin.nunique() or saved.duplicated(["bin_low", "bin_high"]).any():
                raise ValueError("Saved reliability bin identities disagree")
            for b, g in q.groupby("bin"):
                selected = saved.loc[np.isclose(saved.bin_low, BINS[b], atol=1e-14, rtol=0)
                                     & np.isclose(saved.bin_high, min(1, BINS[b + 1]), atol=1e-14, rtol=0)]
                if len(selected) != 1:
                    raise ValueError("Missing saved reliability bin")
                row = selected.iloc[0]
                predicted = g.groupby("repeat")[estimator].mean().mean()
                observed = g.assign(label=g.truth_fraction.ge(.2)).groupby("repeat").label.mean().mean()
                if (row.prediction_events != len(g) or row.unique_rows != g.row_id.nunique()
                    or row.sites != g.site_key.nunique() or row.repeats != g["repeat"].nunique()
                    or not np.allclose([row.predicted, row.observed], [predicted, observed], atol=1e-12, rtol=0)):
                    raise ValueError("Saved reliability means/counts disagree")
                bin_checks.append(dict(reference=reference, branch=branch, scheme=scheme, estimator=estimator,
                    bin_low=BINS[b], bin_high=min(1, BINS[b + 1]), prediction_events=len(g),
                    predicted=predicted, observed=observed, verified=True))
    return pd.DataFrame(results), pd.DataFrame(bin_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root / "expected_value_test"
    output = args.output_dir.resolve()
    if output.is_relative_to(args.source_root.resolve()) or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Keep derived data outside frozen inputs and public code")
    paths = [root / "stage_b/calibration_bins.csv", root / "analyze.py", root / "common.py"]
    paths += [root / f"predictions/stage_b_{r}_{b}.parquet" for r in REFERENCES for b in BRANCHES]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    saved = pd.read_csv(paths[0])
    scores, checks = [], []
    for reference in REFERENCES:
        for branch in BRANCHES:
            frame = pd.read_parquet(root / f"predictions/stage_b_{reference}_{branch}.parquet")
            a, b = extract(frame, saved, reference, branch)
            scores.append(a); checks.append(b)
    if any(digest(p) != hashes[str(p.resolve())] for p in paths):
        raise RuntimeError("Inputs changed during extraction")
    output.mkdir(parents=True, exist_ok=True)
    names = ["probability_brier", "probability_bin_checks"]
    if any((output / f"{name}.csv").exists() for name in names):
        raise FileExistsError("Preserve previous derived outputs")
    for name, frames in zip(names, [scores, checks]):
        data = pd.concat(frames, ignore_index=True)
        path = output / f"{name}.csv"
        data.to_csv(path, index=False)
        path.with_suffix(".provenance.json").write_text(json.dumps(clean(dict(
            sources=hashes, script_sha256=digest(Path(__file__)), output_sha256=digest(path),
            semantics=SEMANTICS, rows=len(data), bins=BINS.tolist())), indent=2, allow_nan=False) + "\n")
        print(name, len(data))


if __name__ == "__main__":
    main()
