import numpy as np
import pandas as pd
import pytest

from paper.derive_probability_diagnostics import BINS, ESTIMATORS, decompose, extract


@pytest.mark.parametrize("p,y", [([0, 1], [0, 1]), ([.5, .5], [0, 1]),
                                 ([.11, .19, .3, .91], [0, 1, 1, 0]), ([1, 1], [0, 0])])
def test_decomposition_conserves_actual_continuous_score(p, y):
    r = decompose(p, y)
    assert r["brier"] == pytest.approx(np.mean((np.array(p) - y) ** 2))
    assert r["brier"] == pytest.approx(r["reliability"] - r["resolution"] + r["uncertainty"]
                                         + r["within_bin_correction"], abs=2e-15)


def test_within_bin_correction_is_not_discarded():
    r = decompose([.11, .19], [0, 1])
    assert r["within_bin_correction"] == pytest.approx(.0016 - .04)
    assert r["brier"] != pytest.approx(r["reliability"] - r["resolution"] + r["uncertainty"])


@pytest.mark.parametrize("p,y", [([], []), ([.5], [0, 1]), ([np.nan], [0]),
                                  ([1.01], [1]), ([.1], [.3]), ([.1], [np.inf])])
def test_invalid_probabilities_rejected(p, y):
    with pytest.raises(ValueError):
        decompose(p, y)


def fixture():
    records = []
    for scheme, repeat, n in [("site_cv", 0, 3), ("site_cv", 1, 2), ("region_out", 0, 4)]:
        for i in range(n):
            records.append(dict(scheme=scheme, repeat=repeat, row_id=str(i), site_key=f"s{i % 2}",
                probability=[.12, .8, .91, .3][i] if repeat == 0 else [.99, .14][i],
                probability_calibrated=[0, 1, .9, .1][i], truth_fraction=[0, .2, .8, 0][i],
                truth_daod=.1, DB_AOD550=.3))
    d = pd.DataFrame(records)
    bins = []
    for scheme, s in d.groupby("scheme"):
        for e in ESTIMATORS:
            q = s.assign(bin=np.searchsorted(BINS, s[e], side="right") - 1)
            for b, g in q.groupby("bin"):
                bins.append(dict(stage="b", reference="ref", branch="branch", scheme=scheme, estimator=e,
                    bin_low=BINS[b], bin_high=min(1, BINS[b + 1]), prediction_events=len(g),
                    unique_rows=g.row_id.nunique(), sites=g.site_key.nunique(), repeats=g["repeat"].nunique(),
                    predicted=g.groupby("repeat")[e].mean().mean(),
                    observed=g.assign(label=g.truth_fraction.ge(.2)).groupby("repeat").label.mean().mean()))
    return d, pd.DataFrame(bins)


def test_equal_repeat_scores_not_ensemble_or_pooled_scores():
    d, bins = fixture()
    scores, checks = extract(d, bins, "ref", "branch")
    raw = scores.loc[scores.scheme.eq("site_cv") & scores.estimator.eq("probability")]
    mean = raw.loc[raw.repeat_summary.eq("mean_over_repeats")].iloc[0]
    assert mean.brier == pytest.approx(raw.loc[raw.repeat_summary.ne("mean_over_repeats"), "brier"].mean())
    q = d.loc[d.scheme.eq("site_cv")]
    assert mean.brier != pytest.approx(np.mean((q.probability - q.truth_fraction.ge(.2)) ** 2))
    assert mean.prediction_events == 5 and mean.unique_rows == 3
    assert checks.verified.all()


def test_float32_scores_use_original_float64_accumulation():
    d, _ = fixture()
    d["probability"] = d.probability.astype(np.float32)
    # Prepare exact source means independently, after the original accumulator's promotion.
    saved = []
    for scheme, s in d.groupby("scheme"):
        for e in ESTIMATORS:
            q = s.assign(bin=np.searchsorted(BINS, s[e], side="right") - 1)
            for b, g in q.groupby("bin"):
                saved.append(dict(stage="b", reference="ref", branch="branch", scheme=scheme, estimator=e,
                    bin_low=BINS[b], bin_high=min(1, BINS[b + 1]), prediction_events=len(g),
                    unique_rows=g.row_id.nunique(), sites=g.site_key.nunique(), repeats=g["repeat"].nunique(),
                    predicted=np.mean([np.mean(z[e].to_numpy(dtype=np.float64)) for _, z in g.groupby("repeat")]),
                    observed=g.assign(label=g.truth_fraction.ge(.2)).groupby("repeat").label.mean().mean()))
    _, checks = extract(d, pd.DataFrame(saved), "ref", "branch")
    assert checks.verified.all()


@pytest.mark.parametrize("change,message", [("count", "means/counts"), ("mean", "means/counts"),
    ("duplicate", "Duplicate"), ("scheme", "scheme"), ("support", "supports differ"),
    ("missing_bin", "bin identities")])
def test_fail_closed_when_saved_sources_disagree(change, message):
    d, bins = fixture()
    if change == "count": bins.loc[0, "prediction_events"] += 1
    if change == "mean": bins.loc[0, "predicted"] += .1
    if change == "duplicate": d = pd.concat([d, d.iloc[[0]]])
    if change == "scheme": d = d.loc[d.scheme.eq("site_cv")]
    if change == "support": d.loc[0, "probability_calibrated"] = np.nan
    if change == "missing_bin": bins = bins.iloc[1:]
    with pytest.raises(ValueError, match=message):
        extract(d, bins, "ref", "branch")
