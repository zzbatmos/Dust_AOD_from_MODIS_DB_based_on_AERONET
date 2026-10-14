"""S11 uses saved metrics, exact supports, and explicit repeat identities."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.numbers import Registry, clean, digest


@pytest.fixture
def plotter():
    pytest.importorskip("cmcrameri")
    pytest.importorskip("colorspacious")
    from paper.figures import figS11_site_month_sensitivity
    return figS11_site_month_sensitivity


def _table(records):
    return dict(records=records, columns=list(records[0]), source_file="/synthetic/saved.csv",
                source_sha256="synthetic-sha256", semantics="Saved mean of repeat-specific metrics.")


@pytest.fixture
def payload(plotter):
    monthly, strata = [], []

    def record(scheme, estimator, minimum, summary, stratum="all", sites=8, cells=20):
        repeats = 2 if scheme == "site_cv" and summary == "mean_over_repeats" else 1
        # Distinct repeat RMSEs; the saved mean is not their RMS or a pooled metric.
        rmse = {"mean_over_repeats": .2, "repeat_0": .1, "repeat_1": .3}[summary]
        return dict(**plotter.FILTERS, scheme=scheme, estimator=estimator, minimum=minimum,
                    repeat_summary=summary, stratum=stratum, sites=sites, repeats=repeats,
                    unique_site_months=cells, prediction_events=cells * repeats, unique_rows=cells * 6,
                    bias=0., bias_lo=-.02 if sites > 1 else 0., bias_hi=.03 if sites > 1 else 0.,
                    bias_valid_bootstraps=400, rmse=rmse, rmse_lo=rmse - .01, rmse_hi=rmse + .02,
                    rmse_valid_bootstraps=400, r=.6, r_lo=.3, r_hi=.9, r_valid_bootstraps=400,
                    unplotted_saved_metric=.123456789)

    for scheme in plotter.SCHEMES:
        for minimum in plotter.MINIMA:
            summaries = ["mean_over_repeats"]
            if minimum == 3:
                summaries += ["repeat_0"] + (["repeat_1"] if scheme == "site_cv" else [])
            for summary in summaries:
                for estimator in plotter.ESTIMATORS + ("E3",):
                    monthly.append(record(scheme, estimator, minimum, summary))
        for group, sites, cells in [("all", 9, 25), ("Siberia_fire_season", 1, 2),
                                    ("East Asian deserts_MAM", 2, 5), ("Sahel", 3, 6),
                                    ("East Asian deserts_DJF", 4, 12), ("Sahara", 5, 22)]:
            for estimator in ("E1", "Eselected", "E5", "E3"):
                strata.append(record(scheme, estimator, 3, "mean_over_repeats", group, sites, cells))
    # Wrong reference, branch, and pairing must never leak into this figure.
    for target in (monthly, strata):
        for key, value in (("reference", "experimental1020"), ("branch", "full8_common"),
                           ("pairing", "including_E6")):
            target.append(dict(target[0], **{key: value, "bias": 999.}))
    return {"tables": {plotter.ROOT + "site_month_metrics": _table(monthly),
                       plotter.ROOT + "stratum_bias": _table(strata)}}


def test_exact_support_full_columns_and_source_provenance(plotter, payload):
    original = copy.deepcopy(payload)
    data, sources = plotter.select(Registry(payload))
    assert len(data) == 93
    for name in plotter.TABLES:
        saved = data.loc[data.source_table.eq(plotter.ROOT + name)]
        table = payload["tables"][plotter.ROOT + name]
        expected = pd.DataFrame(table["records"]).loc[saved.source_record_index].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[table["columns"]].reset_index(drop=True), expected, check_dtype=False)
    assert set(data.reference) == {"old675LR56"}
    assert set(data.branch) == {"hierarchy_own"} and set(data.pairing) == {"core"}
    assert payload == original
    assert [s["table"] for s in sources[:2]] == [plotter.ROOT + n for n in plotter.TABLES]
    assert data.unplotted_saved_metric.eq(.123456789).all()
    assert data.loc[data.estimator.eq("E3"), "plot_role"].eq("source_only").all()


def test_minima_and_repeats_never_averaged_or_fabricated(plotter, payload):
    data, _ = plotter.select(Registry(payload))
    mean = data.loc[data.plot_role.eq("minimum_sensitivity")]
    repeats = data.loc[data.plot_role.eq("repeat_detail")]
    assert len(mean) == 12 and len(repeats) == 8
    assert set(mean.minimum) == {1, 3, 5} and mean.rmse.eq(.2).all()
    assert set(repeats.minimum) == {3}
    assert repeats.loc[repeats.repeat_summary.eq("repeat_0"), "rmse"].eq(.1).all()
    assert repeats.loc[repeats.repeat_summary.eq("repeat_1"), "rmse"].eq(.3).all()
    assert mean.prediction_events.eq(40).all() and repeats.prediction_events.eq(20).all()
    assert data.unique_rows.gt(data.prediction_events).all()
    assert not ((data.minimum != 3) & data.repeat_summary.isin(plotter.REPEATS)).any()


def test_exploratory_selection_missing_e7_and_zero_width_intervals(plotter, payload):
    data, sources = plotter.select(Registry(payload))
    exploratory = data.loc[data.plot_role.eq("exploratory_stratum")]
    assert exploratory.sites.lt(5).all()
    assert not exploratory.stratum.eq("Sahara").any()
    assert not data.loc[data.source_table.str.endswith("stratum_bias"), "estimator"].eq("E7").any()
    assert {item["estimator"] for item in sources[-1]["absent_estimator_strata"]} == {"E7"}
    single = exploratory.loc[exploratory.sites.eq(1)]
    assert single.bias.eq(0).all() and single.bias_interval_available.all()
    assert single.bias_lo.eq(single.bias_hi).all()
    assert "Single-site intervals can collapse" in plotter.caption(data)
    assert "E7 has no saved" in plotter.caption(data)


def test_partial_estimator_stratum_absence_is_not_backfilled(plotter, payload):
    rows = payload["tables"][plotter.ROOT + "stratum_bias"]["records"]
    rows[:] = [r for r in rows if not (r["estimator"] == "E5" and r["stratum"] == "Sahel")]
    data, sources = plotter.select(Registry(payload))
    assert not (data.stratum.eq("Sahel") & data.estimator.eq("E5")).any()
    assert sum(item["estimator"] == "E5" for item in sources[-1]["absent_estimator_strata"]) == 2


def test_missing_metric_and_ci_stay_missing_not_zero(plotter, payload):
    row = payload["tables"][plotter.ROOT + "site_month_metrics"]["records"][0]
    row.update(r=None, r_lo=None, r_hi=None, r_valid_bootstraps=0)
    data, _ = plotter.select(Registry(payload))
    saved = data.loc[data.source_table.str.endswith("site_month_metrics") & data.source_record_index.eq(0)].iloc[0]
    assert pd.isna(saved.r) and not saved.r_interval_available
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
    finally:
        plotter.plt.close(fig)


def test_missing_interval_columns_preserve_saved_points(plotter, payload):
    for metadata in payload["tables"].values():
        metadata["columns"] = [c for c in metadata["columns"] if c not in ("bias_lo", "bias_hi")]
    data, _ = plotter.select(Registry(payload))
    assert data.bias.eq(0).all()
    assert data.bias_lo.isna().all() and data.bias_hi.isna().all()
    assert not data.bias_interval_available.any()


def test_shared_count_labels_cannot_hide_scheme_difference(plotter, payload):
    rows = payload["tables"][plotter.ROOT + "stratum_bias"]["records"]
    for record in rows:
        if record["scheme"] == "region_out" and record["stratum"] == "Sahel":
            record["sites"] = 2
    data, _ = plotter.select(Registry(payload))
    try:
        with pytest.raises(ValueError, match="shared exploratory count labels"):
            plotter.build_figure(data)
    finally:
        plotter.plt.close("all")


@pytest.mark.parametrize("problem", ["duplicate", "missing_repeat", "missing_minimum", "bad_counts",
                                    "wrong_event_unit", "per_repeat_count", "unpaired_counts", "half_ci",
                                    "reversed_ci", "infinite", "negative_rmse", "invalid_r", "zero_bootstraps"])
def test_fail_closed_source_inconsistencies(plotter, payload, problem):
    rows = payload["tables"][plotter.ROOT + "site_month_metrics"]["records"]
    if problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "missing_repeat":
        rows[:] = [r for r in rows if r["repeat_summary"] != "repeat_1"]
    elif problem == "missing_minimum":
        rows[:] = [r for r in rows if r["minimum"] != 5]
    elif problem == "bad_counts":
        rows[0]["sites"] = -1
    elif problem == "wrong_event_unit":
        rows[0]["prediction_events"] = rows[0]["unique_rows"]
    elif problem == "per_repeat_count":
        next(r for r in rows if r["repeat_summary"] == "repeat_0")["repeats"] = 2
    elif problem == "unpaired_counts":
        rows[0]["sites"] -= 1
    elif problem == "half_ci":
        rows[0]["bias_lo"] = None
    elif problem == "reversed_ci":
        rows[0]["bias_lo"] = 1.
    elif problem == "infinite":
        rows[0]["bias"] = np.inf
    elif problem == "negative_rmse":
        rows[0]["rmse"] = -.1
    elif problem == "invalid_r":
        rows[0]["r"] = 2.
    else:
        rows[0]["bias_valid_bootstraps"] = 0
    with pytest.raises(ValueError):
        plotter.select(Registry(payload))


def test_full_saved_endpoints_not_point_centered_errors_and_layout(plotter, payload):
    # Percentile intervals are saved endpoints, not guaranteed to enclose the point.
    payload["tables"][plotter.ROOT + "site_month_metrics"]["records"][0]["bias_lo"] = .01
    data, _ = plotter.select(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
        assert len(fig.axes) == 8
        assert fig.get_figwidth() / plotter.style.MM == pytest.approx(190)
        assert fig.get_figheight() / plotter.style.MM <= 230
        for index, ax in enumerate(fig.axes):
            if index < 6:
                metric = plotter.METRICS[index % 3]
                q = data.loc[data.plot_role.isin(["minimum_sensitivity", "repeat_detail"])]
                lo, hi = ax.get_ylim()
            else:
                metric = "bias"
                q = data.loc[data.plot_role.eq("exploratory_stratum")]
                lo, hi = ax.get_xlim()
            values = q[[metric, f"{metric}_lo", f"{metric}_hi"]].to_numpy(float)
            assert (values[np.isfinite(values)] > lo).all() and (values[np.isfinite(values)] < hi).all()
        text = plotter.caption(data)
        for phrase in ("not predictions or pooled events", "not prediction-event counts",
                       "no cross-table pooling", "strata and region-out sensitivity"):
            assert phrase in text
    finally:
        plotter.plt.close(fig)


@pytest.mark.parametrize("problem", ["off_canvas", "overlap", "small_font"])
def test_layout_refuses_unreadable_text(plotter, payload, problem):
    data, _ = plotter.select(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        if problem == "off_canvas":
            fig.text(-.5, .5, "outside")
        elif problem == "overlap":
            fig.text(.01, .5, "overlap")
            fig.text(.01, .5, "overlap")
        else:
            fig.text(.01, .5, "small", fontsize=5)
        with pytest.raises(ValueError):
            plotter.check_layout(fig)
    finally:
        plotter.plt.close(fig)


def test_cli_export_registry_unchanged_single_stem(plotter, payload, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(clean(payload)))
    before = digest(registry)
    output = tmp_path / "figures"

    def export(fig, out, stem, data, path, evidence, script):
        assert stem == "FigS11_site_month_sensitivity" and path == registry
        assert script == plotter.__file__
        plotter.check_layout(fig)
        Path(out).mkdir()
        data.to_csv(Path(out) / (stem.split("_")[0] + "_data.csv"), index=False)
        (Path(out) / (stem.split("_")[0] + "_provenance.json")).write_text(json.dumps(evidence))

    monkeypatch.setattr(plotter.style, "export", export)
    plotter.main(["--registry", str(registry), "--output-dir", str(output)])
    assert digest(registry) == before
    saved = pd.read_csv(output / "FigS11_data.csv")
    assert len(saved) == 93 and set(saved.plot_role) == {
        "minimum_sensitivity", "repeat_detail", "exploratory_stratum", "source_only"}
    assert (output / "FigS11_caption.md").exists()
    assert (output / "FigS11_provenance.json").exists()
