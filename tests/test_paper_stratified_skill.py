"""Synthetic saved-score selection, support, interval, and layout checks for S9."""
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
    from paper.figures import figS09_stratified_skill
    return figS09_stratified_skill


def table(records, keys):
    return dict(records=records, columns=list(records[0]), filter_columns=keys,
                count_columns=[c for c in ("n", "unique_rows", "sites") if c in records[0]],
                source_file="/synthetic/saved.csv", source_sha256="synthetic-sha256",
                semantics="Saved held-out prediction events; distinct supports.")


@pytest.fixture
def payload(plotter):
    rows, intervals = [], []
    qa = plotter.CONFIG["qa"]["thresholds"][plotter.CONFIG["qa"]["product_gate_index"]]
    for model, population in (plotter.HIERARCHY,) + plotter.FALLBACK + (("full8", "own"),):
        for scheme in plotter.SCHEMES:
            for scope, groups in (("AOD_bin", ["0.2-0.4", "0.4-0.8", ">=0.8"]),
                                  ("region", ["Central Asia", "Amazonia_Cerrado", "other_Europe"])):
                for group in groups:
                    repeat = 2 if scheme == "site_cv" else 1
                    count = 8 if population == "common" else 10
                    missing = group == "Amazonia_Cerrado"
                    sites = 1 if group == "Central Asia" else 3
                    key = dict(reference=plotter.CONFIG["reference"]["name"], model=model,
                               population=population, scheme=scheme, QA=qa, scope=scope, group=group)
                    rows.append(dict(**key, n=count * repeat, unique_rows=count, sites=sites,
                                     eligible_rows=count * repeat, TP=0 if missing else 2 * repeat,
                                     FN=0 if missing else 2 * repeat, FP=0,
                                     TN=(count if missing else count - 4) * repeat,
                                     DAOD_RMSE=.123456789, DAOD_bias=-.0123456789,
                                     recall=None if missing else .5,
                                     unplotted_saved_metric=.987654321))
                    ci = dict(**key, bootstrap_unit="site", replicates=400,
                              DAOD_RMSE_lo=.08, DAOD_RMSE_hi=.21,
                              DAOD_bias_lo=-.035, DAOD_bias_hi=.005,
                              recall_lo=None if missing else .1, recall_hi=None if missing else .9,
                              unplotted_saved_metric_lo=.8, unplotted_saved_metric_hi=1.)
                    if sites == 1:
                        ci.update({k: None for k in ci if k.endswith(("_lo", "_hi"))})
                    intervals.append(ci)
    return {"tables": {plotter.TABLE: table(rows, plotter.KEYS),
                       plotter.INTERVALS: table(intervals, plotter.KEYS)}}


def test_hierarchy_preferred_exact_values_full_columns_and_support(plotter, payload):
    original = copy.deepcopy(payload)
    data, sources = plotter.select_data(Registry(payload))
    assert len(data) == 12
    assert set(data.model) == {"hierarchy_prespecified"}
    assert set(data.population) == {"own"}
    assert set(data.estimator) == {"QA2"}
    assert set(data.aod_bin_variable) == {"AER_AOD550"}
    assert not data.ev_available.any()
    raw = pd.DataFrame(payload["tables"][plotter.TABLE]["records"])
    expected = raw.loc[raw.model.eq("hierarchy_prespecified")].sort_values(plotter.KEYS).reset_index(drop=True)
    actual = data.sort_values(plotter.KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[raw.columns], expected, check_dtype=False)
    assert actual.unplotted_saved_metric_lo.eq(.8).sum() == 10
    assert payload == original
    assert sources[0]["table"] == plotter.TABLE
    assert sources[1]["table"] == plotter.INTERVALS
    assert data.loc[data.scheme.eq("site_cv"), "n"].eq(20).all()
    assert data.loc[data.scheme.eq("region_out"), "n"].eq(10).all()


def test_fallback_is_explicit_distinct_not_pooled(plotter, payload):
    for item in payload["tables"].values():
        item["records"] = [r for r in item["records"] if r["model"] != "hierarchy_prespecified"]
    data, _ = plotter.select_data(Registry(payload))
    assert len(data) == 24
    assert set(data[["model", "population"]].itertuples(index=False, name=None)) == set(plotter.FALLBACK)
    assert set(data.loc[data.model.eq("full8"), "unique_rows"]) == {8}
    assert set(data.loc[data.model.eq("noSSA5"), "unique_rows"]) == {10}
    assert "distinct supports" in plotter.caption(data)
    assert "not a paired" in plotter.caption(data)
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
        assert len(fig.legends[0].get_texts()) == 4
    finally:
        plotter.plt.close(fig)


@pytest.mark.parametrize("kind", ["table", "row", "bounds"])
def test_absent_intervals_are_not_invented(plotter, payload, kind):
    if kind == "table":
        del payload["tables"][plotter.INTERVALS]
    else:
        item = payload["tables"][plotter.INTERVALS]
        if kind == "row":
            item["records"] = [r for r in item["records"] if r["model"] != "hierarchy_prespecified"]
        else:
            for record in item["records"]:
                for key in record:
                    if key.endswith(("_lo", "_hi")):
                        record[key] = None
    data, _ = plotter.select_data(Registry(payload))
    assert not data[[f"{m}_interval_available" for m in plotter.METRICS]].any().any()
    assert data.DAOD_RMSE.notna().all()


def test_missing_recall_is_not_zero_and_single_site_has_no_ci(plotter, payload):
    data, _ = plotter.select_data(Registry(payload))
    assert data.loc[data.group.eq("Amazonia_Cerrado"), "recall"].isna().all()
    assert not data.loc[data.sites.eq(1), "DAOD_RMSE_interval_available"].any()
    fig = plotter.build_figure(data)
    try:
        assert [t.get_text() for t in fig.axes[-1].texts].count("NA") == 2
    finally:
        plotter.plt.close(fig)


@pytest.mark.parametrize("problem", ["duplicate", "partial_hierarchy", "partial_stratum", "counts",
                                    "confusion", "negative_rmse", "bad_recall", "missing_recall",
                                    "infinite", "duplicate_ci", "partial_ci", "reversed_ci",
                                    "wrong_bootstrap", "single_site_ci", "missing_ci_column"])
def test_fail_closed_saved_source_inconsistencies(plotter, payload, problem):
    rows = payload["tables"][plotter.TABLE]["records"]
    cis = payload["tables"][plotter.INTERVALS]["records"]
    if problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem in ("partial_hierarchy", "partial_stratum"):
        rows[:] = [r for r in rows if not (r["model"] == "hierarchy_prespecified" and
                   r["scheme"] == "site_cv" and (problem == "partial_hierarchy" or r["group"] == ">=0.8"))]
    elif problem == "counts":
        rows[0]["unique_rows"] = rows[0]["n"] + 1
    elif problem == "confusion":
        rows[0]["TP"] += 1
    elif problem == "negative_rmse":
        rows[0]["DAOD_RMSE"] = -.1
    elif problem == "bad_recall":
        rows[0]["recall"] = 1.1
    elif problem == "missing_recall":
        rows[0]["recall"] = None
    elif problem == "infinite":
        rows[0]["DAOD_bias"] = np.inf
    elif problem == "duplicate_ci":
        cis.append(cis[0].copy())
    elif problem == "partial_ci":
        cis[0]["DAOD_RMSE_lo"] = None
    elif problem == "reversed_ci":
        cis[0]["DAOD_RMSE_lo"] = 2.
    elif problem == "wrong_bootstrap":
        cis[0]["bootstrap_unit"] = "row"
    elif problem == "single_site_ci":
        cis[3]["DAOD_RMSE_lo"], cis[3]["DAOD_RMSE_hi"] = .1, .2
    else:
        payload["tables"][plotter.INTERVALS]["columns"].remove("DAOD_RMSE_lo")
    with pytest.raises(ValueError):
        plotter.select_data(Registry(payload))


def test_saved_percentile_interval_need_not_contain_point(plotter, payload):
    payload["tables"][plotter.INTERVALS]["records"][0]["DAOD_RMSE_lo"] = .15
    data, _ = plotter.select_data(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
    finally:
        plotter.plt.close(fig)


def test_layout_bounds_saved_endpoints_and_caption(plotter, payload):
    data, _ = plotter.select_data(Registry(payload))
    assert plotter._groups(data, "AOD_bin") == ["0.2-0.4", "0.4-0.8", ">=0.8"]
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
        assert fig.get_figheight() / plotter.style.MM <= 230
        for row, scope in enumerate(plotter.SCOPES):
            for col, metric in enumerate(plotter.METRICS):
                ax = fig.axes[row * 3 + col]
                values = data.loc[data.scope.eq(scope), [metric, f"{metric}_lo", f"{metric}_hi"]].to_numpy(float)
                values = values[np.isfinite(values)]
                low, high = ax.get_xlim()
                assert (values > low).all() and (values < high).all()
        text = plotter.caption(data)
        for phrase in ("AERONET AOD550", "Site-CV regional summaries hold out sites", "not zero recall",
                       "No new fits or bootstrap", "EV is absent", "two repeats", "own eligible"):
            assert phrase in text
    finally:
        plotter.plt.close(fig)


def test_cli_exports_full_data_and_never_changes_registry(plotter, payload, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(clean(payload)))
    before = digest(registry)
    output = tmp_path / "figures"
    captured = {}

    def export(fig, out, stem, data, registry_path, sources, script):
        plotter.check_layout(fig)
        captured.update(stem=stem, sources=sources, script=script)
        Path(out).mkdir()
        data.to_csv(Path(out) / "FigS09_data.csv", index=False)

    monkeypatch.setattr(plotter.style, "export", export)
    plotter.main(["--registry", str(registry), "--output-dir", str(output)])
    assert digest(registry) == before
    saved = pd.read_csv(output / "FigS09_data.csv")
    assert len(saved) == 12 and saved.unplotted_saved_metric.notna().all()
    assert "EV is absent" in (output / "FigS09_caption.md").read_text()
    assert captured["stem"] == "FigS09_stratified_skill"
    assert captured["script"] == plotter.__file__


def test_off_canvas_text_refused(plotter, payload):
    data, _ = plotter.select_data(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        fig.text(-.2, .5, "clipped")
        with pytest.raises(ValueError, match="outside figure"):
            plotter.check_layout(fig)
    finally:
        plotter.plt.close(fig)


def test_overlapping_text_refused(plotter, payload):
    data, _ = plotter.select_data(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        fig.text(.01, .5, "overlap one")
        fig.text(.01, .5, "overlap two")
        with pytest.raises(ValueError, match="Overlapping text"):
            plotter.check_layout(fig)
    finally:
        plotter.plt.close(fig)
