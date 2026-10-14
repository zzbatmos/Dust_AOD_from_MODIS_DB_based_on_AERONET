"""Synthetic registry-only tests for the qualified mixed-scale Figure 5."""
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
    from paper.figures import fig05_site_month
    return fig05_site_month


def table(records):
    return dict(records=records, columns=list(records[0]), source_file="/synthetic/frozen.csv",
                source_sha256="synthetic-only", source_rows=len(records),
                semantics="Saved values; synthetic test fixture.")


@pytest.fixture
def payload(plotter):
    points = []
    for repeat in (0, 1):
        for i in range(10):
            points.append(dict(site_key=f"site_{i}", year=2001, month=1, repeat=repeat,
                               truth_daod=.05 * i, E1_daod=.03 * i + repeat * .01,
                               E5_daod=.04 * i, Eselected_daod=.05 * i + repeat * .02,
                               unique_rows=3, reference="experimental1020", scheme="site_cv", minimum=3, pairing="core",
                               untouched_saved_value=.123456789))

    def score(estimator, stratum="all", sites=10, cells=10):
        return dict(**plotter.MONTHLY, estimator=estimator, stratum=stratum, prediction_events=2 * cells,
                    sites=sites, repeats=2, unique_rows=3 * cells, unique_site_months=cells,
                    # Deliberately unrelated to coordinates: renderer must copy, not recompute, skill.
                    bias=-.0123, rmse=.2345, r=.7654, bias_lo=-.025, bias_hi=.015,
                    untouched_saved_value=.123456789)

    scores = [score(e) for e in plotter.ESTIMATORS]
    strata = []
    for stratum in (*plotter.STRATA, "Sahara_DJF"):
        sites = 1 if stratum == "Central Asia" else 3 if stratum == "Sahel" else 10
        for e in plotter.ESTIMATORS:
            row = score(e, stratum, sites, max(sites, 2))
            if sites == 1:
                row.update(bias_lo=row["bias"], bias_hi=row["bias"])
            strata.append(row)
    regions = ["all", "region=Amazonia_Cerrado", "region=Central Asia", "region=East Asian deserts",
               "region=East China", "region=India", "region=Middle East", "region=SW North America",
               "region=Sahara", "region=Sahel", "region=Siberia", "region=boreal_North_America",
               "region=mainland_SE_Asia", "region=other_Africa", "region=other_Americas",
               "region=other_Asia", "region=other_Europe", "region=southern_Africa"]
    decomposition = []
    for i, stratum in enumerate(regions):
        for component, value in dict(missed_dust_FN=-.02 - i * .001, false_dust_FP=.01,
                                     true_positive_TP=.005, subthreshold_reference_TN=-.015,
                                     TP_fraction_error=10., TP_total_AOD_error=-9.995).items():
            decomposition.append(dict(**plotter.BASE, stratum=stratum, component=component,
                                      prediction_events=60, sites=1 if stratum == "region=Central Asia" else 10,
                                      repeats=2, unique_rows=30, unique_site_months=20,
                                      contribution=value, ci_low=value - .005, ci_high=value + .005,
                                      untouched_saved_value=.123456789))
    rows = {plotter.POINTS: points, plotter.ROOT + "site_month_metrics": scores,
            plotter.ROOT + "stratum_bias": strata, plotter.ROOT + "bias_decomposition_E1": decomposition}
    # Wrong reference/scheme/support must not leak into any panel.
    for name, records in rows.items():
        for key, value in (("reference", "old675LR56"), ("scheme", "region_out")):
            records.append(dict(records[0], **{key: value}))
        if name != plotter.ROOT + "bias_decomposition_E1":
            records.append(dict(records[0], pairing="including_E6"))
    return {"tables": {name: table(records) for name, records in rows.items()}}


def test_exact_sources_filters_and_full_values(plotter, payload):
    original = copy.deepcopy(payload)
    data, evidence = plotter.select(Registry(payload))
    assert payload == original
    assert set(data.reference) == {"experimental1020"}
    assert set(data.scheme) == {"site_cv"}
    for source in evidence[:-1]:
        assert source["source_file"] == "/synthetic/frozen.csv"
        assert source["source_sha256"] == "synthetic-only"
        selected = data.loc[data.source_table.eq(source["table"])]
        metadata = payload["tables"][source["table"]]
        expected = pd.DataFrame(metadata["records"]).loc[selected.source_record_index.astype(int)]
        pd.testing.assert_frame_equal(selected[metadata["columns"]].reset_index(drop=True),
                                      expected.reset_index(drop=True), check_dtype=False)
        assert source["selected_record_indices"] == selected.source_record_index.tolist()
    assert data.untouched_saved_value.eq(.123456789).all()
    assert evidence[-1]["scales"] == dict(abc="site-month", d="collocation", e="site-month")
    assert "approved by author" in evidence[-1]["status"]


def test_repeats_and_annotations_never_reaggregated(plotter, payload):
    data, _ = plotter.select(Registry(payload))
    points = data.loc[data.plot_role.eq("scatter")]
    assert len(points) == 20 and set(points["repeat"]) == {0, 1}
    assert len(points[["site_key", "year", "month"]].drop_duplicates()) == 10
    scores = data.loc[data.plot_role.eq("annotation")]
    assert scores.rmse.eq(.2345).all() and scores.bias.eq(-.0123).all()
    assert scores.r.eq(.7654).all()
    assert not np.isclose(np.sqrt(np.mean((points.E1_daod - points.truth_daod) ** 2)), scores.rmse.iloc[0])


def test_four_terms_signed_stacks_total_and_all_first(plotter, payload):
    data, _ = plotter.select(Registry(payload))
    stacks = data.loc[data.plot_role.eq("stack")]
    assert set(stacks.component) == set(plotter.COMPONENTS)
    extras = data.loc[data.component.isin(["TP_fraction_error", "TP_total_AOD_error"])]
    assert extras.plot_role.eq("source_only").all() and extras.stack_left.isna().all()
    for _, group in stacks.groupby("stratum"):
        assert group.stack_total.iloc[0] == pytest.approx(group.contribution.sum())
        assert np.allclose(group.stack_right - group.stack_left, group.contribution)
        assert group.loc[group.contribution.gt(0), "stack_left"].ge(0).all()
        assert group.loc[group.contribution.lt(0), "stack_left"].le(0).all()
    order = plotter.decomposition_order(data)
    assert order[0] == "all"
    totals = stacks.drop_duplicates("stratum").set_index("stratum").loc[order[1:], "stack_total"]
    assert totals.is_monotonic_increasing


def test_fixed_strata_and_exploratory_not_score_selected(plotter, payload):
    data, _ = plotter.select(Registry(payload))
    strata = data.loc[data.plot_role.eq("stratum_bias")]
    assert set(strata.stratum) == set(plotter.STRATA)
    assert strata.exploratory.eq(strata.sites.lt(5)).all()
    seasonal = data.loc[data.stratum.eq("Sahara_DJF")]
    assert seasonal.plot_role.eq("source_only").all()
    assert not strata.stratum.eq("Amazonia_Cerrado_fire_season").any()


@pytest.mark.parametrize("problem", ["missing_repeat", "duplicate_point", "different_truth", "minimum", "nonfinite",
                                    "score_duplicate", "score_count", "score_missing", "component_missing", "component_duplicate",
                                    "stratum_missing", "stratum_unpaired", "interval_reversed", "interval_missing"])
def test_source_errors_fail_closed(plotter, payload, problem):
    tables = payload["tables"]
    points = tables[plotter.POINTS]["records"]
    scores = tables[plotter.ROOT + "site_month_metrics"]["records"]
    dec = tables[plotter.ROOT + "bias_decomposition_E1"]["records"]
    strata = tables[plotter.ROOT + "stratum_bias"]["records"]
    if problem == "missing_repeat":
        points.pop(0)
    elif problem == "duplicate_point":
        points.append(points[0].copy())
    elif problem == "different_truth":
        points[0]["truth_daod"] = .123
    elif problem == "minimum":
        points[0]["unique_rows"] = 2
    elif problem == "nonfinite":
        points[0]["E1_daod"] = None
    elif problem == "score_duplicate":
        scores.append(scores[0].copy())
    elif problem == "score_count":
        scores[0]["unique_site_months"] += 1
    elif problem == "score_missing":
        scores.pop(0)
    elif problem == "component_missing":
        dec.pop(0)
    elif problem == "component_duplicate":
        dec.append(dec[0].copy())
    elif problem == "stratum_missing":
        strata.pop(0)
    elif problem == "stratum_unpaired":
        strata[0]["sites"] = 5
    elif problem == "interval_reversed":
        strata[0]["bias_lo"] = 1.
    else:
        strata[0]["bias_hi"] = None
    with pytest.raises(ValueError):
        plotter.select(Registry(payload))


def test_missing_registry_source_stops_instead_of_reading_external(plotter, payload):
    del payload["tables"][plotter.POINTS]
    with pytest.raises(KeyError):
        plotter.select(Registry(payload))


def test_geometry_marker_scales_saved_endpoints_and_provisional_caption(plotter, payload):
    # Saved percentile endpoints need not enclose the point estimate.
    payload["tables"][plotter.ROOT + "stratum_bias"]["records"][0].update(bias_lo=.01, bias_hi=.02)
    data, _ = plotter.select(Registry(payload))
    fig = plotter.build_figure(data)
    try:
        plotter.check_layout(fig)
        assert fig.get_figwidth() / plotter.style.MM == pytest.approx(190)
        assert fig.get_figheight() / plotter.style.MM == pytest.approx(150)
        assert len(fig.axes) == 5
        for ax in fig.axes[:3]:
            assert ax.get_xlim() == (0., .8) and ax.get_ylim() == (0., .8)
            assert len(ax.collections[0].get_offsets()) == 20
            assert ax.collections[0].get_alpha() == .5
            assert ax.collections[0].get_sizes().tolist() == [9]
        d, e = fig.axes[3:]
        assert len(d.patches) == 18 * 4
        assert d.get_yticklabels()[0].get_text() == "All"
        assert "COLLOCATION" in d.get_title(loc="right")
        assert "Site-month" in e.get_title(loc="right")
        assert any(line.get_markerfacecolor() == "white" for line in e.lines)
        assert len(e.collections) == len(plotter.STRATA) * 3
        first_interval = e.collections[0].get_segments()[0]
        assert first_interval[:, 0].tolist() == [.01, .02]
        assert "COLLOCATION-scale" in plotter.CAPTION and "not the site-month" in plotter.CAPTION
        assert "not the site-month bias decomposition" in plotter.CAPTION
    finally:
        plotter.plt.close(fig)


def test_outside_axes_retained_in_statistics_and_export(plotter, payload):
    payload["tables"][plotter.POINTS]["records"][0]["Eselected_daod"] = 1.1
    data, _ = plotter.select(Registry(payload))
    points = data.loc[data.plot_role.eq("scatter")]
    assert len(points) == 20 and points.Eselected_outside_axes.sum() == 1
    assert '"Eselected": 1' in plotter.notes(data)
    assert data.loc[data.plot_role.eq("annotation"), "rmse"].eq(.2345).all()


def test_cli_registry_only_immutable_and_provenance(plotter, payload, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(clean(payload)))
    before = digest(registry)
    output = tmp_path / "private_candidate"
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("External CSV read"))
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **kw: pytest.fail("New resampling"))

    def export(fig, out, stem, data, path, evidence, script):
        assert stem == plotter.STEM and path == registry and script == plotter.__file__
        plotter.check_layout(fig)
        Path(out).mkdir()
        data.to_csv(Path(out) / "Fig05_data.csv", index=False)
        (Path(out) / "Fig05_provenance.json").write_text(json.dumps(evidence))

    monkeypatch.setattr(plotter.style, "export", export)
    plotter.main(["--registry", str(registry), "--output-dir", str(output)])
    assert digest(registry) == before
    assert (output / "Fig05_data.csv").exists()
    assert "not the site-month" in (output / "Fig05_caption.md").read_text()
    assert "No new aggregation" in (output / "Fig05_notes.md").read_text()
    assert "source_only" in (output / "Fig05_data.csv").read_text()


def test_author_output_guard(plotter, tmp_path):
    source = tmp_path / "source"
    (source / "01_instructions").mkdir(parents=True)
    with pytest.raises(ValueError, match="author"):
        plotter.make(tmp_path / "missing.json", source / "figures")
