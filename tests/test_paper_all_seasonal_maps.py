"""S13 cache derivation and registry-only maps; synthetic supports and no fitting."""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from paper import derive_all_seasonal_maps as allmaps
from paper.numbers import Registry, digest, ingest
from test_paper_seasonal_maps import cache as fig6_cache, MONTHS, PERIODS, oracle


@pytest.fixture
def cache(fig6_cache):
    c = fig6_cache
    with nc.Dataset(c["stage"] / "comparison_support.nc", "a") as ds:
        core = ds.groups[allmaps.base.GROUP]
        full = ds.groups[allmaps.base.REFERENCE + "__hierarchy__full_common_E6"]
        masks = {name: allmaps.base.read(core[name]) for name in ("mask", "modis_mask", "reference_mean_only_mask")}
        for mask in masks.values():
            mask[:, 70, 35] = 0
        values = dict(masks, counts=allmaps.base.read(core["counts"]))
        values.update({name: allmaps.base.read(core[name]) * .8 for name in allmaps.CORE})
        values["E6"] = c["ev"] * 1.2 + .07
        values["E6"][:, 30, 35] = 0.
        for name, value in values.items():
            dtype = "u1" if name.endswith("mask") else "f8"
            full.createVariable(name, dtype, ("time", "lat", "lon"), zlib=True)[:] = value
    rows = []
    with nc.Dataset(c["cube"]) as cube:
        years = allmaps.base.read(cube["year"])
        song = np.stack([np.asarray(cube["old"][int(np.flatnonzero(years == int(m[:4]))[0]), int(m[5:]) - 1], float)
                         for m in MONTHS])
    with nc.Dataset(c["stage"] / "comparison_support.nc") as ds:
        for support, methods in allmaps.SUPPORT_METHODS.items():
            group = ds.groups[f"{allmaps.base.REFERENCE}__hierarchy__{support}"]
            mask = allmaps.base.read(group["mask"]).astype(bool)
            fields = {e: allmaps.base.read(group[e]) for e in methods}
            for period, dates in PERIODS.items():
                indices = [MONTHS.index(date) for date in dates]
                use = mask[indices].all(axis=0) & (abs(c["lat"][:, None]) < 60)
                means = {e: v[indices].mean(axis=0) for e, v in fields.items()}
                means.update(Song=song[indices].mean(axis=0), CALIOP=c["cal"][indices].mean(axis=0))
                for estimator, value in means.items():
                    rows.append(dict(reference=allmaps.base.REFERENCE, variant="hierarchy", support=support,
                                     reference_mask="prior_joint", region=allmaps.base.REGION,
                                     estimator=estimator, period=period,
                                     **oracle(value, means["CALIOP"], c["area"], use)))
    pd.DataFrame(rows).to_csv(c["stage"] / "seasonal_metrics.csv", index=False)
    c["song"] = song
    return c


@pytest.fixture
def tables(cache):
    return allmaps.derive(cache["root"], cache["land"])


def test_all_methods_seasons_and_original_float64_checks(cache, tables):
    frames, sources, checks = tables
    maps, metrics = frames["all_seasonal_maps"], frames["all_seasonal_map_metrics"]
    assert len(maps) == 8 * 4 * 90 * 72
    assert len(metrics) == 60 and metrics.display_map.sum() == 32
    assert len(checks) == 420 and all(c["status"] == "PASS" for c in checks)
    assert all(digest(Path(p)) == h for p, h in sources.items())
    assert str(Path(allmaps.base.__file__).resolve()) in sources
    assert set(maps.loc[maps.support.eq("full_common_E6"), "estimator"]) == {"E6"}
    assert set(maps.loc[maps.support.eq("core_common"), "estimator"]) == set(allmaps.DISPLAY["core_common"])
    assert not np.array_equal(cache["area"], cache["area"].astype("f4").astype(float))
    for _, frame in maps.groupby(["estimator", "period"]):
        np.testing.assert_array_equal(frame.land_area_weight.to_numpy().reshape(90, 72), cache["area"])
        assert frame.loc[~frame.supported, ["value", "caliop", "difference"]].isna().all().all()


def test_e6_own_support_reference_song_no_transfer_and_djf(cache, tables):
    maps = tables[0]["all_seasonal_maps"]
    e6 = maps.loc[maps.estimator.eq("E6") & maps.period.eq("MAM")].set_index(["lat_index", "lon_index"])
    core = maps.loc[maps.estimator.eq("CALIOP") & maps.period.eq("MAM")].set_index(["lat_index", "lon_index"])
    assert core.loc[(70, 35), "supported"] and not e6.loc[(70, 35), "supported"]
    assert np.isnan(e6.loc[(70, 35), "value"])
    assert e6.loc[(30, 35), "value"] == e6.loc[(30, 35), "caliop"] == e6.loc[(30, 35), "difference"] == 0
    metric = tables[0]["all_seasonal_map_metrics"]
    q = metric.loc[metric.period.eq("MAM") & metric.estimator.eq("CALIOP")].set_index("support")
    assert q.loc["full_common_E6", "candidate_mean"] != q.loc["core_common", "candidate_mean"]
    song = maps.loc[maps.estimator.eq("Song") & maps.period.eq("DJF")].set_index(["lat_index", "lon_index"])
    assert song.loc[(15, 35), "value"] == cache["song"][:3, 15, 35].mean()
    assert song.loc[(15, 35), "caliop"] == cache["cal"][:3, 15, 35].mean()
    assert song.months.eq("2016-12|2017-01|2017-02").all()
    np.testing.assert_array_equal(e6.loc[e6.supported, "difference"],
                                  e6.loc[e6.supported, "value"] - e6.loc[e6.supported, "caliop"])


@pytest.mark.parametrize("problem", ["missing_e6", "missing_group", "bitmap", "nan_e6", "negative_e6", "zero_count"])
def test_e6_source_failures_no_core_substitution(cache, problem):
    with nc.Dataset(cache["stage"] / "comparison_support.nc", "a") as ds:
        key = allmaps.base.REFERENCE + "__hierarchy__full_common_E6"
        group = ds.groups[key]
        if problem == "missing_e6":
            # Rebuild the synthetic group: this HDF build cannot rename variables
            # that use inherited dimensions after reopening the file.
            ds.renameGroup(key, "old_full_common_E6")
            replacement = ds.createGroup(key)
            for name, variable in group.variables.items():
                if name != "E6":
                    replacement.createVariable(name, variable.dtype, variable.dimensions)[:] = variable[:]
        elif problem == "missing_group":
            ds.renameGroup(key, "not_full_common_E6")
        elif problem == "bitmap":
            group["mask"][0, 15, 35] = 2
        elif problem == "nan_e6":
            group["E6"][0, 15, 35] = np.nan
        elif problem == "negative_e6":
            group["E6"][0, 15, 35] = -1
        else:
            group["counts"][0, 15, 35] = 0
    with pytest.raises(ValueError):
        allmaps.derive(cache["root"], cache["land"])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "count", "precision", "wrong_support"])
def test_all_saved_metrics_are_required(cache, problem, tmp_path):
    path = cache["stage"] / "seasonal_metrics.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    index = rows.index[rows.support.eq("full_common_E6") & rows.estimator.eq("Song")][0]
    if problem == "missing":
        rows = rows.drop(index=index)
    elif problem == "duplicate":
        rows = pd.concat([rows, rows.loc[[index]]])
    elif problem == "count":
        rows.loc[index, "n"] += 1
    elif problem == "precision":
        rows.loc[index, "rmse"] += 1e-9
    else:
        rows.loc[index, "support"] = "core_common"
    rows.to_csv(path, index=False)
    out = tmp_path / "not_published"
    with pytest.raises(ValueError):
        allmaps.main(["--source-root", str(cache["root"]), "--land-shapefile", str(cache["land"]),
                      "--output-dir", str(out)])
    assert not out.exists()


def test_cli_sidecars_no_overwrite_and_fig6_untouched(cache, tmp_path):
    out = tmp_path / "derived"
    out.mkdir()
    original = out / "seasonal_maps.csv"
    original.write_text("completed Fig6 data\n")
    arguments = ["--source-root", str(cache["root"]), "--land-shapefile", str(cache["land"])]
    allmaps.main(arguments + ["--output-dir", str(out)])
    assert original.read_text() == "completed Fig6 data\n"
    for name, semantics in allmaps.SEMANTICS.items():
        path = out / (name + ".csv")
        metadata = json.loads(path.with_suffix(".provenance.json").read_text())
        registered = ingest(path, "derived/" + name)
        assert metadata["output_sha256"] == digest(path) == registered["source_sha256"]
        assert metadata["rows"] == registered["source_rows"]
        assert len(metadata["checks"]) == 420 and metadata["semantics"] == semantics
    with pytest.raises(FileExistsError):
        allmaps.main(arguments + ["--output-dir", str(out)])
    for forbidden in (cache["root"] / "derived", Path(allmaps.__file__).parent / "derived"):
        with pytest.raises(ValueError, match="outside"):
            allmaps.main(arguments + ["--output-dir", str(forbidden)])


@pytest.fixture
def renderer():
    pytest.importorskip("cmcrameri")
    pytest.importorskip("colorspacious")
    pytest.importorskip("cartopy")
    from paper.figures import figS13_all_seasonal_maps
    return figS13_all_seasonal_maps


@pytest.fixture
def registry(tables):
    payload = {"tables": {}}
    for name, frame in tables[0].items():
        payload["tables"]["derived/" + name] = dict(
            records=frame.to_dict("records"), columns=list(frame), source_file="/synthetic/" + name + ".csv",
            source_sha256="synthetic", derivation=dict(semantics=allmaps.SEMANTICS[name]))
    return Registry(payload)


def test_renderer_registry_supports_counts_and_geometry(renderer, registry):
    data, grids, evidence = renderer.select(registry)
    assert len(grids) == 32 and len(data) == 207360 and len(evidence) == 2
    for (estimator, _), grid in grids.items():
        assert grid["value"].shape == (90, 72)
        assert grid["x"].shape == (73,) and grid["y"].shape == (91,)
    assert renderer.support_for("E6") == "full_common_E6"
    assert renderer.support_for("Song") == "core_common"


@pytest.mark.parametrize("problem", ["gap", "duplicate", "coordinate", "index", "missing_zero",
                                    "wrong_difference", "wrong_month_count", "bad_flag"])
def test_native_grid_integrity_fails_closed(renderer, tables, problem):
    maps = tables[0]["all_seasonal_maps"]
    frame = maps.loc[maps.estimator.eq("E6") & maps.period.eq("MAM")].copy().reset_index(drop=True)
    supported = frame.index[frame.supported][0]
    if problem == "gap":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "coordinate":
        frame.loc[0, "lon"] = np.nan
    elif problem == "index":
        frame.loc[0, "lat_index"] = 9
    elif problem == "missing_zero":
        frame.loc[0, "value"] = 0
    elif problem == "wrong_difference":
        frame.loc[supported, "difference"] += .1
    elif problem == "wrong_month_count":
        frame.loc[supported, "prior_joint_months"] = 2
    else:
        frame["supported"] = frame.supported.astype(object)
        frame.loc[0, "supported"] = "False"
    with pytest.raises(ValueError):
        renderer.reconstruct(frame)


def test_two_pages_dimensions_fonts_clipping_and_captions(renderer, registry, monkeypatch, tmp_path):
    data, grids, evidence = renderer.select(registry)
    monkeypatch.setattr(renderer.Registry, "read", lambda _: registry)
    monkeypatch.setattr(renderer, "local_geography", lambda _: ({"coastline": (), "ocean": ()}, {}))
    exported = []

    def export(fig, output, stem, frame, registry_path, sources, script, raster):
        renderer.check_layout(fig)
        assert raster and script == renderer.__file__
        assert fig.get_figwidth() / renderer.style.MM == pytest.approx(190)
        assert fig.get_figheight() / renderer.style.MM <= 230
        is_difference = stem.startswith("FigS13b")
        assert len(fig.axes) == (28 if is_difference else 32) + 1
        assert len(frame) == (7 if is_difference else 8) * 4 * 90 * 72
        if is_difference:
            assert not frame.estimator.eq("CALIOP").any()
        exported.append(stem)
        Path(output).mkdir(exist_ok=True)

    monkeypatch.setattr(renderer.style, "export", export)
    path = tmp_path / "registry.json"
    path.write_text("synthetic registry read is mocked\n")
    before = digest(path)
    renderer.make(path, tmp_path / "figures")
    assert exported == ["FigS13a_all_seasonal_maps", "FigS13b_all_seasonal_differences"]
    assert digest(path) == before
    for stem, difference in (("FigS13a", False), ("FigS13b", True)):
        text = (tmp_path / "figures" / (stem + "_caption.md")).read_text()
        assert text == renderer.caption(difference)
        assert "separate full_common_E6" in text and "December 2016" in text
        assert "not paired comparisons" in text
        if difference:
            assert "diagonal hatching" in text


def test_missing_local_geography_never_downloads(renderer, tmp_path):
    with pytest.raises(FileNotFoundError):
        renderer.local_geography(tmp_path)
