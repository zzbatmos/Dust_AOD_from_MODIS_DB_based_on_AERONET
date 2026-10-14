"""Synthetic S15 caches; original source modules must never execute."""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from paper import derive_low_loading_display as loading
from paper.numbers import Registry, digest, ingest
from test_paper_seasonal_maps import cache as seasonal_cache, MONTHS


def oracle(values, area, mask):
    x, w = values[mask], np.broadcast_to(area, values.shape)[mask]
    if not len(x):
        return dict(cell_months=0)
    order = np.argsort(x)
    return dict(cell_months=len(x), mean=np.mean(x), median=np.median(x),
                q05=np.quantile(x, .05), q95=np.quantile(x, .95),
                area_weighted_mean=np.average(x, weights=w),
                area_weighted_median=np.interp(.5, (np.cumsum(w[order]) - w[order] / 2) / w.sum(), x[order]),
                fraction_above_0p01=np.mean(x > .01))


@pytest.fixture
def cache(seasonal_cache):
    c = seasonal_cache
    provenance_path = c["stage"] / "comparison_provenance.json"
    p = json.loads(provenance_path.read_text())
    script = c["root"] / "expected_value_test/compare_global.py"
    script.write_text(script.read_text() + "\ndef distribution(values, weights, mask):\n"
                      " return dict(q05=np.quantile(x,.05), q95=np.quantile(x,.95), flag=(x>.01).mean())\n"
                      "def original_selection():\n lowdust = broad & (cal<.005) & (song<.005) & in2017\n")
    p["script"] = digest(script)
    p["fine_land_coverage"] = .5
    p["regions"]["Amazonia_Cerrado"] = [-40, 20, -5, 0]
    with nc.Dataset(c["stage"] / "comparison_support.nc", "a") as ds:
        group = ds.groups[loading.base.GROUP]
        modis = loading.base.read(group["modis_mask"]).astype(bool)
        group.createVariable("coverage", "f8", ("time", "lat", "lon"))[:] = modis.astype(float)
        fields = {e: loading.base.read(group[e]) for e in ("E1", "Eselected", "E5", "E7", "E8")}
        counts = loading.base.read(group["counts"])
    low = {s: {e: v * (.2 if s == "AOD550_lt_0p2" else .4) for e, v in fields.items()}
           for s in loading.SUBSETS}
    paths = []
    for index, month in enumerate(MONTHS):
        path = c["root"] / f"Aqua_EV_{month}_2x5.nc"
        with nc.Dataset(path, "w") as ds:
            ds.month, ds.fingerprint = month, "synthetic-frozen-model"
            for name in ("lat", "lon"):
                ds.createDimension(name, len(c[name]))
                ds.createVariable(name, "f8", (name,))[:] = c[name]
            group = ds.createGroup("experimental1020").createGroup("hierarchy")
            for e in fields:
                prefix = "core_common__" + e + "__"
                for name, values in dict(mean=fields[e], count=counts,
                                        **{column: low[label][e] for label, column in loading.SUBSETS.items()}).items():
                    var = group.createVariable(prefix + name, "f8", ("lat", "lon"))
                    var[:] = values[index]
                    if name.endswith("contribution"):
                        var.denominator = prefix + "count"
        p["sources"][str(path)] = digest(path)
        path.with_suffix(".sha256.json").write_text(json.dumps(dict(sha256=digest(path), fingerprint="synthetic-frozen-model")))
        paths.append(path)
    provenance_path.write_text(json.dumps(p))
    with nc.Dataset(c["cube"]) as ds:
        song = np.stack([np.asarray(ds["old"][int(m[:4]) - 2016, int(m[5:]) - 1], float) for m in MONTHS])
    year = np.array([m.startswith("2017") for m in MONTHS])[:, None, None]
    domain = loading.region_mask(c["lat"], c["lon"], p["regions"][loading.base.REGION])
    external_low = (c["cal"] < .005) & (song < .005) & year & domain
    amazon = modis & loading.region_mask(c["lat"], c["lon"], p["regions"]["Amazonia_Cerrado"])[None]
    amazon[3:] = False
    regimes = dict(Amazon_DJF=amazon, dual_reference_low_dust=external_low & modis)
    rows, floors = [], []
    for e in fields:
        for region, bounds in p["regions"].items():
            use = modis & year & loading.region_mask(c["lat"], c["lon"], bounds)[None]
            for subset in low:
                rows.append(dict(loading.META, estimator=e, region=region, subset=subset,
                                 **oracle(low[subset][e], c["area"], use),
                                 full_mean_DAOD=oracle(fields[e], c["area"], use).get("area_weighted_mean", np.nan)))
        for regime, use in regimes.items():
            stats = oracle(fields[e], c["area"], use)
            floors.append(dict(loading.META, estimator=e, regime=regime, **stats,
                               median_floor_flag=stats.get("median", np.nan) > .01))
    pd.DataFrame(rows).to_csv(c["stage"] / "low_loading_contributions.csv", index=False)
    pd.DataFrame(floors).to_csv(c["stage"] / "background_floor.csv", index=False)
    pd.DataFrame([dict(loading.META, external_low_cells_before_land_modis=external_low.sum(),
        external_low_after_land_purity=(external_low & (c["area"] > 0)).sum(),
        paired_low_dust_cell_months=regimes["dual_reference_low_dust"].sum(),
        paired_low_dust_spatial_cells=regimes["dual_reference_low_dust"].any(axis=0).sum(),
        Amazon_DJF_cell_months=amazon.sum(), Amazon_DJF_spatial_cells=amazon.any(axis=0).sum())]).to_csv(
            c["stage"] / "floor_selection_audit.csv", index=False)
    return dict(c, paths=paths, fields=fields, low=low, modis=modis, regimes=regimes)


@pytest.fixture
def tables(cache):
    return loading.derive(cache["root"], cache["land"])


def refresh_monthly(cache, index=0):
    path = cache["paths"][index]
    p = cache["stage"] / "comparison_provenance.json"
    meta = json.loads(p.read_text())
    meta["sources"][str(path)] = digest(path)
    p.write_text(json.dumps(meta))
    receipt = path.with_suffix(".sha256.json")
    meta = json.loads(receipt.read_text())
    meta["sha256"] = digest(path)
    receipt.write_text(json.dumps(meta))


def test_available_months_not_fixed_mask_or_cellmonth_global_mean(cache, tables):
    frames, sources, checks = tables
    maps = frames["low_loading_maps"]
    assert len(maps) == 2 * 90 * 72
    assert len(frames["low_loading_summary"]) == 40
    assert len(checks) == 484 and all(x["status"] == "PASS" for x in checks)
    assert all(digest(Path(p)) == h for p, h in sources.items())
    a = maps.loc[maps.subset.eq("AOD550_lt_0p2")].set_index(["lat_index", "lon_index"])
    point = a.loc[(70, 35)]
    assert point.month_count == 11 and point.supported
    use = cache["modis"][1:, 70, 35]
    assert point.value == pytest.approx(cache["low"]["AOD550_lt_0p2"]["Eselected"][1:, 70, 35][use].mean())
    assert a.loc[(30, 35), "value"] == 0 and a.loc[(30, 35), "month_count"] == 12
    assert np.isnan(a.loc[(0, 0), "value"]) and a.loc[(0, 0), "month_count"] == 0
    summary = frames["low_loading_summary"]
    row = summary.loc[summary.estimator.eq("Eselected") & summary.subset.eq("AOD550_lt_0p2") & summary.region.eq(loading.base.REGION)].iloc[0]
    selected = a.loc[a.supported]
    spatial_average = np.average(selected.value, weights=selected.land_area_weight)
    assert not np.isclose(spatial_average, row.area_weighted_mean, atol=1e-10)
    assert maps.denominator.eq("full_paired_count").all()


def test_distinct_floor_supports_december_and_no_qa_zero_filter(cache, tables):
    frames = tables[0]
    samples = frames["low_loading_floor_samples"]
    amazon = samples.loc[samples.regime.eq("Amazon_DJF")]
    low = samples.loc[samples.regime.eq("dual_reference_low_dust")]
    assert set(amazon.month) == {"2016-12", "2017-01", "2017-02"}
    assert low.month.str.startswith("2017").all()
    assert amazon.reference_mask.eq("modis_only").all()
    assert low.reference_mask.eq("reference_mean_only_low_dust").all()
    assert (amazon.loc[amazon.estimator.eq("E1"), "value"] > 0).all()
    assert ((amazon.month == "2017-01") & (amazon.lat_index == 50)).any()  # Ancillary-invalid retained.
    assert len(frames["low_loading_floor_summary"]) == 10
    assert frames["low_loading_floor_audit"].iloc[0].Amazon_DJF_cell_months == 6
    assert loading.frozen_recipe(cache["root"] / "expected_value_test/compare_global.py")["low_dust_threshold"] == .005


@pytest.mark.parametrize("problem", ["missing_month", "hash", "receipt", "denominator", "coordinate", "field", "count", "negative", "exceeds_full", "nan"])
def test_monthly_contract_fails_closed(cache, problem):
    path = cache["paths"][0]
    if problem == "missing_month":
        p = cache["stage"] / "comparison_provenance.json"
        meta = json.loads(p.read_text())
        del meta["sources"][str(path)]
        p.write_text(json.dumps(meta))
    elif problem == "receipt":
        path.with_suffix(".sha256.json").write_text(json.dumps(dict(sha256=digest(path), fingerprint="wrong")))
    else:
        with nc.Dataset(path, "a") as ds:
            g = ds.groups["experimental1020"].groups["hierarchy"]
            prefix = "core_common__Eselected__"
            if problem == "coordinate":
                ds["lat"][0] = 0
            elif problem == "denominator":
                g[prefix + "low_aod_contribution"].denominator = "subset_count"
            elif problem in ("field", "count"):
                g[prefix + ("mean" if problem == "field" else "count")][15, 35] += 1
            else:
                g[prefix + "low_aod_contribution"][15, 35] = {"hash": .8, "negative": -1, "exceeds_full": 5, "nan": np.nan}[problem]
        if problem != "hash":
            refresh_monthly(cache)
    with pytest.raises(ValueError):
        loading.derive(cache["root"], cache["land"])


@pytest.mark.parametrize("name,column", [("low_loading_contributions", "area_weighted_mean"),
                                        ("background_floor", "median"),
                                        ("floor_selection_audit", "paired_low_dust_cell_months")])
def test_saved_summary_mismatch_refused(cache, name, column, tmp_path):
    path = cache["stage"] / (name + ".csv")
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, column] += 1 if column.endswith("cell_months") else .1
    frame.to_csv(path, index=False)
    out = tmp_path / "unpublished"
    with pytest.raises(ValueError):
        loading.main(["--source-root", str(cache["root"]), "--land-shapefile", str(cache["land"]), "--output-dir", str(out)])
    assert not out.exists()


def test_saved_modis_mask_verified_not_substituted(cache):
    with nc.Dataset(cache["stage"] / "comparison_support.nc", "a") as ds:
        ds.groups[loading.base.GROUP]["coverage"][0, 15, 35] = 0
    with pytest.raises(ValueError, match="MODIS-only bitmap"):
        loading.derive(cache["root"], cache["land"])


def test_cli_sidecars_sources_and_no_overwrite(cache, tmp_path):
    out = tmp_path / "derived"
    args = ["--source-root", str(cache["root"]), "--land-shapefile", str(cache["land"]), "--output-dir", str(out)]
    loading.main(args)
    for name in loading.SEMANTICS:
        path = out / (name + ".csv")
        side = json.loads(path.with_suffix(".provenance.json").read_text())
        registered = ingest(path, "derived/" + name)
        assert side["output_sha256"] == digest(path) == registered["source_sha256"]
        assert side["rows"] == registered["source_rows"]
        assert side["script_sha256"] == digest(Path(loading.__file__))
    with pytest.raises(FileExistsError):
        loading.main(args)


def test_empty_distribution_missing_not_zero_and_weighted_convention():
    recipe = dict(low_dust_threshold=.005, descriptive_floor_threshold=.01, q05_probability=.05, q95_probability=.95)
    values, area = np.array([[[0., 2.]]]), np.array([[1., 3.]])
    assert loading.distribution(values, area, np.zeros_like(values, bool), recipe) == {"cell_months": 0}
    result = loading.distribution(values, area, np.ones_like(values, bool), recipe)
    assert result["area_weighted_mean"] == 1.5 and result["mean"] == 1
    assert result["area_weighted_median"] == 1.5


@pytest.fixture
def renderer():
    pytest.importorskip("cartopy")
    from paper.figures import figS15_low_loading
    return figS15_low_loading


@pytest.fixture
def registry(tables):
    return Registry(dict(tables={"derived/" + name: dict(records=frame.to_dict("records"),
        columns=list(frame), source_file="synthetic/" + name, source_sha256="synthetic",
        derivation=dict(semantics=loading.SEMANTICS[name])) for name, frame in tables[0].items()}))


def test_renderer_native_registry_layout_and_supports(renderer, registry, monkeypatch, tmp_path):
    # CSV ingestion parses this single-year column as integer, unlike the
    # in-memory derived frame. Both describe exactly calendar 2017.
    for row in registry.payload["tables"]["derived/low_loading_maps"]["records"]:
        row["period"] = int(row["period"])
    frames, evidence = renderer.select(registry)
    assert len(evidence) == 5 and len(frames["maps"]) == 12960
    monkeypatch.setattr(renderer.Registry, "read", lambda _: registry)
    monkeypatch.setattr(renderer, "local_geography", lambda: ({"coastline": (), "ocean": ()}, {}))
    exports = []

    def export(fig, output, stem, data, registry_path, sources, script, raster):
        renderer.check_layout(fig)
        assert fig.get_figwidth() / renderer.style.MM == pytest.approx(190)
        assert fig.get_figheight() / renderer.style.MM <= 230
        assert raster and set(data.table_kind) == set(frames)
        exports.append(stem)
        Path(output).mkdir(exist_ok=True)

    monkeypatch.setattr(renderer.style, "export", export)
    path = tmp_path / "registry.json"
    path.write_text("mocked registry")
    renderer.make(path, tmp_path / "figures")
    assert exports == ["FigS15_low_loading"]
    caption = (tmp_path / "figures/FigS15_caption.md").read_text()
    for phrase in ("not subset means", "not confidence intervals", "not selected by QA2=0", "December 2016"):
        assert phrase in caption


@pytest.mark.parametrize("problem", ["missing", "duplicate", "zero_missing", "months", "support"])
def test_renderer_invalid_maps_refused(renderer, registry, problem):
    entry = registry.payload["tables"]["derived/low_loading_maps"]
    rows = entry["records"]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "zero_missing":
        rows[0]["value"] = 0
    elif problem == "months":
        rows[0]["month_count"] = 13
    else:
        rows[0]["reference_mask"] = "prior_joint"
    with pytest.raises(ValueError):
        renderer.select(registry)
