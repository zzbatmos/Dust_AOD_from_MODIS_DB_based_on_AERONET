"""Fixed reference arithmetic, distinct archive supports, no source-main execution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper import derive_coarse_fraction_display as coarse
from paper.numbers import Registry, digest, ingest

PREPARE = """
def backfraction(delta):
    return np.clip((delta-.02)*1.30/(.28*(1+delta)),0,1)
def prepare():
    df['old675LR56']=np.clip(backfraction(df.AER_PLDR675)*56/df.AER_LR675,0,1)
    df['experimental1020_native_raw']=backfraction(df.AER_PLDR1020)*lr/df.AER_LR1020
    df['experimental1020_native']=df.experimental1020_native_raw.clip(0,1)
raise RuntimeError('Do not execute or calibrate')
"""
COMPARISON = """
def main():
    validcmf=(total>0)&(coarse>=0)&cmf.between(0,1.05)
    reference_valid=validcmf & data[f'Lidar_Ratio[{wave}nm]'].between(10,150) & data[f'Depolarization_Ratio[{wave}nm]'].between(0,1)
    cmfrows.append({'subset':subset,'wavelength':wave,'n':len(values),'n_gt_0p85':int((values>.85).sum()),'fraction_gt_0p85':float((values>.85).mean()),'p95':values.quantile(.95),'p99':values.quantile(.99),'maximum_before_clip':values.max()})
raise RuntimeError('Do not execute comparison or calibrate')
"""


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "revision_v2_work"
    group = root / "grouped_validation"
    archive = tmp_path / "revision_research/shin_comparison"
    group.mkdir(parents=True)
    archive.mkdir(parents=True)
    (group / "prepare.py").write_text(PREPARE)
    (archive.parent / "compare_shin.py").write_text(COMPARISON)
    n = 13
    frame = pd.DataFrame(dict(Site=["Not_a_historical_six_site"] * n, date=["2025-01-01"] * n,
                              **{"Time(hh:mm:ss)": ["12:00:00"] * n}))
    for wave in coarse.WAVES:
        frame[f"AOD_Extinction-Total[{wave}nm]"] = 1.
        frame[f"AOD_Extinction-Coarse[{wave}nm]"] = np.linspace(.2, .9, n)
        frame[f"Lidar_Ratio[{wave}nm]"] = 50.
        frame[f"Depolarization_Ratio[{wave}nm]"] = .1
        frame.loc[1, f"Depolarization_Ratio[{wave}nm]"] = .6
        frame.loc[1, f"Lidar_Ratio[{wave}nm]"] = 100.
        frame.loc[2, f"Depolarization_Ratio[{wave}nm]"] = -.1
        frame.loc[9, f"Lidar_Ratio[{wave}nm]"] = 10.
        frame.loc[9, f"Depolarization_Ratio[{wave}nm]"] = 0.
        frame.loc[9, f"AOD_Extinction-Coarse[{wave}nm]"] = 0.
        frame.loc[10, f"Lidar_Ratio[{wave}nm]"] = 150.
        frame.loc[10, f"Depolarization_Ratio[{wave}nm]"] = 1.
        frame.loc[11, f"Lidar_Ratio[{wave}nm]"] = np.nan
        frame.loc[12, f"AOD_Extinction-Coarse[{wave}nm]"] = np.nan
    frame.loc[0, "AOD_Extinction-Coarse[675nm]"] = .85
    frame.loc[0, "AOD_Extinction-Coarse[1020nm]"] = .8501
    frame.loc[3, "Lidar_Ratio[675nm]"] = 9.
    frame.loc[4, "Depolarization_Ratio[1020nm]"] = 1.01
    frame.loc[5, "AOD_Extinction-Total[675nm]"] = 0.
    frame.loc[6, "AOD_Extinction-Coarse[675nm]"] = 1.02
    frame.loc[6, "AOD_Extinction-Coarse[1020nm]"] = 1.05
    frame.loc[7, "AOD_Extinction-Coarse[675nm]"] = 1.05
    frame.loc[7, "AOD_Extinction-Coarse[1020nm]"] = 1.051
    frame.loc[8, "AOD_Extinction-Coarse[675nm]"] = -.01
    path = archive / "aeronet_narrow_cache.csv.gz"
    frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    (group / "source_provenance.json").write_text(json.dumps({str(path): dict(sha256=digest(path), bytes=path.stat().st_size)}))
    calibration = dict(median_LR1020_sr=47.308, endmembers=[.02, .30], raw_cache_rows=n)
    (group / "calibration.json").write_text(json.dumps(calibration))
    rows = pd.DataFrame(dict(row_id=[f"row{i}" for i in range(6)]))
    for wave in coarse.WAVES:
        delta = np.array([0, .02, .1, .6, 1, np.nan])
        lr = np.array([50., 10., 50., 100., 10., 50.])
        rows[f"AER_PLDR{wave}"] = delta
        rows[f"AER_LR{wave}"] = lr
        back = np.clip((delta - .02) * 1.30 / (.28 * (1 + delta)), 0, 1)
        raw = back * (56 if wave == 675 else 47.308) / lr
        if wave == 675:
            rows["old675LR56"] = np.clip(raw, 0, 1)
        else:
            rows["experimental1020_native_raw"] = raw
            rows["experimental1020_native"] = np.clip(raw, 0, 1)
    rows.to_parquet(group / "rows.parquet", index=False)
    (group / "schema.json").write_text(json.dumps(dict(rows=len(rows), SDA_wavelength_nm=500)))
    tails = []
    ids = {}
    for wave in coarse.WAVES:
        total, large, lr, pldr = (frame[f"{name}[{wave}nm]"] for name in
            ("AOD_Extinction-Total", "AOD_Extinction-Coarse", "Lidar_Ratio", "Depolarization_Ratio"))
        ratio = large / total
        keep = (total > 0) & (large >= 0) & ratio.between(0, 1.05) & lr.between(10, 150) & pldr.between(0, 1)
        values = ratio[keep]
        ids[wave] = set(values.index)
        tails.append(dict(subset="reference_valid", wavelength=wave, n=len(values), n_gt_0p85=(values > .85).sum(),
            fraction_gt_0p85=(values > .85).mean(), p95=values.quantile(.95), p99=values.quantile(.99), maximum_before_clip=values.max()))
    tails.append(dict(tails[0], subset="six_site_paired", n=999))
    pd.DataFrame(tails).to_csv(archive / "coarse_fraction_upper_tail.csv", index=False)
    return dict(root=root, group=group, archive=archive, path=path, ids=ids)


@pytest.fixture
def tables(cache):
    return coarse.derive(cache["root"], chunksize=2)


def test_distinct_cohorts_strict_tail_zero_and_unclipped_cmf(cache, tables):
    frames, sources, checks, recipe = tables
    points, tail = frames["coarse_fraction_display"], frames["coarse_fraction_tail"]
    assert len(checks) == 20 and all(c["status"] == "PASS" for c in checks)
    assert all(digest(Path(p)) == h for p, h in sources.items())
    assert len(tail) == 2 and tail.subset.eq("reference_valid").all()
    for wave in coarse.WAVES:
        part = points.loc[points.wavelength.eq(wave)].set_index("source_row")
        assert set(part.index) == cache["ids"][wave]
        assert part.loc[9, "native_extinction_fraction"] == 0
        assert part.loc[9, "inversion_cmf"] == 0
        assert part.loc[6, "inversion_cmf"] > 1
        assert part.cohort.eq("reference_valid").all() and part.author_acceptance.eq("approved_2026-10-14").all()
    assert cache["ids"][675] != cache["ids"][1020]
    old = points.loc[points.wavelength.eq(675)]
    assert old.role.eq("sensitivity").all() and old.fixed_lr_sr.eq(56).all()
    primary = points.loc[points.wavelength.eq(1020)]
    assert primary.role.eq("primary_native").all() and primary.fixed_lr_sr.eq(47.308).all()
    assert old.loc[old.source_row.eq(0), "inversion_cmf"].item() == recipe["tail_threshold"]
    assert frames["coarse_fraction_formula_checks"].max_abs_error.eq(0).all()
    assert not any("550" in c or "SDA" in c for c in points)


def test_clip_backscatter_before_lr_and_never_transfer(tables):
    recipe = tables[3]
    raw_back, back, raw, clipped = coarse.native_fraction([.6, 1, -.1], [100, 10, 50], 47.308, recipe)
    assert raw_back[0] > 1 and back[0] == 1
    assert raw[0] == clipped[0] == 47.308 / 100
    assert clipped[0] != np.clip(raw_back[0] * 47.308 / 100, 0, 1)
    assert raw[1] > 1 and clipped[1] == 1
    assert back[2] == raw[2] == clipped[2] == 0


def test_chunk_size_does_not_change_arrays(cache, tables):
    other = coarse.derive(cache["root"], chunksize=1000)
    for name, frame in tables[0].items():
        pd.testing.assert_frame_equal(frame, other[0][name])
    assert tables[1:] == other[1:]


@pytest.mark.parametrize("problem", ["cache_hash", "calibration", "clip_order", "primary_raw", "old_reference",
                                    "tail_count", "duplicate_tail", "missing_tail", "schema", "duplicate_grouped"])
def test_scientific_or_provenance_changes_fail_closed(cache, problem, tmp_path):
    if problem == "cache_hash":
        path = cache["group"] / "source_provenance.json"
        meta = json.loads(path.read_text())
        meta[str(cache["path"])]["sha256"] = "wrong"
        path.write_text(json.dumps(meta))
    elif problem == "calibration":
        path = cache["group"] / "calibration.json"
        meta = json.loads(path.read_text())
        meta["median_LR1020_sr"] += 1
        path.write_text(json.dumps(meta))
    elif problem == "clip_order":
        path = cache["group"] / "prepare.py"
        path.write_text(PREPARE.replace("return np.clip((delta-.02)*1.30/(.28*(1+delta)),0,1)",
                                        "return (delta-.02)*1.30/(.28*(1+delta))"))
    elif problem in ("primary_raw", "old_reference", "duplicate_grouped"):
        path = cache["group"] / "rows.parquet"
        frame = pd.read_parquet(path)
        if problem == "duplicate_grouped":
            frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
        else:
            frame.loc[0, "experimental1020_native_raw" if problem == "primary_raw" else "old675LR56"] += .01
        frame.to_parquet(path, index=False)
    elif problem == "schema":
        path = cache["group"] / "schema.json"
        meta = json.loads(path.read_text())
        meta["rows"] += 1
        path.write_text(json.dumps(meta))
    else:
        path = cache["archive"] / "coarse_fraction_upper_tail.csv"
        frame = pd.read_csv(path)
        if problem == "tail_count":
            frame.loc[0, "n_gt_0p85"] += 1
        elif problem == "duplicate_tail":
            frame = pd.concat([frame, frame.iloc[:1]])
        else:
            frame = frame.iloc[1:]
        frame.to_csv(path, index=False)
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError):
        coarse.main(["--source-root", str(cache["root"]), "--output-dir", str(output)])
    assert not output.exists()


def test_cli_sidecars_and_no_overwrite(cache, tmp_path):
    out = tmp_path / "derived"
    args = ["--source-root", str(cache["root"]), "--output-dir", str(out), "--chunk-size", "2"]
    coarse.main(args)
    for name in coarse.SEMANTICS:
        path = out / (name + ".csv")
        side = json.loads(path.with_suffix(".provenance.json").read_text())
        registered = ingest(path, "derived/" + name)
        assert side["output_sha256"] == digest(path) == registered["source_sha256"]
        assert side["script_sha256"] == digest(Path(coarse.__file__))
        assert side["author_acceptance"] == "approved_2026-10-14" and len(side["checks"]) == 20
        assert registered["records"] and all(row["author_acceptance"] == "approved_2026-10-14"
                                               for row in registered["records"])
    with pytest.raises(FileExistsError):
        coarse.main(args)


@pytest.fixture
def renderer():
    from paper.figures import figS01_coarse_fraction
    return figS01_coarse_fraction


@pytest.fixture
def registry(tables):
    return Registry(dict(tables={"derived/" + name: dict(records=frame.to_dict("records"), columns=list(frame),
        source_file="/synthetic/" + name, source_sha256="synthetic", derivation=dict(
            semantics=coarse.SEMANTICS[name], fixed_recipe=tables[3], author_acceptance="approved_2026-10-14"))
        for name, frame in tables[0].items()}))


def test_renderer_layout_and_approved_qualified_caption(renderer, registry, monkeypatch, tmp_path):
    frames, evidence = renderer.select(registry)
    assert len(evidence) == 3
    monkeypatch.setattr(renderer.Registry, "read", lambda _: registry)
    exports = []

    def export(fig, output, stem, data, registry_path, sources, script, raster):
        renderer.check_layout(fig)
        assert fig.get_figwidth() / renderer.style.MM == pytest.approx(190)
        assert fig.get_figheight() / renderer.style.MM <= 230
        assert raster and set(data.table_kind) == set(frames)
        from matplotlib.text import Text
        text = " ".join(artist.get_text() for artist in fig.findobj(Text)).lower()
        assert "pending" not in text and "candidate" not in text
        assert "not held-out validation" in text
        exports.append(stem)
        Path(output).mkdir(exist_ok=True)

    monkeypatch.setattr(renderer.style, "export", export)
    path = tmp_path / "registry.json"
    path.write_text("mocked registry")
    renderer.make(path, tmp_path / "figures")
    assert exports == ["FigS01_coarse_fraction"]
    caption = (tmp_path / "figures/FigS01_caption.md").read_text()
    assert "pending" not in caption.lower() and "candidate" not in caption.lower()
    for term in ("NOT SDA", "No 550-nm transfer", "separate", "old675LR56 dust-fraction sensitivity",
                 "experimental1020 native", "not the historical six-site paired cohort", "does not make the archive diagnostic independent"):
        assert term in caption


def test_archive_sized_count_annotations_fit(renderer, registry):
    frames, _ = renderer.select(registry)
    frames["tail"]["n"] = 999999
    frames["tail"]["n_gt_0p85"] = 888888
    figure = renderer.build_figure(frames)
    try:
        renderer.check_layout(figure)
    finally:
        renderer.plt.close(figure)


@pytest.mark.parametrize("problem", ["pooled", "clipped_cmf", "accepted", "pending", "pending_sidecar", "wrong_role", "duplicate", "clip_order", "tails"])
def test_registry_integrity_refused(renderer, registry, problem):
    points = registry.payload["tables"]["derived/coarse_fraction_display"]["records"]
    if problem == "pooled":
        points[0]["cohort"] = "six_site_paired"
    elif problem == "clipped_cmf":
        for row in points:
            row["inversion_cmf"] = min(row["inversion_cmf"], 1)
    elif problem in ("accepted", "pending"):
        points[0]["author_acceptance"] = problem
    elif problem == "pending_sidecar":
        registry.payload["tables"]["derived/coarse_fraction_tail"]["derivation"]["author_acceptance"] = "pending"
    elif problem == "wrong_role":
        points[0]["role"] = "primary_native"
    elif problem == "duplicate":
        points.append(points[0])
    elif problem == "clip_order":
        points[0]["native_extinction_raw"] += .1
    else:
        registry.payload["tables"]["derived/coarse_fraction_tail"]["records"][0]["n"] += 1
    with pytest.raises(ValueError):
        renderer.select(registry)
