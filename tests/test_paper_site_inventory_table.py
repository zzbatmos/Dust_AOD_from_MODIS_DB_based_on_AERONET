"""Synthetic cohort/header/provenance tests; no unpublished quantitative fixtures."""
import copy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from paper import derive_site_inventory_table as derive
from paper import site_inventory_table as display
from paper.numbers import Registry, digest


def synthetic():
    common = dict(reference="experimental1020", lat=10.125, lon=-20.25)
    overview = pd.DataFrame([
        dict(common, site="Alpha", site_key="alpha", role="evaluation", region="region_a",
             collocations=2, first_year=2001, last_year=2003, platforms="Aqua;Terra", support="hierarchy_own"),
        dict(common, site="Beta", site_key="beta", role="evaluation", region="region_b",
             collocations=1, first_year=2004, last_year=2004, platforms="Terra", support="hierarchy_own"),
        dict(common, site="Cal_One", site_key="cal_one", role="calibration", region="calibration only",
             calibration_records=2, support="pre2010_reserved_dust_calibration"),
    ])
    records = [dict(row_id=11, site="Alpha", site_key="alpha", region="region_a", time="2001-01-01", platform="Terra"),
               dict(row_id=12, site="Alpha", site_key="alpha", region="region_a", time="2003-01-01", platform="Aqua"),
               dict(row_id=13, site="Beta", site_key="beta", region="region_b", time="2004-01-01", platform="Terra")]
    events = pd.DataFrame([dict(r, scheme="site_cv", repeat=rep, reference="experimental1020",
                                population="own", model="hierarchy_prespecified") for rep in [0, 1] for r in records])
    calibration = pd.DataFrame(dict(Site=["Cal_One"] * 2, date=["2002-02-02"] * 2))
    inventory = pd.DataFrame(dict(site_key=["alpha", "beta", "cal_one"],
                                  CV_region=["region_a", "region_b", "region_c"],
                                  calibration_reserved=[False, False, True]))
    expected = dict(evaluation_sites=2, calibration_sites=1, unique_collocations=3, calibration_records=2)
    return overview, events, calibration, inventory, expected


def header(path, site, pi="Scientist_One_and_Scientist_Two", encoding="utf-8"):
    text = "\n".join(["AERONET Version 3", site, "Version 3: Almucantar Level 2.0 Inversion",
                      "Quality assured", f"All Points,Contact: PI={pi}; PI Email=not-exported@example.org", "UNITS", "data"])
    path.write_bytes(text.encode(encoding))


def inputs(tmp_path):
    overview, events, cal, inventory, expected = synthetic()
    root = tmp_path / "frozen"
    (root / "oof").mkdir(parents=True)
    events.to_parquet(root / "oof/experimental1020__own__hierarchy_prespecified.parquet", index=False)
    cal.to_csv(root / "calibration_selected.csv.gz", index=False)
    inventory.to_csv(root / "site_inventory.csv", index=False)
    (root / "calibration.json").write_text(json.dumps(dict(selected_rows=2, selected_sites=1)))
    op = tmp_path / "overview.csv"
    overview.to_csv(op, index=False)
    side = dict(expected, sources={str(p.resolve()): digest(p) for p in root.rglob("*") if p.is_file()},
                output_sha256=digest(op))
    op.with_suffix(".provenance.json").write_text(json.dumps(side))
    headers = tmp_path / "headers"
    headers.mkdir()
    for site in overview.site:
        header(headers / f"19900101_20250101_{site}.all", site, "Ren\u00e9_D\u2019Elia", "cp1252")
    return op, root, headers


def derived(tmp_path):
    op, root, headers = inputs(tmp_path)
    output = tmp_path / "derived"
    coverage = derive.derive(op, root, headers, output)
    registry = display.temporary_registry(output, tmp_path / "private_preview" / "numbers.json")
    return output, registry, coverage


def test_fixed_cohort_counts_and_year_gaps():
    result = derive.cohort_summary(*synthetic()).set_index("site_key")
    assert result.loc["alpha", "years"] == "2001;2003"
    assert json.loads(result.loc["alpha", "year_counts_json"]) == {"2001": 1, "2003": 1}
    assert result.loc["alpha", "collocations"] == 2  # Not four OOF predictions.
    assert result.loc["cal_one", "calibration_records"] == 2  # Not one unique date.
    assert pd.isna(result.loc["cal_one", "collocations"])
    assert pd.isna(result.loc["alpha", "calibration_records"])
    assert result.loc["cal_one", "region"] == "region_c"
    assert result.loc["cal_one", "overview_region"] == "calibration only"


@pytest.mark.parametrize("case", ["repeat", "duplicate", "identity", "population", "name", "counts", "year", "cohort", "coordinates"])
def test_cohort_corruption_fails(case):
    overview, events, cal, inventory, expected = synthetic()
    if case == "repeat":
        events = events.iloc[:-1]
    elif case == "duplicate":
        events = pd.concat([events, events.iloc[:1]])
    elif case == "identity":
        events.loc[3, "time"] = "2002-01-01"
    elif case == "population":
        events["population"] = "common"
    elif case == "name":
        cal["Site"] = "CAL_ONE"
    elif case == "counts":
        overview.loc[0, "collocations"] = 4
    elif case == "year":
        cal["date"] = "2010-01-01"
    elif case == "cohort":
        overview = overview.iloc[:-1]
    else:
        overview.loc[0, "lat"] = float("nan")
    with pytest.raises(ValueError):
        derive.cohort_summary(overview, events, cal, inventory, expected)


def test_missing_saved_region_is_not_geographically_inferred():
    args = list(synthetic())
    args[3] = args[3].iloc[:2]
    result = derive.cohort_summary(*args).iloc[-1]
    assert result.region == "NOT RECORDED"
    assert result.region_status == "missing_saved_region"


def test_local_pi_encoding_exact_matching_missing_and_conflict(tmp_path):
    path = tmp_path / "19900101_20250101_Alpha.all"
    header(path, "Alpha", "Ren\u00e9_D\u2019Elia", "cp1252")
    meta = derive.read_pi_header(path, "Alpha")
    assert meta["encoding"] == "windows-1252"
    assert meta["principal_investigators"] == "Ren\u00e9_D\u2019Elia"
    assert "Email" not in json.dumps(meta) and "example.org" not in json.dumps(meta)
    evidence = {}
    result = derive.pi_metadata(["Alpha", "alpha", "Absent"], tmp_path, evidence)
    assert result["Alpha"]["pi_status"] == "local_header"
    assert result["alpha"]["pi_status"] == result["Absent"]["pi_status"] == "missing_local_PI"
    assert evidence[str(path.resolve())] == digest(path)
    header(tmp_path / "19900101_20260101_Alpha.all", "Alpha", "Different_Name")
    result = derive.pi_metadata(["Alpha"], tmp_path, {})
    assert result["Alpha"]["pi_status"] == "conflicting_local_headers"
    assert result["Alpha"]["principal_investigators"] == "NOT RECORDED"
    with pytest.raises(ValueError, match="mismatch"):
        derive.read_pi_header(path, "Other")


def test_empty_pi_is_explicit_missing(tmp_path):
    header(tmp_path / "19900101_20250101_Alpha.all", "Alpha", "N/A")
    result = derive.pi_metadata(["Alpha"], tmp_path, {})["Alpha"]
    assert result["pi_status"] == "missing_local_PI"
    assert result["principal_investigators"] == "NOT RECORDED"


def test_derivation_provenance_and_registry_only_full_csv(tmp_path, monkeypatch):
    output, registry, coverage = derived(tmp_path)
    assert coverage["pi_available"] == 3 and coverage["pi_missing_sites"] == []
    csv = output / (derive.NAME + ".csv")
    side = json.loads(csv.with_suffix(".provenance.json").read_text())
    assert side["output_sha256"] == digest(csv)
    assert all(digest(Path(p)) == sha for p, sha in side["sources"].items())
    assert side["required_dimension_additions"] == [] and not side["pi_search"]["network_used"]
    before = digest(registry)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("Registry exporter read external CSV"))
    out = tmp_path / "rendered"
    result = display.export(registry, out)
    assert result["rows"] == 3 and digest(registry) == before
    assert (out / "TableS01.csv").read_bytes() == csv.read_bytes()
    prov = json.loads((out / "TableS01_provenance.json").read_text())
    assert prov["source"]["derivation"] == side
    assert all(digest(out / name) == sha for name, sha in prov["outputs"].items())
    tex = (out / "TableS01.tex").read_text()
    for name in ["Alpha", "Beta", "Cal_One"]:
        assert display.escape(name) in tex
    assert "2001,\\allowbreak{} 2003" in tex
    assert "historical" in tex and "full CSV" in tex and "not MODIS collocations" in tex
    assert prov["font_pt"] >= 7


@pytest.mark.parametrize("case", ["sha", "years", "pi_name", "pi_evidence", "duplicate"])
def test_registry_corruption_fails(tmp_path, case):
    _, path, _ = derived(tmp_path)
    payload = copy.deepcopy(Registry.read(path).payload)
    table = payload["tables"][derive.TABLE]
    row = table["records"][0]
    if case == "sha":
        table["source_sha256"] = "bad"
    elif case == "years":
        row["years"] = "2001;2002;2003"
    elif case == "pi_name":
        row["principal_investigators"] = "Guessed"
    elif case == "pi_evidence":
        values = json.loads(row["pi_evidence_json"])
        values[0]["source_sha256"] = "bad"
        row["pi_evidence_json"] = json.dumps(values)
    else:
        table["records"][1]["site_key"] = row["site_key"]
    with pytest.raises(ValueError):
        display.load(Registry(payload))


def test_refuse_overwrite_and_source_outputs(tmp_path):
    output, registry, _ = derived(tmp_path)
    with pytest.raises(FileExistsError):
        display.temporary_registry(output, registry)
    op, root, headers = tmp_path / "overview.csv", tmp_path / "frozen", tmp_path / "headers"
    with pytest.raises(FileExistsError):
        derive.derive(op, root, headers, output)
    with pytest.raises(ValueError):
        derive.derive(op, root, headers, root / "output")
    with pytest.raises(ValueError):
        display.export(registry, registry.parent / "tables")
    author = tmp_path / "source"
    (author / "01_instructions").mkdir(parents=True)
    with pytest.raises(ValueError):
        display.export(registry, author / "tables")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable")
def test_latex_compiles_accented_pi_headers(tmp_path):
    _, registry, _ = derived(tmp_path)
    output = tmp_path / "pdf"
    result = display.export(registry, output, compile_latex=True)
    assert result["latex"]["status"] == "compiled"
    assert result["latex"]["pages"] >= 1
    assert (output / "TableS01_preview.pdf").exists()
