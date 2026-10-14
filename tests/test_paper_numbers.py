import json

import pandas as pd
import pytest

from paper.numbers import Registry, check_claims, ingest


def registry(tmp_path):
    path = tmp_path / "metrics.csv"
    pd.DataFrame([
        {"reference": "example", "scheme": "site_cv", "minimum": 3,
         "estimator": "A", "unique_rows": 8, "sites": 2, "rmse": .125,
         "rmse_lo": .10, "rmse_hi": .15},
        {"reference": "example", "scheme": "site_cv", "minimum": 3,
         "estimator": "B", "unique_rows": 8, "sites": 2, "rmse": .2,
         "rmse_lo": float("nan"), "rmse_hi": float("nan")},
    ]).to_csv(path, index=False)
    return Registry({"tables": {"example": ingest(path, "example")}})


def test_exact_selector_interval_counts(tmp_path):
    r = registry(tmp_path)
    n = r.number("example", "rmse", estimator="A")
    assert n["value"] == .125
    assert n["interval"] == [.1, .15]
    assert n["N"] == {"unique_rows": 8, "sites": 2}
    assert n["filter"]["minimum"] == 3
    json.dumps(r.payload, allow_nan=False)


def test_missing_or_ambiguous_fails(tmp_path):
    r = registry(tmp_path)
    with pytest.raises(ValueError, match="found 2"):
        r.number("example", "rmse", scheme="site_cv")
    with pytest.raises(ValueError, match="found 0"):
        r.number("example", "rmse", estimator="absent")
    with pytest.raises(KeyError):
        r.number("example", "invented", estimator="A")


def test_claim_comparison_at_draft_precision(tmp_path):
    r = registry(tmp_path)
    result = check_claims(r, [{"claim_id": "test", "table": "example", "metric": "rmse",
                              "filter": {"estimator": "A"}, "draft_value": .125,
                              "decimals": 3}])
    assert result[0]["matches_at_stated_precision"] is True


def test_metric_dimension_can_be_selected(tmp_path):
    r = registry(tmp_path)
    for row in r.payload["tables"]["example"]["records"]:
        row["metric"] = "rmse"
    r.payload["tables"]["example"]["columns"].append("metric")
    r.payload["tables"]["example"]["filter_columns"].append("metric")
    assert r.number("example", "rmse", estimator="A", metric="rmse")["value"] == .125


def test_monthly_multiplies_before_averaging_and_keeps_repeats():
    from paper.derive_site_months import CORE, monthly
    rows = []
    for repeat in (0, 1):
        for index, aod in enumerate((1., 2., 3.)):
            rows.append({"row_id": str(index), "repeat": repeat, "scheme": "site_cv",
                         "site_key": "example", "year": 2017, "month": 1,
                         "truth_daod": .2, "truth_fraction": .1, "DB_AOD550": aod,
                         **{e: index / 2 for e in CORE}})
    result = monthly(pd.DataFrame(rows))
    assert len(result) == 2
    assert result.unique_rows.tolist() == [3, 3]
    assert result.E1_daod.tolist() == [4 / 3, 4 / 3]
    assert not (result.E1_daod == result.E1 * 2).any()
    with pytest.raises(ValueError, match="Duplicate"):
        monthly(pd.DataFrame(rows + [rows[0]]))
def test_physical_units_and_counts():
    from paper.numbers import units_for
    assert units_for("Sd_sr", [], 47.308) == "sr"
    assert units_for("selected_rows", ["selected_rows"], 3998) == "count"
    assert units_for("lat", [], 30.) == "degrees_north"
    assert units_for("lon_lower", [], -90.) == "degrees_east"
    assert units_for("rmse", [], .1) == "1"

