"""Synthetic arithmetic contracts; no unpublished quantitative results or inputs."""
import copy
import json
from pathlib import Path

import pytest

from paper import claim_arithmetic as arithmetic
from paper.numbers import COUNTS, DIMENSIONS, Registry, digest


def synthetic_registry():
    tables = {}

    def table(name, rows):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        tables[name] = {
            "source_file": f"/synthetic/{name}.csv", "source_sha256": "a" * 64,
            "source_rows": len(rows), "columns": columns,
            "filter_columns": [c for c in columns if c in DIMENSIONS],
            "count_columns": [c for c in columns if c in COUNTS or c.startswith("n_")],
            "semantics": "Synthetic fixture only", "records": rows,
        }

    table(arithmetic.DECOMPOSITION, [
        dict(reference=arithmetic.PRIMARY, population="decomposition", scheme="site_cv",
             QA=0.6, A=a, B=b, n_prediction_events=16, unique_rows=8, sites=2,
             delta_RMSE_A_minus_B=-0.25, delta_RMSE_lo=-0.5, delta_RMSE_hi=-0.125)
        for a, b in (("DB_size500proxy", "AER_size500proxy"),
                     ("DB_nonsphericity", "AER_nonsphericity"))
    ])
    site_rows = [
        {**arithmetic.SITE_FILTER, "estimator": estimator, "rmse": rmse,
         "rmse_lo": rmse / 2, "rmse_hi": rmse * 2,
         "unique_rows": 12, "unique_site_months": 4, "prediction_events": 8,
         "sites": 2, "repeats": 2}
        for estimator, rmse in (("Eselected", 0.125), ("E1", 0.25), ("E5", 0.5))
    ]
    # Tempting but scientifically incompatible rows must never be selected.
    site_rows.extend([{**site_rows[0], "minimum": 1, "rmse": 99.0},
                      {**site_rows[0], "repeat_summary": "pooled", "rmse": 99.0},
                      {**site_rows[0], "pairing": "including_E6", "rmse": 99.0}])
    table(arithmetic.SITE_MONTH, site_rows)
    regions = {"Land 60S-60N", *arithmetic.DESERTS, *arithmetic.MIXED_REGIONS,
               *(region for region, _ in arithmetic.SMOKE_SEASONS)}
    seasonal_rows = []
    for region in sorted(regions):
        for period in arithmetic.SEASONS:
            ev = 1.0
            if (region, period) == ("Sahara", "MAM"):
                ev = 2.25
            if (region, period) == ("East China", "JJA"):
                ev = 3.0
            for estimator, mean in (("Eselected", ev), ("E1", 2.5),
                                    ("E8", 0.75), ("E7", 4.0), ("Song", 6.0)):
                row = {**arithmetic.SEASON_FILTER, "region": region, "period": period,
                       "estimator": estimator, "candidate_mean": mean,
                       "reference_mean": 2.0, "n": 4}
                seasonal_rows.append(row)
                if estimator == "Eselected" and region == "Land 60S-60N":
                    seasonal_rows.append({**row, "reference": "old675LR56",
                                          "candidate_mean": mean * 1.5})
    seasonal_rows.append({**seasonal_rows[0], "reference_mask": "mean_only_reference_sensitivity",
                          "candidate_mean": 99.0})
    table(arithmetic.SEASONAL, seasonal_rows)
    table(arithmetic.LOW_LOADING, [
        dict(reference=arithmetic.PRIMARY, variant="hierarchy", support="core_common",
             estimator="Eselected", region="Land 60S-60N", subset="AOD550_lt_0p2",
             area_weighted_mean=0.25, full_mean_DAOD=2.0, cell_months=12),
    ])
    return Registry({"schema_version": 1, "tables": tables})


def indexed(registry):
    return {record["key"]: record for record in arithmetic.derive_claims(registry)}


def source_row(registry, table, **filters):
    matches = [row for row in registry.payload["tables"][table]["records"]
               if all(row.get(key) == value for key, value in filters.items())]
    assert len(matches) == 1
    return matches[0]


def test_signed_reductions_reverse_interval_and_keep_provenance():
    registry = synthetic_registry()
    before = copy.deepcopy(registry.payload)
    results = indexed(registry)
    for target in ("size", "depolarization"):
        rec = results[f"decomposition/{target}/rmse_reduction"]
        assert rec["value"] == 0.25
        assert rec["interval"] == [0.125, 0.5]
        assert rec["interval_type"] == "95% site bootstrap"
        operand = rec["operands"]["delta_RMSE_A_minus_B"]
        assert operand["interval"] == [-0.5, -0.125]
        assert operand["source_sha256"] == "a" * 64
        assert operand["filter"]["A"] == operand["requested_filter"]["A"]
        assert rec["N"]["delta_RMSE_A_minus_B"]["unique_rows"] == 8
    assert registry.payload == before


def test_site_month_improvements_do_not_combine_marginal_intervals():
    results = indexed(synthetic_registry())
    for baseline, expected in (("E1", 50.0), ("E5", 75.0)):
        rec = results[f"site_month/Eselected_vs_{baseline}/rmse_improvement_percent"]
        assert rec["value"] == expected
        assert rec["interval"] is None
        assert rec["operands"]["ev_rmse"]["interval"] == [0.0625, 0.25]
        assert rec["operands"]["baseline_rmse"]["requested_filter"] == {
            **arithmetic.SITE_FILTER, "estimator": baseline}


def test_seasonal_ratios_biases_and_closeness_flags():
    results = indexed(synthetic_registry())
    assert results["global/DJF/Eselected/ratio"]["value"] == 0.5
    assert results["global/DJF/Eselected/relative_bias_percent"]["value"] == -50.0
    assert results["global/DJF/Eselected/bias"]["value"] == -1.0
    key = "desert/Sahara/MAM/"
    assert results[key + "qa2_minus_ev_absolute_bias"]["value"] == 0.25
    assert results[key + "qa2_as_close_or_closer_exception"]["value"] is True
    assert results["desert/Sahara/JJA/qa2_as_close_or_closer_exception"]["value"] is False
    assert results["smoke/Siberia/JJA/E7/ratio"]["value"] == 2.0
    assert results["smoke/Siberia/JJA/Song/bias"]["value"] == 4.0
    assert results["smoke/Siberia/JJA/Eselected/bias"]["value"] == -1.0
    assert results["smoke/Siberia/JJA/E7_vs_EV/ratio"]["value"] == 4.0
    assert results["smoke/Siberia/JJA/Song_vs_EV/mean_difference"]["value"] == 5.0


def test_mixed_flags_are_computed_not_hardcoded():
    registry = synthetic_registry()
    results = indexed(registry)
    key = "mixed/East China/JJA/Eselected/underestimation_exception"
    assert results[key]["value"] is True
    assert results["mixed/Sahel/DJF/Eselected/underestimation_exception"]["value"] is False
    assert results["mixed/Sahel/DJF/E7/underestimation_exception"]["value"] is True
    row = source_row(registry, arithmetic.SEASONAL, reference=arithmetic.PRIMARY,
                     region="East China", period="JJA", estimator="Eselected")
    row["candidate_mean"] = 0.5
    assert indexed(Registry(registry.payload))[key]["value"] is False


def test_sensitivity_and_low_loading_are_descriptive():
    results = indexed(synthetic_registry())
    for period in arithmetic.SEASONS:
        rec = results[f"reference_sensitivity/{period}/old_to_primary_ratio"]
        assert rec["value"] == 1.5
        assert rec["interval"] is None
        assert "Not an isolated lidar-ratio effect" in " ".join(rec["caveats"])
        assert rec["operands"]["old_mean"]["filter"]["reference"] == "old675LR56"
        assert results[f"reference_sensitivity/{period}/increase_percent"]["value"] == 50.0
    rec = results["low_aod/contribution_fraction"]
    assert rec["value"] == 0.125
    assert rec["N"]["low_aod_contribution"] == {"cell_months": 12}
    assert results["low_aod/contribution_percent"]["value"] == 12.5
    assert "Calendar 2017 MODIS-only" in " ".join(rec["caveats"])


def test_every_record_retains_operands_counts_filters_hashes_and_equation():
    records = arithmetic.derive_claims(synthetic_registry())
    assert len(records) == len({rec["key"] for rec in records})
    json.dumps(records, allow_nan=False)
    for rec in records:
        assert rec["equation"] and rec["caveats"] and rec["N"]
        if not rec["key"].startswith("decomposition/"):
            assert rec["interval"] is None and rec["interval_type"] is None
        for role, operand in rec["operands"].items():
            assert rec["N"][role] == operand["N"]
            assert operand["source_file"] and operand["source_sha256"]
            assert operand["filter"] and operand["requested_filter"]
            assert operand["table"] and operand["metric"]
            assert operand["source_record_index"] >= 0


@pytest.mark.parametrize("bad", [0.0, -1.0, None, float("nan"), float("inf"), True])
def test_bad_denominator_fails_instead_of_emitting_invalid_or_infinite_claim(bad):
    registry = synthetic_registry()
    registry.payload["tables"][arithmetic.LOW_LOADING]["records"][0]["full_mean_DAOD"] = bad
    with pytest.raises(ValueError, match="positive|finite numeric"):
        arithmetic.derive_claims(registry)


@pytest.mark.parametrize("interval", [(None, None), (-0.1, -0.5), (None, -0.1)])
def test_missing_or_malformed_contrast_interval_fails(interval):
    registry = synthetic_registry()
    row = registry.payload["tables"][arithmetic.DECOMPOSITION]["records"][0]
    row["delta_RMSE_lo"], row["delta_RMSE_hi"] = interval
    with pytest.raises(ValueError, match="interval|finite numeric"):
        arithmetic.derive_claims(registry)


def test_mismatched_counts_or_reference_mean_fail():
    registry = synthetic_registry()
    row = source_row(registry, arithmetic.SITE_MONTH, estimator="E1")
    row["unique_rows"] += 1
    with pytest.raises(ValueError, match="counts differ"):
        arithmetic.derive_claims(registry)
    registry = synthetic_registry()
    row = source_row(registry, arithmetic.SEASONAL, reference=arithmetic.PRIMARY,
                     region="Sahara", period="MAM", estimator="E1")
    row["reference_mean"] = 3.0
    with pytest.raises(ValueError, match="same reference mean"):
        arithmetic.derive_claims(registry)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_missing_or_ambiguous_source_rows_fail(ambiguous):
    registry = synthetic_registry()
    rows = registry.payload["tables"][arithmetic.DECOMPOSITION]["records"]
    if ambiguous:
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows.pop(0)
    with pytest.raises(ValueError, match="expected 1 row"):
        arithmetic.derive_claims(registry)


def test_cli_private_json_provenance_and_output_guards(tmp_path):
    source = tmp_path / "registry.json"
    source.write_text(json.dumps(synthetic_registry().payload))
    before = digest(source)
    output = tmp_path / "private" / "claims.json"
    arithmetic.main(["--registry", str(source), "--output-file", str(output)])
    payload = json.loads(output.read_text())
    assert payload["registry_sha256"] == before == digest(source)
    assert payload["generator_sha256"] == digest(Path(arithmetic.__file__))
    assert payload["records"]
    forbidden = Path(arithmetic.__file__).resolve().parent / "forbidden_claims.json"
    for destination, message in ((source, "overwrite"), (forbidden, "outside")):
        with pytest.raises(ValueError, match=message):
            arithmetic.main(["--registry", str(source), "--output-file", str(destination)])
    assert digest(source) == before
    assert not forbidden.exists()
