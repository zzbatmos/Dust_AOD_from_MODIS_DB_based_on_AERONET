import numpy as np
import pandas as pd
import pytest

from paper.figures.fig03_information_content import BINS, CAPTION, annual_grid, pilot_counts


def pilot():
    inventory = pd.DataFrame([dict(day="2017-08-29", expected=160, completed=160, failed=0, complete=True)])
    rows = [dict(sample="raw_Aqua2017", month="2017-08", region="All swath", db_type="1", algorithm=algo,
                 aod_bin=bin_, n_valid_aod=10, n_ssa_observed=10, n_ssa_missing=2)
            for algo in ("0", "UNKNOWN") for bin_ in BINS]
    rows.append(dict(sample="raw_Aqua2017", month="2017-08", region="All swath", db_type="ALL", algorithm="ALL",
                     aod_bin="ALL", n_valid_aod=80, n_ssa_observed=80, n_ssa_missing=16))
    # Overlapping marginals are deliberately excluded from the joint extraction.
    rows.append(dict(rows[0], algorithm="ALL", n_valid_aod=20, n_ssa_observed=20, n_ssa_missing=4))
    return pd.DataFrame(rows), inventory


def test_pilot_joint_rows_exclude_marginals_include_unknown_algorithm():
    frame, inventory = pilot()
    result = pilot_counts(frame, inventory)
    assert len(result) == 16
    assert result.n_valid_aod.sum() == 80
    assert np.allclose(result.loc[result.db_type.eq("1"), "fraction"], .8)
    assert result.loc[result.db_type.eq("3"), "fraction"].isna().all()
    assert result.date.eq("2017-08-29").all()


@pytest.mark.parametrize("field,value", [("day", "2017-08-30"), ("complete", False), ("complete", "False"),
                                        ("completed", 159), ("failed", 1)])
def test_pilot_requires_completed_dated_inventory(field, value):
    frame, inventory = pilot()
    inventory = inventory.astype({field: object})
    inventory.loc[0, field] = value
    with pytest.raises(ValueError, match="inventory"):
        pilot_counts(frame, inventory)


def test_pilot_rejects_wrong_month_or_nonclosing_counts():
    frame, inventory = pilot()
    wrong = frame.copy()
    wrong.loc[0, "month"] = "2017-09"
    with pytest.raises(ValueError, match="August"):
        pilot_counts(wrong, inventory)
    wrong = frame.copy()
    wrong.loc[0, "n_ssa_missing"] += 1
    with pytest.raises(ValueError, match="close"):
        pilot_counts(wrong, inventory)
    with pytest.raises(ValueError, match="Ambiguous"):
        pilot_counts(pd.concat([frame, frame.iloc[[0]]]), inventory)


def annual():
    lat, lon = np.meshgrid(np.arange(-89.5, 90), np.arange(-179.5, 180), indexing="ij")
    frame = pd.DataFrame(dict(lat=lat.ravel(), lon=lon.ravel(), year=2017, reference="experimental1020",
                             support="calendar_year_valid_AOD", n_full8=1, n_noSSA5=2, n_ml=3,
                             n_valid_aod=4, full8_fraction_of_valid_aod=.25))
    frame.loc[0, ["n_full8", "n_noSSA5", "n_ml", "n_valid_aod"]] = 0
    frame.loc[0, "full8_fraction_of_valid_aod"] = np.nan
    frame.loc[1, ["n_full8", "n_ml"]] = [0, 2]
    frame.loc[1, "full8_fraction_of_valid_aod"] = 0
    return frame


def test_annual_pooled_valid_aod_denominator_missing_and_zero():
    x, y, values = annual_grid(annual())
    assert len(x) == 361 and len(y) == 181 and values.shape == (180, 360)
    assert np.isnan(values[0, 0]) and values[0, 1] == 0
    assert values[0, 2] == .25  # not full8/n_ml = 1/3


@pytest.mark.parametrize("field,value,match", [("year", 2016, "population"), ("lat", 0, "coordinates"),
                                              ("n_full8", -1, "counts"), ("n_ml", 5, "partition"),
                                              ("full8_fraction_of_valid_aod", .5, "weighted")])
def test_annual_fail_closed(field, value, match):
    frame = annual()
    frame.loc[2, field] = value
    with pytest.raises(ValueError, match=match):
        annual_grid(frame)


def test_caption_keeps_review_and_uncertainty_qualifications():
    assert len(CAPTION.split()) <= 80
    assert "AUC intervals are unavailable" in CAPTION
    assert "not nested Eselected" in CAPTION
    assert "different temporal support" in CAPTION
