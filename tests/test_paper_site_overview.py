import pandas as pd
import pytest

from paper.derive_site_overview import evaluation_sites


def fixture():
    events = pd.DataFrame([dict(row_id="r1", repeat=i, scheme="site_cv", site_key="a",
                               site="A", region="R", time="2017-01-01", platform="Aqua")
                           for i in (0, 1)])
    inventory = pd.DataFrame([dict(site_key="a", lat=20, lon=10, calibration_reserved=False, CV_region="R")])
    return events, inventory


def test_unique_rows_not_prediction_events():
    result = evaluation_sites(*fixture())
    assert result.collocations.sum() == 1
    assert result.first_year.iloc[0] == 2017


def test_reserved_site_fails_closed():
    events, inventory = fixture()
    inventory["calibration_reserved"] = True
    with pytest.raises(ValueError, match="reservation"):
        evaluation_sites(events, inventory)


def test_missing_repeat_fails_closed():
    events, inventory = fixture()
    with pytest.raises(ValueError, match="two events"):
        evaluation_sites(events.iloc[:1], inventory)


def test_changing_identity_fails_closed():
    events, inventory = fixture()
    events.loc[1, "site"] = "B"
    with pytest.raises(ValueError, match="identity"):
        evaluation_sites(events, inventory)
