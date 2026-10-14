import numpy as np
import pandas as pd
import pytest

from paper.derive_monthly_cycles import regional_means


def inputs():
    a = np.array([[0., 2.], [4., np.nan]])
    fields = dict(Eselected=a, CALIOP=a, CALIOP_low=a / 2, CALIOP_high=a * 2)
    return fields, np.array([[True, True], [True, False]]), np.array([[1., 3.], [2., 0.]]), np.array([0, 2]), np.array([0, 5])


def test_area_weighting_zeros_missing_and_upper_exclusive_edges():
    args = inputs()
    r, n, area = regional_means(*args, [-1, 3, -1, 6])
    assert r["Eselected"] == pytest.approx(14 / 6)
    assert n == 3 and area == 6
    r, n, area = regional_means(*args, [-1, 2, -1, 5])
    assert r["Eselected"] == 0 and n == 1 and area == 1


def test_no_supported_cells_is_missing_not_zero():
    r, n, area = regional_means(*inputs(), [10, 12, 10, 12])
    assert all(np.isnan(v) for v in r.values()) and n == 0 and area == 0


@pytest.mark.parametrize("change", ["missing", "weight", "bounds", "mask"])
def test_invalid_support_or_scenario_rejected(change):
    f, m, w, lat, lon = inputs()
    if change == "missing": f["Eselected"][0, 0] = np.nan
    if change == "weight": w[0, 0] = 0
    if change == "bounds": f["CALIOP_low"] = f["CALIOP_high"] + 1
    if change == "mask": m = m.astype(int)
    with pytest.raises(ValueError):
        regional_means(f, m, w, lat, lon, [-1, 3, -1, 6])


def test_months_have_their_own_saved_masks_not_an_annual_intersection():
    f, m, w, lat, lon = inputs()
    first = regional_means(f, m, w, lat, lon, [-1, 3, -1, 6])
    m[0, 1] = False
    second = regional_means(f, m, w, lat, lon, [-1, 3, -1, 6])
    assert first[1] == 3 and second[1] == 2 and first[0]["Eselected"] != second[0]["Eselected"]


class RegistryStub:
    def __init__(self):
        from paper.derive_monthly_cycles import METHODS
        dates = ["2016-12"] + [f"2017-{m:02d}" for m in range(1, 12)]
        self.data = pd.DataFrame([dict(region=f"r{r}", date=d, estimator=e, mean=.1,
                                     n=3, land_area_weight_sum=2., supported=True, sensitivity=False)
                                  for r in range(15) for d in dates for e in METHODS])

    def table(self, *args, **kwargs):
        return self.data.copy()


def test_plot_selector_preserves_months_and_distinct_support_counts():
    from paper.figures.figS14_monthly_cycles import select
    reg = RegistryStub()
    mask = reg.data.date.eq("2016-12")
    reg.data.loc[mask, ["n", "land_area_weight_sum"]] = 0
    reg.data.loc[mask, "supported"] = False
    reg.data.loc[mask, "mean"] = np.nan
    result = select(reg)
    assert result.loc[mask, "mean"].isna().all()
    assert result.date.nunique() == 12


@pytest.mark.parametrize("change,message", [("duplicate", "ambiguous"), ("missing", "Incomplete"),
    ("count", "Unpaired"), ("value", "Invalid supported"), ("zerofill", "zero filled")])
def test_plot_selector_refuses_inconsistent_registry(change, message):
    from paper.figures.figS14_monthly_cycles import select
    reg = RegistryStub()
    if change == "duplicate": reg.data = pd.concat([reg.data, reg.data.iloc[[0]]])
    if change == "missing": reg.data = reg.data.iloc[1:]
    if change == "count": reg.data.loc[0, "n"] += 1
    if change == "value": reg.data.loc[0, "mean"] = np.nan
    if change == "zerofill":
        mask = reg.data.date.eq("2016-12") & reg.data.region.eq("r0")
        reg.data.loc[mask, ["n", "land_area_weight_sum"]] = 0
        reg.data.loc[mask, "supported"] = False
        reg.data.loc[mask, "mean"] = 0
    with pytest.raises(ValueError, match=message):
        select(reg)
