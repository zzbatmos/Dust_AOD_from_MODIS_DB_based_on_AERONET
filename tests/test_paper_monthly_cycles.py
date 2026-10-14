import numpy as np
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
