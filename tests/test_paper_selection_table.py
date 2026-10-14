import pandas as pd
import pytest

from paper.selection_table import validate


class RegistryStub:
    def __init__(self):
        self.frame = pd.DataFrame([dict(reference="experimental1020", population="own", branch="full8",
            scheme="site_cv", split_id="site_r0_f0", repeat=0, selected="E3c", exact_tie="E4",
            criterion="site_month_RMSE", minimum_cell_rows=3, qualifying_sites=2,
            qualifying_cells=10, input_rows=30, paired_rows=30, dropped_rows=0,
            E3c_inner_RMSE=.0999999, E4_inner_RMSE=.1000000)])

    def table(self, name):
        return self.frame.copy()


def test_verbatim_choice_not_reselected_after_rounding():
    registry = RegistryStub()
    d = validate(registry)
    assert d.selected.iloc[0] == "E3c"
    assert f"{d.E3c_inner_RMSE.iloc[0]:.6f}" == f"{d.E4_inner_RMSE.iloc[0]:.6f}"
    pd.testing.assert_frame_equal(d, registry.frame)


@pytest.mark.parametrize("field,value,message", [
    ("selected", "E4", "contradicts"),
    ("selected", "new", "Unknown"),
    ("exact_tie", "E3c", "Unknown"),
    ("criterion", "outer_RMSE", "Unexpected"),
    ("minimum_cell_rows", 5, "Unexpected"),
    ("paired_rows", 31, "ledger"),
    ("qualifying_sites", 1.5, "Invalid inner support"),
    ("E4_inner_RMSE", float("nan"), "Invalid saved"),
])
def test_invalid_choice_or_support_is_rejected(field, value, message):
    registry = RegistryStub()
    if field == "qualifying_sites":
        registry.frame[field] = registry.frame[field].astype(float)
    registry.frame.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        validate(registry)


def test_exact_tie_is_E4_and_ambiguous_fold_fails():
    registry = RegistryStub()
    registry.frame.loc[0, "E3c_inner_RMSE"] = .1
    registry.frame.loc[0, "selected"] = "E4"
    assert validate(registry).selected.iloc[0] == "E4"
    registry.frame = pd.concat([registry.frame, registry.frame])
    with pytest.raises(ValueError, match="ambiguous"):
        validate(registry)
