import numpy as np
import pandas as pd
import pytest

from paper.derive_collocation_overview import FEATURES5, FEATURES8, REFERENCE, extract


def inputs():
    records = []
    for i in range(10):
        a440, a675 = .1 + i * .1, .08 + i * .08
        aod = np.exp(np.log(a440) + (np.log(a675) - np.log(a440)) * np.log(550 / 440) / np.log(675 / 440))
        records.append(dict(row_id=str(i), site_key=f"s{i % 3}", platform="Terra" if i % 2 else "Aqua",
                            time=f"{2017 + i % 2}-01-01", calibration_reserved=i == 9,
                            AER_AOD440=a440, AER_AOD675=a675, AER_AOD550=aod,
                            AER_AE440_675=-np.log(a440 / a675) / np.log(440 / 675),
                            **{REFERENCE: .3}, **{f: .5 for f in FEATURES8}))
    d = pd.DataFrame(records)
    d.loc[8, REFERENCE] = np.nan
    d.loc[7, "DB_AOD550"] = np.nan
    d.loc[6, "DB_AOD412"] = np.nan
    d.loc[5, "DB_SSA412"] = np.nan
    population = pd.DataFrame([dict(target=REFERENCE, features=f, train_valid_rows=n,
                                    evaluation_valid_rows=n, train_sites=3)
                               for f, n in [("full8", 5), ("noSSA5", 6)]])
    events = pd.DataFrame([dict(row_id=str(i), scheme="site_cv", repeat=r) for r in (0, 1) for i in range(6)])
    return d, population, events, dict(rows=10, sites=3)


def test_ledger_identity_partition_and_distinct_panel_populations():
    d, p, e, s = inputs()
    before = d.copy(deep=True)
    out = extract(d, p, e, s)
    assert out["collocation_ledger"].rows.tolist() == [10, 9, 8, 7, 6, 5, 1]
    stats = out["collocation_product_statistics"].set_index("panel")
    assert stats.loc["aod", "rows"] == 9
    assert stats.loc["ae", "rows"] == 10
    assert out["collocation_product_pairs"].calibration_reserved.sum() == 1
    assert len(out["collocation_product_pairs"]) == 10
    assert out["collocation_year_platform"].groupby("sample").rows.sum().to_dict() == {
        "all_collocations": 10, "hierarchy_eligible": 6}
    assert out["collocation_cohort_summary"].rows_2017.tolist() == [5, 3]
    pd.testing.assert_frame_equal(d, before)


def test_display_outliers_remain_in_all_pair_statistics():
    d, p, e, s = inputs()
    d.loc[0, "DB_AE"] = 20
    out = extract(d, p, e, s)
    stats = out["collocation_product_statistics"].set_index("panel").loc["ae"]
    assert stats.rows == 10 and stats.n_outside_view == 1
    assert out["collocation_product_pairs"].DB_AE.max() == 20
    assert stats.rmse > 5


@pytest.mark.parametrize("change,message", [
    ("duplicate", "duplicate collocation"), ("schema", "schema counts"),
    ("spectral", "spectral derivation"), ("reservation", "calibration reservation"),
    ("target", "already bounded"), ("time", "collocation year"),
    ("population", "eligibility differs"), ("oof", "row identities"),
])
def test_inconsistent_sources_fail(change, message):
    d, p, e, s = inputs()
    if change == "duplicate": d.loc[1, "row_id"] = d.row_id.iloc[0]
    elif change == "schema": s["rows"] = 11
    elif change == "spectral": d.loc[0, "AER_AOD550"] = 3
    elif change == "reservation": d["calibration_reserved"] = d.calibration_reserved.astype(int)
    elif change == "target": d.loc[0, REFERENCE] = 1.1
    elif change == "time": d.loc[0, "time"] = None
    elif change == "population": p.loc[0, "train_valid_rows"] = 4
    elif change == "oof": e.loc[0, "row_id"] = "not_training"
    with pytest.raises(ValueError, match=message):
        extract(d, p, e, s)


def test_oof_repeats_are_verified_not_counted_as_collocations():
    d, p, e, s = inputs()
    with pytest.raises(ValueError, match="OOF event"):
        extract(d, p, pd.concat([e, e.iloc[[0]]]), s)
    with pytest.raises(ValueError, match="OOF event"):
        extract(d, p, e.loc[e["repeat"].eq(0)], s)


class RegistryStub:
    def __init__(self):
        self.frames = {"derived/" + k: v for k, v in extract(*inputs()).items()}

    def table(self, name):
        return self.frames[name].copy()


def test_plot_selectors_preserve_populations_and_counts():
    from paper.figures.figS04_collocation_ledger import select as ledger_select
    from paper.figures.figS06_product_comparison import select as product_select
    r = RegistryStub()
    frames = ledger_select(r)
    assert len(frames["collocation_year_platform"]) == 8
    pairs, statistics = product_select(r)
    assert len(pairs) == 10 and statistics.rows.tolist() == [9, 10]


@pytest.mark.parametrize("table,field,value,selector,message", [
    ("collocation_ledger", "rows", 1, "ledger", "not nested"),
    ("collocation_year_platform", "rows", 99, "ledger", "Year totals"),
    ("collocation_cohort_summary", "fraction_2017", .9, "ledger", "2017 share"),
    ("collocation_product_statistics", "n_outside_view", 99, "product", "Display-window"),
    ("collocation_product_statistics", "rows", 99, "product", "counts disagree"),
])
def test_plot_selectors_reject_inconsistent_display_sources(table, field, value, selector, message):
    from paper.figures.figS04_collocation_ledger import select as ledger_select
    from paper.figures.figS06_product_comparison import select as product_select
    r = RegistryStub()
    r.frames["derived/" + table].loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        (ledger_select if selector == "ledger" else product_select)(r)
