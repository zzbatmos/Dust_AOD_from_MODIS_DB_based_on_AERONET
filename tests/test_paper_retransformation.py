import pandas as pd
import pytest

from paper.figures.figS12_retransformation import BRANCHES, REFERENCES, ROOT, TABLES, select


class RegistryStub:
    def __init__(self):
        records = [dict(reference=r, branch=b, population="own", scheme=s,
                        split_id="fold0", repeat=0, training_rows=100,
                        evaluation_positive_rows=2, positive_observed_mean=.5, f_pos_mean=.6,
                        f_pos_star_mean=.55, selection_criterion="site_month_RMSE",
                        target="dust fraction (NOT FMF)", coefficient_order="a + b*DB_AE + c*DB_AE^2",
                        a=.4, b=-.2, c=.1)
                   for r in REFERENCES for b in BRANCHES for s in ("site_cv", "region_out")]
        self.frames = {ROOT + k: pd.DataFrame(records) for k in TABLES}
        self.payload = {"tables": {k: dict(source_file=k + ".csv", source_sha256="a" * 64)
                                   for k in self.frames}}

    def table(self, name, **filters):
        d = self.frames[name]
        for key, value in filters.items():
            d = d.loc[d[key].eq(value)]
        return d.copy()


def test_saved_folds_not_averaged_and_zero_positive_missing_preserved():
    registry = RegistryStub()
    d = registry.frames[ROOT + TABLES[0]]
    d.loc[0, "evaluation_positive_rows"] = 0
    d.loc[0, ["positive_observed_mean", "f_pos_mean", "f_pos_star_mean"]] = float("nan")
    frames, exported, sources = select(registry)
    assert len(exported) == 16 and len(sources) == 2
    assert frames[TABLES[0]].f_pos_mean.isna().sum() == 1
    assert frames[TABLES[1]].b.eq(-.2).all()


@pytest.mark.parametrize("table,field,value,message", [
    (0, "training_rows", 99, "populations disagree"),
    (0, "selection_criterion", "collocation_RMSE", "final Stage B"),
    (0, "evaluation_positive_rows", 0, "must remain missing"),
    (0, "f_pos_mean", 1.1, "Invalid saved"),
    (1, "target", "FMF", "dust-fraction target"),
    (1, "coefficient_order", "quadratic first", "coefficient ordering"),
    (1, "a", float("nan"), "Nonfinite"),
])
def test_wrong_or_invalid_diagnostics_fail(table, field, value, message):
    registry = RegistryStub()
    registry.frames[ROOT + TABLES[table]].loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        select(registry)


def test_missing_or_duplicate_fold_fails():
    registry = RegistryStub()
    registry.frames[ROOT + TABLES[1]] = registry.frames[ROOT + TABLES[1]].iloc[1:]
    with pytest.raises(ValueError, match="populations disagree"):
        select(registry)
    registry = RegistryStub()
    name = ROOT + TABLES[0]
    registry.frames[name] = pd.concat([registry.frames[name], registry.frames[name].iloc[[0]]])
    with pytest.raises(ValueError, match="ambiguous"):
        select(registry)
