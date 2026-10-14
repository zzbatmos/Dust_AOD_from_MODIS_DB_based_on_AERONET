import pandas as pd
import pytest

from paper.figures.figS16_reference_means import FILTERS, GLOBAL, METHODS, REFS, SEASONS, select


class RegistryStub:
    def __init__(self):
        self.frame = pd.DataFrame([
            dict(FILTERS, reference=r, estimator=m, region=g, period=s, n=10,
                 candidate_mean=.1 if m == "CALIOP" else .05, reference_mean=.1)
            for r in REFS for m in (*METHODS, "CALIOP")
            for g in (GLOBAL, "Sahara", "latitude_sensitivity") for s in SEASONS
        ])

    def table(self, name, **filters):
        frame = self.frame.copy()
        for key, value in filters.items():
            frame = frame.loc[frame[key].eq(value)]
        return frame


def test_preserves_values_and_excludes_named_latitude_sensitivity():
    registry = RegistryStub()
    before = registry.frame.copy()
    selected = select(registry)
    assert len(selected) == 48
    assert set(selected.region) == {GLOBAL, "Sahara"}
    assert set(selected.reference) == set(REFS)
    pd.testing.assert_frame_equal(before, registry.frame)


@pytest.mark.parametrize("field,value,message", [
    ("n", 11, "different saved support"),
    ("reference_mean", .2, "different saved support"),
    ("candidate_mean", float("nan"), "Invalid saved"),
    ("candidate_mean", -.1, "Invalid saved"),
    ("period", "annual", "Incomplete four-season"),
])
def test_invalid_support_or_value_fails(field, value, message):
    registry = RegistryStub()
    registry.frame.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        select(registry)


def test_missing_and_duplicate_data_fail():
    registry = RegistryStub()
    registry.frame = pd.concat([registry.frame, registry.frame.iloc[[0]]])
    with pytest.raises(ValueError, match="ambiguous"):
        select(registry)
    registry = RegistryStub()
    registry.frame = registry.frame.loc[~registry.frame.reference.eq(REFS[0])]
    with pytest.raises(ValueError, match="Incomplete reference"):
        select(registry)


def test_caliop_identity_checked():
    registry = RegistryStub()
    registry.frame.loc[registry.frame.estimator.eq("CALIOP"), "candidate_mean"] = .9
    with pytest.raises(ValueError, match="CALIOP mean identity"):
        select(registry)


@pytest.mark.parametrize("count", [0, 1])
def test_undefined_saved_means_remain_missing_with_original_count(count):
    registry = RegistryStub()
    chosen = registry.frame.region.eq("Sahara") & registry.frame.period.eq("DJF")
    registry.frame.loc[chosen, ["candidate_mean", "reference_mean"]] = float("nan")
    registry.frame.loc[chosen, "n"] = count
    data = select(registry)
    saved = data.loc[data.region.eq("Sahara") & data.period.eq("DJF")]
    assert len(saved) == 6
    assert saved.candidate_mean.isna().all()
    assert saved.n.eq(count).all()
