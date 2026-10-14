"""Ensure temporal displays preserve distinct supports and saved statistics."""
import copy

import pandas as pd
import pytest

from paper.figures.figS18_temporal import FILTERS, METHODS, ROOT, SCHEMES, select_tables


class RegistryStub:
    def __init__(self):
        self.frames = {}
        self.payload = {"tables": {}}
        for kind in ("seasonal_summary", "anomalies", "support"):
            records = []
            for scheme in SCHEMES:
                repeats = 2 if scheme == "site_cv" else 1
                common = dict(FILTERS, scheme=scheme, expected_repeats=repeats)
                if kind == "support":
                    for prefix, count in (("seasonal", 7), ("anomaly", 36)):
                        common.update({f"{prefix}_{key}": value for key, value in dict(
                            sites=count, prediction_events=100 * repeats, unique_rows=1000,
                            unique_site_months=100, repeats=repeats).items()})
                    records.append(common)
                else:
                    for method in METHODS:
                        records.append(dict(common, estimator=method, sites=7 if kind == "seasonal_summary" else 36,
                            prediction_events=100 * repeats, unique_rows=1000, unique_site_months=100,
                            repeats=repeats, summary_defined=True, finite_r_sites=5 if method == "E1" else 7,
                            median_site_r=.5, q05_site_r=-.2, q95_site_r=.9,
                            statistic_semantics="distribution statistics; not CIs", r_defined=True,
                            r=.4, r_lo=.1, r_hi=.8))
            self.frames[ROOT + kind] = pd.DataFrame(records)
            self.payload["tables"][ROOT + kind] = dict(source_file=f"{kind}.csv", source_sha256="a" * 64)

    def table(self, name, **filters):
        frame = self.frames[name]
        for key, value in filters.items():
            frame = frame.loc[frame[key].eq(value)]
        return frame.copy()


def test_preserves_saved_values_and_distinct_supports_without_mutation():
    registry = RegistryStub()
    before = copy.deepcopy(registry.frames)
    frames, exported, sources = select_tables(registry)
    assert len(exported) == 22
    assert len(sources) == 3
    assert set(frames["seasonal_summary"].sites) == {7}
    assert set(frames["anomalies"].sites) == {36}
    assert set(frames["seasonal_summary"].finite_r_sites) == {5, 7}
    assert frames["anomalies"].r.eq(.4).all()
    for name in before:
        pd.testing.assert_frame_equal(registry.frames[name], before[name])


@pytest.mark.parametrize("kind,field,value,message", [
    ("seasonal_summary", "sites", 36, "support mismatch"),
    ("anomalies", "unique_site_months", 99, "support mismatch"),
    ("support", "seasonal_prediction_events", 999, "support mismatch"),
    ("anomalies", "expected_repeats", 3, "repeat semantics"),
    ("seasonal_summary", "statistic_semantics", "confidence intervals", "must not"),
    ("seasonal_summary", "finite_r_sites", 8, "Undefined seasonal"),
    ("seasonal_summary", "q05_site_r", .8, "Reversed"),
    ("anomalies", "r_lo", float("nan"), "Invalid saved"),
    ("anomalies", "r_hi", 1.1, "Invalid saved"),
    ("anomalies", "r", 1.1, "Invalid anomaly"),
    ("anomalies", "r_defined", False, "Undefined anomaly"),
])
def test_bad_or_mixed_summary_fails(kind, field, value, message):
    registry = RegistryStub()
    registry.frames[ROOT + kind].loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        select_tables(registry)


@pytest.mark.parametrize("kind", ["seasonal_summary", "anomalies", "support"])
def test_duplicate_or_missing_summary_fails(kind):
    for duplicate in (False, True):
        registry = RegistryStub()
        frame = registry.frames[ROOT + kind]
        registry.frames[ROOT + kind] = pd.concat([frame, frame.iloc[[0]]]) if duplicate else frame.iloc[1:]
        with pytest.raises(ValueError, match="Missing or ambiguous"):
            select_tables(registry)


def test_unrelated_reference_and_pairing_are_not_pooled():
    registry = RegistryStub()
    for name, frame in registry.frames.items():
        registry.frames[name] = pd.concat([frame, frame.assign(reference="old675LR56"),
                                           frame.assign(pairing="including_E6")])
    _, exported, _ = select_tables(registry)
    assert len(exported) == 22
    assert set(exported.reference) == {"experimental1020"}
    assert set(exported.pairing) == {"core"}
