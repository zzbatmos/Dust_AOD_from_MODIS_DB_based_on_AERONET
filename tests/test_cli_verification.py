"""Numerical product verification; mocked I/O and no inference or fitting."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dbdust import cli
from dbdust.config import CONFIG
from dbdust.product_l3 import MonthlyAccumulator


NAMES = CONFIG['outputs']['monthly']
COUNTS = [name for name in NAMES if name.startswith('n_')]
STATISTICS = [name for name in NAMES if name not in COUNTS]
FRACTIONS = [name for name in NAMES if name.startswith('frac_')]
CONTRIBUTIONS = [name for name in NAMES if name.startswith('ev_contribution_')]


@pytest.fixture
def product():
    """One supported, one valid-AOD-only, and one empty cell."""
    data = {
        'n_valid_aod': [4, 1, 0], 'n_ml': [2, 0, 0],
        'n_full8': [1, 0, 0], 'n_noSSA5': [1, 0, 0],
        'aod550_db': [1., np.nan, np.nan],
        'daod550_ev': [0.25, np.nan, np.nan],
        'daod550_qa2': [0.5, np.nan, np.nan],
        'dust_probability_mean': [0.5, np.nan, np.nan],
        'frac_full8': [0.5, np.nan, np.nan],
        'frac_ml_missing': [0.5, 1., np.nan],
        'frac_qa1': [0.5, np.nan, np.nan],
        'frac_qa2': [0.5, np.nan, np.nan],
        'frac_qa3': [0.5, np.nan, np.nan],
        'ev_contribution_aod_lt_0p2': [0.01, np.nan, np.nan],
        'ev_contribution_p_lt_0p3': [0.01, np.nan, np.nan],
    }
    return xr.Dataset({name: (('time', 'lat', 'lon'), np.asarray(data[name]).reshape(1, 1, -1))
                       for name in NAMES})


@pytest.fixture
def verify(monkeypatch):
    monkeypatch.setattr(cli, 'EVPredictor', lambda: SimpleNamespace(manifest_sha256='verified-models'))

    def run(ds):
        monkeypatch.setattr(cli.xr, 'open_dataset', lambda path: ds)
        cli.verify_main(['mock-product.nc'])

    return run


def change(ds, name, value, cell=0):
    ds[name] = ds[name].astype(float)
    ds[name].values[0, 0, cell] = value


def test_valid_product_and_distinct_estimators(product, verify, capsys):
    original = product.copy(deep=True)
    assert product.daod550_qa2.values[0, 0, 0] > product.daod550_ev.values[0, 0, 0]
    verify(product)
    assert 'Product value/count invariants passed' in capsys.readouterr().out
    xr.testing.assert_identical(product, original)


def test_artifact_only_verification(monkeypatch, capsys):
    monkeypatch.setattr(cli, 'EVPredictor', lambda: SimpleNamespace(manifest_sha256='verified-models'))

    def unexpected_open(*args, **kwargs):
        pytest.fail('Artifact-only verification must not open a product')

    monkeypatch.setattr(cli.xr, 'open_dataset', unexpected_open)
    cli.verify_main([])
    output = capsys.readouterr().out
    assert 'Artifact manifest verified: verified-models' in output
    assert 'Product value/count invariants passed' not in output


@pytest.mark.parametrize('name', NAMES)
def test_missing_variable_rejected(product, verify, name, capsys):
    with pytest.raises(ValueError, match='Missing monthly variables'):
        verify(product.drop_vars(name))
    assert 'invariants passed' not in capsys.readouterr().out


@pytest.mark.parametrize('name', COUNTS)
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1., 0.5,
                               int(np.iinfo(np.int32).max) + 1])
def test_invalid_counts_rejected(product, verify, name, bad):
    change(product, name, bad)
    with pytest.raises(ValueError, match=name + '.*counts'):
        verify(product)


@pytest.mark.parametrize('name', STATISTICS)
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_supported_statistics_rejected(product, verify, name, bad):
    change(product, name, bad)
    with pytest.raises(ValueError, match=name + '.*finite'):
        verify(product)


@pytest.mark.parametrize('name', STATISTICS)
@pytest.mark.parametrize('bad', [0., 1., np.inf, -np.inf])
def test_empty_cell_statistics_must_be_missing(product, verify, name, bad):
    change(product, name, bad, cell=2)
    with pytest.raises(ValueError, match=name + '.*missing'):
        verify(product)


@pytest.mark.parametrize('name', [name for name in STATISTICS if name != 'frac_ml_missing'])
def test_no_ml_cell_statistics_must_be_missing(product, verify, name):
    change(product, name, 0., cell=1)
    with pytest.raises(ValueError, match=name + '.*missing'):
        verify(product)


@pytest.mark.parametrize('bad', [0., 0.5, np.nan, np.inf])
def test_no_ml_missing_fraction_must_be_one(product, verify, bad):
    change(product, 'frac_ml_missing', bad, cell=1)
    with pytest.raises(ValueError, match='frac_ml_missing'):
        verify(product)


@pytest.mark.parametrize('name', STATISTICS)
def test_negative_statistics_rejected(product, verify, name):
    change(product, name, -1.)
    with pytest.raises(ValueError, match=name + '.*nonnegative'):
        verify(product)


@pytest.mark.parametrize('name', [*FRACTIONS, 'dust_probability_mean'])
def test_fraction_probability_upper_bounds(product, verify, name):
    change(product, name, np.nextafter(1., np.inf))
    with pytest.raises(ValueError, match=name + r'.*\[0, 1\]'):
        verify(product)


@pytest.mark.parametrize('name', ['daod550_ev', 'daod550_qa2'])
def test_daod_cannot_exceed_aod(product, verify, name):
    change(product, name, np.nextafter(float(product.aod550_db.values[0, 0, 0]), np.inf))
    with pytest.raises(ValueError, match=name + ' exceeds aod550_db'):
        verify(product)


@pytest.mark.parametrize('name', CONTRIBUTIONS)
def test_subset_contributions_cannot_exceed_ev(product, verify, name):
    change(product, name, np.nextafter(float(product.daod550_ev.values[0, 0, 0]), np.inf))
    with pytest.raises(ValueError, match=name + ' exceeds daod550_ev'):
        verify(product)


def test_ml_count_cannot_exceed_valid_aod(product, verify):
    change(product, 'n_valid_aod', 1)
    with pytest.raises(ValueError, match='ML count exceeds'):
        verify(product)


def test_branch_counts_must_sum_to_ml(product, verify):
    change(product, 'n_full8', 0)
    with pytest.raises(ValueError, match='Branch count mismatch'):
        verify(product)


def test_branch_count_addition_cannot_wrap_int32(product, verify):
    for name in COUNTS:
        product[name] = product[name].astype(np.int32)
        product[name].values[0, 0, 0] = np.iinfo(np.int32).max
    with pytest.raises(ValueError, match='Branch count mismatch'):
        verify(product)


@pytest.mark.parametrize('name', ['frac_full8', 'frac_ml_missing'])
def test_count_fraction_identities(product, verify, name):
    change(product, name, 0.25)
    with pytest.raises(ValueError, match=name + ' disagrees with its count ratio'):
        verify(product)


@pytest.mark.parametrize('higher', ['frac_qa2', 'frac_qa3'])
def test_qa_fraction_ordering(product, verify, higher):
    change(product, higher, 1.)
    with pytest.raises(ValueError, match='QA fractions must be nonincreasing'):
        verify(product)


def test_no_qa_gate_passes_requires_zero_qa_daod(product, verify):
    for i in range(CONFIG['qa']['product_gate_index'], len(CONFIG['qa']['thresholds'])):
        change(product, 'frac_qa' + str(i + 1), 0.)
    with pytest.raises(ValueError, match='must be zero when no retrieval passes'):
        verify(product)
    change(product, 'daod550_qa2', 0.)
    verify(product)


def test_zero_qa_daod_does_not_imply_no_gate_passes(product, verify):
    change(product, 'daod550_qa2', 0.)
    verify(product)


def test_all_zero_optical_depths_are_valid_on_support(product, verify):
    for name in ['aod550_db', 'daod550_ev', 'daod550_qa2', *CONTRIBUTIONS]:
        change(product, name, 0.)
    verify(product)


def test_zero_or_full_ev_subset_contributions_are_valid(product, verify):
    for name in CONTRIBUTIONS:
        change(product, name, float(product.daod550_ev.values[0, 0, 0]))
    verify(product)
    for name in CONTRIBUTIONS:
        change(product, name, 0.)
    verify(product)


def test_fraction_endpoints_and_full_branch_support(product, verify):
    change(product, 'n_valid_aod', 2)
    change(product, 'frac_ml_missing', 0.)
    for full8 in (0, 2):
        change(product, 'n_full8', full8)
        change(product, 'n_noSSA5', 2 - full8)
        change(product, 'frac_full8', full8 / 2)
        verify(product)


def test_bad_dimensions_and_nonnumeric_values(product, verify):
    wrong_dims = product.copy(deep=True)
    wrong_dims['daod550_ev'] = wrong_dims.daod550_ev.transpose('lat', 'time', 'lon')
    with pytest.raises(ValueError, match='monthly dimensions'):
        verify(wrong_dims)
    wrong_type = product.copy(deep=True)
    wrong_type['n_ml'] = wrong_type.n_ml.astype(str)
    with pytest.raises(ValueError, match='real numeric'):
        verify(wrong_type)
    with pytest.raises(ValueError, match='monthly dimensions'):
        verify(xr.concat([product, product], dim='time'))


@pytest.mark.parametrize('grid', list(CONFIG['grids']))
def test_real_accumulator_output_passes(grid, verify):
    p = np.asarray([0., *CONFIG['qa']['thresholds'], 1.])
    aod = np.arange(len(p), dtype=float)
    conditional = np.full(len(p), 0.5)
    gate = CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']]
    frame = pd.DataFrame({
        'lat': np.zeros(len(p)), 'lon': np.zeros(len(p)),
        'branch': np.full(len(p), CONFIG['branches']['full8']['code']),
        'dust_probability': p, 'qa_tier': np.searchsorted(CONFIG['qa']['thresholds'], p, side='right'),
        'dust_fraction_conditional': conditional, 'aod550_db': aod,
        'daod550_ev': aod * conditional, 'daod550_qa2': np.where(p >= gate, aod * conditional, 0),
        'db_aerosol_type': np.zeros(len(p)),
    })
    accumulator = MonthlyAccumulator(grid)
    verify(accumulator.to_dataset('201701'))
    verify(accumulator.update(frame).to_dataset('201701'))
