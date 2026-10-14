"""Boundary and local frozen-artifact regression checks; no fits."""
import os
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from dbdust.branches import assign_branches, qa_tiers
from dbdust.config import CONFIG
from dbdust.modis_db import decode_science, decode_aerosol_type
from dbdust.predictor import EVPredictor
from dbdust.reference import construct_reference


def input_frame():
    return pd.DataFrame({key: np.ones(4) for key in CONFIG['branches']['full8']['features']})


def test_every_missing_feature_combination():
    names = CONFIG['branches']['full8']['features']
    base = pd.DataFrame(np.ones((2**len(names), len(names))), columns=names)
    for i, name in enumerate(names):
        base.loc[(np.arange(len(base)) >> i) & 1 != 0, name] = np.nan
    expected = np.where(base[names].notna().all(axis=1), 1,
                        np.where(base[names[:5]].notna().all(axis=1), 2, 0))
    np.testing.assert_array_equal(assign_branches(base), expected)


def test_qa_edges():
    boundaries = CONFIG['qa']['thresholds']
    values = [np.nan, 0, *boundaries, 1]
    np.testing.assert_array_equal(qa_tiers(values), [-1, 0, 1, 2, 3, 3])
    for i, edge in enumerate(boundaries):
        assert qa_tiers([np.nextafter(edge, -np.inf)])[0] == i


def test_science_and_qa_decode():
    attrs = {'_FillValue': -9999, 'valid_range': [0, 5000], 'scale_factor': .001}
    np.testing.assert_allclose(decode_science(np.array([0, 5000, -9999, -1, 5001]), attrs),
                               [0, 5, np.nan, np.nan, np.nan], equal_nan=True)
    q = np.zeros((1, 4, 6), dtype=np.int8)
    q[..., CONFIG['inputs']['qa_byte_index']] = np.arange(4) << CONFIG['inputs']['aerosol_type_shift']
    np.testing.assert_array_equal(decode_aerosol_type(q, (1, 4)), [[0, 1, 2, 3]])


def test_reference_endmembers():
    c = CONFIG['reference']
    out = construct_reference([c['nondust_pldr'], c['dust_pldr']],
                              c['dust_lidar_ratio_sr'], 1, 1, 1)
    np.testing.assert_allclose(out['backscatter_fraction'], [0, 1])
    np.testing.assert_allclose(out['fraction550'], [0, 1])


def test_missing_stays_missing():
    frame = input_frame()
    frame.loc[1, 'DB_SSA412'] = np.nan
    frame.loc[2, 'DB_AE'] = np.nan
    frame.loc[3, 'DB_AOD550'] = np.nan
    out = EVPredictor().predict_frame(frame)
    np.testing.assert_array_equal(out.branch, [1, 2, 0, -1])
    assert out.loc[2:, ['daod550_qa2', 'daod550_ev', 'dust_probability']].isna().all().all()
    assert out.loc[2, 'aod550_db'] == 1
    assert np.isnan(out.loc[3, 'aod550_db'])


def private_root():
    root = os.environ.get('DBDUST_VALIDATION_ROOT')
    if not root:
        pytest.skip('Set DBDUST_VALIDATION_ROOT to the existing private revision_v2_work directory')
    return Path(root)


def test_cached_reference():
    df = pd.read_parquet(private_root()/'grouped_validation/rows.parquet')
    actual = construct_reference(df.AER_PLDR1020, df.AER_LR1020, df.AER_AOD1020,
                                 df.AER_AOD440, df.AER_AOD675)['fraction550']
    np.testing.assert_allclose(actual, df.experimental1020, rtol=0, atol=1e-12, equal_nan=True)


def test_deployment_on_all_cached_rows():
    root = private_root()
    path = root/'expected_value_test/deploy_ev/predict.py'
    spec = importlib.util.spec_from_file_location('frozen_deploy', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    rows = pd.read_parquet(root/'grouped_validation/rows.parquet')
    expected = mod.EVPredictor('experimental1020').predict_frame(rows)
    actual = EVPredictor().predict_frame(rows)
    for field, original in [('dust_probability', 'probability'),
                            ('dust_fraction_conditional', 'positive_fraction'),
                            ('daod550_qa2', 'E1_daod'), ('daod550_ev', 'Eselected_daod')]:
        np.testing.assert_allclose(actual[field], expected[original], rtol=0, atol=1e-6, equal_nan=True)
