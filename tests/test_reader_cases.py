"""Independent raw-reader equality tests using retained case granules."""
import os
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

from dbdust.modis_db import read_granule
from dbdust.config import CONFIG


def test_retained_raw_cases_match_original_reader():
    validation = os.environ.get('DBDUST_VALIDATION_ROOT')
    inputs = os.environ.get('DBDUST_CASE_INPUT_ROOT')
    if not validation or not inputs:
        pytest.skip('Retained raw-case and private validation paths not configured')
    path = Path(validation)/'coverage/audit.py'
    spec = importlib.util.spec_from_file_location('old_raw_case_audit', path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    names = ['MOD04_L2.A2025073.1705.061.2025074013638.hdf',
             'MOD04_L2.A2024266.1345.061.2024268021127.hdf',
             'MYD04_L2.A2017241.2015.061.2018039202859.hdf']
    for name in names:
        found = list(Path(inputs).glob('**/'+name))
        assert found, f'Missing configured golden case {name}'
        original = old.read_hdf(found[0])
        frame, shape, _ = read_granule(found[0])
        good = old.geo_valid(original) & np.isfinite(original['aod550']) & (original['aod550'] >= 0)
        np.testing.assert_array_equal(np.isfinite(frame.DB_AOD550).to_numpy().reshape(shape), good)
        values = {'DB_AOD550':original['aod550'], 'DB_AE':original['ae'],
                  'db_aerosol_type':original['db_type']}
        for i, wave in enumerate(CONFIG['inputs']['spectral_wavelengths_nm']):
            values[f'DB_AOD{wave}'] = original['spectral_aod'][i]
            values[f'DB_SSA{wave}'] = original['ssa'][i]
        for key, value in values.items():
            expected = np.where(good, value, np.nan).ravel()
            np.testing.assert_allclose(frame[key], expected, atol=0, rtol=0, equal_nan=True)
