"""Recreate case outputs against separately retained local golden files."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from dbdust.modis_db import read_granule
from dbdust.predictor import EVPredictor, file_hash
from dbdust.product_l2 import write_l2


@pytest.mark.parametrize('case', ['Kansas_20250314', 'Amazonia_20240922', 'NorthAmerica_20170829'])
def test_golden_case(tmp_path, case):
    root = os.environ.get('DBDUST_CASE_INPUT_ROOT')
    gold = os.environ.get('DBDUST_GOLDEN_ROOT')
    if not root or not gold:
        pytest.skip('Set DBDUST_CASE_INPUT_ROOT and DBDUST_GOLDEN_ROOT to existing local case/golden artifacts')
    gold = Path(gold)
    record = next(x for x in json.loads((gold/'manifest.json').read_text())['cases'] if x['case'] == case)
    matches = list(Path(root).glob('**/'+record['source_file']))
    if not matches:
        pytest.fail('Configured case input root does not contain required HDF')
    source = next((p for p in matches if file_hash(p) == record['source_sha256']), None)
    assert source is not None, 'No source matches the frozen raw checksum'
    expected = gold/record['output_file']
    assert file_hash(expected) == record['output_sha256']
    frame, _, _ = read_granule(source)
    output = tmp_path/'golden.nc'
    write_l2(EVPredictor().predict_frame(frame), output)
    with xr.open_dataset(expected) as ref, xr.open_dataset(output) as actual:
        assert dict(actual.sizes) == dict(ref.sizes)
        for name in ref.variables:
            tolerance = 0 if name in ('branch', 'qa_tier', 'db_aerosol_type') else 1e-6
            np.testing.assert_allclose(actual[name], ref[name], rtol=0, atol=tolerance, equal_nan=True)
