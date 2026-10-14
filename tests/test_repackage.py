"""Converter gates reject stale evidence and mislabeled grids before output."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from dbdust.config import CONFIG, config_path
from dbdust.predictor import file_hash, verify_manifest
from dbdust.product_l3 import MonthlyAccumulator


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repackage', ROOT/'tools/repackage_stage_c.py')
repackage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repackage)


def valid_report():
    return dict(reproduction_status='PASS', source_commit='verified-commit',
        config_sha256=file_hash(config_path()), model_manifest_sha256=verify_manifest(ROOT/'models/v2.0.0'),
        converter_sha256=file_hash(ROOT/'tools/repackage_stage_c.py'),
        code_sha256={f'dbdust/{p.name}':file_hash(p) for p in (ROOT/'dbdust').glob('*.py')})


def test_matching_acceptance():
    repackage.validate_acceptance(valid_report(), 'verified-commit', ROOT/'models/v2.0.0')


@pytest.mark.parametrize('field', ['reproduction_status', 'source_commit', 'config_sha256',
                                  'model_manifest_sha256', 'code_sha256', 'converter_sha256'])
def test_stale_acceptance_rejected(field):
    report = valid_report()
    report[field] = 'incorrect'
    with pytest.raises(ValueError, match=field):
        repackage.validate_acceptance(report, 'verified-commit', ROOT/'models/v2.0.0')
    del report[field]
    with pytest.raises(ValueError, match=field):
        repackage.validate_acceptance(report, 'verified-commit', ROOT/'models/v2.0.0')


@pytest.mark.parametrize('error', ['descending_latitude', 'shifted_longitude', 'transposed_statistic'])
def test_grid_errors_rejected_before_write(tmp_path, monkeypatch, error):
    source = tmp_path/'source.nc'
    source.write_bytes(b'mocked-source')
    source.with_suffix('.sha256.json').write_text('{"sha256":"mocked", "fingerprint":"f"}')
    monkeypatch.setattr(repackage, 'file_hash', lambda path: 'mocked')
    template = MonthlyAccumulator('2x5').to_dataset('2017-01')
    lat = template.lat.values.copy()
    lon = template.lon.values.copy()
    if error == 'descending_latitude':
        lat = lat[::-1]
    if error == 'shifted_longitude':
        lon += 1
    root = xr.Dataset(coords={'lat':lat, 'lon':lon}, attrs={'month':'2017-01','fingerprint':'f'})
    fields = ['valid_aod_count', 'p_count', 'full8_count', 'noSSA5_count',
              'gate04_count', 'gate06_count', 'gate07_count', 'own__Eselected__sum',
              'own__E1__sum', 'aod_sum', 'p_sum', 'own__Eselected__low_aod_sum',
              'own__Eselected__low_p_sum', 'none_count', 'minimal2_count',
              'ml_missing_count', 'own__Eselected__count', 'own__E1__count']
    group = xr.Dataset({name: (('lat','lon'), np.zeros((len(lat),len(lon)))) for name in fields})
    group['valid_aod_count'] = (('lon','lat'), np.zeros((len(lon),len(lat))))
    monkeypatch.setattr(repackage.xr, 'open_dataset', lambda p, **kw: group if kw else root)
    writer = MagicMock()
    monkeypatch.setattr(repackage.MonthlyAccumulator, 'write', writer)
    with pytest.raises(ValueError, match='coordinate/order|dimension order'):
        repackage.convert(source, tmp_path/'output', '2017-01','2x5','commit','f', ROOT/'models/v2.0.0')
    writer.assert_not_called()
