"""Pixel products from predictor frames; no inference or DAOD rescaling.

``write_l2(frame, path, provenance=None, **provenance_fields)`` streams a
pandas frame to an arbitrary NetCDF path. Required columns are ``lat``, ``lon``
and CONFIG['outputs']['pixel']; each row is one pixel, independent of its index.
Coordinates and outputs must be scalar numeric values. Invalid AOD/geolocation
is missing throughout; valid branch-zero rows retain AOD and aerosol type only.
Malformed *supported* predictions raise ValueError, rather than creating a new
scientific support mask. Memory beyond the input is bounded by runtime.batch_size.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import xarray as xr

from .config import CONFIG


def _fill():
    return float(CONFIG['outputs']['missing_float'])


def _attrs(provenance=None, **fields):
    """Shared product provenance; supplied fields cannot replace definitions."""
    supplied = dict(provenance or {})
    supplied.update(fields)
    if 'commit' in supplied:
        supplied.setdefault('source_code_commit', supplied['commit'])
    attrs = {
        'source_code_commit': 'unknown',
        'model_manifest_hash': 'unknown',
    }
    for key, value in supplied.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        elif isinstance(value, (bool, np.bool_)):
            value = str(value).lower()
        elif not isinstance(value, (str, int, float, np.number)):
            value = str(value)
        attrs[key] = value
    attrs.update({
        'Conventions': 'CF-1.8',
        'version': CONFIG['version'],
        'title': f"DBDust v{CONFIG['version']} dust aerosol optical depth product",
        'history': f"Packaged with DBDust v{CONFIG['version']} from predictor outputs; "
                   f"source_code_commit={attrs['source_code_commit']}; "
                   f"model_manifest_hash={attrs['model_manifest_hash']}",
        'reference_definition': CONFIG['reference']['name'],
        'reference_configuration': json.dumps(CONFIG['reference'], sort_keys=True),
        'branch_rule': 'first input-valid configured branch; 0 none; '
                       + json.dumps(CONFIG['branches'], sort_keys=True),
        'screening': json.dumps(CONFIG['inputs'], sort_keys=True),
        'missing_conventions': 'Invalid AOD/geolocation: all outputs missing; '
                               'branch 0: ML outputs missing, not zero; '
                               'valid QA-gate rejection: DAOD QA2 zero.',
    })
    return attrs


def _pixel_attrs():
    unit = CONFIG['outputs']['units']
    attrs = {
        name: {'long_name': name.replace('_', ' '), 'units': unit}
        for name in CONFIG['outputs']['pixel']
    }
    attrs['branch'].update(
        flag_values=np.asarray([0] + [b['code'] for b in CONFIG['branches'].values()],
                               dtype=np.int8),
        flag_meanings='none ' + ' '.join(CONFIG['branches']),
    )
    attrs['qa_tier'].update(
        flag_values=np.arange(len(CONFIG['qa']['thresholds']) + 1, dtype=np.int8),
        flag_meanings='below_qa1 qa1_only qa2_only qa3',
        thresholds=np.asarray(CONFIG['qa']['thresholds']),
    )
    attrs['db_aerosol_type'].update(
        flag_values=np.arange(CONFIG['inputs']['aerosol_type_mask'] + 1, dtype=np.int8),
        flag_meanings='mixed dust smoke sulfate',
    )
    attrs['dust_probability']['long_name'] = 'Raw classifier dust probability'
    attrs['dust_fraction_conditional']['long_name'] = 'Unsmeared positive-regressor dust fraction'
    attrs['daod550_ev']['long_name'] = 'Expected-value dust aerosol optical depth'
    attrs['daod550_qa2']['long_name'] = 'QA2-gated conditional dust aerosol optical depth'
    for name in ('aod550_db', 'daod550_ev', 'daod550_qa2'):
        attrs[name]['wavelength_nm'] = CONFIG['reference']['output_wavelength_nm']
    for name in ('dust_probability', 'dust_fraction_conditional'):
        attrs[name]['valid_range'] = np.asarray([0., 1.])
    return attrs


def _coordinate_attrs():
    return {
        'lat': {'standard_name': 'latitude', 'long_name': 'Latitude', 'units': 'degrees_north'},
        'lon': {'standard_name': 'longitude', 'long_name': 'Longitude', 'units': 'degrees_east'},
    }


def _check_columns(frame):
    required = ['lat', 'lon', *CONFIG['outputs']['pixel']]
    if not frame.columns.is_unique:
        raise ValueError('Pixel frame must have unique column names')
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing pixel columns: ' + ', '.join(sorted(missing)))
    return required


def _normalise(frame):
    """Validate one bounded chunk and canonicalize missing-value sentinels."""
    values = {}
    for name in _check_columns(frame):
        try:
            values[name] = frame[name].to_numpy(dtype=float, na_value=np.nan, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain scalar numeric values') from exc
    lat, lon, aod = (values[name] for name in ('lat', 'lon', 'aod550_db'))
    inputs = CONFIG['inputs']
    lat_min, lat_max = inputs['latitude_bounds']
    lon_min, lon_max = inputs['longitude_bounds']
    lat_ok = np.isfinite(lat) & (lat >= lat_min)
    lat_ok &= lat < lat_max if inputs['latitude_upper_exclusive'] else lat <= lat_max
    lon_ok = np.isfinite(lon) & (lon >= lon_min) & (lon <= lon_max)
    valid = lat_ok & lon_ok & np.isfinite(aod) & (aod >= inputs['minimum_aod'])
    branch = values['branch']
    codes = [b['code'] for b in CONFIG['branches'].values()]
    if np.any(valid & ~np.isin(branch, [0, *codes])):
        raise ValueError('Valid-AOD pixels require branch 0 or a configured branch code')
    supported = valid & np.isin(branch, codes)
    for name in ('dust_probability', 'dust_fraction_conditional', 'daod550_qa2', 'daod550_ev'):
        x = values[name]
        if np.any(supported & ~np.isfinite(x)):
            raise ValueError(f'Supported pixels require finite {name}')
        upper = aod if name.startswith('daod') else 1
        if np.any(supported & ((x < 0) | (x > upper))):
            raise ValueError(f'Supported {name} outside its physical range')
        x[~supported] = _fill()
    probability = values['dust_probability']
    tier = values['qa_tier']
    expected = np.searchsorted(CONFIG['qa']['thresholds'], probability, side='right')
    if np.any(supported & (tier != expected)):
        raise ValueError('qa_tier disagrees with raw probability thresholds')
    gate = CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']]
    if np.any(supported & (probability < gate) & (values['daod550_qa2'] != 0)):
        raise ValueError('Supported QA2 gate rejections must be zero')
    missing = CONFIG['qa']['missing_tier']
    branch[~valid] = missing
    tier[~supported] = missing
    aerosol = values['db_aerosol_type']
    aerosol_missing = ~np.isfinite(aerosol) | (aerosol == missing) | ~valid
    if np.any(~aerosol_missing & ((aerosol < 0) | (aerosol > inputs['aerosol_type_mask'])
                                 | (aerosol != np.floor(aerosol)))):
        raise ValueError('Unknown db_aerosol_type flag')
    aerosol[aerosol_missing] = missing
    aod[~valid] = _fill()
    lat[~lat_ok] = _fill()
    lon[~lon_ok] = _fill()
    for name in ('branch', 'qa_tier', 'db_aerosol_type'):
        values[name] = values[name].astype(np.int8)
    return values, valid, supported


def _chunks(frame):
    _check_columns(frame)
    size = CONFIG['runtime']['batch_size']
    for start in range(0, len(frame), size):
        yield start, _normalise(frame.iloc[start:start + size])


@contextmanager
def _atomic_path(path):
    """Do not leave a partial product or replace a good file on validation failure."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.nc',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def to_dataset(frame, provenance=None, **provenance_fields):
    """Return an in-memory pixel Dataset; use write_l2 for bounded-memory writing."""
    values, _, _ = _normalise(frame)
    attrs = _pixel_attrs()
    coords = {name: ('pixel', values.pop(name), meta)
              for name, meta in _coordinate_attrs().items()}
    ds = xr.Dataset({name: ('pixel', value, attrs[name]) for name, value in values.items()},
                    coords=coords, attrs=_attrs(provenance, **provenance_fields))
    ds.attrs['product_level'] = 'L2'
    for name in ds.variables:
        ds[name].encoding['_FillValue'] = (CONFIG['qa']['missing_tier']
                                           if ds[name].dtype.kind == 'i' else _fill())
    return ds


def write_l2(frame, path, provenance=None, **provenance_fields):
    """Write a pixel frame in bounded chunks, atomically, returning the Path.

    ``commit`` (also exposed as ``source_code_commit``) and
    ``model_manifest_hash`` may be supplied in provenance or as keyword fields.
    No row filtering or probability/fraction-to-DAOD multiplication is performed.
    """
    from netCDF4 import Dataset

    _check_columns(frame)
    metadata = {**_coordinate_attrs(), **_pixel_attrs()}
    flags = {'branch', 'qa_tier', 'db_aerosol_type'}
    with _atomic_path(path) as temporary:
        with Dataset(temporary, 'w', format='NETCDF4') as nc:
            nc.setncatts(_attrs(provenance, **provenance_fields))
            nc.product_level = 'L2'
            nc.createDimension('pixel', None)
            variables = {}
            for name, attrs in metadata.items():
                flag = name in flags
                var = nc.createVariable(name, 'i1' if flag else 'f8', ('pixel',),
                                        fill_value=CONFIG['qa']['missing_tier'] if flag else _fill(),
                                        zlib=True, chunksizes=(CONFIG['runtime']['batch_size'],))
                var.setncatts(attrs)
                if name not in _coordinate_attrs():
                    var.coordinates = 'lat lon'
                variables[name] = var
            for start, (values, _, _) in _chunks(frame):
                stop = start + len(values['lat'])
                for name, value in values.items():
                    variables[name][start:stop] = value
    return Path(path)


write_pixels = write_l2
write = write_l2
