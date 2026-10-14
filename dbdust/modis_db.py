"""Collection 6.1 land inputs, decoded exactly as the frozen application."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CONFIG


def decode_science(raw, attrs):
    raw = np.asarray(raw)
    values = raw.astype(float)
    bad = ~np.isfinite(values)
    if '_FillValue' in attrs:
        bad |= raw == attrs['_FillValue']
    if 'valid_range' in attrs:
        limits = np.asarray(attrs['valid_range'], dtype=float).ravel()
        bad |= (values < limits.min()) | (values > limits.max())
    offset = float(attrs.get('add_offset', 0))
    if offset != 0:
        raise ValueError('Nonzero HDF offset outside verified frozen input contract')
    values *= float(attrs.get('scale_factor', 1))
    values[bad] = np.nan
    return values


def band_first(array, shape, bands):
    array = np.asarray(array)
    if array.shape == (bands, *shape):
        return array
    if array.shape == (*shape, bands):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f'Unexpected band shape: {array.shape}')


def valid_geolocation(lat, lon):
    c = CONFIG['inputs']
    lo, hi = c['latitude_bounds']
    west, east = c['longitude_bounds']
    return (np.isfinite(lat) & np.isfinite(lon) & (lat >= lo) & (lat < hi)
            & (lon >= west) & (lon <= east))


def decode_aerosol_type(raw, shape):
    c = CONFIG['inputs']
    q = band_first(raw, shape, c['qa_byte_count'])[c['qa_byte_index']].astype(np.uint8)
    return (q >> c['aerosol_type_shift']) & c['aerosol_type_mask']


def read_granule(path):
    """Return flattened frame, swath shape, and source attributes.

    Keep invalid cells in swath order, with all predictor fields missing.
    This is equivalent to dropping these pixels for the L3 accumulator.
    """
    from pyhdf.SD import SD, SDC
    c = CONFIG['inputs']
    hdf = SD(str(path), SDC.READ)
    arrays, attrs = {}, {}
    try:
        for name in ('aod550', 'spectral_aod', 'ssa', 'ae', 'latitude', 'longitude'):
            sds = hdf.select(c[name])
            try:
                attrs[name] = sds.attributes()
                arrays[name] = decode_science(sds.get(), attrs[name])
            finally:
                sds.endaccess()
        shape = arrays['aod550'].shape
        if len(shape) != 2:
            raise ValueError('Expected a two-dimensional swath')
        for name in ('spectral_aod', 'ssa'):
            arrays[name] = band_first(arrays[name], shape, len(c['spectral_wavelengths_nm']))
        arrays['ssa'] = np.clip(arrays['ssa'], *c['ssa_bounds'])
        qa = hdf.select(c['qa'])
        try:
            aerosol_type = decode_aerosol_type(qa.get(), shape)
        finally:
            qa.endaccess()
    finally:
        hdf.end()
    lat, lon, aod = arrays['latitude'], arrays['longitude'], arrays['aod550']
    valid = valid_geolocation(lat, lon) & np.isfinite(aod) & (aod >= c['minimum_aod'])
    fields = dict(DB_AOD550=aod, DB_AE=arrays['ae'], db_aerosol_type=aerosol_type)
    for i, wave in enumerate(c['spectral_wavelengths_nm']):
        fields[f'DB_AOD{wave}'] = arrays['spectral_aod'][i]
        fields[f'DB_SSA{wave}'] = arrays['ssa'][i]
    frame = pd.DataFrame({key: np.where(valid, value, np.nan).ravel() for key,value in fields.items()})
    frame['lat'] = lat.ravel()
    frame['lon'] = lon.ravel()
    attrs['source_file'] = Path(path).name
    return frame, shape, attrs
