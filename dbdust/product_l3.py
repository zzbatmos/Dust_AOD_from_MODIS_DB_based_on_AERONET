"""Streaming retrieval-count monthly products on the configured global grids.

Use ``MonthlyAccumulator(grid_name='1x1').update(frame)`` for each granule or
chunk in ONE month, then ``to_dataset(month, provenance)`` or
``write(path, month, provenance)``. Inputs obey product_l2's pixel-frame contract;
DAOD inputs are already optical depths. The caller assigns retrievals to months.
State is O(grid cells), scratch space O(configured batch size + grid cells),
independent of the number of granules. No input frames are retained.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import CONFIG
from .product_l2 import _atomic_path, _attrs, _chunks, _coordinate_attrs, _fill


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator,
                     out=np.full(denominator.shape, _fill(), dtype=float),
                     where=denominator > 0)


def _cf_count(values, name):
    """CF 1.8 permits int32 counts; refuse narrowing that would lose data."""
    limits = np.iinfo(np.int32)
    if np.any(values < limits.min) or np.any(values > limits.max):
        raise OverflowError(f'{name} exceeds the CF 1.8 int32 count range')
    return values.astype(np.int32)


def _month_bounds(month):
    text = str(month)
    if text.isdigit() and len(text) == len('YYYYMM'):
        text = text[:len('YYYY')] + '-' + text[len('YYYY'):]
    try:
        period = pd.Period(text, freq='M')
        if pd.isna(period):
            raise ValueError('missing month')
        return period.start_time.to_datetime64(), (period + 1).start_time.to_datetime64()
    except (TypeError, ValueError) as exc:
        raise ValueError('month must identify a calendar month, e.g. 2017-01 or 201701') from exc


class MonthlyAccumulator:
    """Fixed-size float64 sums and int64 counts, with no equal-day weighting.

    Counts are true zeros in empty cells. All n_ml-denominator statistics are
    missing where n_ml is zero. frac_ml_missing is missing only where
    n_valid_aod is zero, and is one in valid-AOD-only cells. Supported branches
    must have complete finite predictions; invalid inputs are screened exactly
    as configured. A failed update can have applied earlier valid chunks; do
    not reuse the accumulator after an update raises an exception.
    """

    def __init__(self, grid_name='1x1'):
        if grid_name not in CONFIG['grids']:
            raise ValueError(f'Unknown grid {grid_name!r}; choose from {tuple(CONFIG["grids"])}')
        self.grid_name = grid_name
        self.dy, self.dx = CONFIG['grids'][grid_name]
        self.lat_min, self.lat_max = CONFIG['inputs']['latitude_bounds']
        self.lon_min, self.lon_max = CONFIG['inputs']['longitude_bounds']
        self.lat_edges = np.arange(self.lat_min, self.lat_max + self.dy, self.dy, dtype=float)
        self.lon_edges = np.arange(self.lon_min, self.lon_max + self.dx, self.dx, dtype=float)
        self.shape = (len(self.lat_edges) - 1, len(self.lon_edges) - 1)
        self._size = int(np.prod(self.shape))
        counts = ['n_valid_aod', 'n_ml', *('n_' + name for name in CONFIG['branches']),
                  *('frac_qa' + str(i) for i, _ in enumerate(CONFIG['qa']['thresholds'], start=1))]
        self._counts = {name: np.zeros(self._size, dtype=np.int64) for name in counts}
        sums = ['daod550_ev', 'daod550_qa2', 'aod550_db', 'dust_probability_mean',
                'ev_contribution_aod_lt_0p2', 'ev_contribution_p_lt_0p3']
        self._sums = {name: np.zeros(self._size, dtype=np.float64) for name in sums}

    def update(self, frame):
        """Accumulate a pandas pixel frame in bounded chunks; return self."""
        for _, (values, valid, supported) in _chunks(frame):
            positions = np.flatnonzero(valid)
            lat = values['lat'][positions]
            lon = (values['lon'][positions] - self.lon_min) % (self.lon_max - self.lon_min)
            row = np.floor((lat - self.lat_min) / self.dy).astype(np.intp)
            # If the configured latitude upper endpoint is inclusive, put it
            # in the last cell; longitude's upper endpoint wraps to the first.
            row = np.minimum(row, self.shape[0] - 1)
            col = np.floor(lon / self.dx).astype(np.intp)
            cells = row * self.shape[1] + col
            self._counts['n_valid_aod'] += np.bincount(cells, minlength=self._size)
            ml = supported[positions]
            cells = cells[ml]
            positions = positions[ml]
            self._counts['n_ml'] += np.bincount(cells, minlength=self._size)
            for name, branch in CONFIG['branches'].items():
                take = values['branch'][positions] == branch['code']
                self._counts['n_' + name] += np.bincount(cells[take], minlength=self._size)
            p = values['dust_probability'][positions]
            for i, threshold in enumerate(CONFIG['qa']['thresholds'], start=1):
                self._counts['frac_qa' + str(i)] += np.bincount(cells[p >= threshold],
                                                              minlength=self._size)
            aod, ev = (values[name][positions] for name in ('aod550_db', 'daod550_ev'))
            weights = {
                'daod550_ev': ev,
                'daod550_qa2': values['daod550_qa2'][positions],
                'aod550_db': aod,
                'dust_probability_mean': p,
                'ev_contribution_aod_lt_0p2': np.where(aod < CONFIG['diagnostics']['low_aod_threshold'], ev, 0),
                'ev_contribution_p_lt_0p3': np.where(p < CONFIG['diagnostics']['low_probability_threshold'], ev, 0),
            }
            for name, weight in weights.items():
                self._sums[name] += np.bincount(cells, weights=weight, minlength=self._size)
        return self

    def to_dataset(self, month, provenance=None, **provenance_fields):
        """Return an independent CF-described monthly Dataset (time, lat, lon).

        month accepts YYYYMM, YYYY-MM, or a datetime-like value. Time identifies
        the interval midpoint and has bounds [month start, next month start).
        Provenance accepts commit/source_code_commit and model_manifest_hash.
        Count snapshots are int32 for CF 1.8, while internal counts stay int64.
        Raises OverflowError before serialization if any count cannot fit int32.
        """
        start, end = _month_bounds(month)
        midpoint = start + (end - start) // (1 + 1)
        lat = (self.lat_edges[:-1] + self.lat_edges[1:]) / (1 + 1)
        lon = (self.lon_edges[:-1] + self.lon_edges[1:]) / (1 + 1)
        coords = _coordinate_attrs()
        for name, axis in (('lat', 'Y'), ('lon', 'X')):
            coords[name].update(axis=axis, bounds=name + '_bounds')
        attrs = _attrs(provenance, **provenance_fields)
        attrs.update(product_level='L3M', grid_name=self.grid_name,
                     weighting=CONFIG['aggregation']['weighting'],
                     aggregation='Arithmetic mean over supported retrievals; not equal-day weighting.',
                     missing_conventions=attrs['missing_conventions'] +
                     ' Empty-cell counts are zero. Zero denominators give missing statistics.',
                     time_coverage_start=str(start), time_coverage_end=str(end))
        ds = xr.Dataset(coords={'lat': ('lat', lat, coords['lat']),
                                'lon': ('lon', lon, coords['lon']),
                                'time': ('time', [midpoint], {'standard_name': 'time', 'axis': 'T',
                                                            'bounds': 'time_bounds'})}, attrs=attrs)
        ds['lat_bounds'] = (('lat', 'bounds'), np.column_stack((self.lat_edges[:-1], self.lat_edges[1:])))
        ds['lon_bounds'] = (('lon', 'bounds'), np.column_stack((self.lon_edges[:-1], self.lon_edges[1:])))
        ds['time_bounds'] = (('time', 'bounds'), np.asarray([[start, end]]))
        n_ml, n_valid = self._counts['n_ml'], self._counts['n_valid_aod']
        fields = {name: _ratio(total, n_ml) for name, total in self._sums.items()}
        fields.update({name: _cf_count(value, name) if name.startswith('n_') else _ratio(value, n_ml)
                       for name, value in self._counts.items()})
        fields['frac_full8'] = _ratio(self._counts['n_full8'], n_ml)
        fields['frac_ml_missing'] = _ratio(n_valid - n_ml, n_valid)
        for name in CONFIG['outputs']['monthly']:
            is_count = name.startswith('n_')
            metadata = {'long_name': name.replace('_', ' '), 'units': CONFIG['outputs']['units']}
            if not is_count:
                metadata['denominator'] = 'n_valid_aod' if name == 'frac_ml_missing' else 'n_ml'
                metadata['cell_methods'] = 'time: mean (weighted by retrieval count)'
            if name in self._sums and name.startswith('ev_contribution'):
                metadata['comment'] = 'Subset DAOD sum divided by full supported retrieval count, not subset count.'
                metadata['subset_threshold'] = CONFIG['diagnostics'][
                    'low_aod_threshold' if 'aod_lt' in name else 'low_probability_threshold']
            if name.startswith('frac_qa'):
                metadata['probability_threshold'] = CONFIG['qa']['thresholds'][int(name[-1]) - 1]
            if name in ('daod550_ev', 'daod550_qa2', 'aod550_db'):
                metadata['wavelength_nm'] = CONFIG['reference']['output_wavelength_nm']
            ds[name] = (('time', 'lat', 'lon'), fields[name].reshape((1, *self.shape)), metadata)
            ds[name].encoding.update(_FillValue=None if is_count else _fill(), zlib=True)
        for name in ('lat', 'lon', 'lat_bounds', 'lon_bounds'):
            ds[name].encoding['_FillValue'] = None
        for name in ('time', 'time_bounds'):
            ds[name].encoding.update(units='days since ' + str(start).split('T')[0],
                                     calendar='proleptic_gregorian', _FillValue=None, dtype='float64')
        return ds

    def write(self, path, month, provenance=None, **provenance_fields):
        """Atomically write to the caller's arbitrary path and return that Path."""
        with self.to_dataset(month, provenance, **provenance_fields) as ds:
            with _atomic_path(path) as temporary:
                ds.to_netcdf(temporary, engine='netcdf4', format='NETCDF4')
        return Path(path)


def monthly_filename(month, grid_name='1x1'):
    """Suggested release filename; writers never override the supplied path."""
    if grid_name not in CONFIG['grids']:
        raise ValueError(f'Unknown grid {grid_name!r}')
    start, _ = _month_bounds(month)
    return f"{CONFIG['outputs']['product_prefix']}_{grid_name}_{pd.Timestamp(start):%Y%m}.nc"
