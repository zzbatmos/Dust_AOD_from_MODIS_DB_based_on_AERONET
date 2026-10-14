"""Synthetic product tests: no artifacts, fitting, data access, or downloads."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from netCDF4 import Dataset

from dbdust.config import CONFIG
from dbdust.product_l2 import to_dataset, write_l2
from dbdust.product_l3 import MonthlyAccumulator, monthly_filename


def pixels(aod=(0.1, 0.2, 0.6), probability=(0.2, 0.6, 0.7), branch=None):
    aod = np.asarray(aod, dtype=float)
    p = np.asarray(probability, dtype=float)
    if branch is None:
        branch = np.full(len(aod), CONFIG['branches']['full8']['code'])
    conditional = np.full(len(aod), 0.5)
    gate = CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']]
    return pd.DataFrame({
        'lat': np.full(len(aod), 0.25), 'lon': np.full(len(aod), 0.25),
        'branch': branch, 'dust_probability': p,
        'qa_tier': np.searchsorted(CONFIG['qa']['thresholds'], p, side='right'),
        'dust_fraction_conditional': conditional,
        'daod550_qa2': np.where(p >= gate, aod * conditional, 0),
        'daod550_ev': aod * 0.25, 'aod550_db': aod,
        'db_aerosol_type': np.ones(len(aod)),
    })


def cell(ds, lat=0.25, lon=0.25):
    return ds.sel(lat=lat, lon=lon, method='nearest').isel(time=0)


@pytest.mark.parametrize('grid_name', list(CONFIG['grids']))
def test_month_sums_counts_and_missing(grid_name):
    frame = pixels(branch=[1, 2, 1])
    unsupported = pixels(aod=[10.], probability=[0.9], branch=[0])
    unsupported_only = unsupported.assign(lat=10.25, lon=10.25)
    invalid = pixels(aod=[np.nan, -1., np.inf], probability=[0.7] * 3)
    invalid['branch'] = CONFIG['qa']['missing_tier']
    accumulator = MonthlyAccumulator(grid_name)
    # Unequal granule sizes distinguish retrieval means from equal-granule means.
    accumulator.update(frame.iloc[:1]).update(frame.iloc[1:])
    accumulator.update(unsupported).update(unsupported_only).update(invalid)
    ds = accumulator.to_dataset('2017-01', {'commit': 'abc', 'model_manifest_hash': 'def'})
    assert set(CONFIG['outputs']['monthly']) <= set(ds.data_vars)
    got = cell(ds)
    assert got.n_valid_aod == 4
    assert got.n_ml == 3
    assert got.n_full8 == 2
    assert got.n_noSSA5 == 1
    assert float(got.frac_full8) == pytest.approx(2 / 3)
    assert float(got.frac_ml_missing) == pytest.approx(1 / 4)
    for name in ('daod550_ev', 'daod550_qa2', 'aod550_db'):
        assert float(got[name]) == pytest.approx(frame[name].sum() / len(frame))
    assert float(got.dust_probability_mean) == pytest.approx(frame.dust_probability.mean())
    assert float(got.frac_qa1) == pytest.approx(2 / 3)
    assert float(got.frac_qa2) == pytest.approx(2 / 3)
    assert float(got.frac_qa3) == pytest.approx(1 / 3)
    assert float(got.ev_contribution_aod_lt_0p2) == pytest.approx(0.025 / 3)
    assert float(got.ev_contribution_p_lt_0p3) == pytest.approx(0.025 / 3)
    unsupported_cell = cell(ds, 10.25, 10.25)
    assert unsupported_cell.n_valid_aod == 1
    assert unsupported_cell.n_ml == 0
    assert unsupported_cell.frac_ml_missing == 1
    for name in CONFIG['outputs']['monthly']:
        if not name.startswith('n_') and name != 'frac_ml_missing':
            assert np.isnan(unsupported_cell[name])
    empty_cell = cell(ds, -40., -40.)
    for name in CONFIG['outputs']['monthly']:
        if name.startswith('n_'):
            assert empty_cell[name] == 0
        else:
            assert np.isnan(empty_cell[name])
    assert (ds.n_ml <= ds.n_valid_aod).all()
    assert (ds.n_full8 + ds.n_noSSA5 == ds.n_ml).all()
    supported = ds.n_ml > 0
    for name in ('daod550_ev', 'daod550_qa2'):
        assert (ds[name].where(supported, 0) >= 0).all()
        assert (ds[name].where(supported, 0) <= ds.aod550_db.where(supported, 0)).all()
    assert ds.attrs['source_code_commit'] == 'abc'
    assert ds.attrs['model_manifest_hash'] == 'def'
    assert ds.attrs['weighting'] == CONFIG['aggregation']['weighting']


def test_qa_zero_is_not_missing():
    frame = pixels(aod=[0., 0.1], probability=[0.1, 0.2])
    ds = MonthlyAccumulator().update(frame).to_dataset('201701')
    got = cell(ds)
    assert got.n_ml == 2
    assert got.daod550_qa2 == 0
    assert np.isfinite(got.daod550_ev)
    assert cell(ds, 5., 5.).daod550_qa2.isnull()


def test_all_thresholds_and_subset_boundaries():
    p = np.asarray([CONFIG['diagnostics']['low_probability_threshold'],
                    *CONFIG['qa']['thresholds']])
    aod = np.full(len(p), CONFIG['diagnostics']['low_aod_threshold'])
    ds = MonthlyAccumulator().update(pixels(aod, p)).to_dataset('2017-01')
    got = cell(ds)
    assert got.ev_contribution_aod_lt_0p2 == 0
    assert got.ev_contribution_p_lt_0p3 == 0
    for i, threshold in enumerate(CONFIG['qa']['thresholds'], start=1):
        assert float(got['frac_qa' + str(i)]) == pytest.approx(np.mean(p >= threshold))


@pytest.mark.parametrize('grid_name', list(CONFIG['grids']))
def test_geolocation_edges_and_invalids(grid_name):
    frame = pixels(aod=np.ones(8), probability=np.full(8, 0.7))
    frame['lat'] = [-90., -90., 90., -90.01, 0., np.nan, 0., 0.]
    frame['lon'] = [-180., 180., 0., 0., 180.01, 0., np.inf, np.nan]
    ds = MonthlyAccumulator(grid_name).update(frame).to_dataset('201701')
    assert ds.n_valid_aod.sum() == 2
    assert ds.n_ml.sum() == 2
    assert ds.n_ml.isel(time=0, lat=0, lon=0) == 2
    dy, dx = CONFIG['grids'][grid_name]
    assert ds.lat[0] == -90 + dy / 2
    assert ds.lat[-1] == 90 - dy / 2
    assert ds.lon[0] == -180 + dx / 2
    assert ds.lon[-1] == 180 - dx / 2


def test_l2_roundtrip_flags_missing_and_arbitrary_path(tmp_path):
    frame = pixels(aod=[0.1, 0.2, 0.4, np.nan, -1., np.inf, 0.4],
                   probability=[0.2, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7],
                   branch=[1, 2, 0, -1, -1, -1, 1])
    frame.loc[6, 'lat'] = np.nan
    original = frame.copy(deep=True)
    path = tmp_path / 'arbitrary-pixels.extension'
    assert write_l2(frame, path, commit='abc', model_manifest_hash='def') == path
    pd.testing.assert_frame_equal(frame, original)
    with xr.open_dataset(path) as ds:
        assert ds.sizes['pixel'] == len(frame)
        assert set(CONFIG['outputs']['pixel']) == set(ds.data_vars)
        assert ds.branch[0] == 1
        assert ds.branch[1] == 2
        assert ds.branch[2] == 0
        assert ds.aod550_db[2] == frame.aod550_db[2]
        assert ds.db_aerosol_type[2] == frame.db_aerosol_type[2]
        for name in ('dust_probability', 'qa_tier', 'dust_fraction_conditional', 'daod550_qa2', 'daod550_ev'):
            assert np.isnan(ds[name][2])
        assert ds.daod550_qa2[0] == 0
        assert ds.daod550_ev[0] == frame.daod550_ev[0]
        for name in CONFIG['outputs']['pixel']:
            assert ds[name][3:].isnull().all()
        assert ds.lat.attrs['standard_name'] == 'latitude'
        assert ds.lon.attrs['units'] == 'degrees_east'
        assert ds.branch.attrs['flag_meanings'] == 'none full8 noSSA5'
        assert ds.attrs['source_code_commit'] == 'abc'
        assert ds.attrs['model_manifest_hash'] == 'def'
    with Dataset(path) as nc:
        for name in CONFIG['outputs']['pixel']:
            assert '_FillValue' in nc[name].ncattrs()
        assert nc['branch'].dtype == np.dtype('int8')
        assert nc['qa_tier']._FillValue == CONFIG['qa']['missing_tier']
        assert nc['daod550_ev'].coordinates == 'lat lon'


@pytest.mark.parametrize('grid_name', list(CONFIG['grids']))
def test_l3_netcdf_cf_metadata_and_time(tmp_path, grid_name):
    accumulator = MonthlyAccumulator(grid_name).update(pixels())
    path = tmp_path / 'any-month-file.nc'
    accumulator.write(path, '202002', {'commit': 'abc', 'model_manifest_hash': 'def'})
    with xr.open_dataset(path) as ds:
        assert ds.time_bounds.values[0, 0] == np.datetime64('2020-02-01')
        assert ds.time_bounds.values[0, 1] == np.datetime64('2020-03-01')
        assert ds.time_bounds.values[0, 0] < ds.time.values[0] < ds.time_bounds.values[0, 1]
        assert ds.time.attrs['bounds'] == 'time_bounds'
        assert ds.lat.attrs['bounds'] == 'lat_bounds'
        assert ds.lon.attrs['bounds'] == 'lon_bounds'
        assert ds.attrs['Conventions'].startswith('CF-')
        assert ds.attrs['source_code_commit'] == 'abc'
        assert ds.attrs['model_manifest_hash'] == 'def'
        for name in CONFIG['outputs']['monthly']:
            if name.startswith('n_'):
                assert ds[name].dtype == np.dtype('int32')
        assert ds.n_ml.sum() == 3
        for name in CONFIG['outputs']['monthly']:
            assert ds[name].attrs['units'] == CONFIG['outputs']['units']
        for name in ('daod550_ev', 'daod550_qa2', 'aod550_db'):
            assert ds[name].attrs['cell_methods'].startswith('time: mean')
            assert ds[name].where(ds.n_ml == 0).isnull().all()
    with Dataset(path) as nc:
        # CF bounds inherit these attributes; xarray removes redundant copies.
        for attribute in ('calendar', 'units'):
            parent = nc['time'].getncattr(attribute)
            assert getattr(nc['time_bounds'], attribute, parent) == parent
        assert '_FillValue' not in nc['n_ml'].ncattrs()
        assert np.isnan(nc['daod550_ev']._FillValue)
        for name in CONFIG['outputs']['monthly']:
            if name.startswith('n_'):
                assert nc[name].dtype == np.dtype('int32')
    assert all(value.dtype == np.dtype('int64') for value in accumulator._counts.values())
    assert monthly_filename('2020-02', grid_name) == f"{CONFIG['outputs']['product_prefix']}_{grid_name}_202002.nc"


def test_chunked_and_unpartitioned_agree(monkeypatch, tmp_path):
    frame = pd.concat([pixels()] * 5, ignore_index=True)
    original = MonthlyAccumulator('2x5').update(frame).to_dataset('201701')
    monkeypatch.setitem(CONFIG['runtime'], 'batch_size', 2)
    chunked = MonthlyAccumulator('2x5')
    for offset in range(0, len(frame), 4):
        chunked.update(frame.iloc[offset:offset + 4])
    xr.testing.assert_allclose(original, chunked.to_dataset('201701'))
    path = tmp_path / 'chunked.nc'
    write_l2(frame, path)
    with xr.open_dataset(path) as ds:
        for name in CONFIG['outputs']['pixel']:
            np.testing.assert_allclose(ds[name], frame[name])
    # Snapshot must not alias mutable accumulator state.
    snapshot = chunked.to_dataset('201701')
    chunked.update(frame)
    assert snapshot.n_ml.sum() == len(frame)


def test_empty_frames_and_nullable_values(tmp_path):
    empty = pixels().iloc[:0]
    ds = MonthlyAccumulator().update(empty).to_dataset('201701')
    assert ds.n_ml.sum() == 0
    assert ds.daod550_ev.isnull().all()
    path = tmp_path / 'empty.nc'
    write_l2(empty, path)
    with xr.open_dataset(path) as product:
        assert product.sizes['pixel'] == 0
    frame = pixels(aod=[1.], probability=[0.1], branch=[0]).astype('Float64')
    frame.loc[0, 'db_aerosol_type'] = pd.NA
    frame.loc[0, 'dust_probability'] = pd.NA
    product = to_dataset(frame)
    assert product.aod550_db[0] == 1
    assert product.dust_probability.isnull().all()
    assert product.db_aerosol_type[0] == CONFIG['qa']['missing_tier']


@pytest.mark.parametrize(('column', 'value', 'message'), [
    ('branch', 99, 'branch'),
    ('dust_probability', np.nan, 'finite'),
    ('dust_probability', 1.1, 'range'),
    ('dust_fraction_conditional', np.inf, 'finite'),
    ('dust_fraction_conditional', -0.1, 'range'),
    ('daod550_ev', 9., 'range'),
    ('daod550_qa2', 0.01, 'rejections'),
    ('qa_tier', -1, 'qa_tier'),
    ('db_aerosol_type', 0.5, 'aerosol_type'),
    ('lat', [1., 2.], 'scalar numeric'),
])
def test_bad_supported_input_fails(column, value, message, tmp_path):
    frame = pixels(aod=[0.1], probability=[0.2])
    if isinstance(value, list):
        frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    with pytest.raises(ValueError, match=message):
        MonthlyAccumulator().update(frame)
    path = tmp_path / 'bad.nc'
    with pytest.raises(ValueError, match=message):
        write_l2(frame, path)
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_atomic_l2_preserves_existing_file_on_late_error(monkeypatch, tmp_path):
    path = tmp_path / 'pixels.nc'
    write_l2(pixels(), path)
    before = path.read_bytes()
    monkeypatch.setitem(CONFIG['runtime'], 'batch_size', 1)
    bad = pixels()
    bad.loc[2, 'branch'] = 99
    with pytest.raises(ValueError):
        write_l2(bad, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_contract_errors():
    with pytest.raises(ValueError, match='Missing pixel columns'):
        MonthlyAccumulator().update(pixels().drop(columns='lat'))
    with pytest.raises(ValueError, match='unique'):
        to_dataset(pd.concat([pixels(), pixels()[['lat']]], axis=1))
    with pytest.raises(ValueError, match='Unknown grid'):
        MonthlyAccumulator('unsupported')
    with pytest.raises(ValueError, match='month'):
        MonthlyAccumulator().to_dataset('not-a-month')
    with pytest.raises(ValueError, match='month'):
        MonthlyAccumulator().to_dataset('NaT')


def test_config_controls_thresholds_and_grid(monkeypatch):
    # Deliberately vary constants to catch copied scientific literals.
    config = deepcopy(CONFIG)
    config['qa']['thresholds'] = [0.1, 0.3, 0.8]
    config['diagnostics']['low_aod_threshold'] = 0.5
    config['diagnostics']['low_probability_threshold'] = 0.5
    config['grids']['test'] = [10, 20]
    for key in ('qa', 'diagnostics', 'grids'):
        monkeypatch.setitem(CONFIG, key, config[key])
    frame = pixels(aod=[0.4], probability=[0.4])
    ds = MonthlyAccumulator('test').update(frame).to_dataset('201701')
    got = cell(ds)
    assert ds.sizes['lat'] == 18
    assert ds.sizes['lon'] == 18
    assert got.frac_qa2 == 1
    assert got.frac_qa3 == 0
    assert got.ev_contribution_aod_lt_0p2 == got.daod550_ev
    assert got.ev_contribution_p_lt_0p3 == got.daod550_ev


def test_bounded_scratch_and_fixed_accumulator_state(monkeypatch, tmp_path):
    import dbdust.product_l2 as product_l2

    monkeypatch.setitem(CONFIG['runtime'], 'batch_size', 2)
    normalise = product_l2._normalise
    observed_sizes = []

    def track_chunk(frame):
        observed_sizes.append(len(frame))
        return normalise(frame)

    monkeypatch.setattr(product_l2, '_normalise', track_chunk)
    accumulator = MonthlyAccumulator('2x5')
    state = [*accumulator._counts.values(), *accumulator._sums.values()]
    identities = [id(value) for value in state]
    bytes_before = sum(value.nbytes for value in state)
    frame = pd.concat([pixels()] * 10, ignore_index=True)
    for _ in range(3):
        accumulator.update(frame)
    write_l2(frame, tmp_path / 'bounded.nc')
    assert observed_sizes and max(observed_sizes) <= CONFIG['runtime']['batch_size']
    state_after = [*accumulator._counts.values(), *accumulator._sums.values()]
    assert [id(value) for value in state_after] == identities
    assert sum(value.nbytes for value in state_after) == bytes_before
    assert not any(isinstance(value, pd.DataFrame) for value in vars(accumulator).values())
    assert accumulator.to_dataset('201701').n_ml.sum() == len(frame) * 3


def test_l2_in_memory_dataset_serialization_matches_stream(tmp_path):
    frame = pixels(branch=[1, 2, 0])
    frame.loc[2, 'db_aerosol_type'] = np.nan
    in_memory_path = tmp_path / 'memory.nc'
    streamed_path = tmp_path / 'stream.nc'
    to_dataset(frame, commit='abc').to_netcdf(in_memory_path, engine='netcdf4')
    write_l2(frame, streamed_path, commit='abc')
    with xr.open_dataset(in_memory_path) as memory, xr.open_dataset(streamed_path) as stream:
        xr.testing.assert_identical(memory, stream)


@pytest.mark.parametrize('name', [name for name in CONFIG['outputs']['monthly'] if name.startswith('n_')])
@pytest.mark.parametrize('outside', [int(np.iinfo(np.int32).max) + 1,
                                   int(np.iinfo(np.int32).min) - 1])
def test_count_overflow_refused_before_writing(tmp_path, name, outside):
    accumulator = MonthlyAccumulator('2x5').update(pixels())
    path = tmp_path / 'month.nc'
    accumulator.write(path, '201701')
    before = path.read_bytes()
    accumulator._counts[name][0] = outside
    with pytest.raises(OverflowError, match=name + '.*int32'):
        accumulator.to_dataset('201701')
    with pytest.raises(OverflowError, match=name + '.*int32'):
        accumulator.write(path, '201701')
    assert path.read_bytes() == before
    missing_path = tmp_path / 'not-created.nc'
    with pytest.raises(OverflowError, match=name + '.*int32'):
        accumulator.write(missing_path, '201701')
    assert list(tmp_path.iterdir()) == [path]
    assert accumulator._counts[name].dtype == np.dtype('int64')
    assert accumulator._counts[name][0] == outside


def test_count_int32_max_roundtrips_without_changing_internal_state(tmp_path):
    accumulator = MonthlyAccumulator('2x5')
    maximum = np.iinfo(np.int32).max
    accumulator._counts['n_valid_aod'][0] = maximum
    ds = accumulator.to_dataset('201701')
    assert ds.n_valid_aod.dtype == np.dtype('int32')
    assert ds.n_valid_aod.values.flat[0] == maximum
    path = tmp_path / 'max-count.nc'
    accumulator.write(path, '201701')
    with xr.open_dataset(path) as written:
        assert written.n_valid_aod.values.flat[0] == maximum
    assert accumulator._counts['n_valid_aod'].dtype == np.dtype('int64')
    assert accumulator._counts['n_valid_aod'][0] == maximum


@pytest.mark.parametrize('provenance', [None, {'commit': 'abc', 'model_manifest_hash': 'def'},
                                      {'title': '', 'history': '  '}])
def test_deterministic_nonempty_cf_global_metadata(tmp_path, provenance):
    frame = pixels()
    accumulator = MonthlyAccumulator('2x5').update(frame)
    pixel_ds = to_dataset(frame, provenance)
    monthly_ds = accumulator.to_dataset('201701', provenance)
    for key in ('title', 'history'):
        assert isinstance(pixel_ds.attrs[key], str) and pixel_ds.attrs[key].strip()
        assert pixel_ds.attrs[key] == to_dataset(frame, provenance).attrs[key]
        assert monthly_ds.attrs[key] == accumulator.to_dataset('201701', provenance).attrs[key]
    for ds in (pixel_ds, monthly_ds):
        assert ds.attrs['source_code_commit'] in ds.attrs['history']
        assert ds.attrs['model_manifest_hash'] in ds.attrs['history']
    pixel_path, monthly_path = tmp_path / 'pixel.nc', tmp_path / 'month.nc'
    write_l2(frame, pixel_path, provenance)
    accumulator.write(monthly_path, '201701', provenance)
    for path, expected in ((pixel_path, pixel_ds), (monthly_path, monthly_ds)):
        with Dataset(path) as nc:
            assert nc.title == expected.attrs['title']
            assert nc.history == expected.attrs['history']
