"""Synthetic native-grid caches; no original source execution or downloads."""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from paper import derive_seasonal_maps as seasonal
from paper.numbers import digest, ingest


PERIODS = {'DJF': ['2016-12', '2017-01', '2017-02'],
           'MAM': ['2017-03', '2017-04', '2017-05'],
           'JJA': ['2017-06', '2017-07', '2017-08'],
           'SON': ['2017-09', '2017-10', '2017-11']}
CORE = ['E1', 'Eselected', 'E5', 'E7', 'E8']
MONTHS = ['2016-12'] + [f'2017-{m:02d}' for m in range(1, 13)]


def oracle(candidate, reference, area, use):
    x, y, weights = candidate[use], reference[use], area[use]
    a, b = np.average(x, weights=weights), np.average(y, weights=weights)
    return dict(n=int(use.sum()), candidate_mean=a, reference_mean=b, bias=a - b,
                rmse=np.sqrt(np.average((x - y) ** 2, weights=weights)),
                mae=np.average(abs(x - y), weights=weights),
                r=np.average((x - a) * (y - b), weights=weights) /
                np.sqrt(np.average((x - a) ** 2, weights=weights) *
                        np.average((y - b) ** 2, weights=weights)))


@pytest.fixture
def cache(tmp_path):
    shapefile = pytest.importorskip('shapefile')
    pytest.importorskip('shapely')
    root = tmp_path / 'revision_v2_work'
    stage = root / 'expected_value_test/stage_c'
    stage.mkdir(parents=True)
    prior = tmp_path / 'revision_research'
    (prior / 'song_comparison').mkdir(parents=True)
    land = tmp_path / 'local_land.shp'
    with shapefile.Writer(str(land)) as writer:
        writer.field('name', 'C')
        writer.poly([[(-20, -65), (-20, 65), (20, 65), (20, -65), (-20, -65)]])
        writer.record('synthetic')
    land.with_suffix('.prj').write_text('GEOGCS["WGS 84"]')
    lat, lon = np.arange(-89., 90., 2.), np.arange(-177.5, 180., 5.)
    area, fraction = seasonal.original_geometry(land, lat, lon)
    iy, ix = np.indices(area.shape)
    ev = np.stack([.005 * (i + 1) + .0003 * iy + .0001 * ix for i in range(len(MONTHS))])
    ev[:, 30, 35] = 0.
    modis = np.zeros_like(ev, dtype=bool)
    for row in (15, 30, 50, 70, 76):
        modis[:, row, 35] = True
    modis[2, 70, 35] = False
    mean_only = modis.copy()
    mask = modis.copy()
    mask[1, 50, 35] = False
    support = stage / 'comparison_support.nc'
    with nc.Dataset(support, 'w') as ds:
        for name, size in [('time', 13), ('lat', 90), ('lon', 72)]:
            ds.createDimension(name, size)
        ds.months = json.dumps(MONTHS)
        ds.CALIOP_spectral_transfer = 'DAOD550=DAOD532*(550/532)^(-0.06)'
        group = ds.createGroup(seasonal.GROUP)
        values = dict(mask=mask, modis_mask=modis, reference_mean_only_mask=mean_only,
                      counts=np.broadcast_to(np.arange(1, 14)[:, None, None], ev.shape))
        values.update({key: ev if key == 'Eselected' else ev + 1 for key in CORE})
        for name, data in values.items():
            ds_type = 'u1' if name.endswith('mask') else 'f8'
            group.createVariable(name, ds_type, ('time', 'lat', 'lon'), zlib=True)[:] = data
        # An alternative support must never be used when the primary is absent.
        ds.createGroup(seasonal.REFERENCE + '__hierarchy__full_common_E6')
    cal = np.empty((2, 12, 90, 72), dtype=np.float32)
    for year in range(2):
        for month in range(12):
            cal[year, month] = .009 * (month + 1) + .002 * year + .0002 * iy + .0001 * ix
    cal[1, 11] += .8  # Wrong December must produce a conspicuous difference.
    cal[:, :, 30, 35] = 0.
    cube = prior / 'song_comparison/matched_native_grid.nc'
    with nc.Dataset(cube, 'w') as ds:
        for name, coordinates in [('year', [2016, 2017]), ('month', np.arange(1, 13)), ('lat', lat), ('lon', lon)]:
            ds.createDimension(name, len(coordinates))
            ds.createVariable(name, 'f8', (name,))[:] = coordinates
        for name, data in [('land_area_weight', area), ('land_fraction', fraction)]:
            ds.createVariable(name, 'f4', ('lat', 'lon'), zlib=True)[:] = data
        for name in ('cal_mean', 'old', 'cal_low', 'cal_high', 'cal_total', 'old_total'):
            var = ds.createVariable(name, 'f4', ('year', 'month', 'lat', 'lon'), zlib=True)
            var[:] = cal
            if name == 'cal_low':
                var[1, 0, 50, 35] = np.nan
    script = root / 'expected_value_test/compare_global.py'
    script.write_text(f'PERIODS = {PERIODS!r}\nCORE = {CORE!r}\nraise RuntimeError("Do not execute")\n')
    (prior / 'compare_song.py').write_text('raise RuntimeError("Do not execute geometry source")\n')
    provenance = dict(sources={str(cube): digest(cube)}, script=digest(script),
                      spatial_weight='0.5degree NaturalEarth110m land-center area',
                      seasonal_weight='equal complete months', land_purity=.95,
                      regions={seasonal.REGION: [-60, 60, -180, 180],
                               'example': [-20, 30, -10, 10],
                               'boreal_50_75N_sensitivity': [50, 75, -20, 20]})
    (stage / 'comparison_provenance.json').write_text(json.dumps(provenance))
    rows = []
    factor = (seasonal.CONFIG['reference']['output_wavelength_nm'] / 532) ** (-seasonal.CONFIG['reference']['dust_angstrom_exponent'])
    cal_months = np.stack([cal[int(date[:4]) - 2016, int(date[5:]) - 1].astype(float) * factor for date in MONTHS])
    for period, dates in PERIODS.items():
        indices = [MONTHS.index(date) for date in dates]
        use = np.all(mask[indices], axis=0) & (abs(lat[:, None]) < 60)
        rows.append(dict(seasonal.META, period=period,
                         **oracle(ev[indices].mean(axis=0), cal_months[indices].mean(axis=0), area, use)))
    # Decoy row tests exact primary selection rather than positional slicing.
    rows.insert(0, dict(rows[0], reference_mask='mean_only_reference_sensitivity', n=999))
    pd.DataFrame(rows).to_csv(stage / 'seasonal_metrics.csv', index=False)
    return dict(root=root, stage=stage, land=land, cube=cube, area=area,
                ev=ev, cal=cal_months, mask=mask, lat=lat, lon=lon)


def refresh_cube_hash(cache):
    path = cache['stage'] / 'comparison_provenance.json'
    provenance = json.loads(path.read_text())
    provenance['sources'][str(cache['cube'])] = digest(cache['cube'])
    path.write_text(json.dumps(provenance))


def test_complete_month_intersection_zero_missing_coordinates_and_weights(cache):
    tables, sources, checks = seasonal.derive(cache['root'], cache['land'])
    maps, metrics = tables['seasonal_maps'], tables['seasonal_map_metrics']
    assert len(maps) == 4 * 90 * 72
    assert metrics.n.tolist() == [2, 4, 4, 4]
    assert len(checks) == 28 and all(c['status'] == 'PASS' for c in checks)
    assert all(Path(p).is_absolute() and digest(Path(p)) == h for p, h in sources.items())
    for period, dates in PERIODS.items():
        frame = maps.loc[maps.period.eq(period)]
        selected = [MONTHS.index(date) for date in dates]
        expected = cache['mask'][selected].all(axis=0) & (abs(cache['lat'][:, None]) < 60)
        np.testing.assert_array_equal(frame.supported.to_numpy().reshape(90, 72), expected)
        assert frame.loc[~frame.supported, ['ev', 'caliop', 'difference']].isna().all().all()
        assert frame.loc[frame.supported, ['ev', 'caliop', 'difference']].notna().all().all()
        np.testing.assert_array_equal(frame.land_area_weight.to_numpy().reshape(90, 72), cache['area'])
        point = frame.loc[frame.lat_index.eq(30) & frame.lon_index.eq(35)].iloc[0]
        assert point.supported and point.ev == point.caliop == point.difference == 0
        assert (point.lat, point.lon, point.lat_lower, point.lon_upper) == (-29, -2.5, -30, 0)
        np.testing.assert_array_equal(frame.lat.to_numpy().reshape(90, 72)[:, 0], cache['lat'])
        np.testing.assert_array_equal(frame.lon.to_numpy().reshape(90, 72)[0], cache['lon'])
    djf = maps.loc[maps.period.eq('DJF')].set_index(['lat_index', 'lon_index'])
    assert djf.loc[(15, 35), 'months'] == '2016-12|2017-01|2017-02'
    assert djf.loc[(15, 35), 'ev'] == pytest.approx(cache['ev'][:3, 15, 35].mean())
    assert djf.loc[(15, 35), 'caliop'] == pytest.approx(cache['cal'][:3, 15, 35].mean())
    assert djf.loc[(50, 35), 'prior_joint_months'] == 2
    assert djf.loc[(50, 35), 'mean_only_months'] == 3
    assert djf.loc[(70, 35), 'modis_months'] == 2
    assert not djf.loc[(76, 35), 'in_domain']
    assert not djf.loc[(76, 35), 'supported']
    boxes = tables['seasonal_region_boxes'].set_index('region')
    assert tuple(boxes.loc['example', ['lat_lower', 'lat_upper', 'lon_lower', 'lon_upper']]) == (-20, 30, -10, 10)
    assert boxes.loc['boreal_50_75N_sensitivity', 'sensitivity']


def test_geometry_preserves_original_precision(cache):
    # At a completely land-covered native cell, all fine weights contribute.
    row, col = 50, 35
    fine_lat = np.array([10.25, 10.75, 11.25, 11.75])
    expected = ((np.sin(np.deg2rad(fine_lat + .25)) - np.sin(np.deg2rad(fine_lat - .25))) * 10).sum()
    assert cache['area'][row, col] == pytest.approx(expected, abs=1e-15)
    rounded = cache['area'].astype('f4').astype(float)
    assert not np.array_equal(cache['area'], rounded)
    mask = cache['mask'][3:6].all(axis=0) & (abs(cache['lat'][:, None]) < 60)
    x, y = cache['ev'][3:6].mean(axis=0), cache['cal'][3:6].mean(axis=0)
    assert abs(oracle(x, y, cache['area'], mask)['candidate_mean'] -
               oracle(x, y, rounded, mask)['candidate_mean']) > seasonal.CHECK_ATOL


@pytest.mark.parametrize('problem', ['missing_group', 'bitmap', 'wrong_month', 'dimension_order', 'nonfinite_ev'])
def test_invalid_support_cache_refused(cache, problem):
    with nc.Dataset(cache['stage'] / 'comparison_support.nc', 'a') as ds:
        group = ds.groups[seasonal.GROUP]
        if problem == 'missing_group':
            ds.renameGroup(seasonal.GROUP, 'not_primary')
        elif problem == 'bitmap':
            group['mask'][0, 15, 35] = 2
        elif problem == 'wrong_month':
            ds.months = json.dumps(['2017-12'] + MONTHS[1:])
        elif problem == 'dimension_order':
            ds.renameDimension('lat', 'latitude')
        else:
            group['Eselected'][0, 15, 35] = np.nan
    with pytest.raises(ValueError):
        seasonal.derive(cache['root'], cache['land'])


@pytest.mark.parametrize('problem', ['coordinates', 'geometry', 'ancillary', 'cube_hash', 'missing_year'])
def test_invalid_cube_refused_without_substitution(cache, problem):
    with nc.Dataset(cache['cube'], 'a') as ds:
        if problem == 'coordinates':
            ds['lon'][:] = ds['lon'][::-1]
        elif problem == 'geometry':
            ds['land_area_weight'][50, 35] *= 2
        elif problem == 'missing_year':
            ds['year'][0] = 2015
        elif problem == 'ancillary':
            ds['cal_high'][0, 11, 15, 35] = np.nan
        else:
            ds.note = 'unrecorded change'
    if problem != 'cube_hash':
        refresh_cube_hash(cache)
    with pytest.raises(ValueError):
        seasonal.derive(cache['root'], cache['land'])


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'count', 'precision'])
def test_saved_metric_disagreement_prevents_publication(cache, tmp_path, problem):
    path = cache['stage'] / 'seasonal_metrics.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    if problem == 'missing':
        frame = frame.iloc[:-1]
    elif problem == 'duplicate':
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif problem == 'count':
        frame.loc[1, 'n'] += 1
    else:
        frame.loc[1, 'rmse'] += 1e-9
    frame.to_csv(path, index=False)
    output = tmp_path / 'not_published'
    with pytest.raises(ValueError):
        seasonal.main(['--source-root', str(cache['root']), '--land-shapefile', str(cache['land']),
                       '--output-dir', str(output)])
    assert not output.exists()


def test_local_geometry_missing_fails_without_downloader(cache):
    cache['land'].with_suffix('.shx').unlink()
    with pytest.raises(FileNotFoundError):
        seasonal.derive(cache['root'], cache['land'])


def test_spectral_and_source_definition_guards(cache):
    for description in ('not a transfer', 'DAOD550=DAOD532*(550/532)^(-0.1)'):
        with pytest.raises(ValueError):
            seasonal.spectral_factor(description)
    script = cache['root'] / 'expected_value_test/compare_global.py'
    script.write_text(script.read_text().replace('2016-12', '2017-12'))
    with pytest.raises(ValueError, match='script hash'):
        seasonal.derive(cache['root'], cache['land'])
    with pytest.raises(ValueError, match='seasonal/core'):
        seasonal.source_literals(script)


def test_cli_hashes_registry_compatibility_determinism_and_protection(cache, tmp_path):
    directories = [tmp_path / 'derived_a', tmp_path / 'derived_b']
    arguments = ['--source-root', str(cache['root']), '--land-shapefile', str(cache['land'])]
    for output in directories:
        seasonal.main(arguments + ['--output-dir', str(output)])
        assert len(list(output.iterdir())) == 6
        for name, semantics in seasonal.SEMANTICS.items():
            path = output / (name + '.csv')
            metadata = json.loads(path.with_suffix('.provenance.json').read_text())
            registered = ingest(path, 'derived/' + name)
            assert metadata['output_sha256'] == digest(path) == registered['source_sha256']
            assert metadata['semantics'] == semantics
            assert metadata['rows'] == registered['source_rows']
            assert metadata['script_sha256'] == digest(Path(seasonal.__file__))
            assert len(metadata['checks']) == 28
            assert all(digest(Path(p)) == h for p, h in metadata['sources'].items())
    for path in directories[0].iterdir():
        assert path.read_bytes() == (directories[1] / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        seasonal.main(arguments + ['--output-dir', str(directories[0])])
    for path in (cache['root'] / 'forbidden', Path(seasonal.__file__).parent / 'forbidden'):
        with pytest.raises(ValueError, match='outside'):
            seasonal.main(arguments + ['--output-dir', str(path)])
