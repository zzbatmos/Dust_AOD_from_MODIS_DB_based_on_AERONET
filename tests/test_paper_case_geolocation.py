"""Full-swath extraction and native-grid rendering, using synthetic cases."""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from paper import derive_case_geolocation as geo
from paper.numbers import Registry, digest, ingest


FILENAMES = ['MOD04_L2.A2025073.1705.061.2025074013638.hdf',
             'MOD04_L2.A2024266.1345.061.2024268021127.hdf',
             'MYD04_L2.A2017241.2015.061.2018039202859.hdf']


@pytest.fixture
def golden(tmp_path):
    root = tmp_path / 'golden'
    root.mkdir()
    records = []
    for (case, (_, _, bounds)), filename in zip(geo.CASES.items(), FILENAMES):
        west, east, south, north = bounds
        lon, lat = np.meshgrid(np.linspace(west, east, 4), np.linspace(south, north, 3))
        lon[0, 0] = west - 1
        path = root / f'DBDust_v{geo.CONFIG["version"]}_{case}_L2.nc'
        with nc.Dataset(path, 'w') as ds:
            ds.createDimension('pixel', lat.size)
            ds.swath_shape = '[3, 4]'
            ds.source_file = filename
            ds.source_sha256 = 'synthetic-retained-input-hash'
            ds.reference_definition = geo.CONFIG['reference']['name']
            ds.model_manifest_hash = geo.CONFIG['artifacts']['manifest_sha256']
            for name, data in [('lat', lat), ('lon', lon)]:
                ds.createVariable(name, 'f8', ('pixel',), fill_value=np.nan)[:] = data.ravel()
        # Deliberately no science fields: geometry extraction must not read them.
        records.append(dict(case=case, source_file=filename, source_sha256='synthetic-retained-input-hash',
                            output_file=path.name, output_sha256=digest(path), swath_shape=[3, 4]))
    (root / 'manifest.json').write_text(json.dumps(dict(status='PASS', cases=records)))
    return root


def mutate_golden(root, change, refresh=True):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    product = root / manifest['cases'][0]['output_file']
    with nc.Dataset(product, 'a') as ds:
        change(ds)
    if refresh:
        manifest['cases'][0]['output_sha256'] = digest(product)
        path.write_text(json.dumps(manifest))


def test_all_pixels_shape_indices_and_closed_zoom(golden):
    table, sources, summary = geo.derive(golden)
    assert len(table) == 36
    assert set(table) == {'case', 'lat', 'lon', 'pixel_index', 'swath_row', 'swath_col',
                          'swath_rows', 'swath_cols', 'inside_zoom', 'geolocation_finite', 'geolocation_valid'}
    assert all(digest(Path(p)) == value for p, value in sources.items())
    for case in geo.CASES:
        group = table.loc[table.case.eq(case)]
        np.testing.assert_array_equal(group.pixel_index, np.arange(12))
        np.testing.assert_array_equal(group.swath_row, np.arange(12) // 4)
        np.testing.assert_array_equal(group.swath_col, np.arange(12) % 4)
        assert group.inside_zoom.tolist() == [False] + [True] * 11
        assert group.geolocation_finite.all() and group.geolocation_valid.all()
    assert all(s['swath_pixels'] == 12 and s['inside_zoom_pixels'] == 11 for s in summary)


def test_missing_and_invalid_geolocation_retained_explicitly(golden):
    def change(ds):
        ds['lat'][5] = np.nan
        ds['lon'][6] = np.nan
        ds['lat'][7] = 91
    mutate_golden(golden, change)
    table, _, summary = geo.derive(golden)
    first = table.loc[table.case.eq('Kansas_20250314')].set_index('pixel_index')
    assert len(first) == 12
    assert np.isnan(first.loc[5, 'lat']) and np.isnan(first.loc[6, 'lon'])
    assert first.loc[7, 'lat'] == 91
    assert not first.loc[[5, 6, 7], 'inside_zoom'].any()
    assert not first.loc[[5, 6], 'geolocation_finite'].any()
    assert first.loc[7, 'geolocation_finite'] and not first.loc[7, 'geolocation_valid']
    assert summary[0]['missing_geolocation_pixels'] == 2
    assert summary[0]['invalid_geolocation_pixels'] == 3


@pytest.mark.parametrize('problem', ['hash', 'shape', 'source', 'date', 'duplicate_manifest'])
def test_golden_inconsistencies_refused(golden, problem):
    if problem == 'hash':
        mutate_golden(golden, lambda ds: ds.setncattr('change', 'unrecorded'), refresh=False)
    elif problem == 'shape':
        mutate_golden(golden, lambda ds: ds.setncattr('swath_shape', '[2, 6]'))
    elif problem == 'source':
        mutate_golden(golden, lambda ds: ds.setncattr('source_sha256', 'different'))
    else:
        path = golden / 'manifest.json'
        manifest = json.loads(path.read_text())
        if problem == 'date':
            manifest['cases'][0]['source_file'] = FILENAMES[1]
        else:
            manifest['cases'].append(manifest['cases'][0])
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        geo.derive(golden)


def test_cli_hashes_determinism_protection_and_case_display_untouched(golden, tmp_path):
    directories = [tmp_path / 'derived_a', tmp_path / 'derived_b']
    original = {'case_display.csv': 'completed science file\n', 'case_display.provenance.json': '{"immutable":true}\n'}
    for output in directories:
        output.mkdir()
        for name, text in original.items():
            (output / name).write_text(text)
        geo.main(['--golden-root', str(golden), '--output-dir', str(output)])
        for name, text in original.items():
            assert (output / name).read_text() == text
        path = output / 'case_geolocation.csv'
        metadata = json.loads(path.with_suffix('.provenance.json').read_text())
        registered = ingest(path, 'derived/case_geolocation')
        assert metadata['output_sha256'] == digest(path) == registered['source_sha256']
        assert metadata['script_sha256'] == digest(Path(geo.__file__))
        assert metadata['rows'] == registered['source_rows'] == 36
        assert metadata['semantics'] == geo.SEMANTICS
    for suffix in ('.csv', '.provenance.json'):
        name = 'case_geolocation' + suffix
        assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes()
    with pytest.raises(FileExistsError):
        geo.main(['--golden-root', str(golden), '--output-dir', str(directories[0])])
    for forbidden in (golden / 'derived', Path(geo.__file__).parent / 'derived'):
        with pytest.raises(ValueError, match='outside'):
            geo.main(['--golden-root', str(golden), '--output-dir', str(forbidden)])


@pytest.fixture
def renderer():
    pytest.importorskip('cmcrameri')
    pytest.importorskip('colorspacious')
    pytest.importorskip('cartopy')
    from paper.figures import figS17_case_inputs
    return figS17_case_inputs


@pytest.fixture
def registered_frames(golden):
    geometry, _, _ = geo.derive(golden)
    records = []
    for case, (date, platform, bounds) in geo.CASES.items():
        q = geometry.loc[geometry.case.eq(case) & geometry.inside_zoom].copy()
        q['aod550_db'] = q.pixel_index / 10
        q['DB_AE'] = q.pixel_index / 5
        q['branch'] = q.pixel_index % 3
        q['db_aerosol_type'] = q.pixel_index % 4
        q.loc[q.pixel_index.eq(2), ['aod550_db', 'DB_AE', 'branch', 'db_aerosol_type']] = np.nan
        q.loc[q.pixel_index.eq(3), ['aod550_db', 'DB_AE']] = 0.
        q = q.assign(date=date, platform=platform, granule_start_utc=date + 'T12:00:00Z',
                     zoom_west=bounds[0], zoom_east=bounds[1], zoom_south=bounds[2], zoom_north=bounds[3])
        records.append(q)
    return pd.concat(records, ignore_index=True), geometry


def test_reconstruct_preserves_values_indices_zeros_and_gaps(renderer, registered_frames):
    display, geometry = registered_frames
    before = display.copy(deep=True)
    swath = renderer.reconstruct(display.sample(frac=1, random_state=0), geometry.sample(frac=1, random_state=1), 'Kansas_20250314')
    assert swath['lat'].shape == (3, 4)
    for field in renderer.FIELDS:
        values = swath['values'][field].ravel()
        assert np.isnan(values[0]) and np.isnan(values[2])
        selected = display.loc[display.case.eq('Kansas_20250314')]
        np.testing.assert_array_equal(values[selected.pixel_index.to_numpy()], selected[field])
    assert swath['values']['aod550_db'].ravel()[3] == 0
    pd.testing.assert_frame_equal(display, before)


@pytest.mark.parametrize('problem', ['nonfinite_outside', 'nonfinite_inside', 'bad_range', 'geo_gap', 'geo_duplicate',
                                    'wrong_row', 'science_gap', 'science_duplicate', 'coordinate_mismatch',
                                    'inside_flag', 'shape', 'category', 'bounds'])
def test_renderer_refuses_inexact_geometry_or_science(renderer, registered_frames, problem):
    display, geometry = (f.copy() for f in registered_frames)
    if problem == 'nonfinite_outside':
        geometry.loc[0, 'lon'] = np.nan
    elif problem == 'nonfinite_inside':
        geometry.loc[1, 'lat'] = np.nan
    elif problem == 'bad_range':
        geometry.loc[0, 'lat'] = 91
    elif problem == 'geo_gap':
        geometry = geometry.drop(index=0)
    elif problem == 'geo_duplicate':
        geometry = pd.concat([geometry, geometry.iloc[:1]])
    elif problem == 'wrong_row':
        geometry.loc[0, 'swath_row'] = 1
    elif problem == 'science_gap':
        display = display.drop(index=0)
    elif problem == 'science_duplicate':
        display = pd.concat([display, display.iloc[:1]])
    elif problem == 'coordinate_mismatch':
        display.loc[0, 'lat'] += .00001
    elif problem == 'inside_flag':
        geometry.loc[0, 'inside_zoom'] = True
    elif problem == 'shape':
        geometry.loc[0, 'swath_rows'] = 4
    elif problem == 'category':
        display.loc[0, 'branch'] = 7
    else:
        display.loc[0, 'zoom_west'] -= 1
    with pytest.raises(ValueError):
        renderer.reconstruct(display, geometry, 'Kansas_20250314')


def test_draw_panel_uses_native_mesh_and_masked_values(renderer, registered_frames):
    from unittest.mock import Mock
    display, geometry = registered_frames
    swath = renderer.reconstruct(display, geometry, 'Kansas_20250314')
    ax = Mock()
    renderer.draw_panel(ax, swath, 'aod550_db')
    assert ax.pcolormesh.call_count == 2
    args, kwargs = ax.pcolormesh.call_args
    np.testing.assert_array_equal(args[0], swath['lon'])
    np.testing.assert_array_equal(args[1], swath['lat'])
    assert kwargs['shading'] == 'nearest'
    assert np.ma.is_masked(args[2].ravel()[0])
    assert np.ma.is_masked(args[2].ravel()[2])
    assert np.ma.is_masked(args[2].ravel()[3])  # Zero has its own grey mesh.
    np.testing.assert_array_equal(args[2].compressed(), swath['values']['aod550_db'][swath['values']['aod550_db'] > 0])
    swath['lat'][0, 0] = np.nan
    ax.reset_mock()
    with pytest.raises(ValueError, match='Nonfinite'):
        renderer.draw_panel(ax, swath, 'branch')
    ax.pcolormesh.assert_not_called()


def test_no_coastline_download(renderer, tmp_path, monkeypatch):
    monkeypatch.setitem(renderer.cartopy.config, 'data_dir', tmp_path / 'absent')
    monkeypatch.setitem(renderer.cartopy.config, 'pre_existing_data_dir', tmp_path / 'also_absent')
    with pytest.raises(FileNotFoundError, match='no download'):
        renderer.local_coastlines()


def test_synthetic_figure_exports_contract_without_publishing_real_figure(renderer, registered_frames, tmp_path, monkeypatch):
    display, geometry = registered_frames
    tables = {}
    for name, frame in [('case_display', display), ('case_geolocation', geometry)]:
        tables['derived/' + name] = dict(records=frame.to_dict('records'), columns=list(frame),
                                         source_file=name + '.csv', source_sha256='synthetic', derivation={})
    registry = Registry(dict(tables=tables))
    monkeypatch.setattr(renderer.Registry, 'read', lambda _: registry)
    monkeypatch.setattr(renderer, 'local_coastlines', lambda: ([], {}))
    captured = {}

    def export(fig, output, stem, data, registry_path, evidence, script, raster=False):
        from cartopy.mpl.gridliner import Gridliner

        # Exercise a synthetic canvas only; never export a production figure.
        fig.canvas.draw()
        captured.update(stem=stem, data=data, raster=raster, evidence=evidence)
        assert all(t.get_fontsize() >= 6.5 for t in fig.findobj(renderer.matplotlib.text.Text)
                   if t.get_visible() and t.get_text())
        axes = [ax for ax in fig.axes if hasattr(ax, 'projection')]
        assert len(axes) == 12
        assert fig.get_figheight() / renderer.style.MM <= 230
        canvas_renderer = fig.canvas.get_renderer()
        for index, ax in enumerate(axes):
            row, col = divmod(index, 4)
            grid = next(artist for artist in ax.get_children() if isinstance(artist, Gridliner))
            assert grid.bottom_labels and grid.left_labels == (col == 0)
            west, east, _, _ = list(geo.CASES.values())[row][2]
            expected = {grid.xformatter(lon) for lon in np.arange(-180, 181, 5) if west < lon < east}
            labels = [text for text in grid.bottom_label_artists if text.get_visible()]
            assert {text.get_text() for text in labels} == expected
            if row < 2:
                next_heading = fig.texts[row + 1].get_window_extent(canvas_renderer)
                assert all(text.get_window_extent(canvas_renderer).y0 > next_heading.y1 for text in labels)
        Path(output).mkdir()

    monkeypatch.setattr(renderer.style, 'export', export)
    output = tmp_path / 'synthetic'
    renderer.make(tmp_path / 'not_a_real_registry.json', output)
    assert captured['stem'] == 'FigS17_case_inputs' and captured['raster']
    assert len(captured['data']) == len(geometry)
    assert captured['data'].loc[~captured['data'].inside_zoom, list(renderer.FIELDS)].isna().all().all()
    caption = (output / 'FigS17_caption.md').read_text()
    assert 'AOD 0-4' in caption and 'Angstrom exponent 0-2' in caption
    assert 'no hatching' in caption
