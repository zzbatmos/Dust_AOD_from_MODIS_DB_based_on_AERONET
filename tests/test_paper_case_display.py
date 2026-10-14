"""Synthetic golden/cache fixtures; no model inference or HDF downloads."""
from datetime import datetime, timedelta
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from paper import derive_case_display as cases
from paper.numbers import digest, ingest


FILENAMES = ['MOD04_L2.A2025073.1705.061.2025074013638.hdf',
             'MOD04_L2.A2024266.1345.061.2024268021127.hdf',
             'MYD04_L2.A2017241.2015.061.2018039202859.hdf']


def physical_text(case, filename):
    date, platform, _ = cases.CASES[case]
    hhmm = filename.split('.')[2]
    beginning = datetime.fromisoformat(date).replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]))
    ending = beginning + timedelta(minutes=5)
    values = dict(LOCALGRANULEID=filename, ASSOCIATEDPLATFORMSHORTNAME=platform,
                  RANGEBEGINNINGDATE=date, RANGEBEGINNINGTIME=beginning.strftime('%H:%M:%S.000000'),
                  RANGEENDINGDATE=ending.date().isoformat(), RANGEENDINGTIME=ending.strftime('%H:%M:%S.000000'))
    return '\n'.join(f'OBJECT = {k}\n NUM_VAL = 1\n VALUE = "{v}"\nEND_OBJECT = {k}' for k, v in values.items())


@pytest.fixture
def cached(tmp_path, monkeypatch):
    workspace = tmp_path / 'project'
    root = workspace / 'JGRA_revison/revision_v2_work'
    golden = workspace / 'JGRA_revison/resubmission/golden_cases'
    golden.mkdir(parents=True)
    (root / 'compact_cases').mkdir(parents=True)
    (root / 'expected_value_test/deploy_ev').mkdir(parents=True)
    spec = golden.parent / 'source/01_instructions/FIGURE_SPEC_AND_STYLE_GUIDE.md'
    spec.parent.mkdir(parents=True)
    spec.write_text('Synthetic Fig8 specification fixture.\n')
    for path in [root / 'expected_value_test/deploy_ev/predict.py', root / 'expected_value_test/report_global.py']:
        path.write_text('raise RuntimeError("Never execute old sources")\n')
    definitions, provenance, records, frames, stats = [], [], [], {}, []
    for number, ((case, (_, _, bounds)), filename) in enumerate(zip(cases.CASES.items(), FILENAMES), start=1):
        west, east, south, north = bounds
        lat, lon = np.full(12, (south + north) / 2), np.full(12, (west + east) / 2)
        lat[:2], lon[:2] = [south, north], [west, east]
        lat[10], lon[11] = np.nan, west - 1
        aod = np.arange(1, 13) / 10
        aod[[2, 10]], aod[4] = np.nan, 0.
        frame = pd.DataFrame(dict(lat=lat, lon=lon, DB_AOD550=aod,
                                  DB_AE=[.1, .5, np.nan, 1.6, 1., 3., 1.6, .8, .9, .7, np.nan, .6],
                                  db_aerosol_type=np.where(np.isfinite(aod), 1., np.nan)))
        for feature in cases.FEATURES:
            if feature != 'DB_AE':
                frame[feature] = np.where(np.isfinite(aod), .8, np.nan)
        frame.loc[7, 'DB_SSA412'] = np.nan
        relative = 'case_inputs/' + filename
        hdf = workspace / relative
        hdf.parent.mkdir(exist_ok=True)
        hdf.write_text(physical_text(case, filename))
        frames[filename] = frame
        shape = (3, 4)
        definitions.append((case, relative, 'ignored_thumbnail_wrong_date.png', bounds))
        provenance.append(dict(path=relative, sha256=digest(hdf)))
        values = dict(lat=lat, lon=lon, aod550_db=aod,
                      db_aerosol_type=frame.db_aerosol_type.to_numpy(),
                      branch=np.where(np.isfinite(aod), 2., np.nan))
        values['branch'][:2], values['branch'][3] = 1., 0.
        ml = np.isin(values['branch'], [1, 2])
        for name in ('dust_probability', 'dust_fraction_conditional'):
            values[name] = np.where(ml, .5, np.nan)
        values['qa_tier'] = np.where(ml, 1., np.nan)
        values['daod550_qa2'] = np.where(ml, 0., np.nan)
        values['daod550_ev'] = np.where(ml, aod / 3, np.nan)
        path = golden / f'DBDust_v{cases.CONFIG["version"]}_{case}_L2.nc'
        with nc.Dataset(path, 'w') as ds:
            ds.createDimension('pixel', 12)
            ds.source_file, ds.source_sha256 = filename, digest(hdf)
            ds.swath_shape = json.dumps(shape)
            ds.reference_definition = cases.CONFIG['reference']['name']
            ds.model_manifest_hash = cases.CONFIG['artifacts']['manifest_sha256']
            for name, data in values.items():
                ds.createVariable(name, 'f8', ('pixel',), fill_value=np.nan)[:] = data
        records.append(dict(case=case, source_file=filename, source_sha256=digest(hdf),
                            output_file=path.name, output_sha256=digest(path), swath_shape=list(shape)))
        c0, c1, c2 = cases.CONFIG['baselines']['li_ginoux_coefficients']
        ae = frame.DB_AE.to_numpy()
        lg = (1 - np.clip(c2 * ae**2 + c1 * ae + c0, 0, 1)) * aod
        np.savez(root / f'compact_cases/case_{number}_fields.npz',
                 lat=lat.reshape(shape), lon=lon.reshape(shape), total=aod.reshape(shape),
                 lg25_coarse=lg.reshape(shape), screened_lg=np.zeros(shape), ml_qa2=np.ones(shape))
        good = np.isfinite(lg[:10])
        stats.append(dict(case=case, method='lg25_coarse', evaluated_pixels=int(good.sum()),
                          zero_pixels=int((lg[:10][good] == 0).sum()),
                          valid_total_pixels=int((aod[:10] > 0).sum()), mean_own_support=lg[:10][good].mean()))
    (root / 'compact_cases.py').write_text(f'CASES = {definitions!r}\nraise RuntimeError("Do not execute")\n')
    (root / 'compact_cases/provenance.json').write_text(json.dumps(provenance))
    pd.DataFrame(stats).to_csv(root / 'compact_cases/case_statistics.csv', index=False)
    (golden / 'manifest.json').write_text(json.dumps(dict(status='PASS', cases=records)))
    monkeypatch.setattr(cases, 'read_granule', lambda p: (frames[p.name].copy(), (3, 4), {}))
    monkeypatch.setattr(cases, 'physical_metadata', lambda p, case: cases.parse_physical_metadata(p.read_text(), p.name, case))
    return dict(root=root, golden=golden, frames=frames, workspace=workspace)


def alter_npz(cached, **changes):
    path = cached['root'] / 'compact_cases/case_1_fields.npz'
    with np.load(path) as source:
        data = {k: source[k].copy() for k in source.files}
    for key, value in changes.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value(data[key])
    np.savez(path, **data)


def test_exact_zoom_zero_missing_baseline_support_and_features(cached):
    table, sources, metadata = cases.derive(cached['root'], cached['golden'])
    assert len(table) == 30 and len(metadata['checks']) == 36
    assert all(c['status'] == 'PASS' for c in metadata['checks'])
    assert all(digest(Path(p)) == h for p, h in sources.items())
    for case in cases.CASES:
        frame = table.loc[table.case.eq(case)].set_index('pixel_index')
        assert frame.index.tolist() == list(range(10))
        assert not frame.loc[2, ['dust_probability', 'daod550_ev', 'lg25_daod', 'aod550_db']].notna().any()
        assert frame.loc[2, 'difference_missing']
        assert frame.loc[3, 'branch'] == 0 and frame.loc[3, 'lg25_daod'] > 0
        assert frame.loc[3, 'difference_missing'] and not frame.loc[3, 'ml_available']
        assert frame.loc[4, 'daod550_ev'] == frame.loc[4, 'lg25_daod'] == frame.loc[4, 'ev_minus_lg25'] == 0
        assert frame.loc[5, 'lg25_daod'] == 0
        assert frame.loc[6, 'DB_AE'] > 1.42 and frame.loc[6, 'lg25_daod'] > 0
        assert frame.loc[0, 'full8_branch'] and not frame.loc[7, 'ssa_all_available']
        assert frame.loc[7, 'ml_available']
        assert frame.loc[9, 'swath_row'] == 2 and frame.loc[9, 'swath_col'] == 1
        assert frame.lg25_status.eq('verified_saved_E7').all()
        np.testing.assert_allclose(frame.ev_minus_lg25, frame.daod550_ev - frame.lg25_daod, equal_nan=True)
    assert {c['platform'] for c in metadata['cases']} == {'Terra', 'Aqua'}
    assert all(c['unlocatable_swath_pixels'] == 1 and not c['unavailable_fields'] for c in metadata['cases'])
    assert metadata['cases'][1]['date'] == '2024-09-22'


@pytest.mark.parametrize('problem', ['lg_field', 'lg_file', 'AE_feature'])
def test_missing_lg_is_flagged_not_recomputed_or_replaced(cached, problem):
    if problem == 'lg_field':
        alter_npz(cached, lg25_coarse=None)
    elif problem == 'lg_file':
        (cached['root'] / 'compact_cases/case_1_fields.npz').unlink()
    else:
        cached['frames'][FILENAMES[0]].drop(columns='DB_AE', inplace=True)
    table, _, metadata = cases.derive(cached['root'], cached['golden'])
    first = table.loc[table.case.eq('Kansas_20250314')]
    assert first.lg25_daod.isna().all() and first.ev_minus_lg25.isna().all()
    assert first.lg25_status.eq(cases.UNAVAILABLE).all()
    assert first.daod550_ev.notna().any()
    assert 'lg25_daod' in metadata['cases'][0]['unavailable_fields']
    assert metadata['cases'][0]['lg25_limitation']


def test_optional_ssa_absence_does_not_reselect_branch(cached):
    cached['frames'][FILENAMES[0]].drop(columns='DB_SSA412', inplace=True)
    table, _, metadata = cases.derive(cached['root'], cached['golden'])
    first = table.loc[table.case.eq('Kansas_20250314')]
    assert first.DB_SSA412.isna().all()
    assert first.full8_branch.sum() == 2
    assert first.lg25_status.eq('verified_saved_E7').all()
    assert 'DB_SSA412' in metadata['cases'][0]['unavailable_fields']


@pytest.mark.parametrize('problem', ['screened', 'old_ae_cut', 'missing_zero', 'coordinates', 'shape'])
def test_saved_mismatch_fails_closed(cached, problem):
    if problem == 'screened':
        alter_npz(cached, lg25_coarse=lambda a: np.where(np.isfinite(a), 0., a))
    elif problem == 'old_ae_cut':
        ae = cached['frames'][FILENAMES[0]].DB_AE.to_numpy().reshape(3, 4)
        alter_npz(cached, lg25_coarse=lambda a: np.where(ae >= 1.42, 0., a))
    elif problem == 'missing_zero':
        alter_npz(cached, lg25_coarse=lambda a: np.where(a == 0, np.nan, a))
    elif problem == 'coordinates':
        alter_npz(cached, lon=lambda a: np.nextafter(a, np.inf))
    else:
        alter_npz(cached, lat=lambda a: a.T)
    with pytest.raises(ValueError):
        cases.derive(cached['root'], cached['golden'])


def test_reader_coordinate_mismatch_is_not_nearest_matched(cached):
    cached['frames'][FILENAMES[0]].loc[0, 'lat'] += .001
    with pytest.raises(ValueError, match='Exact swath/value mismatch'):
        cases.derive(cached['root'], cached['golden'])


@pytest.mark.parametrize('problem', ['date', 'platform', 'granule', 'missing', 'ambiguous'])
def test_physical_metadata_refuses_mismatch(problem):
    case, filename = 'Amazonia_20240922', FILENAMES[1]
    text = physical_text(case, filename)
    if problem == 'date':
        text = text.replace('2024-09-22', '2025-09-22')
    elif problem == 'platform':
        text = text.replace('Terra', 'Aqua')
    elif problem == 'granule':
        text = text.replace(filename, FILENAMES[0])
    elif problem == 'missing':
        text = text.replace('RANGEBEGINNINGTIME', 'ABSENT')
    else:
        text += '\n' + text
    with pytest.raises(ValueError):
        cases.parse_physical_metadata(text, filename, case)


@pytest.mark.parametrize('problem', ['input_hash', 'golden_hash', 'duplicate_manifest', 'bounds', 'saved_count'])
def test_provenance_and_counts_fail_before_publication(cached, tmp_path, problem):
    if problem == 'input_hash':
        path = cached['workspace'] / 'case_inputs' / FILENAMES[0]
        path.write_text(path.read_text() + '\n')
    elif problem == 'golden_hash':
        record = json.loads((cached['golden'] / 'manifest.json').read_text())['cases'][0]
        with nc.Dataset(cached['golden'] / record['output_file'], 'a') as ds:
            ds.changed = 'yes'
    elif problem == 'duplicate_manifest':
        path = cached['golden'] / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['cases'].append(manifest['cases'][0])
        path.write_text(json.dumps(manifest))
    elif problem == 'bounds':
        path = cached['root'] / 'compact_cases.py'
        path.write_text(path.read_text().replace('-110', '-111'))
    else:
        path = cached['root'] / 'compact_cases/case_statistics.csv'
        saved = pd.read_csv(path)
        saved.loc[0, 'evaluated_pixels'] += 1
        saved.to_csv(path, index=False)
    output = tmp_path / 'not_published'
    with pytest.raises(ValueError):
        cases.main(['--source-root', str(cached['root']), '--golden-root', str(cached['golden']), '--output-dir', str(output)])
    assert not output.exists()


def test_cli_deterministic_hashes_registry_and_output_protection(cached, tmp_path):
    arguments = ['--source-root', str(cached['root']), '--golden-root', str(cached['golden'])]
    directories = [tmp_path / 'derived_a', tmp_path / 'derived_b']
    for output in directories:
        cases.main(arguments + ['--output-dir', str(output)])
        path = output / 'case_display.csv'
        metadata = json.loads(path.with_suffix('.provenance.json').read_text())
        registered = ingest(path, 'derived/case_display')
        assert metadata['output_sha256'] == digest(path) == registered['source_sha256']
        assert metadata['script_sha256'] == digest(Path(cases.__file__))
        assert metadata['rows'] == registered['source_rows'] == 30
        assert metadata['semantics'] == cases.SEMANTICS
        assert all(digest(Path(p)) == h for p, h in metadata['sources'].items())
    for path in directories[0].iterdir():
        assert path.read_bytes() == (directories[1] / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        cases.main(arguments + ['--output-dir', str(directories[0])])
    for forbidden in [cached['root'] / 'derived', cached['golden'] / 'derived', Path(cases.__file__).parent / 'derived']:
        with pytest.raises(ValueError, match='outside'):
            cases.main(arguments + ['--output-dir', str(forbidden)])
