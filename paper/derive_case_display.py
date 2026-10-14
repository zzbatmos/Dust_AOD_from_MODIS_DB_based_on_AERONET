"""Register Fig. 8/S17 pixels from golden products and matched saved LG25.

Run ``python -m paper.derive_case_display --source-root REVISION_V2_WORK
--golden-root GOLDEN_CASES --output-dir PRIVATE_DERIVED_DIR``. Optional
``--figure-spec`` overrides the default golden-root sibling source spec.

No model or old case script is imported/executed. The frozen input reader only
decodes retained HDFs. Saved lg25_coarse is validated against the frozen E7
equation; validation never fills or replaces the saved field. Missing saved
LG25 is explicitly unavailable and requires a separately authorized analytic
derivation. E6, old AE<1.42, and old model outputs are never substitutes.

The output includes every geolocated pixel within the closed reviewer bounds,
not just retrievals. Flat pixel index and original swath row/column preserve
identity. No resampling, new land mask, branch selection or RGB processing.
"""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd

from dbdust.config import CONFIG, config_path
from dbdust import modis_db
from dbdust.modis_db import read_granule
from paper.numbers import digest


# Display domains and physical identities from Fig. 8, not retrieval screens.
CASES = {
    'Kansas_20250314': ('2025-03-14', 'Terra', (-110, -93, 28, 45)),
    'Amazonia_20240922': ('2024-09-22', 'Terra', (-70, -53, -27, -9)),
    'NorthAmerica_20170829': ('2017-08-29', 'Aqua', (-125, -100, 39, 50)),
}
FEATURES = ['DB_AE', *[f'DB_SSA{w}' for w in CONFIG['inputs']['spectral_wavelengths_nm']],
            *[f'DB_AOD{w}' for w in CONFIG['inputs']['spectral_wavelengths_nm']]]
UNAVAILABLE = 'unavailable_requires_separate_analytic_derivation'
SEMANTICS = (
    'Fig8/S17 closed reviewer zoom boxes; one row per original geolocated swath pixel, '
    'including missing AOD and ML outputs. No land, confidence, SSA or common-support screen '
    'is applied to the display rows. Golden values are copied without recomputation; exact '
    'zeros remain zero and missing values remain missing. DB features use the frozen HDF '
    'reader, including its fill/range/scale handling, SSA clipping and invalid-AOD masking. '
    'full8_branch comes from golden branch, not a new eligibility decision; ssa_all_available '
    'is only a diagnostic. LG25/E7 is the saved unscreened coarse-AOD field, validated against '
    'AOD550*(1-clip(c2*AE^2+c1*AE+c0,0,1)) using configured coefficients. No FMF<=0.7, '
    'SSA412<SSA470 or old AE<1.42 screen is applied. EV-minus-LG25 exists only where both '
    'saved fields are finite; other panels retain their own support. Absent saved LG25 is '
    'explicitly unavailable, never analytically backfilled. No predictions, fits, branch '
    'reselection, downloads, resampling or RGB. Times are granule interval times, not per-pixel '
    'acquisition times. Pixels with unlocatable geolocation cannot be assigned to a zoom box '
    'and are counted separately in provenance. These are development cases, not new validation.'
)


def case_definitions(path):
    """Read old source's literal case paths/extents without executing it."""
    definitions = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in node.targets):
            definitions.append(ast.literal_eval(node.value))
    if len(definitions) != 1 or len(definitions[0]) != len(CASES):
        raise ValueError('Missing or ambiguous source case definitions')
    return definitions[0]


def parse_physical_metadata(text, filename, case):
    """Cross-check physical HDF metadata with granule name and requested date."""
    def value(name):
        blocks = re.findall(r'(?m)^\s*OBJECT\s*=\s*' + name +
                            r'\s*\n(.*?)^\s*END_OBJECT\s*=\s*' + name + r'\s*$', text, re.S | re.M)
        values = [v for block in blocks for v in re.findall(r'\bVALUE\s*=\s*"([^"]+)"', block)]
        if len(values) != 1:
            raise ValueError('Missing or ambiguous physical metadata: ' + name)
        return values[0]

    pattern = r'(MOD|MYD)04_L2\.A(\d{4})(\d{3})\.(\d{4})\.(\d{3})\.\d+\.hdf'
    match = re.fullmatch(pattern, filename)
    if match is None:
        raise ValueError('Unexpected physical granule filename')
    product, year, day, hhmm, collection = match.groups()
    start = datetime(int(year), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(day) - 1)
    start = start.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]))
    if start.year != int(year) or start.strftime('%j') != day:
        raise ValueError('Invalid granule day of year')
    platform = 'Terra' if product == 'MOD' else 'Aqua'
    expected_date, expected_platform, _ = CASES[case]
    beginning = datetime.fromisoformat(value('RANGEBEGINNINGDATE') + 'T' + value('RANGEBEGINNINGTIME')).replace(tzinfo=timezone.utc)
    ending = datetime.fromisoformat(value('RANGEENDINGDATE') + 'T' + value('RANGEENDINGTIME')).replace(tzinfo=timezone.utc)
    if (value('LOCALGRANULEID') != filename or value('ASSOCIATEDPLATFORMSHORTNAME') != platform
            or platform != expected_platform or start.date().isoformat() != expected_date
            or beginning != start or ending <= beginning or collection != CONFIG['inputs']['collection']):
        raise ValueError('Physical date/platform/granule metadata mismatch')
    return dict(date=expected_date, platform=platform,
                granule_start_utc=beginning.isoformat().replace('+00:00', 'Z'),
                granule_end_utc=ending.isoformat().replace('+00:00', 'Z'), granule=filename)


def physical_metadata(path, case):
    from pyhdf.SD import SD, SDC
    hdf = SD(str(path), SDC.READ)
    try:
        attrs = hdf.attributes()
    finally:
        hdf.end()
    keys = sorted(k for k in attrs if k.startswith('CoreMetadata.'))
    if not keys:
        raise ValueError('Missing HDF physical metadata')
    return parse_physical_metadata('\n'.join(str(attrs[k]) for k in keys), path.name, case)


def exact(actual, expected, label, checks):
    if not np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True):
        raise ValueError('Exact swath/value mismatch: ' + label)
    checks.append(dict(check=label, status='PASS'))


def validate_saved_e7(saved, frame, checks):
    """Validate only; return no generated baseline and do not repair support."""
    aod, ae = frame.DB_AOD550.to_numpy(float), frame.DB_AE.to_numpy(float)
    c0, c1, c2 = CONFIG['baselines']['li_ginoux_coefficients']
    supported = np.isfinite(aod) & (aod >= CONFIG['inputs']['minimum_aod']) & np.isfinite(ae)
    expected = np.where(supported, (1 - np.clip(c2 * ae**2 + c1 * ae + c0, 0, 1)) * aod, np.nan)
    if not np.array_equal(saved, expected, equal_nan=True):
        raise ValueError('Saved LG25 is not exact unscreened E7 on frozen support; '
                         'do not substitute E6/AE<1.42 or silently derive missing values')
    checks.append(dict(check='saved_lg25_matches_unscreened_E7_values_and_support', status='PASS'))


def derive(source_root, golden_root, figure_spec=None):
    """Return display frame, input hash map and case-specific provenance."""
    root, golden = Path(source_root).resolve(), Path(golden_root).resolve()
    workspace = root.parents[1]  # Same ROOT as original revision_v2_work/compact_cases.py.
    sources, checks, summaries, frames = {}, [], [], []

    def track(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = digest(path)
        return path

    track(config_path())
    track(modis_db.__file__)
    track(figure_spec or golden.parent / 'source/01_instructions/FIGURE_SPEC_AND_STYLE_GUIDE.md')
    definitions = case_definitions(track(root / 'compact_cases.py'))
    track(root / 'expected_value_test/deploy_ev/predict.py')
    track(root / 'expected_value_test/report_global.py')
    old_provenance = json.loads(track(root / 'compact_cases/provenance.json').read_text())
    statistics_path = root / 'compact_cases/case_statistics.csv'
    statistics = pd.read_csv(track(statistics_path), float_precision='round_trip') if statistics_path.exists() else None
    manifest = json.loads(track(golden / 'manifest.json').read_text())
    records = manifest['cases']
    if (manifest.get('status') != 'PASS' or len(records) != len(CASES)
            or {r['case'] for r in records} != set(CASES)):
        raise ValueError('Golden manifest must contain exactly the three verified cases')
    for case, (_, _, bounds) in CASES.items():
        record = next(r for r in records if r['case'] == case)
        matched = [(i, item) for i, item in enumerate(definitions, start=1)
                   if Path(item[1]).name == record['source_file']]
        if len(matched) != 1:
            raise ValueError('Missing or ambiguous retained case input: ' + case)
        number, (label, relative, _, extent) = matched[0]
        if tuple(extent) != bounds:
            raise ValueError('Source bounds differ from Fig8 reviewer zoom: ' + case)
        hdf = track(workspace / relative)
        old = [r for r in old_provenance if r.get('path') == relative]
        if len(old) != 1 or old[0]['sha256'] != sources[str(hdf)] or record['source_sha256'] != sources[str(hdf)]:
            raise ValueError('Retained HDF hash differs from golden/case provenance')
        identity = physical_metadata(hdf, case)
        frame, shape, _ = read_granule(hdf)
        if list(shape) != record['swath_shape'] or len(shape) != 2 or len(frame) != np.prod(shape):
            raise ValueError('Retained swath shape differs from golden manifest')
        output_name = f'DBDust_v{CONFIG["version"]}_{case}_L2.nc'
        if record['output_file'] != output_name:
            raise ValueError('Unexpected golden filename')
        path = track(golden / output_name)
        if sources[str(path)] != record['output_sha256']:
            raise ValueError('Golden output hash mismatch')
        with nc.Dataset(path) as ds:
            if (ds.source_file != hdf.name or ds.source_sha256 != sources[str(hdf)]
                    or json.loads(ds.swath_shape) != list(shape)
                    or ds.reference_definition != CONFIG['reference']['name']
                    or ds.model_manifest_hash != CONFIG['artifacts']['manifest_sha256']):
                raise ValueError('Golden metadata mismatch')
            columns = ['lat', 'lon', *CONFIG['outputs']['pixel']]
            values = {}
            for name in columns:
                if ds[name].shape != (len(frame),):
                    raise ValueError('Unexpected golden pixel variable shape: ' + name)
                values[name] = np.asarray(np.ma.filled(ds[name][:].astype(float), np.nan))
        for name in ('lat', 'lon'):
            exact(values[name], frame[name], case + '/HDF/' + name, checks)
        for name, source_name in [('aod550_db', 'DB_AOD550'), ('db_aerosol_type', 'db_aerosol_type')]:
            exact(values[name], frame[source_name], case + '/HDF/' + name, checks)
        west, east, south, north = bounds
        geo = modis_db.valid_geolocation(values['lat'], values['lon'])
        inside = geo & (values['lon'] >= west) & (values['lon'] <= east) & (values['lat'] >= south) & (values['lat'] <= north)
        if not inside.any():
            raise ValueError('Empty requested zoom: ' + case)
        lg = np.full(len(frame), np.nan)
        unavailable = [name for name in FEATURES if name not in frame]
        status, reason = UNAVAILABLE, 'Exact saved lg25_coarse field absent; separate analytic derivation needed.'
        saved_path = root / f'compact_cases/case_{number}_fields.npz'
        if saved_path.exists():
            with np.load(track(saved_path), allow_pickle=False) as saved:
                for name in ('lat', 'lon', 'total'):
                    if saved[name].shape != tuple(shape):
                        raise ValueError('Saved case-array shape mismatch: ' + name)
                    target = 'aod550_db' if name == 'total' else name
                    exact(saved[name].ravel(), values[target], case + '/saved/' + name, checks)
                if 'lg25_coarse' in saved:
                    if saved['lg25_coarse'].shape != tuple(shape):
                        raise ValueError('Saved LG25 swath shape mismatch')
                    if 'DB_AE' not in frame:
                        reason = 'Cannot verify saved LG25 without retained AE; no substitution.'
                    else:
                        lg = saved['lg25_coarse'].ravel().copy()
                        validate_saved_e7(lg, frame, checks)
                        status, reason = 'verified_saved_E7', ''
        if status == UNAVAILABLE:
            unavailable += ['lg25_daod', 'ev_minus_lg25']
        elif statistics is not None:
            row = statistics.loc[statistics['case'].eq(label) & statistics.method.eq('lg25_coarse')]
            if len(row) != 1:
                raise ValueError('Missing or ambiguous saved LG25 zoom summary')
            good = inside & np.isfinite(lg)
            observed = dict(evaluated_pixels=int(good.sum()), zero_pixels=int((good & (lg == 0)).sum()),
                            valid_total_pixels=int((inside & (values['aod550_db'] > 0)).sum()))
            for key, value in observed.items():
                exact(value, row.iloc[0][key], case + '/saved_summary/' + key, checks)
            # CSV round-trip tolerance, not a scientific threshold.
            if not np.isclose(lg[good].mean(), row.iloc[0].mean_own_support, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError('Saved LG25 zoom mean mismatch')
            checks.append(dict(check=case + '/saved_summary/mean_own_support', status='PASS'))
        for name in FEATURES:
            values[name] = frame[name].to_numpy(float) if name in frame else np.full(len(frame), np.nan)
        values['lg25_daod'] = lg
        paired = np.isfinite(values['daod550_ev']) & np.isfinite(lg)
        values['ev_minus_lg25'] = np.where(paired, values['daod550_ev'] - lg, np.nan)
        values['full8_branch'] = values['branch'] == CONFIG['branches']['full8']['code']
        values['ml_available'] = np.isin(values['branch'], [b['code'] for b in CONFIG['branches'].values()])
        values['ssa_all_available'] = np.logical_and.reduce([np.isfinite(values[f'DB_SSA{w}']) for w in CONFIG['inputs']['spectral_wavelengths_nm']])
        values['ev_missing'] = ~np.isfinite(values['daod550_ev'])
        values['lg25_missing'] = ~np.isfinite(lg)
        values['difference_missing'] = ~paired
        index = np.arange(len(frame))
        values.update(pixel_index=index, swath_row=index // shape[1], swath_col=index % shape[1])
        display = pd.DataFrame({key: value[inside] for key, value in values.items()})
        display = display.assign(case=case, **identity, swath_rows=shape[0], swath_cols=shape[1],
                                 zoom_west=west, zoom_east=east, zoom_south=south, zoom_north=north,
                                 lg25_status=status)
        frames.append(display)
        summaries.append(dict(case=case, **identity, swath_shape=list(shape), zoom_bounds=list(bounds),
                              zoom_boundary_rule='closed_all_edges', zoom_pixels=int(inside.sum()),
                              unlocatable_swath_pixels=int((~geo).sum()),
                              missing_aod_in_zoom=int((inside & ~np.isfinite(values['aod550_db'])).sum()),
                              ev_pixels_in_zoom=int((inside & ~values['ev_missing']).sum()),
                              lg25_pixels_in_zoom=int((inside & np.isfinite(lg)).sum()),
                              paired_pixels_in_zoom=int((inside & paired).sum()),
                              lg25_status=status, lg25_limitation=reason, unavailable_fields=unavailable,
                              optional_feature_finite_pixels={name: int(np.isfinite(display[name]).sum()) for name in FEATURES}))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError('Source changed during derivation: ' + path)
    return pd.concat(frames, ignore_index=True), sources, dict(cases=summaries, checks=checks,
        unprovided_fields=['true_color_RGB', 'per_pixel_acquisition_time'],
        lg25_formula_coefficients=CONFIG['baselines']['li_ginoux_coefficients'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--golden-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-spec', type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    forbidden = [args.source_root.resolve(), args.golden_root.resolve(), Path(__file__).resolve().parents[1]]
    if any(output.is_relative_to(p) for p in forbidden):
        raise ValueError('Write display arrays outside frozen sources, golden inputs and public worktree')
    targets = [output / ('case_display' + suffix) for suffix in ('.csv', '.provenance.json')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Case display already exists; choose a fresh output directory')
    table, sources, details = derive(args.source_root, args.golden_root, args.figure_spec)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.case-', dir=output) as temporary:
        path = Path(temporary) / 'case_display.csv'
        table.to_csv(path, index=False)
        metadata = dict(output_sha256=digest(path), sources=sources, semantics=SEMANTICS,
                        script_sha256=digest(Path(__file__)), rows=len(table), **details)
        path.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        for staged in sorted(Path(temporary).iterdir()):
            os.replace(staged, output / staged.name)
    print(f'Wrote {len(table)} case-display pixels; {len(details["checks"])} checks passed')
    for case in details['cases']:
        if case['unavailable_fields']:
            print(case['case'], 'unavailable:', ', '.join(case['unavailable_fields']))


if __name__ == '__main__':
    main()
