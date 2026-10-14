"""Register Fig. 6 maps from immutable Stage C caches, without running sources.

CLI: python -m paper.derive_seasonal_maps --source-root REVISION_V2_WORK
--output-dir PRIVATE_DERIVED_DIR [--land-shapefile EXISTING_NE_110M_LAND.shp].

The comparison cache has dimensions but no coordinates. Coordinates and CALIOP
come from the unique matched_native_grid.nc recorded in comparison_provenance.
The saved primary mask is never reconstructed or replaced. Original float64
area weights are reproduced from local geometry, checked against BOTH cached
float32 geometry arrays, and used only for metrics/diagnostics. No downloader,
model, original source module, or plotting module is invoked. Shapely and pyshp
are needed for that original-precision geometry check.

derive() returns tables, source hashes, and saved-metric checks. CSVs retain the
entire native grid with explicit indices, bounds, support flags and missing
values outside the three-month primary intersection. Existing outputs are
never overwritten. Region boxes are exact recorded bounds, not new regions.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd

from dbdust.config import CONFIG, config_path
from paper.numbers import digest


REFERENCE = CONFIG['reference']['name']
GROUP = REFERENCE + '__hierarchy__core_common'
REGION = 'Land 60S-60N'
DEFAULT_LAND = Path.home() / '.local/share/cartopy/shapefiles/natural_earth/physical/ne_110m_land.shp'
# Numerical reproducibility tolerance, not a selection threshold.
CHECK_ATOL = 1e-12
META = dict(reference=REFERENCE, variant='hierarchy', support='core_common',
            reference_mask='prior_joint', region=REGION, estimator='Eselected')
SEMANTICS = {
    'seasonal_maps': 'Native 2x5 grid; coordinates from the provenance-selected CALIOP cube, '
        'not absent comparison-cache coordinate variables. EV=Eselected; CALIOP532 transferred '
        'to 550 using the recorded/configured dust AE before equal three-month averaging. '
        'Difference=EV-CALIOP. All three fields have exactly the saved prior_joint monthly '
        'mask intersection and recorded Land 60S-60N bounds. DJF uses December 2016, not '
        'December 2017. Unsupported values are missing; supported zero remains zero. '
        'Full 90x72 grid per season in latitude-major order; indices and cell edges are explicit. '
        'Monthly mask counts, cached land fraction and original float64 land weights are '
        'diagnostics only, never replacement masks. CALIOP is a comparison, not truth.',
    'seasonal_map_metrics': 'Area-weighted spatial statistics on each complete three-month '
        'primary intersection. Original float64 0.5-degree NaturalEarth110m land-center area '
        'weights, not rounded cached weights or cosine-latitude approximations. n counts '
        'spatial cells, not retrievals or cell-months. Every metric checked against the exact '
        'experimental1020/hierarchy/core_common/prior_joint/Eselected/Land 60S-60N saved rows.',
    'seasonal_region_boxes': 'Exact comparison_provenance region bounds, with lower edges '
        'inclusive and upper edges exclusive. Includes the global domain and explicitly '
        'named 50_75N sensitivities; these are not substitutes for primary boreal boxes '
        'clipped at 60N. Outlines do not imply pixel fire screening or independent regions.',
}


def read(variable):
    """Match original compare_global.read: promote before subsequent arithmetic."""
    return np.asarray(np.ma.filled(variable[:].astype(float), np.nan), dtype=float)


def source_literals(path):
    """Read literal definitions without importing/executing original source."""
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PERIODS', 'CORE'):
                    if target.id in result:
                        raise ValueError('Ambiguous source constant ' + target.id)
                    result[target.id] = ast.literal_eval(node.value)
    expected = {'DJF': ['2016-12', '2017-01', '2017-02'],
                'MAM': ['2017-03', '2017-04', '2017-05'],
                'JJA': ['2017-06', '2017-07', '2017-08'],
                'SON': ['2017-09', '2017-10', '2017-11']}
    if result.get('PERIODS') != expected or result.get('CORE') != ['E1', 'Eselected', 'E5', 'E7', 'E8']:
        raise ValueError('Unexpected original seasonal/core definitions')
    return result['PERIODS'], result['CORE']


def spectral_factor(description):
    match = re.fullmatch(r'DAOD(\d+)=DAOD(\d+)\*\((\d+)/(\d+)\)\^\(-([\d.]+)\)', description)
    if match is None:
        raise ValueError('Unrecognized recorded spectral transfer')
    out, incoming, numerator, denominator, ae = map(float, match.groups())
    if (out != numerator or incoming != denominator or incoming <= 0
            or out != CONFIG['reference']['output_wavelength_nm']
            or ae != CONFIG['reference']['dust_angstrom_exponent']):
        raise ValueError('Spectral transfer differs from frozen configuration')
    return (out / incoming) ** (-ae)


def original_geometry(shapefile_path, lat, lon):
    """Reproduce compare_song.geometry weights, never a replacement support mask.

    The 0.5-degree sample spacing and strict polygon interior rule are the
    existing source geometry, not new science or a newly selected coastline.
    Only an explicit existing shapefile is read; no Cartopy resource resolver.
    """
    import shapefile
    import shapely
    from shapely.geometry import shape

    step = .5
    dy, dx = CONFIG['grids']['2x5']
    fine_lat = np.arange(lat[0] - dy / 2 + step / 2, lat[-1] + dy / 2, step)
    fine_lon = np.arange(lon[0] - dx / 2 + step / 2, lon[-1] + dx / 2, step)
    yy, xx = np.meshgrid(fine_lat, fine_lon, indexing='ij')
    with shapefile.Reader(str(shapefile_path)) as reader:
        polygons = shapely.union_all([shape(s.__geo_interface__) for s in reader.shapes()])
    land = shapely.contains_xy(polygons, xx, yy)
    weights = np.broadcast_to((np.sin(np.deg2rad(fine_lat + step / 2)) -
                               np.sin(np.deg2rad(fine_lat - step / 2)))[:, None], land.shape)

    def block(values):
        return values.reshape(len(lat), int(dy / step), len(lon), int(dx / step)).sum(axis=(-3, -1))

    area = block(weights * land)
    return area, area / block(weights)


def weighted_stats(candidate, reference, weights, mask):
    """Original compare_song.stats arithmetic and spatial reduction order."""
    if not np.isfinite(candidate[mask]).all() or not np.isfinite(reference[mask]).all():
        raise ValueError('Nonfinite data on saved seasonal support')
    a, b, w = candidate[mask], reference[mask], weights[mask]
    if len(a) < 2 or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Insufficient or invalid weighted seasonal support')
    w = w / w.sum()
    am, bm = (a * w).sum(), (b * w).sum()
    va, vb = ((a - am) ** 2 * w).sum(), ((b - bm) ** 2 * w).sum()
    return dict(n=len(a), candidate_mean=am, reference_mean=bm, bias=am - bm,
                rmse=np.sqrt(((a - b) ** 2 * w).sum()), mae=(abs(a - b) * w).sum(),
                r=((a - am) * (b - bm) * w).sum() / np.sqrt(va * vb) if va * vb > 0 else np.nan)


def _check(actual, expected, label, checks, exact=False):
    good = actual == expected if exact else np.isclose(actual, expected, atol=CHECK_ATOL, rtol=0, equal_nan=True)
    if not good:
        raise ValueError(f'Saved comparison mismatch: {label}: {actual!r} != {expected!r}')
    checks.append(dict(check=label, status='PASS'))


def _mask(group, name):
    values = read(group[name])
    if not np.isin(values, [0, 1]).all():
        raise ValueError('Invalid saved bitmap: ' + name)
    return values.astype(bool)


def derive(source_root, land_shapefile=DEFAULT_LAND):
    """Return checked maps/metrics/exact boxes, without writing or inference."""
    root = Path(source_root).resolve()
    stage = root / 'expected_value_test/stage_c'
    sources, checks = {}, []

    def track(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = digest(path)
        return path

    track(config_path())
    provenance = json.loads(track(stage / 'comparison_provenance.json').read_text())
    script = track(root / 'expected_value_test/compare_global.py')
    if sources[str(script)] != provenance['script']:
        raise ValueError('Original comparison script hash mismatch')
    periods, core = source_literals(script)
    cubes = [Path(p) for p in provenance['sources'] if Path(p).name == 'matched_native_grid.nc']
    if len(cubes) != 1 or not cubes[0].is_absolute():
        raise ValueError('Missing or ambiguous provenance-selected CALIOP cube')
    cube = track(cubes[0])
    if sources[str(cube)] != provenance['sources'][str(cubes[0])]:
        raise ValueError('CALIOP cube hash mismatch')
    # Actual sibling root, not the release worktree or a guessed data directory.
    track(cube.parent.parent / 'compare_song.py')
    land_shapefile = Path(land_shapefile).resolve()
    for suffix in ('.shp', '.shx', '.dbf', '.prj'):
        track(land_shapefile.with_suffix(suffix))
    if land_shapefile.with_suffix('.cpg').exists():
        track(land_shapefile.with_suffix('.cpg'))
    saved = pd.read_csv(track(stage / 'seasonal_metrics.csv'), float_precision='round_trip')
    for key, value in META.items():
        saved = saved.loc[saved[key].eq(value)]
    if len(saved) != len(periods) or saved.period.duplicated().any() or set(saved.period) != set(periods):
        raise ValueError('Require exactly four unambiguous primary seasonal metric rows')
    regions = provenance['regions']
    if regions.get(REGION) != [-60, 60, -180, 180]:
        raise ValueError('Unexpected primary land-domain bounds')
    if (provenance['spatial_weight'] != '0.5degree NaturalEarth110m land-center area'
            or provenance['seasonal_weight'] != 'equal complete months'):
        raise ValueError('Unexpected comparison weighting semantics')
    with nc.Dataset(track(stage / 'comparison_support.nc')) as ds:
        months = json.loads(ds.months)
        expected_months = ['2016-12'] + [f'2017-{m:02d}' for m in range(1, 13)]
        if months != expected_months:
            raise ValueError('Unexpected comparison-cache months/order')
        factor = spectral_factor(ds.CALIOP_spectral_transfer)
        if GROUP not in ds.groups:
            raise ValueError('Missing exact primary comparison group')
        group = ds.groups[GROUP]
        mask = _mask(group, 'mask')
        modis = _mask(group, 'modis_mask')
        mean_only = _mask(group, 'reference_mean_only_mask')
        fields = {name: read(group[name]) for name in core}
        counts = read(group['counts'])
        for name in ['mask', 'modis_mask', 'reference_mean_only_mask', 'counts', *core]:
            if group[name].dimensions != ('time', 'lat', 'lon'):
                raise ValueError('Unexpected comparison-cache dimension order')
    with nc.Dataset(cube) as ds:
        for name in ('lat', 'lon', 'year', 'month'):
            if ds[name].dimensions != (name,):
                raise ValueError('Unexpected cube coordinate dimensions')
        lat, lon, years, calendar_months = (read(ds[k]) for k in ('lat', 'lon', 'year', 'month'))
        dy, dx = CONFIG['grids']['2x5']
        for name, values, bounds, step in (
            ('lat', lat, CONFIG['inputs']['latitude_bounds'], dy),
            ('lon', lon, CONFIG['inputs']['longitude_bounds'], dx),
        ):
            if not np.array_equal(values, np.arange(bounds[0] + step / 2, bounds[1], step)):
                raise ValueError('Unexpected native grid coordinates: ' + name)
        if len(np.unique(years)) != len(years) or not np.array_equal(calendar_months, np.arange(1, 13)):
            raise ValueError('Ambiguous cube calendar coordinates')
        if mask.shape != (len(months), len(lat), len(lon)):
            raise ValueError('Support and cube grid shapes differ')
        cached_area, cached_fraction = read(ds['land_area_weight']), read(ds['land_fraction'])
        area_dtype, fraction_dtype = ds['land_area_weight'].dtype, ds['land_fraction'].dtype
        external = {k: [] for k in ('cal_mean', 'old', 'cal_low', 'cal_high', 'cal_total', 'old_total')}
        for name in ('land_area_weight', 'land_fraction'):
            if ds[name].dimensions != ('lat', 'lon'):
                raise ValueError('Unexpected cube geometry dimensions')
        for name in external:
            if ds[name].dimensions != ('year', 'month', 'lat', 'lon'):
                raise ValueError('Unexpected CALIOP/ancillary dimension order')
        for month in months:
            year, mon = map(int, month.split('-'))
            matches = np.flatnonzero(years == year)
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous cube year: ' + str(year))
            for name in external:
                values = ds[name][int(matches[0]), mon - 1].astype(float)
                external[name].append(np.asarray(np.ma.filled(values, np.nan)))
        external = {k: np.stack(v) for k, v in external.items()}
    area, fraction = original_geometry(land_shapefile, lat, lon)
    if (not np.array_equal(area.astype(area_dtype).astype(float), cached_area)
            or not np.array_equal(fraction.astype(fraction_dtype).astype(float), cached_fraction)):
        raise ValueError('Local geometry differs from frozen geometry cache; no replacement allowed')
    cal = external['cal_mean'] * factor
    ancillary = np.logical_and.reduce([np.isfinite(external[k]) & (external[k] >= 0)
                                      for k in ('cal_low', 'cal_high', 'cal_total', 'old_total')])
    if not np.array_equal(mask, mean_only & ancillary) or (mean_only & ~modis).any():
        raise ValueError('Saved masks violate prior_joint support semantics')
    if (not np.isfinite(counts[modis]).all() or (counts[modis] <= 0).any()
            or any(not np.isfinite(v[modis]).all() for v in fields.values())):
        raise ValueError('Invalid cached core fields/counts on MODIS support')
    if any((mask & (~np.isfinite(v) | (v < 0))).any() for v in external.values()):
        raise ValueError('Invalid external data on saved primary support')
    lo, hi, left, right = regions[REGION]
    domain = (lat[:, None] >= lo) & (lat[:, None] < hi) & (lon[None, :] >= left) & (lon[None, :] < right)
    iy, ix = np.indices(area.shape)
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    maps, metrics = [], []
    for period, dates in periods.items():
        indices = [months.index(date) for date in dates]
        use = mask[indices].all(axis=0) & domain
        ev, reference = fields['Eselected'][indices].mean(axis=0), cal[indices].mean(axis=0)
        statistics = weighted_stats(ev, reference, area, use)
        row = saved.loc[saved.period.eq(period)].iloc[0]
        for key, value in statistics.items():
            _check(value, row[key], period + '/' + key, checks, exact=key == 'n')
        metrics.append(dict(META, period=period, months='|'.join(dates), **statistics))
        columns = dict(lat_index=iy, lon_index=ix, lat=yy, lon=xx,
                       lat_lower=yy - dy / 2, lat_upper=yy + dy / 2,
                       lon_lower=xx - dx / 2, lon_upper=xx + dx / 2,
                       ev=np.where(use, ev, np.nan), caliop=np.where(use, reference, np.nan),
                       difference=np.where(use, ev - reference, np.nan), supported=use,
                       in_domain=domain, prior_joint_months=mask[indices].sum(axis=0),
                       modis_months=modis[indices].sum(axis=0),
                       mean_only_months=mean_only[indices].sum(axis=0),
                       land_fraction=cached_fraction, land_area_weight=area,
                       has_land=cached_fraction > 0,
                       original_land_purity_eligible=fraction >= provenance['land_purity'])
        maps.append(pd.DataFrame({k: v.ravel() for k, v in columns.items()}).assign(
            **META, period=period, months='|'.join(dates)))
    boxes = []
    for name, bounds in regions.items():
        if (len(bounds) != 4 or not np.isfinite(bounds).all()
                or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]):
            raise ValueError('Invalid exact region bounds: ' + name)
        boxes.append(dict(region=name, lat_lower=bounds[0], lat_upper=bounds[1],
                          lon_lower=bounds[2], lon_upper=bounds[3],
                          sensitivity=name.endswith('_50_75N_sensitivity'),
                          boundary_rule='lower_inclusive_upper_exclusive'))
    tables = dict(seasonal_maps=pd.concat(maps, ignore_index=True),
                  seasonal_map_metrics=pd.DataFrame(metrics), seasonal_region_boxes=pd.DataFrame(boxes))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError('Frozen source changed during derivation: ' + path)
    return tables, sources, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--land-shapefile', type=Path, default=DEFAULT_LAND)
    args = parser.parse_args(argv)
    output, source = args.output_dir.resolve(), args.source_root.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.is_relative_to(source):
        raise ValueError('Write derived arrays outside public worktree and frozen source tree')
    targets = [output / (name + suffix) for name in SEMANTICS for suffix in ('.csv', '.provenance.json')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Seasonal outputs already exist; choose a fresh output directory')
    tables, sources, checks = derive(source, args.land_shapefile)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.seasonal-', dir=output) as temporary:
        for name, table in tables.items():
            path = Path(temporary) / (name + '.csv')
            table.to_csv(path, index=False)
            metadata = dict(output_sha256=digest(path), sources=sources, semantics=SEMANTICS[name],
                            script_sha256=digest(Path(__file__)), rows=len(table), checks=checks,
                            numerical_check_atol=CHECK_ATOL)
            path.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        for path in sorted(Path(temporary).iterdir()):
            os.replace(path, output / path.name)
    print(f'Wrote {len(tables)} seasonal CSVs to {output}; {len(checks)} saved comparisons passed')


if __name__ == '__main__':
    main()
