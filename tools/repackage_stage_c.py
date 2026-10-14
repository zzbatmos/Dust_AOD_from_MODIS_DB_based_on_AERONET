"""Lossless remapping of verified Stage C sufficient statistics, without fitting.

Use only after completing the release's reproduction acceptance checks.
The source group is the primary reference's prespecified hierarchy. This
converter refuses missing-ML cells because the older source stores only an
all-AOD sum; it never assumes that sum is an ML-supported sum without proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from dbdust.config import CONFIG, config_path
import dbdust
from dbdust.predictor import file_hash, verify_manifest
from dbdust.product_l3 import MonthlyAccumulator, monthly_filename


def validate_acceptance(report, source_commit, model_root):
    package = Path(dbdust.__file__).resolve().parent
    actual_code = {f'dbdust/{p.name}': file_hash(p) for p in package.glob('*.py')}
    required = dict(reproduction_status='PASS', source_commit=source_commit,
        config_sha256=file_hash(config_path()), model_manifest_sha256=verify_manifest(model_root),
        converter_sha256=file_hash(Path(__file__)), code_sha256=actual_code)
    for name, value in required.items():
        if report.get(name) != value:
            raise ValueError(f'Acceptance report is missing or mismatched: {name}')


def convert(source, destination, month, grid, commit, fingerprint, model_root):
    sidecar = json.loads(source.with_suffix('.sha256.json').read_text())
    if file_hash(source) != sidecar['sha256'] or sidecar['fingerprint'] != fingerprint:
        raise ValueError('Source checksum/fingerprint mismatch')
    acc = MonthlyAccumulator(grid)
    with xr.open_dataset(source) as root:
        if root.attrs['fingerprint'] != fingerprint or root.attrs['month'] != month:
            raise ValueError('Source provenance/month mismatch')
        for coord, edges in [('lat', acc.lat_edges), ('lon', acc.lon_edges)]:
            centers = (edges[:-1]+edges[1:])/2
            if coord not in root.coords or root[coord].dims != (coord,) or not np.array_equal(root[coord].values, centers):
                raise ValueError(f'Source {coord} coordinate/order differs from the configured grid')
    mapping = {
        'n_valid_aod': 'valid_aod_count', 'n_ml': 'p_count',
        'n_full8': 'full8_count', 'n_noSSA5': 'noSSA5_count',
        'frac_qa1': 'gate04_count', 'frac_qa2': 'gate06_count', 'frac_qa3': 'gate07_count',
        'daod550_ev': 'own__Eselected__sum', 'daod550_qa2': 'own__E1__sum',
        'aod550_db': 'aod_sum', 'dust_probability_mean': 'p_sum',
        'ev_contribution_aod_lt_0p2': 'own__Eselected__low_aod_sum',
        'ev_contribution_p_lt_0p3': 'own__Eselected__low_p_sum',
    }
    with xr.open_dataset(source, group=CONFIG['reference']['name']+'/hierarchy') as ds:
        for variable in {*mapping.values(), 'none_count', 'minimal2_count', 'ml_missing_count',
                         'own__Eselected__count', 'own__E1__count'}:
            if variable not in ds or ds[variable].dims != ('lat', 'lon'):
                raise ValueError(f'Unexpected statistic dimension order: {variable}')
        for field in ('none_count', 'minimal2_count', 'ml_missing_count'):
            if not np.all(ds[field].values == 0):
                raise ValueError('Cannot infer ML-supported AOD sum when support differs')
        np.testing.assert_array_equal(ds.valid_aod_count, ds.p_count)
        np.testing.assert_array_equal(ds.p_count, ds.own__Eselected__count)
        np.testing.assert_array_equal(ds.p_count, ds.own__E1__count)
        np.testing.assert_array_equal(ds.p_count, ds.full8_count+ds.noSSA5_count)
        for name, original in mapping.items():
            values = ds[original].values
            if values.shape != acc.shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Unexpected sufficient statistic {original}')
            if name in acc._counts:
                if not np.array_equal(values, values.astype(np.int64)):
                    raise ValueError('Nonintegral count')
                acc._counts[name][:] = values.ravel().astype(np.int64)
            else:
                acc._sums[name][:] = values.ravel()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/monthly_filename(month, grid)
    if target.exists():
        raise FileExistsError(target)
    acc.write(target, month, dict(source_code_commit=commit,
        model_manifest_hash=verify_manifest(model_root), source_file=source.name,
        source_sha256=sidecar['sha256'], source_fingerprint=fingerprint,
        conversion='Repackage verified native-grid sufficient statistics; no interpolation or new inference',
        source_variable_mapping=mapping))
    return dict(file=target.name, sha256=file_hash(target), source=source.name,
                source_sha256=sidecar['sha256'], mapping=mapping)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--expected-fingerprint', required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--acceptance-report', type=Path, required=True,
                        help='Private report showing all required reproduction gates PASS')
    args = parser.parse_args()
    report = json.loads(args.acceptance_report.read_text())
    validate_acceptance(report, args.source_commit, args.models)
    months = ['2016-12']+[f'2017-{m:02d}' for m in range(1, 13)]
    results = []
    for month in months:
        for grid in CONFIG['grids']:
            source = args.source_root/month/f'Aqua_EV_{month}_{grid}.nc'
            results.append(convert(source, args.output, month, grid, args.source_commit,
                                   args.expected_fingerprint, args.models))
    (args.output/'MANIFEST.json').write_text(json.dumps(dict(
        acceptance_report_sha256=file_hash(args.acceptance_report),
        source_commit=args.source_commit, products=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
