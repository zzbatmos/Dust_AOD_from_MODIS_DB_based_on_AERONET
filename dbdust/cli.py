"""Local-file inference and aggregation commands; no download or fitting."""
import argparse
from pathlib import Path
import re

import numpy as np
import xarray as xr

from .config import CONFIG, config_path
from .modis_db import read_granule
from .predictor import EVPredictor, file_hash
from .product_l2 import write_l2
from .product_l3 import MonthlyAccumulator


def _provenance(args, predictor):
    return dict(source_code_commit=args.source_commit,
                model_manifest_hash=predictor.manifest_sha256,
                configuration_sha256=file_hash(config_path()))


def apply_main(argv=None):
    parser = argparse.ArgumentParser(description='Apply frozen DBDust to one retained L2 HDF')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; choose a new file rather than overwriting')
    predictor = EVPredictor()
    frame, shape, _ = read_granule(args.input)
    result = predictor.predict_frame(frame)
    provenance = _provenance(args, predictor)
    provenance.update(source_file=args.input.name, source_sha256=file_hash(args.input), swath_shape=shape)
    write_l2(result, args.output, provenance)
    print(args.output)


def l3_main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate a local list of L2 files for one month')
    parser.add_argument('--manifest', type=Path, required=True, help='One HDF path per line, no duplicates')
    parser.add_argument('--month', required=True, help='YYYY-MM')
    parser.add_argument('--grid', choices=list(CONFIG['grids']), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    files = [Path(line.strip()).expanduser().resolve() for line in args.manifest.read_text().splitlines() if line.strip()]
    if not files or len(files) != len(set(files)):
        parser.error('Manifest must contain unique inputs')
    import datetime
    acquisitions = set()
    for path in files:
        match = re.match(r'MYD04_L2\.A(\d{4})(\d{3})\.(\d{4})\.061\.', path.name)
        if not match:
            parser.error(f'Expected Aqua Collection 6.1 input: {path.name}')
        year, day, time = match.groups()
        date = datetime.datetime.strptime(year+day, '%Y%j')
        key = (year, day, time)
        if date.strftime('%Y-%m') != args.month or key in acquisitions:
            parser.error('Wrong month or duplicate acquisition in manifest')
        acquisitions.add(key)
    model = EVPredictor()
    acc = MonthlyAccumulator(args.grid)
    for path in files:
        frame, _, _ = read_granule(path)
        acc.update(model.predict_frame(frame))
    provenance = _provenance(args, model)
    provenance.update(input_manifest_sha256=file_hash(args.manifest), input_files=len(files),
                      completeness='Caller-supplied manifest; calendar completeness is not inferred')
    acc.write(args.output, args.month, provenance)
    print(args.output)


def _verify_monthly(ds):
    """Check frozen monthly numerical invariants, without changing any data."""
    names = CONFIG['outputs']['monthly']
    if set(names) - set(ds.data_vars):
        raise ValueError('Missing monthly variables')
    values = {}
    for name in names:
        if ds[name].dims != ('time', 'lat', 'lon') or ds.sizes['time'] != 1:
            raise ValueError(f'{name} must have monthly dimensions (time=1, lat, lon)')
        array = ds[name].values
        if array.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must contain real numeric values')
        values[name] = array

    counts = {}
    for name in names:
        if not name.startswith('n_'):
            continue
        array = values[name]
        if (not np.isfinite(array).all() or (array < 0).any()
                or (array > np.iinfo(np.int32).max).any()
                or (array != np.floor(array)).any()):
            raise ValueError(f'{name} must contain finite nonnegative integral int32 counts')
        # Promote before adding branches: on-disk counts are signed int32.
        counts[name] = array.astype(np.int64)
    n_ml, n_valid = counts['n_ml'], counts['n_valid_aod']
    if (n_ml > n_valid).any():
        raise ValueError('ML count exceeds valid AOD count')
    if (counts['n_full8'] + counts['n_noSSA5'] != n_ml).any():
        raise ValueError('Branch count mismatch')

    for name in names:
        if name in counts:
            continue
        denominator = (counts[CONFIG['aggregation']['missing_fraction_denominator']]
                       if name == 'frac_ml_missing' else n_ml)
        supported = denominator > 0
        array = values[name]
        if not np.isfinite(array[supported]).all():
            raise ValueError(f'{name} must be finite on positive-denominator support')
        if not np.isnan(array[~supported]).all():
            raise ValueError(f'{name} must be missing where its denominator is zero')
        if (array[supported] < 0).any():
            raise ValueError(f'{name} must be nonnegative')
        if name.startswith('frac_') or name == 'dust_probability_mean':
            if (array[supported] > 1).any():
                raise ValueError(f'{name} must be in [0, 1]')

    supported = n_ml > 0
    for name in ('daod550_ev', 'daod550_qa2'):
        if (values[name][supported] > values['aod550_db'][supported]).any():
            raise ValueError(f'{name} exceeds aod550_db')
    for name in names:
        if name.startswith('ev_contribution_'):
            if (values[name][supported] > values['daod550_ev'][supported]).any():
                raise ValueError(f'{name} exceeds daod550_ev')

    for name, numerator, denominator in (
        ('frac_full8', counts['n_full8'], counts[CONFIG['aggregation']['fractions_denominator']]),
        ('frac_ml_missing', n_valid - n_ml, counts[CONFIG['aggregation']['missing_fraction_denominator']]),
    ):
        positive = denominator > 0
        expected = numerator[positive] / denominator[positive]
        if (values[name][positive] != expected).any():
            raise ValueError(f'{name} disagrees with its count ratio')

    qa_names = ['frac_qa' + str(i) for i, _ in enumerate(CONFIG['qa']['thresholds'], start=1)]
    for lower, higher in zip(qa_names, qa_names[1:]):
        if (values[higher][supported] > values[lower][supported]).any():
            raise ValueError(f'QA fractions must be nonincreasing: {higher} exceeds {lower}')
    gate_fraction = values[qa_names[CONFIG['qa']['product_gate_index']]]
    if (values['daod550_qa2'][supported & (gate_fraction == 0)] != 0).any():
        raise ValueError('daod550_qa2 must be zero when no retrieval passes its QA gate')


def verify_main(argv=None):
    parser = argparse.ArgumentParser(description='Verify artifacts and optional L3 product invariants')
    parser.add_argument('product', type=Path, nargs='?')
    args = parser.parse_args(argv)
    predictor = EVPredictor()
    if args.product:
        with xr.open_dataset(args.product) as ds:
            _verify_monthly(ds)
    print('Artifact manifest verified:', predictor.manifest_sha256)
    if args.product:
        print('Product value/count invariants passed (not a complete CF checker).')
