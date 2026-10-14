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


def verify_main(argv=None):
    parser = argparse.ArgumentParser(description='Verify artifacts and optional L3 product invariants')
    parser.add_argument('product', type=Path, nargs='?')
    args = parser.parse_args(argv)
    predictor = EVPredictor()
    if args.product:
        with xr.open_dataset(args.product) as ds:
            if set(CONFIG['outputs']['monthly']) - set(ds.data_vars):
                raise ValueError('Missing monthly variables')
            for name in ('daod550_ev', 'daod550_qa2'):
                good = np.isfinite(ds[name])
                if bool(((ds[name] < 0) | (ds[name] > ds.aod550_db)).where(good, False).any()):
                    raise ValueError(f'Unphysical {name}')
                if bool(ds[name].where(ds.n_ml == 0).notnull().any()):
                    raise ValueError('Unsupported cells are not missing')
            if bool((ds.n_ml > ds.n_valid_aod).any()):
                raise ValueError('ML count exceeds valid AOD count')
            if bool((ds.n_full8+ds.n_noSSA5 != ds.n_ml).any()):
                raise ValueError('Branch count mismatch')
    print('Artifact manifest verified:', predictor.manifest_sha256)
    if args.product:
        print('Product value/count invariants passed (not a complete CF checker).')
