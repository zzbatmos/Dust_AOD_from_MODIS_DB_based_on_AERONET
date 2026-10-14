"""Register complete golden swath geometry without touching case_display.

CLI: python -m paper.derive_case_geolocation --golden-root GOLDEN_CASES
--output-dir PRIVATE_DERIVED_DIR. No HDF decoding, science fields, fitting,
interpolation, coordinate repair, or registry edits. Missing coordinates are
retained explicitly; renderers must reject them before native pcolormesh.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd

from dbdust.config import CONFIG, config_path
from dbdust.modis_db import valid_geolocation
from paper import derive_case_display
from paper.derive_case_display import CASES
from paper.numbers import digest


SEMANTICS = (
    'Complete original golden swaths, including pixels outside the reviewer zoom and '
    'pixels with missing geolocation. No scientific variables or interpolated/repaired '
    'coordinates. pixel_index is zero-based C-order; swath_row/col and swath_rows/cols '
    'permit exact 2-D reconstruction. inside_zoom uses valid golden geolocation and '
    'closed reviewer bounds, independent of retrieval availability. geolocation_finite '
    'requires both coordinates finite; geolocation_valid additionally checks frozen '
    'physical bounds. Missing coordinates remain missing and inside_zoom=False. '
    'A native pcolormesh renderer must fail if any full-swath coordinates are nonfinite, '
    'rather than filling them or dropping rows. case_display is not read or changed.'
)


def derive(golden_root):
    """Return the complete geometry frame, source hashes and per-case audit."""
    golden = Path(golden_root).resolve()
    sources, frames, summaries = {}, [], []

    def track(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = digest(path)
        return path

    track(config_path())
    track(derive_case_display.__file__)
    manifest = json.loads(track(golden / 'manifest.json').read_text())
    records = manifest['cases']
    if (manifest.get('status') != 'PASS' or len(records) != len(CASES)
            or {r['case'] for r in records} != set(CASES)):
        raise ValueError('Require exactly the three verified golden cases')
    for case, (date, platform, bounds) in CASES.items():
        record = next(r for r in records if r['case'] == case)
        filename = f'DBDust_v{CONFIG["version"]}_{case}_L2.nc'
        if record['output_file'] != filename:
            raise ValueError('Golden filename differs from case identity')
        path = track(golden / filename)
        if sources[str(path)] != record['output_sha256']:
            raise ValueError('Golden file hash mismatch: ' + case)
        granule = re.fullmatch(r'(MOD|MYD)04_L2\.A(\d{4})(\d{3})\.\d{4}\.(\d{3})\.\d+\.hdf', record['source_file'])
        if granule is None:
            raise ValueError('Unrecognized golden source granule')
        product, year, day, collection = granule.groups()
        observed_date = datetime(int(year), 1, 1) + timedelta(days=int(day) - 1)
        if (observed_date.strftime('%Y-%m-%d') != date
                or ('Terra' if product == 'MOD' else 'Aqua') != platform
                or collection != CONFIG['inputs']['collection']):
            raise ValueError('Golden case date/platform/collection mismatch')
        with nc.Dataset(path) as ds:
            shape = json.loads(ds.swath_shape)
            if (len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
                    or shape != record['swath_shape']):
                raise ValueError('Invalid or mismatched golden swath shape')
            if (ds.source_file != record['source_file'] or ds.source_sha256 != record['source_sha256']
                    or ds.reference_definition != CONFIG['reference']['name']
                    or ds.model_manifest_hash != CONFIG['artifacts']['manifest_sha256']):
                raise ValueError('Golden source metadata mismatch')
            coords = {}
            for name in ('lat', 'lon'):
                if ds[name].shape != (int(np.prod(shape)),):
                    raise ValueError('Golden coordinate size differs from swath shape')
                coords[name] = np.asarray(np.ma.filled(ds[name][:].astype(float), np.nan))
        lat, lon = coords['lat'], coords['lon']
        finite = np.isfinite(lat) & np.isfinite(lon)
        valid = valid_geolocation(lat, lon)
        west, east, south, north = bounds
        inside = valid & (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        index = np.arange(len(lat), dtype=np.int64)
        frames.append(pd.DataFrame(dict(case=case, lat=lat, lon=lon, pixel_index=index,
                                        swath_row=index // shape[1], swath_col=index % shape[1],
                                        swath_rows=shape[0], swath_cols=shape[1], inside_zoom=inside,
                                        geolocation_finite=finite, geolocation_valid=valid)))
        summaries.append(dict(case=case, shape=shape, swath_pixels=len(lat),
                              inside_zoom_pixels=int(inside.sum()),
                              missing_geolocation_pixels=int((~finite).sum()),
                              invalid_geolocation_pixels=int((~valid).sum()),
                              zoom_bounds=list(bounds), zoom_boundary_rule='closed_all_edges',
                              source_file=record['source_file'], source_sha256=record['source_sha256']))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError('Golden source changed during extraction: ' + path)
    return pd.concat(frames, ignore_index=True), sources, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--golden-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(p) for p in (args.golden_root.resolve(), Path(__file__).resolve().parents[1])):
        raise ValueError('Write geometry outside golden inputs and public worktree')
    targets = [output / ('case_geolocation' + suffix) for suffix in ('.csv', '.provenance.json')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Case geolocation already exists; choose a fresh output directory')
    table, sources, summaries = derive(args.golden_root)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.case-geo-', dir=output) as temporary:
        path = Path(temporary) / 'case_geolocation.csv'
        table.to_csv(path, index=False)
        metadata = dict(output_sha256=digest(path), sources=sources, semantics=SEMANTICS,
                        script_sha256=digest(Path(__file__)), rows=len(table), cases=summaries)
        path.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        for staged in sorted(Path(temporary).iterdir()):
            os.replace(staged, output / staged.name)
    print(f'Wrote {len(table)} full-swath geometry pixels to {output}')


if __name__ == '__main__':
    main()
