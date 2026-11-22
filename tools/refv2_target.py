"""Load the fixed matched Level-2 point target; never recompute or select it."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_matched_rows(path, expected_sha256):
    """Require the registered matched-input file before constructing a target."""
    path = Path(path)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError('Matched-input SHA-256 mismatch')
    frame = pd.read_parquet(path)
    required = {'row_id', 'site_key', 'granule', 'R1_quality',
                'calibration_reserved', 'C0_f', 'C5_f'}
    if required - set(frame):
        raise ValueError('Missing frozen matching/target columns')
    if frame.row_id.duplicated().any() or not frame.R1_quality.eq(20).all():
        raise ValueError('Expected unique Level 2 matched records')
    if frame.calibration_reserved.any():
        raise ValueError('Reserved calibration records cannot enter training')
    return frame


def point_target(frame, reference='C5'):
    """C5_f is the target. Monte Carlo percentiles remain uncertainty only."""
    if reference not in ('C0', 'C5'):
        raise ValueError('Only the fixed C5 target and C0 identity control are allowed')
    result = frame.copy()
    target = result[reference + '_f'].to_numpy(float)
    if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError('Point target must be finite in [0, 1]; no row filtering')
    result['model_target'] = target
    result['dust_label'] = (target >= 0.2).astype(np.int8)
    return result
