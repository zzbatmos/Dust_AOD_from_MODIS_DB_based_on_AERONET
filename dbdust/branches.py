"""Input-only branch assignment, preserving missing predictions."""
import numpy as np
from .config import CONFIG


def valid_features(frame, names):
    good = np.ones(len(frame), dtype=bool)
    for name in names:
        if name not in frame:
            return np.zeros(len(frame), dtype=bool)
        values = np.asarray(frame[name], dtype=float)
        good &= np.isfinite(values)
        if "AOD" in name:
            good &= values >= CONFIG['inputs']['minimum_aod']
        if "SSA" in name:
            low, high = CONFIG['inputs']['ssa_bounds']
            good &= (values >= low) & (values <= high)
    return good


def assign_branches(frame):
    result = np.zeros(len(frame), dtype=np.int8)
    for spec in CONFIG['branches'].values():
        use = (result == 0) & valid_features(frame, spec['features'])
        result[use] = spec['code']
    return result


def qa_tiers(probability):
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(p)
    if np.any(valid & ((p < 0) | (p > 1))):
        raise ValueError('Probability outside [0, 1]')
    out = np.full(p.shape, CONFIG['qa']['missing_tier'], dtype=np.int8)
    out[valid] = np.searchsorted(CONFIG['qa']['thresholds'], p[valid], side='right')
    return out
