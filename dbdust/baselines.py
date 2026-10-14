"""Analysis-only partitions, never part of the DBDust product definition."""
import numpy as np
from .branches import valid_features
from .config import CONFIG


def li_ginoux(frame, screened=True):
    """Frozen E6 (FMF and SSA screened) or E7 (unscreened sensitivity).

    A failed screen is zero. A missing required input is missing. The frozen
    experiment used FMF <= 0.7, not rounded AE < 1.42; do not conflate these.
    """
    spec = CONFIG['baselines']
    a, b, c = spec['li_ginoux_coefficients']
    ae = frame.DB_AE.to_numpy(float)
    fmf = np.clip(c*ae**2+b*ae+a, 0, 1)
    valid = valid_features(frame, ['DB_AOD550', 'DB_AE'])
    fraction = 1-fmf
    if screened:
        first, second = spec['screened_ssa_order']
        valid &= valid_features(frame, [first, second])
        if first not in frame or second not in frame:
            return np.full(len(frame), np.nan)
        passed = (fmf <= spec['fmf_maximum']) & (frame[first].to_numpy() < frame[second].to_numpy())
        fraction = np.where(passed, fraction, 0)
    return np.where(valid, fraction*frame.DB_AOD550.to_numpy(float), np.nan)
