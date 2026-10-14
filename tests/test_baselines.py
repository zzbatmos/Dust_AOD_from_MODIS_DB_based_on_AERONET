import numpy as np
import pandas as pd

from dbdust.baselines import li_ginoux
from dbdust.config import CONFIG


def test_frozen_baseline_screening_and_missing():
    frame = pd.DataFrame(dict(DB_AOD550=[1, 1, 1, 1, 1],
        DB_AE=[0, 0, 0, 2, 0], DB_SSA412=[.90, .95, np.nan, .90, .90],
        DB_SSA470=[.95, .90, .95, .95, .95]))
    frame.loc[4, 'DB_AOD550'] = np.nan
    screened = li_ginoux(frame)
    np.testing.assert_allclose(screened, [.949, 0, np.nan, 0, np.nan], equal_nan=True)
    assert np.isfinite(li_ginoux(frame, screened=False)[2])


def test_fmf_boundary_is_not_silently_rounded_to_ae_cutoff():
    c = CONFIG['baselines']
    a,b,q = c['li_ginoux_coefficients']
    ae = (-b+np.sqrt(b*b-4*q*(a-c['fmf_maximum'])))/(2*q)
    assert ae > 1.42
    frame = pd.DataFrame(dict(DB_AOD550=[1], DB_AE=[(ae+1.42)/2],
                              DB_SSA412=[.90], DB_SSA470=[.95]))
    assert li_ginoux(frame)[0] > 0
