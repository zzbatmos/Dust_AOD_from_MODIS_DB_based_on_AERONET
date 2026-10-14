"""Frozen reference reconstruction, with the original clipping order."""
import numpy as np
from .config import CONFIG


def construct_reference(pldr, lidar_ratio, aod_native, aod_short, aod_long):
    c = CONFIG['reference']
    d, lr, an, ash, alo = np.broadcast_arrays(*[
        np.asarray(x, dtype=float) for x in (pldr, lidar_ratio, aod_native, aod_short, aod_long)])
    good = np.isfinite(d) & np.isfinite(lr) & np.isfinite(an) & np.isfinite(ash) & np.isfinite(alo)
    good &= (lr > 0) & (an > 0) & (ash > 0) & (alo > 0) & (d > -1)
    low, high = c['interpolation_wavelengths_nm']
    with np.errstate(divide='ignore', invalid='ignore'):
        back = np.clip((d-c['nondust_pldr'])*(1+c['dust_pldr']) /
                       ((c['dust_pldr']-c['nondust_pldr'])*(1+d)), 0, 1)
        fn = np.clip(back*c['dust_lidar_ratio_sr']/lr, 0, 1)
        a550 = np.exp(np.log(ash)+(np.log(alo)-np.log(ash))*
                      np.log(c['output_wavelength_nm']/low)/np.log(high/low))
        raw = fn*an*(c['wavelength_nm']/c['output_wavelength_nm'])**c['dust_angstrom_exponent']/a550
        fraction = np.clip(raw, 0, 1)
    return {k: np.where(good, v, np.nan) for k, v in dict(
        backscatter_fraction=back, native_fraction=fn, aod550=a550,
        fraction550=fraction, daod550=fraction*a550).items()}
