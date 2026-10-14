# DBDust 2.0.0 algorithm

This release packages existing fits, not a new optimization. It estimates a
reference-defined dust optical depth from MODIS Deep Blue (DB) retrievals. It
does not independently observe nonsphericity from MODIS radiances. All scientific
constants and feature orders are in `configs/dbdust_v2.0.0.yaml`.

## Reference

The training reference uses AERONET Version 3 Level 2.0 almucantar inversions.
At the reference wavelength, let d be particle linear depolarization ratio and
S_p the retrieved lidar ratio. The implemented sequence is

```
b = clip((d - d_nd) * (1 + d_d) / ((d_d - d_nd) * (1 + d)), 0, 1)
f_native = clip(b * S_d / S_p, 0, 1)
DAOD_native = f_native * AOD_native
DAOD_transferred = DAOD_native * (lambda_native/lambda_output)**AE_dust
f_output = clip(DAOD_transferred / AOD_output, 0, 1)
DAOD_reference = f_output * AOD_output
```

AOD_output is logarithmically interpolated between the configured AERONET input
wavelengths. The backscatter fraction is clipped **before** the lidar-ratio
conversion, matching the validated implementation. This detail was omitted from
the initial specification. Endmembers, calibrated dust lidar ratio and transfer
exponent are frozen. The lidar-ratio calibration sites are reserved by a
deterministic hash rule and excluded from model training and evaluation.
This implements the depolarization partition of Tesche et al. (2009) and
Shin et al. (2019), not a new physical inversion.

## Satellite inputs and spatial support

The input is Collection 6.1 `MOD04_L2` or `MYD04_L2`. Training includes both;
the demonstration monthly product is Aqua. Use all-quality
`Deep_Blue_Aerosol_Optical_Depth_550_Land`, not its Best Estimate counterpart.
Mask file fill values and packed valid-range violations before scaling; valid
AOD must be nonnegative and geolocation valid. No additional DB confidence or
useful-flag cutoff is applied. The verified products have zero scaling offset;
unverified nonzero offsets are rejected, not guessed. SSA is clipped to its
physical bounds after masking to handle floating-point scaling at unity.
`Quality_Assurance_Land` contains the diagnostic aerosol type, decoded with the
configured byte index, bit shift and mask. It is not a predictor.

Training uses independently averaged fields in a 25-km site neighborhood;
application uses pixel vectors. Equal field screening does not eliminate this
spatial-support difference or recover the original per-field joint pixel support.

## Inference

Choose full8 when all eight configured predictors are finite and physically
valid. Otherwise choose noSSA5 when its five predictors are valid. Otherwise
return missing ML fields, not zero. Selection depends only on input availability.
There is no minimal2 fallback, veto, rescue or aerosol-type-conditional branch.

Each branch retains a classifier and positive-fraction regressor. The classifier
estimates the probability of exceeding the configured reference fraction label.
The positive regressor predicts logit fraction, transformed by the logistic
function; this unsmeared conditional fraction defines the thresholded QA fields.
QA tiers use raw, not isotonic-calibrated, probability at the configured boundaries.

For averages, the selected full8 estimator is E3c:

```
f_EV = P_calibrated * f_positive_smeared + (1-P_calibrated) * f_negative_mean
```

The positive term averages the inverse-logit over saved empirical residuals,
using the verified interpolation table inside its range and exact empirical
integration outside. The selected noSSA5 estimator is E4, a direct squared-error
fraction regressor clipped to physical bounds. These choices were made using
inner held-out validation and are not changed during application. Thus the EV
field is a **hybrid of two frozen estimator forms**, not universally P times a
conditional fraction. Every output fraction is multiplied by DB total AOD once.

Thresholded QA2 reports zero for supported pixels below its probability gate.
EV retains their estimated contribution and is intended for averaging; it is
not a declaration that every weak signal is real dust.

## Validation and aggregation

The existing evaluation uses site-grouped five-fold validation with two repeats,
leave-region-out tests, shared-granule purging, nested calibration/selection and
site-bootstrap uncertainty. Evaluate against reference fraction times AERONET
AOD, not fraction times DB AOD. All-noncalibration deployment fits are never
scored as held-out models. Quantitative evidence belongs in the paper and its
private reproducibility outputs until publication.

Monthly products use retrieval-count weighting on the two configured grids,
not an unweighted average of daily means. Total AOD, probabilities and QA
fractions use the same ML-supported population. Counts of valid AOD and missing
ML support are reported separately. Cells with no support remain missing.

## References

Tesche et al. (2009), JGR Atmospheres, doi:10.1029/2009JD011862.

Shin et al. (2019), Atmospheric Measurement Techniques 12, 607-618,
doi:10.5194/amt-12-607-2019.
