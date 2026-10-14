# DBDust 2.0.0 model card

**Status:** release candidate under reproduction testing; not a published release.
The authors must approve merge, tag and archive publication separately.

Use raw probability and QA tiers for event screening. Use the expected-value
DAOD for spatial or temporal means. Always retain the branch and coverage fields.
The reference is `experimental1020`; older `old675LR56` and minimal2 artifacts
are sensitivity analyses, not part of the product. v1 remains unchanged.

## Domain and limitations

- Reference-bearing training inversions are high-loading AERONET observations;
  low-AOD application is extrapolation, not established validation.
- Training observations are clustered around AERONET sites. Site-grouped tests
  address some geographic leakage, not all geographic representativeness.
- The absolute DAOD scale depends on the calibrated dust lidar ratio, mixing
  assumptions and spectral transfer. The reference is an estimate, not error-free truth.
- DB aerosol-model assumptions affect all input features. Smoke typed as dust
  upstream can remain indistinguishable from dust; this is retrieval-space
  estimation, not an independent nonsphericity detector.
- Missing SSA changes the branch and estimator form. Compare branch composition
  before interpreting regional or temporal changes.
- Mixed dust, smoke and pollution regimes remain difficult. A positive EV at
  low loading may be a background floor; no nonzero field is automatically validated dust.
- QA gating suppresses weak-probability contributions. Its mean differs from
  the EV mean by design and should not be called the same climatology.
- The 2017 demonstration is not a validation of trends over the full MODIS record.
- Training neighborhood means and application pixels have different spatial support.

## Provenance and use

Small model files are hash checked against `models/v2.0.0/MANIFEST.sha256`.
Exact parameters and feature order are in the configuration and sanitized model
metadata. Raw AERONET/MODIS data, unpublished evaluation tables, manuscripts and
review correspondence are not included in this repository. Local regression
tests read existing private artifacts only when explicitly configured.

See `ALGORITHM.md`, `DATA_FORMAT.md`, and `FREEZE_REPORT.md`. Cite the final
published paper/archive once approved; do not invent a new release DOI.
