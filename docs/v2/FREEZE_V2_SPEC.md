# DBDust v2.0.0: freeze specification

Instructions for the implementation agent. Version of this spec: 2026-10-13.
Working name of the algorithm: **DBDust** (the authors may rename it before the
release is tagged; a rename is a find-and-replace and must happen before tagging).

## 0. Purpose and definition of "frozen"

The v2 revision work (site-grouped validation, predictor ablation, the
retrieval-space versus AERONET-space decomposition, and the expected-value
experiment with its 2017 global application) produced the method the authors
will describe in the paper. It currently lives in several experimental
directories (`grouped_validation/`, `expected_value_test/` including
`stage_b_support/` and `global_application/`). This task turns that method into
one versioned, documented, reproducible release, **v2.0.0**, without changing
any science.

"Frozen" means all of the following are fixed, hashed and versioned together:

1. the reference definition and every constant in it;
2. the input specification (MODIS fields, scaling, screening);
3. the branch logic and feature order;
4. every model artifact (XGBoost files, calibration maps, smearing tables,
   constants) and the training-membership lists that produced them;
5. the output definitions (pixel and monthly gridded), including missing-value
   semantics;
6. the software environment;
7. the reference outputs used for regression tests.

After v2.0.0 is tagged, any change to items 1 to 6 requires a new version number
and an entry in `CHANGELOG.md`.

## 1. Ground rules

- **No new science.** No refitting, no hyperparameter search, no new thresholds,
  no new selection between estimators, no new masks. Package what was already
  fitted and selected. If you find something that seems to require refitting,
  stop and ask (Section 12).
- **Reproduce, then package.** Every artifact placed in the release must be
  shown to reproduce the outputs that the experimental code produced (Section 8).
- **Keep v1 intact.** `v1.0.0` is already tagged on `main` (commit `89aaa72`).
  Do not rewrite history, do not delete or move v1 scripts in this release.
  The v2 code goes in a new package directory alongside them.
- **This repository is public.** Do not commit manuscript drafts, response
  letters, reviewer comments, internal assessments, unpublished result tables,
  credentials (`.netrc`, tokens), raw MODIS or AERONET files, or large data.
  Large artifacts go to the archive (Section 11). Code, configuration, small
  model files and documentation belong here.
- **Do not create a GitHub release or tag on the remote without author
  approval.** If the repository is connected to Zenodo, publishing a GitHub
  release can mint a DOI automatically.
- Record every decision you make while packaging in `docs/v2/FREEZE_REPORT.md`
  (Section 13).

## 2. Frozen scientific definition

### 2.1 AERONET dust reference (primary: `experimental1020`)

Source: AERONET Version 3 Level 2.0 almucantar inversions (which implies
AOD440 > 0.4 for every reference-bearing record).

1. Dust backscatter fraction at 1020 nm from the particle linear
   depolarization ratio (PLDR), two-component mixing (Tesche et al., 2009;
   Shin et al., 2019a):
   `b_d = ((d - d_nd) * (1 + d_d)) / ((d_d - d_nd) * (1 + d))`,
   with endmembers `d_nd = 0.02`, `d_d = 0.30`, where `d` is the AERONET PLDR at
   1020 nm.
2. Dust extinction fraction at 1020 nm: `f_1020 = clip(b_d * S_d / S_p, 0, 1)`,
   where `S_p` is the AERONET particle lidar ratio at 1020 nm and
   `S_d = 47.308 sr`.
3. `S_d` calibration (already done; freeze it, do not redo it): record-weighted
   median of AERONET LR1020 over 3,998 pre-2010 records at 27 reserved
   calibration sites, selecting records with dust backscatter fraction > 0.9 and
   LR1020 in [10, 150] sr. All 160 reserved collocation sites are excluded from
   training and evaluation. Site reservation uses the SHA-256 threshold rule in
   `grouped_validation/calibration.json`.
4. Transfer to 550 nm: `DAOD_1020 = f_1020 * AOD_1020`;
   `DAOD_550 = DAOD_1020 * (1020/550)**0.06`; `AOD_550` is log-interpolated
   from AERONET AOD at 440 and 675 nm; `f_550 = clip(DAOD_550 / AOD_550, 0, 1)`.
5. Dust label: `f_550 >= 0.2`.
6. Reference DAOD for evaluation: `f_550 * AERONET AOD_550` (never the MODIS AOD).

Verify that the code implements exactly these equations (including where the
clips are applied) and that this section matches the code. If the code differs
from this text in any way, the code that produced the validated results wins,
this document is corrected, and the difference is recorded in the freeze report.

**Sensitivity reference (`old675LR56`)**: the v1 reference (675 nm, `S_d = 56 sr`,
675-nm fraction used as the 550-nm fraction). Keep its models and outputs in a
clearly labeled `sensitivity/` location. It is not part of the v2.0.0 product.

### 2.2 Satellite inputs

- MODIS Collection 6.1 Deep Blue Level 2 aerosol product over land
  (`MYD04_L2` for the 2017 product; the training collocations include both
  `MOD04_L2` and `MYD04_L2`; document the exact split).
- Fields (apply HDF scale factors, offsets and fill values from the file
  attributes): `Deep_Blue_Aerosol_Optical_Depth_550_Land`;
  `Deep_Blue_Spectral_Aerosol_Optical_Depth_Land` at 412, 470, 660 nm;
  `Deep_Blue_Angstrom_Exponent_Land`;
  `Deep_Blue_Spectral_Single_Scattering_Albedo_Land` at 412, 470, 660 nm;
  the Deep Blue aerosol type (Mixed, Dust, Smoke, Sulfate) as decoded in the
  experimental code (QA byte 4, bits 3 to 4). Confirm the exact SDS used for
  the aerosol type and document it.
- Screening used in the 2017 global application: finite, nonnegative DB AOD550
  with valid geolocation; no additional DB confidence threshold (confidence and
  "useful" flags are recorded as diagnostics only).

**Required consistency check (blocking):** confirm from code and data that the
training collocations used the same AOD field and the same screening as the
global application. Report the fraction of training rows that would fail the
application screening and vice versa. If the two differ in a way that changes
which retrievals are eligible, stop and ask before continuing.

### 2.3 Branches

Feature order is fixed:

| Branch | Features (in this order) | Used when |
|---|---|---|
| `full8` | AOD550, AOD412, AOD470, AOD660, AE, SSA412, SSA470, SSA660 | all eight finite |
| `noSSA5` | AOD550, AOD412, AOD470, AOD660, AE | not `full8`, and all five finite |
| none | | otherwise: machine-learning outputs are missing (not zero) |

The `minimal2` fallback (AOD550, AE) is **not** part of the v2.0.0 product. Keep
it only as a labeled sensitivity. The branch is decided from input validity
only, never from a prediction.

### 2.4 Model artifacts per branch

All primary artifacts are the `experimental1020` all-noncalibration fits that
were frozen for deployment in `expected_value_test` (Stage B deployment freeze)
and used for the 2017 application. For each of `full8` and `noSSA5`:

| Artifact | Description |
|---|---|
| Classifier | XGBoost binary classifier for `f_550 >= 0.2` (fixed configuration: 192 trees, depth 4; record every other parameter) |
| Positive regressor | XGBoost regressor of logit(`f_550`) trained on dust-labeled rows only |
| Smearing table | empirical logit-residual set (or its exact interpolation table) used to compute the retransformation-corrected positive fraction `f_pos*` |
| Isotonic map | probability calibration map from inner cross-validation (used by the mixture estimator) |
| `f_neg` | mean reference fraction of non-dust training rows |
| Direct regressor | XGBoost regressor of `f_550` on all rows (squared error), clipped to [0, 1] |
| Selected expected-value form | `full8`: calibrated mixture (E3c); `noSSA5`: direct regression (E4). These choices came from inner validation only and are frozen. |

Record for each artifact: file name, SHA-256, training-membership list
(row IDs and site IDs), the source commit that produced it, and the library
versions used to fit it.

### 2.5 Pixel-level outputs (definitions)

Let `AOD` be DB AOD550 for the pixel and `P` the raw classifier probability.

| Output | Definition |
|---|---|
| `branch` | 0 none, 1 `full8`, 2 `noSSA5` |
| `dust_probability` | raw classifier probability `P` (not the isotonic-calibrated value) |
| `qa_tier` | 0 if `P < 0.4`; 1 if `0.4 <= P < 0.6`; 2 if `0.6 <= P < 0.7`; 3 if `P >= 0.7` |
| `dust_fraction_conditional` | unsmeared positive-regressor fraction `f_pos` |
| `daod550_qa2` | `f_pos * AOD` if `P >= 0.6`, else 0 (this is estimator E1 exactly) |
| `daod550_ev` | expected-value fraction from the selected form for the pixel's branch, times `AOD` (estimator Eselected exactly) |
| `aod550_db` | echo of the input AOD |
| `db_aerosol_type` | echo of the decoded Deep Blue aerosol type |

Missing semantics: if `AOD` is missing, every output is missing. If `AOD` is
valid and `branch == 0`, all machine-learning outputs are missing (NaN), never
zero. A valid pixel rejected by the QA2 gate has `daod550_qa2 = 0`.
All fractions are multiplied by `AOD` exactly once.

### 2.6 Monthly gridded product (L3)

- Grids: 1 x 1 degree (centers from -89.5 to 89.5 and -179.5 to 179.5) and
  2 x 5 degree (centers from -89 to 89 and -177.5 to 177.5), the latter for
  comparison with CALIOP.
- Weighting: arithmetic mean over all supported retrievals in the cell and
  month (retrieval-count weighting). This differs from equal-day weighting in
  older products; state it in the file attributes.
- Variables (names are proposals; use CF-compliant names, units "1" for
  dimensionless quantities, and `_FillValue` for missing):
  `daod550_ev`, `daod550_qa2` (zeros included), `aod550_db` (mean over
  ML-supported retrievals), `n_valid_aod`, `n_ml`, `n_full8`, `n_noSSA5`,
  `frac_full8` (of `n_ml`), `frac_ml_missing` (of `n_valid_aod`),
  `dust_probability_mean`, `frac_qa1`, `frac_qa2`, `frac_qa3` (fractions of
  `n_ml` with `P` at or above 0.4, 0.6, 0.7), and the low-loading diagnostics
  `ev_contribution_aod_lt_0p2` and `ev_contribution_p_lt_0p3` (subset sums
  divided by the full supported count).
- Cells with no supported retrieval are missing, not zero.
- One file per grid per month, named
  `DBDust_v2.0.0_Aqua_L3M_{1x1|2x5}_YYYYMM.nc`, with global attributes for
  version, reference definition, branch rule, weighting, screening, missing
  conventions, source code commit and model manifest hash.

The baselines (DB type-only, published and screened Li-Ginoux, DB-space
Angstrom-exponent refit) and the comparison supports remain in the analysis
outputs, not in the product files.

## 3. Target repository layout

```
dbdust/                      # installable package (pyproject.toml at repo root)
  __init__.py                # __version__ = "2.0.0"
  config.py                  # loads configs/dbdust_v2.0.0.yaml; no hidden constants
  reference.py               # AERONET reference construction (Section 2.1)
  modis_db.py                # reading, scaling, screening, aerosol-type decoding
  branches.py                # branch logic and feature order
  predictor.py               # EVPredictor: loads artifacts, returns pixel outputs
  product_l2.py              # per-granule pixel output writer (used for case studies)
  product_l3.py              # monthly accumulation and NetCDF writer
  baselines.py               # analysis-only baselines (not product)
  cli.py                     # dbdust-apply, dbdust-l3, dbdust-verify
configs/dbdust_v2.0.0.yaml   # every constant in Sections 2.1 to 2.6
models/v2.0.0/               # small model files + MANIFEST.sha256 (or pointers if large)
models/sensitivity/          # old675LR56 and minimal2 artifacts, clearly labeled
tests/                       # unit, regression, golden-case tests (Section 8)
docs/v2/ALGORITHM.md         # algorithm description (ATBD-style)
docs/v2/MODEL_CARD.md        # intended use, training domain, known limitations
docs/v2/DATA_FORMAT.md       # pixel and L3 variables, units, conventions
docs/v2/FREEZE_REPORT.md     # what you did, hashes, test results, open issues
CHANGELOG.md
CITATION.cff
environment.yml and requirements-lock.txt
```

v1 scripts stay where they are. Add a "Versions" section to `README.md` that
explains v1.0.0 (submitted method) versus v2.0.0 (this release) in a few lines
and points to `docs/v2/`.

## 4. Configuration file

`configs/dbdust_v2.0.0.yaml` is the single source of truth. It contains at least:
reference wavelength, endmembers, `S_d`, transfer exponent, interpolation rule,
clip rules, label threshold; MODIS field names and screening rule; branch
definitions and feature order; QA thresholds (0.4, 0.6, 0.7); the selected
expected-value form per branch; model file names and their SHA-256; output
variable names, units and missing conventions; grid definitions; weighting rule.
No constant may be hard-coded elsewhere. A unit test must fail if a module
uses a numeric literal that duplicates a config value.

## 5. Software environment

Pin the environment that produced the validated predictions: Python version,
`xgboost`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `xarray`, `netCDF4`,
`pyhdf` (or the HDF4 reader actually used), and anything else imported.
Provide `environment.yml` and a fully pinned `requirements-lock.txt`. Record
the platform (macOS arm64) in the freeze report. Note that predictions can
differ in the last bits across platforms and library versions (Section 8 sets
tolerances).

## 6. Provenance

`models/v2.0.0/MANIFEST.sha256` lists every artifact with its hash. The freeze
report links each artifact to: the source directory and commit that produced
it (including preregistration commit `e40f71a` for the expected-value
experiment), its training-membership file, and the verification manifests
already produced (`grouped_validation/verification_manifest.json`,
`expected_value_test/stage_b_support/verification.json`).

## 7. Documentation to write

- `ALGORITHM.md`: the method in the order of Sections 2.1 to 2.6, with
  equations, the validation design (site-grouped five-fold cross-validation
  repeated with seeds 20261012 and 20261013, leave-region-out, granule purging,
  site bootstrap), and why the expected-value field exists (thresholded fields
  are not conditional means and are biased for averaging).
- `MODEL_CARD.md`: intended uses (pixel probability and QA tiers for event and
  screening use; expected-value field for averages); training domain
  (AERONET AOD440 > 0.4; retrievals at AERONET sites); known limitations
  (absolute scale depends on the dust lidar ratio of the reference; mixed
  dust, pollution and smoke regimes; heavy smoke that Deep Blue types as dust;
  low-AOD retrievals are an extrapolation; the 2017 application is a
  demonstration year). Point to the paper for quantitative performance once
  published; do not put unpublished numbers in the public repository.
- `DATA_FORMAT.md`: every pixel and L3 variable with definition, units,
  missing semantics and an example `ncdump -h`.
- `CHANGELOG.md`: v1.0.0 to v2.0.0 differences: reference (from 675 nm with
  56 sr and a wavelength-constant fraction, to 1020 nm with the calibrated
  47.308 sr and an explicit spectral transfer to 550 nm); validation (random
  split replaced by site-grouped and leave-region-out cross-validation);
  predictors (eight predictors everywhere replaced by the `full8`/`noSSA5`
  hierarchy with a branch flag); expected-value field added; missing kept
  missing (v1 converted missing predictions to zero); QA gating on the raw
  probability with tiers 0.4, 0.6 and 0.7.
- `CITATION.cff` with all authors.

## 8. Verification and regression tests (acceptance criteria)

All tests run with `pytest` from the repo root and must pass before the
release is proposed.

1. **Unit tests.** Reference equations on hand-computed inputs (including
   clip edges); MODIS scaling and fill handling on a synthetic record; branch
   assignment (every combination of missing inputs); `qa_tier` boundaries
   (0.4, 0.6, 0.7 exactly); E1 and expected-value formulas; L3 accumulator
   (sums, counts, missing-versus-zero) on a synthetic month.
2. **Reference reproduction.** Rebuilding the `experimental1020` targets from
   the cached AERONET inputs reproduces the training-table values exactly
   (bit-identical or `|diff| <= 1e-12`).
3. **Cross-validation reproduction.** Running the packaged predictor on the
   cached out-of-fold inputs reproduces the stored out-of-fold probabilities
   and fractions (`|diff| <= 1e-6`), and recomputing the metrics reproduces the
   saved metric tables in `grouped_validation/` and `expected_value_test/`
   to the precision stored. Do not copy metric values into this repository;
   the test reads them from their existing local location and is skipped
   (with a clear message) when that location is absent.
4. **Deployment reproduction.** Re-run the packaged L2-to-L3 pipeline for two
   complete months (2017-01 and 2017-07) of Aqua data and compare with the
   Stage C monthly outputs for the `experimental1020` hierarchy: counts equal
   exactly; sums within relative 1e-6; QA-tier differences only where
   `|P - threshold| < 1e-6`. If full months are too slow, use a fixed,
   documented random sample of at least 300 granules per month and justify it.
5. **Golden cases.** Produce pixel outputs (Section 2.5) for the three case
   granules used in the paper: Kansas, 14 March 2025 (Terra); Amazonia,
   22 September 2024 (Terra; note the RGB thumbnail file name carries a 2025
   date and must be corrected); North America, 29 August 2017. Store their
   checksums; the test re-creates them and compares.
6. **Product checks.** CF-compliance check on one L3 file; value ranges
   (`0 <= daod550 <= aod550_db`, `0 <= P <= 1`); `n_ml <= n_valid_aod`;
   missing cells are missing in every DAOD variable; zero counts give missing
   means.

## 9. Producing the paper's product files

If Sections 6 and 8 confirm that the packaged code and artifacts are identical
(by hash) to those that produced the Stage C run, you may repackage the 13
existing Stage C months (December 2016 to December 2017) into the v2.0.0 product
files of Section 2.6 without reprocessing, and record the mapping from Stage C
variables to product variables. Otherwise rerun those months with the tagged
code. Either way, regenerate the case-study pixel files with the tagged code.

## 10. Git workflow

1. This spec arrives on branch `v2-freeze-spec`. Branch `release/v2.0.0` from
   `main` (or from `v2-freeze-spec`), implement, and commit in small logical
   commits.
2. Push `release/v2.0.0` and open a pull request to `main` describing what is
   included and the test results. Do not merge it yourself.
3. After the authors approve: merge, tag `v2.0.0` on the merge commit, and only
   then (with explicit approval) create the GitHub release and the archive
   version.

## 11. Archive

Prepare, but do not publish, a new version of the existing Zenodo record
(v1 is record 21223044) containing: the tagged source, model artifacts and
manifest, configuration, the AERONET-MODIS collocation table with reference
values (with AERONET attribution and the list of sites and principal
investigators), the 2017 L3 product files, `DATA_FORMAT.md` and the model card.
Reserve the DOI if possible so it can be cited in the paper's Open Research
section. Check that redistributing the collocation table is consistent with
AERONET's data policy, and record the conclusion.

## 12. Stop and ask the authors before

- continuing after the blocking consistency check in Section 2.2 fails;
- refitting, recalibrating or reselecting anything;
- changing any threshold, endmember, constant, branch rule or output definition;
- accepting a reproduction failure outside the tolerances in Section 8;
- merging to `main`, tagging on the remote, creating a GitHub release, or
  publishing the archive;
- redistributing any third-party data whose license or policy is unclear.

## 13. Report back

Write `docs/v2/FREEZE_REPORT.md` in the repository (technical: hashes, test
results, decisions, deviations, open issues) and a short status note in the
authors' working directory as instructed separately. The report must list
anything in this spec that did not match the code, and how it was resolved.

## References

Shin, S.-K., Tesche, M., Noh, Y., and Müller, D. (2019a). Technical note:
Absorption aerosol optical depth components from AERONET observations of mixed
dust plumes. Atmospheric Measurement Techniques, 12, 607-618.
https://doi.org/10.5194/amt-12-607-2019

Tesche, M., Ansmann, A., Müller, D., Althausen, D., Engelmann, R.,
Freudenthaler, V., and Groß, S. (2009). Vertically resolved separation of dust
and smoke over Cape Verde using multiwavelength Raman and polarization lidars
during Saharan Mineral Dust Experiment 2008. Journal of Geophysical Research:
Atmospheres, 114, D13202. https://doi.org/10.1029/2009JD011862
