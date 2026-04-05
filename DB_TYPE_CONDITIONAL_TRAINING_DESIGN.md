# DB Aerosol-Type-Conditional Dust Training Design

## Purpose

This document defines the next-generation MODIS Deep Blue dust-detection and
dust-AOD training strategy for this repo.

The key change is to use the **actual Deep Blue aerosol type flag** from
`Quality_Assurance_Land` as a **routing variable and prior QA signal**, while
still using **AERONET DPR/LR-derived dust information as the truth source**.

This avoids two weak assumptions:

1. treating the MODIS DB product as a generic feature table with no retrieval
   branch structure
2. treating the MODIS DB aerosol type flag itself as truth

Instead, the design is:

- **AERONET DPR/LR** provides the trusted dust labels and dust-AOD target
- **DB aerosol type** provides retrieval-regime context
- **type-specific ML models** learn within each DB regime
- **final QA** is derived from both DB aerosol type and ML outputs


## Motivation

The DB algorithm does not retrieve all pixels in the same way.

From the QA plan and the two test-granule diagnostics:

- `Deep Blue Aerosol Type` is encoded in `Quality_Assurance_Land`
- the delivered DB fields have different information content by aerosol type
- `Dust` pixels tend to have near-fixed low AE and more complete SSA
- `Smoke`/`Mixed` pixels show more variable AE behavior

This means:

- the same predictor does not mean the same thing across aerosol types
- a single global classifier/regressor is suboptimal
- DB aerosol type is scientifically useful, but imperfect

Our own comparisons already showed:

- DB aerosol type helps separate heavy smoke scenes from dust
- but it cannot be trusted as a standalone dust mask
- some real AERONET dust cases still appear inside DB `Smoke`
- some DB `Dust` pixels appear in smoke-heavy scenes


## Core Training Idea

### Truth source

Use AERONET inversion and SDA data as the truth source:

- AERONET inversion (`AERONET_INV_Level2_v3`) provides:
  - AOD at 440/675/1020 nm
  - SSA
  - lidar ratio
  - depolarization ratio
- AERONET SDA (`AERONET_SDA_v1.5`) provides:
  - total AOD at 500 nm
  - fine mode fraction
  - AE

Dust truth remains based on the DPR/LR method already used elsewhere in the
repo.

### DB aerosol type role

Use DB aerosol type as:

- a routing variable
- a prior QA signal
- an information-content flag

Do **not** use DB aerosol type as the final truth label.


## Proposed Product Logic

For each collocated MODIS DB pixel:

1. determine its DB aerosol type from `Quality_Assurance_Land`
2. route it to a type-specific classifier and regressor
3. estimate:
   - dust probability
   - dust fraction
   - dust AOD at 550 nm
4. combine DB-type prior and ML output into a final dust-detection QA class

Final outputs should include at minimum:

- `dust_probability`
- `dust_fraction`
- `dust_aod_550`
- `dust_detection_qa`
- `db_aerosol_type`


## First-Version QA Concept

The first operational QA concept is:

- `QA0`: non-dust
- `QA1`: possible dust / ambiguous
- `QA2`: likely dust
- `QA3`: confident dust

DB aerosol type provides the prior:

- `Dust` -> high prior
- `Mixed` -> ambiguous prior
- `Smoke` / `Sulfate` -> low prior

But final QA should be based on both:

- DB aerosol type
- ML dust probability
- feature completeness / validity

So DB type guides the QA, but does not fully determine it.


## Why Type-Specific Models

### Dust branch

For DB `Dust`, AE is often nearly fixed and therefore less informative.
Useful predictors are more likely to be:

- spectral AOD
- SSA values
- SSA slopes
- QA completeness / confidence

### Mixed branch

For DB `Mixed`, the problem is hardest and all predictors remain relevant:

- AOD
- AE
- SSA
- feature availability
- QA confidence

### Smoke / Sulfate branch

For DB `Smoke` / `Sulfate`, the key problem is avoiding false dust while still
capturing mixed dust cases.

Useful predictors are likely:

- AE
- SSA spectral behavior
- AOD ratios
- QA confidence

Because `Sulfate` is usually rare, the first version combines:

- `Smoke`
- `Sulfate`

into one branch:

- `smoke_sulfate`


## Training Targets

### Detection target

Primary detection target:

- `dust_label = 1` if `AERONET dust_fraction_675 >= threshold`

Initial threshold:

- `dust_fraction_675 >= 0.2`

This is the same threshold used in earlier two-stage detection experiments.

### Regression targets

The first version trains two quantitative targets:

1. `dust_fraction_550`
2. `dust_aod_550`

The main physically consistent output should be:

- `dust_aod_550_from_fraction = predicted_dust_fraction_550 * MODIS_DB_AOD550`

But keeping a direct `dust_aod_550` target in the training table is still
useful for diagnostics and later model comparisons.


## Feature Design

### Shared MODIS DB predictors

Core retrieval variables:

- `MODIS_DB_AOD550`
- `MODIS_DB_AOD412`
- `MODIS_DB_AOD470`
- `MODIS_DB_AOD660`
- `MODIS_DB_AE`
- `MODIS_DB_SSA412`
- `MODIS_DB_SSA470`
- `MODIS_DB_SSA660`

Derived spectral predictors:

- `AOD412_over_470`
- `AOD470_over_660`
- `AOD550_over_660`
- `SSA470_minus_412`
- `SSA660_minus_470`

Missingness flags:

- `has_ssa412`
- `has_ssa470`
- `has_ssa660`
- `has_ae`

### QA-derived predictors

From `Quality_Assurance_Land` summaries:

- dominant DB aerosol type
- dominant DB algorithm flag
- useful-pixel fraction
- confidence histograms
- aerosol-type histograms

These help capture whether the collocated DB pixels are:

- internally consistent
- mixed even within the 25 km footprint
- dominated by useful or low-confidence retrievals


## First-Version Model Layout

Train separate models for:

- `dust`
- `mixed`
- `smoke_sulfate`

For each branch:

1. classifier:
   - predict dust presence
2. regressor:
   - predict dust fraction / dust AOD

The first version should start with:

- one classifier per branch
- one fraction regressor per branch

Direct dust-AOD regression can remain a diagnostic secondary product.


## Data Source for Training

Use the merged all-site file:

- [AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl](/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl)

This file now contains:

- AERONET inversion fields
- nearest AERONET SDA fields
- MODIS DB collocation statistics
- DB QA summaries and decoded arrays


## First-Version Implementation Scope

The first implementation will:

1. build a flat training dataframe from the merged all-site file
2. route each record by actual DB aerosol type
3. train branch-specific XGBoost models
4. save:
   - training dataframe
   - cross-tabs
   - per-branch metrics
   - feature importance
   - metadata

The first implementation is primarily a training and evaluation pipeline.
Application to L2/L3 products can be added after the branch-specific behavior
is understood.


## Important Caveats

1. `usefulness` in the DB QA field appears conservative for some collocations.
   Therefore the first implementation should **not** require useful pixels only.
   It should prefer:
   - dominant aerosol type from useful pixels if available
   - otherwise dominant aerosol type from all collocated pixels

2. AERONET truth remains uncertain in mixed aerosol scenes.
   The new routing strategy reduces this problem but does not eliminate it.

3. Branch sample sizes must be checked carefully.
   `Sulfate` is expected to be rare and should remain merged into the
   `smoke_sulfate` branch unless sample counts become large enough.


## Expected Advantages

This design should improve:

- smoke-vs-dust separation
- physical interpretability of feature importance
- consistency between MODIS retrieval regime and ML feature usage
- user-facing QA

It should also reduce the current mismatch where:

- one model is forced to handle dust, smoke, and mixed scenes with the same
  mapping


## Expected Next Steps After First-Version Training

1. compare branch-specific performance against the old AE-proxy routing
2. inspect confusion inside DB `Smoke` and DB `Dust`
3. calibrate branch probabilities
4. define final QA thresholds
5. apply the branch-routed model to the heavy dust and heavy smoke test
   granules
6. then apply to annual L3 products
