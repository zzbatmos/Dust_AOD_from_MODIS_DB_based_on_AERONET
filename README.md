# Dust AOD From MODIS Deep Blue Based on AERONET

## Overview

This repository develops a dust aerosol optical depth (dust AOD) retrieval
workflow anchored in AERONET inversion products and transferred to MODIS Deep
Blue (DB) through machine learning.

The scientific idea is:

1. Use AERONET lidar depolarization ratio (DPR) and lidar ratio (LR) retrievals
   to derive a physically constrained dust fraction and dust AOD.
2. Validate that AERONET-derived dust AOD against AERONET coarse-mode AOD.
3. Use the AERONET-derived dust AOD as the target for machine-learning models.
4. Apply the trained models to MODIS DB observations and compare them with the
   empirical Li and Ginoux method and with previous studies.
5. Extend the workflow to an experimental two-stage dust model with QA tiers.

The repository contains both research scripts extracted from the original
notebook and more production-style drivers for Level-2 and Level-3 MODIS DB
products.

## Data Sources

### AERONET inversion archive

The AERONET inversion archive provides:

- total AOD
- coarse/fine AOD
- lidar depolarization ratio (DPR)
- lidar ratio (LR)
- single-scattering albedo (SSA)

at multiple wavelengths including `440`, `675`, and `1020 nm`.

### AERONET SDA and MODIS collocations

The MODIS-side training uses saved collocations between AERONET and MODIS DB:

- `AERONET_MOD04_L2_collocation_with_SDA_data.pkl`
- `AERONET_MYD04_L2_collocation_with_SDA_data.pkl`

These contain collocated AERONET inversion quantities, nearest AERONET SDA
quantities, and MODIS DB retrieval features.

### MODIS Deep Blue products

The repository supports:

- Level-2 swath products: `MOD04_L2`, `MYD04_L2`
- Level-3 daily products: `MOD08_D3`, `MYD08_D3`

## 1. Dust AOD Derived From AERONET Data

The first part of the workflow derives dust AOD directly from AERONET
retrievals using DPR and LR.

Method summary:

1. Convert DPR to a dust fraction proxy using the Shin et al. (2019)
   depolarization framework.
2. Estimate the dust fraction as:

   `dust_fraction = Rd * LR_dust / LR`

3. Clip the fraction to the physical range `[0, 1]`.
4. Derive dust AOD from total AOD:

   `dust_AOD = total_AOD * dust_fraction`

The scripts involved are:

- [aeronet_dpr_utils.py](/home/ec2-user/Research/Codex/aeronet_dpr_utils.py)
- [analyze_aeronet_dpr_sites.py](/home/ec2-user/Research/Codex/analyze_aeronet_dpr_sites.py)
- [derive_dust_dominant_dpr_lr_stats.py](/home/ec2-user/Research/Codex/derive_dust_dominant_dpr_lr_stats.py)
- [derive_aeronet_dust_aod.py](/home/ec2-user/Research/Codex/derive_aeronet_dust_aod.py)

These scripts are used to:

- inspect DPR distributions by site
- identify dust-dominant cases
- derive dust-dominant DPR/LR statistics
- calculate dust AOD at `440`, `675`, and `1020 nm`

## 2. Comparison With AERONET Coarse-Mode AOD

The DPR/LR-derived dust AOD is compared against AERONET coarse-mode AOD as a
consistency check.

This step is implemented in:

- [derive_aeronet_dust_aod.py](/home/ec2-user/Research/Codex/derive_aeronet_dust_aod.py)

Outputs include:

- scatter plots of DPR/LR dust AOD vs. AERONET coarse-mode AOD
- summary statistics at `440`, `675`, and `1020 nm`

This comparison is an important validation step because the DPR/LR dust-AOD
method is original to this project.

## 3. AERONET-to-AERONET Prediction

Before training on MODIS DB, the workflow tests whether AERONET-observable
quantities can reproduce the DPR/LR-derived dust target.

This script performs that sanity test:

- [train_aeronet_dust_aod500_xgb.py](/home/ec2-user/Research/Codex/train_aeronet_dust_aod500_xgb.py)

Current design:

- target: `dust_AOD_500`
- target derived from wavelength-dependent AERONET dust AOD
- XGBoost regression in log space

The script compares two setups:

- without DPR/LR features
- with DPR/LR features

Interpretation:

- including DPR/LR improves the fit strongly
- but that also introduces target leakage, because the target is derived from
  DPR/LR
- the no-DPR/LR case is therefore the more honest proxy-learning benchmark

## 4. Training MODIS DB to AERONET Dust AOD

The central machine-learning task is to predict AERONET-derived dust AOD from
collocated MODIS DB retrievals.

There are two training branches in the repository.

### Legacy branch: dust fraction at 675 nm

- [extract_modis_db_dust_model.py](/home/ec2-user/Research/Codex/extract_modis_db_dust_model.py)

This branch trains on a `675 nm` dust-fraction target and tests multiple target
transforms, including `log1p`, `logit`, and weighted-bin training.

### Current branch: direct dust AOD at 550 nm

- [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py)

This is the main recommended MODIS training workflow because it aligns the ML
product with:

- MODIS DB total AOD at `550 nm`
- Li and Ginoux comparison products
- monthly products used in previous studies

Method summary:

1. Derive AERONET dust AOD at `440`, `675`, and `1020 nm`.
2. Interpolate to `500 nm`.
3. Step to `550 nm` using the derived dust spectral slope.
4. Train an XGBoost model from MODIS DB spectral AOD, AE, and SSA features.

## 5. Applying the ML Method and Li-Ginoux Method to MODIS Data

The repository supports both local case-study application and larger-scale
batch processing.

### Local Level-2 case studies

- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

This script can:

- read a local MODIS DB swath granule or download one via Earthdata
- plot MODIS DB spectral AOD, AE, and SSA
- apply the direct `550 nm` XGBoost dust-AOD model
- apply the Li and Ginoux empirical coarse-mode dust-AOD method
- generate comparison panels, side-by-side maps, and difference maps

### Level-2 and Level-3 production workflows

- [DAOD_from_DB_L2_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L2_XGBoost.py)
- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)

Additional utilities:

- [monthly_mean_l3_daod_compare.py](/home/ec2-user/Research/Codex/monthly_mean_l3_daod_compare.py)
- [plot_l3_daod_global_map.py](/home/ec2-user/Research/Codex/plot_l3_daod_global_map.py)
- [run_modis_l3_daod_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_daod_backfill.py)
- [supervise_modis_l3_daod_jobs.py](/home/ec2-user/Research/Codex/supervise_modis_l3_daod_jobs.py)

Important operational convention used in these workflows:

- if DB total AOD is missing, dust AOD is `NaN`
- if DB total AOD exists but no dust estimate is produced, dust AOD is set to `0`
- if DB total AOD exists and a dust estimate is produced, keep the positive value

### Experimental two-stage and QA branch

The repository also contains an experimental two-stage branch:

- `two-stage-dust-model`

Main scripts:

- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)
- [apply_two_stage_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_two_stage_dust_model_to_modis_db.py)
- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)
- [TWO_STAGE_METHOD.md](/home/ec2-user/Research/Codex/TWO_STAGE_METHOD.md)

Current QA interpretation:

- `QA>=1`: broad product, intentionally reduced to direct XGBoost
- `QA>=2`: moderate-confidence two-stage dust
- `QA>=3`: strict/high-confidence two-stage dust

This gives users a practical choice between:

- broad weak-dust coverage
- increasingly conservative smoke-resistant dust products

### Li and Ginoux reference branch

The repository also contains scripts to analyze Li and Ginoux style FMF-based
coarse-mode dust estimates:

- [replicate_li_ginoux_figure2_from_collocations.py](/home/ec2-user/Research/Codex/replicate_li_ginoux_figure2_from_collocations.py)
- [train_apply_modis_fmf_gam.py](/home/ec2-user/Research/Codex/train_apply_modis_fmf_gam.py)

## 6. Comparison With Previous Studies

The repository includes direct comparison against the monthly Aqua MODIS dust
AOD product from Song et al. (2021).

Reference file:

- `Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc`

Scripts:

- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)
- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)

What these scripts do:

1. Aggregate our daily Aqua Level-3 dust-AOD products into a Song-style monthly
   NetCDF with dimensions `(year, month, lat, lon)`.
2. Apply the required averaging rule:
   - ignore days with missing total AOD
   - if total AOD exists but dust AOD is missing, count the day and use dust AOD = `0`
3. Compare our monthly DAOD and TAOD with the Song monthly product over land.
4. Generate global monthly mean time series, seasonal cycles, climatology maps,
   and difference maps.

Important implementation detail:

- Song stores latitude ascending from `-89.5` to `89.5`
- our monthly builder initially preserved descending latitude from the daily
  MODIS files
- the comparison script reorders our monthly fields to the Song convention
  before masking and plotting

One useful benchmark from Song et al. (2021):

- the published MODIS land mean DAOD over `60S–60N` for `2007–2019` is about
  `0.103` in their Table 4

Our current Aqua comparison on the same weighted land domain gives:

- Song monthly file: about `0.102`
- XGBoost: about `0.069`
- Li-Ginoux: about `0.071`

## Main Supporting Documents

For more detail, see:

- [WORKFLOW_MEMO.md](/home/ec2-user/Research/Codex/WORKFLOW_MEMO.md)
- [SESSION_NOTES.md](/home/ec2-user/Research/Codex/SESSION_NOTES.md)
- [TWO_STAGE_METHOD.md](/home/ec2-user/Research/Codex/TWO_STAGE_METHOD.md)

## Current Status

The repository now supports:

- AERONET DPR/LR dust-AOD derivation
- AERONET validation against coarse-mode AOD
- AERONET-to-AERONET ML sanity checks
- MODIS DB to AERONET dust-AOD XGBoost training
- Level-2 and Level-3 application of XGBoost and Li-Ginoux methods
- comparison with the Song et al. monthly Aqua dust-AOD product

The main remaining scientific question is why the current Aqua land-mean DAOD
from our XGBoost and Li-Ginoux workflows is lower than the Song et al. monthly
reference by roughly `31-33 %` over `60S–60N` for `2007–2019`.
