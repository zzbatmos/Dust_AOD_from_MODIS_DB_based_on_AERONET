# Dust AOD Workflow Memo

## Purpose

This project develops a dust aerosol optical depth (dust AOD) estimation workflow anchored in AERONET inversion products and then transfers that information to MODIS Deep Blue (DB) through machine learning.

The core idea is:

1. Use AERONET lidar depolarization ratio (DPR) and lidar ratio (LR) retrievals to derive a physically motivated dust fraction and dust AOD.
2. Use those AERONET-derived dust quantities as training targets.
3. Train XGBoost models so dust AOD can be estimated from MODIS DB aerosol products.
4. Apply the trained model to real MODIS granules and compare against simpler reference methods such as Li and Ginoux.

This memo is intended to let future sessions pick up the work quickly and extend it without re-deriving the current structure.

## Scientific Scope

The workflow focuses on the following scientific tasks:

1. Derive dust LR and DPR statistics from AERONET inversion products, following the dust-fraction framework in Shin et al. (2019).
2. Derive dust-dominant DPR and LR statistics and then derive dust fraction and dust AOD from AERONET total AOD using an original DPR/LR-based method.
3. Compare the DPR/LR-derived dust AOD against AERONET coarse-mode AOD for reference and validation.
4. Use AERONET-to-AERONET XGBoost training as a sanity check.
5. Train XGBoost models against collocated MODIS DB products.
6. Apply the trained MODIS model to a real Terra MODIS case on March 14, 2025.

## Data Foundations

### AERONET inversion archive

Primary inversion directory:

- `/home/ec2-user/Research/AERONET_INV_Level2_v3.0_allsites`

These files provide:

- total AOD
- coarse/fine extinction AOD
- DPR
- LR
- SSA

at multiple wavelengths, including 440, 675, and 1020 nm.

### AERONET SDA and MODIS collocations

The MODIS-side ML workflow uses two saved collocation pickles:

- `../AERONET_MOD04_L2_collocation_with_SDA_data.pkl`
- `../AERONET_MYD04_L2_collocation_with_SDA_data.pkl`

These contain collocated AERONET inversion quantities, nearest AERONET SDA quantities, and MODIS DB retrieval statistics.

### Notebook anchor

The original exploratory notebook is:

- `/home/ec2-user/Research/Codex/AERONET_DPR_analysis.ipynb`

Many standalone scripts in this repo were extracted or generalized from that notebook.

## Main Workflow Components

### 1. AERONET DPR site analysis

Goal:

- inspect DPR distributions by site
- reproduce notebook-style site selection and violin/histogram views

Script:

- [analyze_aeronet_dpr_sites.py](/home/ec2-user/Research/Codex/analyze_aeronet_dpr_sites.py)

Shared helper:

- [aeronet_dpr_utils.py](/home/ec2-user/Research/Codex/aeronet_dpr_utils.py)

Notes:

- supports all sites or the Shin et al. selected-site subset
- plots DPR at a chosen wavelength

### 2. Dust-dominant DPR and LR statistics

Goal:

- identify dust-dominant cases using the 1020 nm DPR-based dust fraction
- summarize both DPR and LR for dust-dominant cases

Script:

- [derive_dust_dominant_dpr_lr_stats.py](/home/ec2-user/Research/Codex/derive_dust_dominant_dpr_lr_stats.py)

Notes:

- dust-dominant selection is based on `Rd_1020 > threshold`
- the threshold used in the extracted workflow is typically `0.89`
- output includes both DPR and LR statistics because both are needed later in the dust-AOD derivation

### 3. AERONET dust AOD from DPR/LR

Goal:

- derive dust fraction and dust AOD from AERONET inversion quantities
- compare dust AOD with AERONET coarse-mode AOD
- produce wavelength-dependent validation figures

Script:

- [derive_aeronet_dust_aod.py](/home/ec2-user/Research/Codex/derive_aeronet_dust_aod.py)

Method summary:

- compute `Rd` from DPR using the Shin et al. depolarization relation
- estimate dust fraction as `Rd * LR_dust / LR`
- clip to physical range `[0, 1]`
- derive dust AOD as `total AOD * dust fraction`

This is an original method in this project, not a direct copy of an existing published dust-AOD algorithm.

Important output already generated:

- [/home/ec2-user/Research/Codex/aeronet_dust_aod_allsites](/home/ec2-user/Research/Codex/aeronet_dust_aod_allsites)

This directory contains all-site comparisons for 440, 675, and 1020 nm.

### 4. Derived dust spectral quantities

Goal:

- derive a representative dust spectral slope from the wavelength-dependent DPR/LR dust-AOD estimates
- support interpolation/extrapolation to intermediate wavelengths such as 500 or 550 nm

Current practice:

- use log-log interpolation between 440 and 675 nm
- optionally use a three-wavelength log-log fit using 440, 675, and 1020 nm

This interpolation logic is currently embedded in:

- [train_aeronet_dust_aod500_xgb.py](/home/ec2-user/Research/Codex/train_aeronet_dust_aod500_xgb.py)
- [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py)

### 5. AERONET-to-AERONET XGBoost sanity test

Goal:

- test whether AERONET-only observable quantities can reproduce the DPR/LR-derived dust target
- use this as a sanity check before training on MODIS DB inputs

Script:

- [train_aeronet_dust_aod500_xgb.py](/home/ec2-user/Research/Codex/train_aeronet_dust_aod500_xgb.py)

Current design:

- target: `dust_AOD_500`
- target built from AERONET-derived dust AOD at 440, 675, and 1020 nm
- training uses `log1p` target transform

Important comparison built into the script:

- train without DPR/LR features
- train with DPR/LR features

Interpretation:

- including DPR/LR gives much better scores
- but that mostly reflects target leakage, because the target itself is constructed from DPR/LR
- the no-DPR/LR version is the more honest proxy-learning test

### 6. MODIS DB comparison against collocated AERONET

Goal:

- compare collocated MODIS DB products with AERONET reference quantities
- inspect AOD, AE, and SSA behavior with cleaner density plots

Script:

- [compare_collocated_aeronet_modis_db.py](/home/ec2-user/Research/Codex/compare_collocated_aeronet_modis_db.py)

Generated output directory:

- [/home/ec2-user/Research/Codex/aeronet_modis_db_comparisons](/home/ec2-user/Research/Codex/aeronet_modis_db_comparisons)

This script is more flexible than the notebook because it supports:

- Terra only
- Aqua only
- combined Terra+Aqua
- preset comparison groups
- custom variable pairs

### 7. MODIS DB XGBoost training using AERONET-derived dust targets

There are now two main ML training branches.

#### Older branch: dust fraction at 675 nm

Script:

- [extract_modis_db_dust_model.py](/home/ec2-user/Research/Codex/extract_modis_db_dust_model.py)

This branch trained a MODIS-to-dust model using:

- target based on dust fraction at 675 nm

Tested variants included:

- `log1p`
- `logit`
- weighted-bin training

Best result from that line was the `log1p` baseline.

#### Current wavelength-aligned branch: direct dust AOD at 550 nm

Script:

- [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py)

Purpose:

- align the ML dust product with Li and Ginoux and with MODIS DB AOD at 550 nm

Method:

1. derive AERONET dust AOD at 440, 675, and 1020 nm using the DPR/LR method
2. interpolate to 500 nm
3. step to 550 nm using the spectral slope derived from 440 and 675 nm
4. train XGBoost from MODIS DB AOD/AE/SSA features

Chosen training setup:

- use the best target transform discovered earlier: `log1p`
- do not use the older `logit` or weighted-bin variants for the 550 nm target

Current saved model directory:

- [/home/ec2-user/Research/Codex/modis_db_dust_aod550_xgb](/home/ec2-user/Research/Codex/modis_db_dust_aod550_xgb)

Key files:

- [modis_db_dust_aod550_xgb.json](/home/ec2-user/Research/Codex/modis_db_dust_aod550_xgb/modis_db_dust_aod550_xgb.json)
- [modis_db_dust_aod550_metrics.json](/home/ec2-user/Research/Codex/modis_db_dust_aod550_xgb/modis_db_dust_aod550_metrics.json)
- [modis_db_dust_aod550_feature_importance.csv](/home/ec2-user/Research/Codex/modis_db_dust_aod550_xgb/modis_db_dust_aod550_feature_importance.csv)

### 8. FMF-based side branch

There is also a parallel branch focused on FMF and Li and Ginoux style coarse-mode estimates.

Key scripts:

- [replicate_li_ginoux_figure2_from_collocations.py](/home/ec2-user/Research/Codex/replicate_li_ginoux_figure2_from_collocations.py)
- [train_apply_modis_fmf_gam.py](/home/ec2-user/Research/Codex/train_apply_modis_fmf_gam.py)

Purpose:

- replicate and test Li and Ginoux style AE-to-FMF parameterizations
- compare against GAM alternatives

This branch is useful for comparison and interpretation, but it is separate from the main DPR/LR-based dust-AOD target workflow.

## March 14, 2025 Terra MODIS Case

Case-study directory:

- [/home/ec2-user/Research/Codex/dust_model_application_2025-03-14](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14)

Granule:

- `MOD04_L2.A2025073.1705.061.2025074013638.hdf`

Application script:

- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

Current capabilities of the application script:

- reads a local HDF or downloads via Earthaccess
- plots MODIS DB reference products
- applies legacy dust-fraction models
- applies the new direct dust-AOD-550 model
- computes Li-Ginoux coarse-mode dust AOD proxy
- builds comparison panels and scatter plots

Important update:

- the script now supports the direct `550 nm` ML dust-AOD model and compares it directly against Li-Ginoux at the same wavelength

Current case summary:

- legacy `log1p` fraction model gives much smaller dust AOD
- direct `dust_aod550_log1p` model gives dust AOD at 550 nm
- Li-Ginoux gives a same-wavelength coarse-mode dust proxy for direct comparison

## Main Script Map

### Core AERONET processing

- [aeronet_dpr_utils.py](/home/ec2-user/Research/Codex/aeronet_dpr_utils.py)
- [analyze_aeronet_dpr_sites.py](/home/ec2-user/Research/Codex/analyze_aeronet_dpr_sites.py)
- [derive_dust_dominant_dpr_lr_stats.py](/home/ec2-user/Research/Codex/derive_dust_dominant_dpr_lr_stats.py)
- [derive_aeronet_dust_aod.py](/home/ec2-user/Research/Codex/derive_aeronet_dust_aod.py)

### AERONET-only ML

- [train_aeronet_dust_aod500_xgb.py](/home/ec2-user/Research/Codex/train_aeronet_dust_aod500_xgb.py)

### Collocation extraction and MODIS-target ML

- [extract_modis_db_dust_model.py](/home/ec2-user/Research/Codex/extract_modis_db_dust_model.py)
- [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py)
- [compare_collocated_aeronet_modis_db.py](/home/ec2-user/Research/Codex/compare_collocated_aeronet_modis_db.py)
- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

### Li and Ginoux / FMF branch

- [replicate_li_ginoux_figure2_from_collocations.py](/home/ec2-user/Research/Codex/replicate_li_ginoux_figure2_from_collocations.py)
- [train_apply_modis_fmf_gam.py](/home/ec2-user/Research/Codex/train_apply_modis_fmf_gam.py)

## Current Recommended Workflow

For future work, the main recommended path is:

1. Use the AERONET inversion archive to maintain and validate the DPR/LR dust-AOD derivation.
2. Use [derive_dust_dominant_dpr_lr_stats.py](/home/ec2-user/Research/Codex/derive_dust_dominant_dpr_lr_stats.py) to update dust-dominant DPR/LR statistics if the AERONET archive changes.
3. Use [derive_aeronet_dust_aod.py](/home/ec2-user/Research/Codex/derive_aeronet_dust_aod.py) to regenerate dust-AOD validation against AERONET coarse-mode AOD.
4. Use [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py) as the main MODIS DB training script, because the target is now aligned at 550 nm.
5. Use [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py) for case studies and direct Li-Ginoux comparison.

## Known Scientific / Technical Caveats

1. The DPR/LR-derived dust-AOD target depends on assumptions about dust-dominant LR statistics.
2. Including DPR/LR as ML input features in AERONET-to-AERONET training creates target leakage if the target is built from DPR/LR.
3. The MODIS DB ML model coverage is limited by the need for all required input features, especially SSA channels.
4. Dust spectral interpolation from 440/675/1020 to 500 or 550 nm is currently handled through log-log scaling; future sessions may want to test more constrained spectral assumptions.
5. The Li and Ginoux comparisons are useful references, but they represent a coarse-mode proxy rather than the same physical retrieval chain as the DPR/LR method.

## Suggested Next Extensions

Likely next steps for future sessions:

1. Refine the 550 nm dust-AOD target interpolation and test whether the 440-675 two-point slope or the full 3-point fit is more stable.
2. Add more rigorous train/test split strategies for MODIS training, such as leave-site-out or leave-region-out validation.
3. Build fallback MODIS models that do not require SSA inputs, to improve spatial coverage.
4. Improve the comparison plotting and panel generation for publication-style figures.
5. Add README-level documentation for external users of the GitHub repository.

## Related Project Notes

There is also a shorter session summary here:

- [SESSION_NOTES.md](/home/ec2-user/Research/Codex/SESSION_NOTES.md)

That file is useful as a compact chronological session log. This memo is the higher-level workflow guide.
