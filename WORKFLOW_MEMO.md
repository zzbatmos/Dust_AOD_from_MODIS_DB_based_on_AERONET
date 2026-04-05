# Dust AOD Workflow Memo

## Purpose

This project develops a dust aerosol optical depth (dust AOD) estimation workflow anchored in AERONET inversion products and then transfers that information to MODIS Deep Blue (DB) through machine learning.

The core idea is:

1. Use AERONET lidar depolarization ratio (DPR) and lidar ratio (LR) retrievals to derive a physically motivated dust fraction and dust AOD.
2. Use those AERONET-derived dust quantities as training targets.
3. Train XGBoost models so dust AOD can be estimated from MODIS DB aerosol products.
4. Apply the trained model to real MODIS granules and compare against simpler reference methods such as Li and Ginoux.
5. Extend the ML workflow to a two-stage dust-detection plus dust-amount framework with QA tiers.
6. Build long-term L3 daily, monthly, and trend products for Terra and Aqua.

This memo is intended to let future sessions pick up the work quickly and extend it without re-deriving the current structure.

## Scientific Scope

The workflow focuses on the following scientific tasks:

1. Derive dust LR and DPR statistics from AERONET inversion products, following the dust-fraction framework in Shin et al. (2019).
2. Derive dust-dominant DPR and LR statistics and then derive dust fraction and dust AOD from AERONET total AOD using an original DPR/LR-based method.
3. Compare the DPR/LR-derived dust AOD against AERONET coarse-mode AOD for reference and validation.
4. Use AERONET-to-AERONET XGBoost training as a sanity check.
5. Train XGBoost models against collocated MODIS DB products.
6. Apply the trained MODIS model to a real Terra MODIS case on March 14, 2025.
7. Build QA-tiered yearly L3 products so users can choose broad or conservative dust screening.
8. Use DB aerosol-type QA structure to build aerosol-type-conditional dust detection and dust-fraction estimation.
9. Apply the three L2 methods day by day over Aqua 2017 and aggregate them to daily and monthly `1x1` products.

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

This was extended to a QA-augmented archive and a merged all-site file:

- `/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_files_with_QA`
- `/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl`

The QA archive preserves the old DB summary statistics and adds:

- raw masked `Quality_Assurance_Land`
- decoded usefulness, confidence, aerosol type, and algorithm flag
- aerosol-type and confidence histograms
- dominant aerosol type and dominant algorithm flag

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

### 7b. DB-aerosol-type-conditional branch

This newer branch uses the actual DB aerosol type decoded from
`Quality_Assurance_Land` as a routing variable and prior, while still using
AERONET-derived dust targets as truth.

Design note:

- [DB_TYPE_CONDITIONAL_TRAINING_DESIGN.md](/home/ec2-user/Research/Codex/DB_TYPE_CONDITIONAL_TRAINING_DESIGN.md)

Training scripts:

- [train_modis_db_type_conditional_detection_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_type_conditional_detection_xgb.py)
- [train_modis_db_type_conditional_from_qa_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_type_conditional_from_qa_xgb.py)

Application and comparison scripts:

- [apply_db_type_conditional_dust_model.py](/home/ec2-user/Research/Codex/apply_db_type_conditional_dust_model.py)
- [compare_two_stage_vs_db_type_conditional.py](/home/ec2-user/Research/Codex/compare_two_stage_vs_db_type_conditional.py)

Core structure:

1. route each sample by DB aerosol type:
   - `Dust`
   - `Mixed`
   - `Smoke/Sulfate`
2. train type-specific dust classifiers against AERONET dust truth
3. train type-specific dust-fraction regressors
4. derive final QA from DB type plus ML output

Scientific interpretation from the first QA-aware training pass:

- DB `Dust` is the cleanest branch
- DB `Mixed` is intermediate
- DB `Smoke` still contains real AERONET dust cases

So DB aerosol type should be used as a structured prior and routing variable,
not as a hard truth mask.

### 8. FMF-based side branch

There is also a parallel branch focused on FMF and Li and Ginoux style coarse-mode estimates.

Key scripts:

- [replicate_li_ginoux_figure2_from_collocations.py](/home/ec2-user/Research/Codex/replicate_li_ginoux_figure2_from_collocations.py)
- [train_apply_modis_fmf_gam.py](/home/ec2-user/Research/Codex/train_apply_modis_fmf_gam.py)

Purpose:

- replicate and test Li and Ginoux style AE-to-FMF parameterizations
- compare against GAM alternatives

This branch is useful for comparison and interpretation, but it is separate from the main DPR/LR-based dust-AOD target workflow.

### 9. Two-stage branch and QA extension

There is now an experimental branch that extends the single-stage MODIS ML
workflow to a two-stage design:

1. stage 1: dust detection
2. stage 2: dust-fraction estimation
3. final derivation: `dust_AOD_550 = dust_fraction * MODIS_DB_AOD550`

Core scripts:

- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)
- [apply_two_stage_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_two_stage_dust_model_to_modis_db.py)
- [tune_two_stage_dust_model.py](/home/ec2-user/Research/Codex/tune_two_stage_dust_model.py)
- [TWO_STAGE_METHOD.md](/home/ec2-user/Research/Codex/TWO_STAGE_METHOD.md)

Branch:

- `two-stage-dust-model`

This branch was initially developed separately, but the tested workflow has now
been merged into `main`.

### 10. Terra 2024 L3 two-stage QA products

The two-stage workflow now has L3 yearly support.

Scripts:

- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_daod_methods_year.py](/home/ec2-user/Research/Codex/compare_l3_daod_methods_year.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)

Current 2024 Terra QA interpretation:

- `QA>=1`: broad product, intentionally reduced to direct XGBoost
- `QA>=2`: moderate-confidence two-stage dust
- `QA>=3`: strict/high-confidence two-stage dust

2024 Terra annual means:

- Li-Ginoux: `0.0602`
- direct XGBoost: `0.0670`
- `QA>=1`: `0.0670`
- `QA>=2`: `0.0491`
- `QA>=3`: `0.0420`

Current output/figure locations:

- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage_2024](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage_2024)
- [/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024](/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024)
- [tracked QA climatology figures](/home/ec2-user/Research/Codex/comparison_to_two_stage_QA_2024)

### 11. Full L3 comparison backfill and QA archive

The repo now has a complete daily L3 comparison archive for both Terra and Aqua
covering the current processed range.

Backfill scripts:

- [run_modis_l3_comparison_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_comparison_backfill.py)
- [build_modis_l3_two_stage_qa.py](/home/ec2-user/Research/Codex/build_modis_l3_two_stage_qa.py)

Main output directories:

- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_LiGinoux)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_XGBoost)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStageQA](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStageQA)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_LiGinoux)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_XGBoost)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStage](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStage)
- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStageQA](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStageQA)

Notes:

- Terra raw two-stage includes one extra boundary file for `2001-12-31`
- in-range `2002-2024` Terra and Aqua daily archives are complete for Li-Ginoux, direct XGBoost, raw two-stage, and QA-tier products

### 12. Monthly products

Monthly aggregation script:

- [build_l3_monthly_products.py](/home/ec2-user/Research/Codex/build_l3_monthly_products.py)

Monthly outputs:

- [/home/ec2-user/Research/Codex/modis_l3_monthly_products](/home/ec2-user/Research/Codex/modis_l3_monthly_products)

Important structure:

- Li-Ginoux monthly files provide `dust_aod`, `total_aod`, and `n_days`
- QA monthly files provide `dust_aod_qa1`, `dust_aod_qa2`, `dust_aod_qa3`, `total_aod`, and `n_days`
- dimensions are `(year, month, lat, lon)`

Aggregation rule:

- if daily total AOD is missing, ignore the day
- if total AOD exists but daily dust AOD is absent, count the day and set daily dust AOD to zero in the monthly mean

### 13. Trend and significance analysis

Trend script:

- [compute_l3_monthly_trends.py](/home/ec2-user/Research/Codex/compute_l3_monthly_trends.py)

Method:

1. For each land grid box, compute monthly climatology from the monthly product.
2. Deseasonalize by subtracting that climatology from the monthly series.
3. Fit a linear trend to the monthly anomalies.
4. Report trend in dust AOD per decade.
5. Compute p values for the slope and retain significance maps for `p < 0.05`.

Trend outputs:

- [/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024)

The trend files now include both dust-AOD and total-AOD trend variables:

- `dust_aod_*_trend_per_decade`
- `dust_aod_*_p_value`
- `dust_aod_*_significant_p_lt_0p05`
- `total_aod_trend_per_decade`
- `total_aod_p_value`
- `total_aod_significant_p_lt_0p05`

### 14. Regional trend diagnostics

Regional scripts:

- [analyze_sahel_trends.py](/home/ec2-user/Research/Codex/analyze_sahel_trends.py)
- [analyze_regime_shift_regions.py](/home/ec2-user/Research/Codex/analyze_regime_shift_regions.py)

These support:

- drawing region boxes on the trend maps
- selecting representative positive- or negative-trend grid boxes
- plotting deseasonalized monthly time series for `QA1`, `QA2`, and `QA3`
- testing regime-shift hypotheses with period-mean and breakpoint-style comparisons

Current interpretation from Aqua `QA2`:

- southern Sahel band: positive dust-AOD shift centered around `2015`
- northern band: marked dust-AOD decrease concentrated in `2020-2024`

### 15. External context and QA-flag tests

Niamey duststorm archive extractor:

- [extract_niamey_duststorm_dates.py](/home/ec2-user/Research/Codex/extract_niamey_duststorm_dates.py)

DB aerosol-type QA diagnostics:

- [plot_modis_db_aerosol_type_flag.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_flag.py)
- [plot_modis_db_aerosol_type_diagnostics.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_diagnostics.py)
- [compare_db_flag_vs_two_stage.py](/home/ec2-user/Research/Codex/compare_db_flag_vs_two_stage.py)
- [DB_AEROSOL_TYPE_NOTE.md](/home/ec2-user/Research/Codex/DB_AEROSOL_TYPE_NOTE.md)

Main finding:

- DB aerosol type is useful as an aerosol-regime prior
- it is not reliable enough to use by itself as the final dust mask
- the QA-aware ML methods are more selective than the raw DB dust flag in smoke-heavy scenes

### 16. Aqua 2017 L2 day-by-day three-method workflow

Goal:

- process all Aqua `MYD04_L2` granules in `2017` without storing a full year of raw HDF downloads
- compute Li-Ginoux, tuned two-stage, and DB-type-aware outputs day by day
- save per-granule `.npz` outputs, then aggregate them to daily and monthly `1x1` grids

Main scripts:

- [run_aqua_l2_day_three_methods_noplot.py](/home/ec2-user/Research/Codex/run_aqua_l2_day_three_methods_noplot.py)
- [run_l2_year_three_methods_noplot.py](/home/ec2-user/Research/Codex/run_l2_year_three_methods_noplot.py)
- [build_l2_daily_monthly_1deg.py](/home/ec2-user/Research/Codex/build_l2_daily_monthly_1deg.py)

Plotting/diagnostic scripts:

- [plot_aqua_l2_day_global_mosaic.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_global_mosaic.py)
- [plot_aqua_l2_day_qa_levels.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_qa_levels.py)
- [plot_aqua_l2_day_raw_vs_qa_mosaic.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_raw_vs_qa_mosaic.py)
- [plot_aqua_l2_day_dbtype_qa2_minus_raw.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_dbtype_qa2_minus_raw.py)
- [plot_l2_monthly_jja_comparison.py](/home/ec2-user/Research/Codex/plot_l2_monthly_jja_comparison.py)

Main year-run root:

- `/home/ec2-user/Research/Codex/aqua_l2_2017_three_methods_noplot_by_day/MYD04_L2_2017-01-01_2017-12-31`

Status:

- all `365` days completed successfully
- total per-granule `.npz` files: `28221`

Daily `1x1` output:

- `/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_daily/MYD04_L2_three_methods_daily_1deg_2017.nc`

Monthly `1x1` output:

- `/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_monthly/MYD04_L2_three_methods_monthly_1deg_2017.nc`

Averaging semantics:

- invalid total AOD pixels are excluded
- for QA-thresholded methods, pixels with valid total AOD but QA below threshold contribute `dust_aod = 0`
- monthly means are simple averages of the daily `1x1` means

Current example monthly figure:

- `/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_monthly/MYD04_L2_three_methods_monthly_1deg_2017_JJA_comparison.png`

This was used to compare the Sahel signal against independently reported
duststorm days from Timeanddate. The archive only starts in `2010-10`, but it
does confirm Niamey duststorm conditions on `2017-04-17`.

MODIS aerosol-type QA-flag tools:

- [plot_modis_db_aerosol_type_flag.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_flag.py)
- [compare_db_flag_vs_two_stage.py](/home/ec2-user/Research/Codex/compare_db_flag_vs_two_stage.py)

Current interpretation:

- the Deep Blue aerosol-type flag is useful as auxiliary QA or smoke context
- it is not reliable enough to serve as a standalone dust mask
- the tuned two-stage detector behaves better on the heavy smoke test case

Operational interpretation:

- `QA>=1` is the broad weak-dust product
- `QA>=2` and `QA>=3` provide progressively stricter smoke-resistant products

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

## Extended MODIS Application Work

### 9. L2 and L3 production scripts

The project now includes production-style daily application scripts for both
Li-Ginoux and XGBoost.

Scripts:

- [DAOD_from_DB_L2-v4.py](/home/ec2-user/Research/DAOD_from_DB_L2-v4.py)
- [DAOD_from_DB_L3-v4.py](/home/ec2-user/Research/DAOD_from_DB_L3-v4.py)
- [DAOD_from_DB_L2_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L2_XGBoost.py)
- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)

Important operational convention:

- if DB total AOD is missing, dust AOD is `NaN`
- if DB total AOD exists but the method does not produce dust AOD, set dust AOD to `0`
- if DB total AOD exists and the method produces dust AOD, keep the positive value

This convention is now used consistently in the L2 and L3 workflows so that
Li-Ginoux and XGBoost products are directly comparable.

### 10. September 2024 monthly L3 comparison

The L3 workflows were applied to Terra daily `MOD08_D3` for September 2024 and
then averaged to a monthly mean using `NaN`-ignoring averaging.

Scripts:

- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)
- [monthly_mean_l3_daod_compare.py](/home/ec2-user/Research/Codex/monthly_mean_l3_daod_compare.py)

Output directories:

- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_LiGinoux](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_LiGinoux)
- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_XGBoost](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_XGBoost)
- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_monthly_mean](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_monthly_mean)

The plotting routine was tuned for presentation:

- Robinson projection
- side colorbar instead of overlaid colorbar
- yellow-brown dust colormap

This case is useful because September 2024 includes major Amazon fire/smoke
activity, making it a good stress test for dust-versus-smoke separation.

### 11. September 22, 2024 Terra Amazon granule

The Terra granule `MOD04_L2.A2024266.1345.061.2024268021127.hdf` was used as a
local L2 case study over the Amazon basin.

Output directory:

- [/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon)

This case was used to improve the L2 visualization workflow. The plotting
routine in [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)
was updated from scatter plotting to swath-style `pcolormesh` so the granule
edges and pixel continuity are rendered more clearly.

Additional publication-style outputs were created:

- shared-scale two-panel Li-Ginoux vs XGBoost dust-AOD comparison
- `ML - Li-Ginoux` difference map with a diverging colormap
- full MODIS DB reference maps for spectral AOD, AE, and SSA

### 12. Long-run L3 backfill status

A resumable monthly backfill driver and supervisor were added for long L3 runs:

- [run_modis_l3_daod_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_daod_backfill.py)
- [supervise_modis_l3_daod_jobs.py](/home/ec2-user/Research/Codex/supervise_modis_l3_daod_jobs.py)

Primary output directory:

- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill](/home/ec2-user/Research/Codex/modis_l3_daod_backfill)

Important state at the latest check:

- Aqua (`MYD08_D3`) backfills are complete from `2002-07-04` through `2024-12-31`
- Terra (`MOD08_D3`) backfills are substantial but still incomplete
- within each platform, the Li-Ginoux and XGBoost date sets match exactly
- some expected Terra and Aqua dates remain missing, so future sessions should
  validate date coverage rather than assuming the run fully finished

Because Earthdata sessions can expire and tmux supervision proved unreliable,
future long runs should treat missing-month retries and explicit coverage
checks as part of the normal workflow.

### 13. Song2021-style monthly Aqua comparison

The workflow now includes a direct comparison against the Song et al. (2021)
monthly Aqua MODIS land dust-AOD product.

Reference file:

- [/home/ec2-user/Research/Codex/Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc)

New scripts:

- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)
- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)

Purpose:

- aggregate our Aqua daily L3 DAOD outputs into a Song-style monthly NetCDF
  with dimensions `(year, month, lat, lon)`
- compare our monthly DAOD and TAOD fields against the Song reference product
  on the same grid

Important averaging rule:

- if daily total AOD is missing, that day is ignored
- if daily total AOD exists but daily dust AOD is missing, that day is counted
  and dust AOD is set to zero for the monthly mean

Generated monthly files:

- [/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_XGB_2003_2019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_XGB_2003_2019_Monthly_1degX1deg.nc)
- [/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_LiGinoux_2003_2019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_LiGinoux_2003_2019_Monthly_1degX1deg.nc)

Comparison outputs:

- [/home/ec2-user/Research/Codex/song2021_comparison](/home/ec2-user/Research/Codex/song2021_comparison)

Important implementation detail:

- Song stores latitude ascending from `-89.5` to `89.5`
- our monthly files were written with descending latitude
- the comparison script now reorders our fields to the Song latitude
  convention before masking and plotting

Current comparison benchmark:

- the relevant published paper benchmark is Song et al. 2021 Table 4
- for MODIS land DAOD over `60S-60N` during `2007-2019`, the paper reports
  about `0.103`
- the copied Song monthly product reproduces this closely at `0.1021`
- our current Aqua means on the same weighted land domain are:
  - XGBoost `0.0694`
  - Li-Ginoux `0.0708`

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
- [DAOD_from_DB_L2_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L2_XGBoost.py)
- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)
- [monthly_mean_l3_daod_compare.py](/home/ec2-user/Research/Codex/monthly_mean_l3_daod_compare.py)
- [run_modis_l3_daod_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_daod_backfill.py)
- [supervise_modis_l3_daod_jobs.py](/home/ec2-user/Research/Codex/supervise_modis_l3_daod_jobs.py)
- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)
- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)

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
6. Use the `DAOD_from_DB_L3_*` scripts for long daily production runs, but verify date coverage after completion because Earthdata/network interruptions can silently leave gaps.
7. Use [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py) and [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py) when comparing our Aqua climatology against Song et al. monthly products.

## Known Scientific / Technical Caveats

1. The DPR/LR-derived dust-AOD target depends on assumptions about dust-dominant LR statistics.
2. Including DPR/LR as ML input features in AERONET-to-AERONET training creates target leakage if the target is built from DPR/LR.
3. The MODIS DB ML model coverage is limited by the need for all required input features, especially SSA channels.
4. Dust spectral interpolation from 440/675/1020 to 500 or 550 nm is currently handled through log-log scaling; future sessions may want to test more constrained spectral assumptions.
5. The Li and Ginoux comparisons are useful references, but they represent a coarse-mode proxy rather than the same physical retrieval chain as the DPR/LR method.
6. Long L3 Earthdata runs can stall due to authentication expiry, missing-month responses, or detached-session failures. Coverage validation is therefore part of the workflow, not an optional afterthought.
7. Comparison against Song et al. requires attention to grid conventions: Song stores latitude ascending, whereas our monthly builder initially preserved descending latitude from the daily MODIS files.

## Suggested Next Extensions

Likely next steps for future sessions:

1. Refine the 550 nm dust-AOD target interpolation and test whether the 440-675 two-point slope or the full 3-point fit is more stable.
2. Add more rigorous train/test split strategies for MODIS training, such as leave-site-out or leave-region-out validation.
3. Build fallback MODIS models that do not require SSA inputs, to improve spatial coverage.
4. Improve the comparison plotting and panel generation for publication-style figures.
5. Harden the long L3 backfill system so it can re-authenticate and resume more reliably without relying on tmux session state.
6. Diagnose why our current Aqua land-mean DAOD is about 31-33 % lower than the Song monthly reference over `60S-60N` for `2007-2019`.
7. Add README-level documentation for external users of the GitHub repository.

## Related Project Notes

There is also a shorter session summary here:

- [SESSION_NOTES.md](/home/ec2-user/Research/Codex/SESSION_NOTES.md)

That file is useful as a compact chronological session log. This memo is the higher-level workflow guide.
