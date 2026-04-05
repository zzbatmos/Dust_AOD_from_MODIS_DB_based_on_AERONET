# Session Notes

## Scope

This session extended the project from single-case notebook extraction into a
more complete operational workflow for L2 and L3 MODIS Deep Blue dust-AOD
production, comparison, visualization, long-term aggregation, and trend
analysis.

The major additions were:

- direct `550 nm` XGBoost dust-AOD training and application
- L2 and L3 production-style application scripts
- September 2024 monthly L3 Terra comparison
- long backfill jobs for Terra and Aqua daily L3 products
- improved L2 swath plotting for local case studies
- Song2021-style monthly Aqua aggregation and comparison
- experimental two-stage MODIS DB dust workflow on branch `two-stage-dust-model`
- Terra 2024 L3 two-stage QA-tier annual products and comparison figures
- full `2002-2024` Terra and Aqua L3 two-stage plus QA backfills
- monthly `(year, month, lat, lon)` L3 products for Li-Ginoux and QA tiers
- deseasonalized grid-box trend analysis with p values
- Sahel and north-band regional regime-shift diagnostics
- MODIS DB aerosol-type QA flag decoding and comparison against two-stage dust detection
- QA-augmented all-site MODIS DB collocation archive
- DB-aerosol-type-conditional dust-detection and dust-fraction training
- Aqua 2017 L2 day-by-day three-method processing and 1x1 degree daily/monthly products

## Main Code Added or Updated

### L2/L3 production scripts

- [DAOD_from_DB_L2_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L2_XGBoost.py)
- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)

These scripts mirror the older Li-Ginoux batch workflow but apply the direct
`550 nm` XGBoost dust-AOD model where appropriate.

Dust-AOD semantics were unified across methods:

- if `DB AOD550` is missing: dust AOD is `NaN`
- if `DB AOD550 > 0` but the method has no valid estimate: dust AOD is `0`
- if `DB AOD550 > 0` and a valid estimate exists: keep the positive dust AOD

### L3 monthly and backfill utilities

- [monthly_mean_l3_daod_compare.py](/home/ec2-user/Research/Codex/monthly_mean_l3_daod_compare.py)
- [plot_l3_daod_global_map.py](/home/ec2-user/Research/Codex/plot_l3_daod_global_map.py)
- [run_modis_l3_daod_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_daod_backfill.py)
- [supervise_modis_l3_daod_jobs.py](/home/ec2-user/Research/Codex/supervise_modis_l3_daod_jobs.py)
- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)
- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)
- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_daod_methods_year.py](/home/ec2-user/Research/Codex/compare_l3_daod_methods_year.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)
- [build_l3_monthly_products.py](/home/ec2-user/Research/Codex/build_l3_monthly_products.py)
- [compute_l3_monthly_trends.py](/home/ec2-user/Research/Codex/compute_l3_monthly_trends.py)

These were used for:

- daily L3 application over long time ranges
- monthly averaging with `NaN`-ignoring means
- large-scale monitoring and resumable reruns
- Song-style `(year, month, lat, lon)` monthly aggregation for Aqua 2003-2019
- land-masked comparison against the Song et al. 2021 monthly product

### L2 local-application plotting improvements

- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

Important visualization change:

- L2 plots were switched from point-based scatter plots to swath-style
  `pcolormesh` rendering

This improves:

- continuity at the granule edge
- readability of dense swath products
- side-by-side comparison of Li-Ginoux and XGBoost maps

## QA-Augmented Collocation Archive

The collocation workflow was extended so the saved MODIS DB collocation records
retain full QA information from `Quality_Assurance_Land`.

New scripts:

- [augment_collocation_with_db_qa.py](/home/ec2-user/Research/Codex/augment_collocation_with_db_qa.py)
- [run_collocation_db_qa_backfill.py](/home/ec2-user/Research/Codex/run_collocation_db_qa_backfill.py)
- [build_all_site_collocation_with_qa_and_sda.py](/home/ec2-user/Research/Codex/build_all_site_collocation_with_qa_and_sda.py)

Important outputs:

- QA-augmented per-site files:
  - [/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_files_with_QA](/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_files_with_QA)
- merged all-site file:
  - [AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl](/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA.pkl)
- merge summary:
  - [AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA_summary.json](/home/ec2-user/Research/Codex/AERONET_MODIS_DB_collocation_all_sites_with_QA_and_SDA_summary.json)

The QA augmentation preserves the legacy DB collocation statistics exactly and
adds:

- raw masked `Quality_Assurance_Land`
- decoded masked arrays for usefulness, confidence, aerosol type, and
  algorithm flag
- aerosol-type and confidence histograms
- dominant aerosol type and dominant algorithm flag

The Capo Verde test reproduced the old collocation values exactly:

- [capo_verde_mod04_collocation_repro_check.json](/home/ec2-user/Research/Codex/capo_verde_mod04_collocation_repro_check.json)

## DB-Aerosol-Type-Conditional Training

This branch added a DB-type-aware training strategy that treats the DB aerosol
type as a routing variable and prior, but not as dust truth.

Design note:

- [DB_TYPE_CONDITIONAL_TRAINING_DESIGN.md](/home/ec2-user/Research/Codex/DB_TYPE_CONDITIONAL_TRAINING_DESIGN.md)

Training scripts:

- [train_modis_db_type_conditional_detection_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_type_conditional_detection_xgb.py)
- [train_modis_db_type_conditional_from_qa_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_type_conditional_from_qa_xgb.py)

Application and comparison scripts:

- [apply_db_type_conditional_dust_model.py](/home/ec2-user/Research/Codex/apply_db_type_conditional_dust_model.py)
- [compare_two_stage_vs_db_type_conditional.py](/home/ec2-user/Research/Codex/compare_two_stage_vs_db_type_conditional.py)
- [compare_db_flag_vs_two_stage.py](/home/ec2-user/Research/Codex/compare_db_flag_vs_two_stage.py)
- [plot_modis_db_aerosol_type_flag.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_flag.py)
- [plot_modis_db_aerosol_type_diagnostics.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_diagnostics.py)
- [DB_AEROSOL_TYPE_NOTE.md](/home/ec2-user/Research/Codex/DB_AEROSOL_TYPE_NOTE.md)

Main training finding:

- DB `Dust` is the cleanest branch
- DB `Mixed` is intermediate
- DB `Smoke` still contains real AERONET dust cases

So DB aerosol type is useful as a structured prior and routing variable, but
not as a hard dust truth mask.

## Aqua 2017 L2 Day-by-Day Three-Method Processing

To test the new method on more realistic daily swath data, Aqua `MYD04_L2` was
processed day by day for all of `2017`.

Driver scripts:

- [run_aqua_l2_day_three_methods_noplot.py](/home/ec2-user/Research/Codex/run_aqua_l2_day_three_methods_noplot.py)
- [run_l2_year_three_methods_noplot.py](/home/ec2-user/Research/Codex/run_l2_year_three_methods_noplot.py)

Single-day diagnostic plotting scripts:

- [plot_aqua_l2_day_global_mosaic.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_global_mosaic.py)
- [plot_aqua_l2_day_qa_levels.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_qa_levels.py)
- [plot_aqua_l2_day_raw_vs_qa_mosaic.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_raw_vs_qa_mosaic.py)
- [plot_aqua_l2_day_dbtype_qa2_minus_raw.py](/home/ec2-user/Research/Codex/plot_aqua_l2_day_dbtype_qa2_minus_raw.py)

Year-run root:

- [/home/ec2-user/Research/Codex/aqua_l2_2017_three_methods_noplot_by_day/MYD04_L2_2017-01-01_2017-12-31](/home/ec2-user/Research/Codex/aqua_l2_2017_three_methods_noplot_by_day/MYD04_L2_2017-01-01_2017-12-31)

Year-run status:

- all `365` days completed successfully
- total per-granule `.npz` files: `28221`

Each day folder contains:

- `run_metadata.json`
- `three_method_summary.csv`
- `three_method_skips.csv`
- `granules/*.npz`

The `.npz` files preserve:

- total AOD
- Li-Ginoux DAOD
- two-stage probability, fraction, and DAOD
- DB-type-aware probability, fraction, QA, and QA-thresholded DAOD
- DB aerosol type code

## Aqua 2017 1x1 Degree Daily and Monthly Products

The saved Aqua 2017 no-plot outputs were aggregated to daily and monthly `1x1`
degree products.

Builder:

- [build_l2_daily_monthly_1deg.py](/home/ec2-user/Research/Codex/build_l2_daily_monthly_1deg.py)

Outputs:

- daily:
  - [MYD04_L2_three_methods_daily_1deg_2017.nc](/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_daily/MYD04_L2_three_methods_daily_1deg_2017.nc)
- monthly:
  - [MYD04_L2_three_methods_monthly_1deg_2017.nc](/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_monthly/MYD04_L2_three_methods_monthly_1deg_2017.nc)

Averaging rule:

- if total AOD is invalid, the pixel is excluded
- for QA-thresholded methods, pixels with valid total AOD but QA below the
  threshold contribute `dust_aod = 0`
- monthly means are simple averages of the daily `1x1` means, ignoring daily
  `NaN` cells

Current comparison figure built from the monthly file:

- [MYD04_L2_three_methods_monthly_1deg_2017_JJA_comparison.png](/home/ec2-user/Research/Codex/aqua_l2_2017_1deg_monthly/MYD04_L2_three_methods_monthly_1deg_2017_JJA_comparison.png)
- plotting script:
  - [plot_l2_monthly_jja_comparison.py](/home/ec2-user/Research/Codex/plot_l2_monthly_jja_comparison.py)

## September 2024 Terra L3 Monthly Case

Purpose:

- inspect dust-vs-smoke behavior during the major Amazon fire season

Daily outputs:

- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_LiGinoux](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_LiGinoux)
- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_XGBoost](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_XGBoost)

Monthly mean outputs:

- [/home/ec2-user/Research/Codex/MOD08_D3_2024-09_monthly_mean](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_monthly_mean)

Important plotting adjustments:

- Robinson projection
- side colorbar instead of overlaid top colorbar
- yellow-brown dust colormap
- smaller colorbar to match the map better

Main comparison figure:

- [September 2024 monthly mean comparison](/home/ec2-user/Research/Codex/MOD08_D3_2024-09_monthly_mean/MOD08_D3_DustAOD_monthly_mean_compare_2024-09.png)

## September 22, 2024 Terra Amazon L2 Granule

Granule:

- [MOD04_L2.A2024266.1345.061.2024268021127.hdf](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127.hdf)

Output directory:

- [/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon)

Important outputs:

- [shared-scale Li-Ginoux vs XGBoost dust-AOD comparison](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_dust_aod_two_panel_shared_scale.png)
- [ML minus Li-Ginoux difference map](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_dust_aod_ml_minus_li_ginoux.png)
- [comparison panel](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_comparison_panel.png)

Reference MODIS DB retrieval plots generated for this granule:

- spectral AOD at `412`, `470`, `550`, `660 nm`
- `AE`
- `SSA412`, `SSA470`, `SSA660`

Key plotting conclusion:

- scatter plotting was too sparse and visually broken for this swath
- `pcolormesh` is the better default for L2 granule figures

## Long L3 Backfill Runs

Primary output directory:

- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill](/home/ec2-user/Research/Codex/modis_l3_daod_backfill)

Current directory layout:

- [MOD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_LiGinoux)
- [MOD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_XGBoost)
- [MYD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_LiGinoux)
- [MYD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_XGBoost)

Status after the later reruns and comparison-backfill completion:

- [MOD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_LiGinoux) is complete for the in-range Terra record
- [MOD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_XGBoost) is complete for the in-range Terra record
- [MOD08_D3_TwoStage](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage) is complete for the in-range Terra record
- [MOD08_D3_TwoStageQA](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStageQA) is complete for the in-range Terra record
- [MYD08_D3_LiGinoux](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_LiGinoux) is complete for the in-range Aqua record
- [MYD08_D3_XGBoost](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_XGBoost) is complete for the in-range Aqua record
- [MYD08_D3_TwoStage](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStage) is complete for the in-range Aqua record
- [MYD08_D3_TwoStageQA](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MYD08_D3_TwoStageQA) is complete for the in-range Aqua record

Important detail:

- raw Terra two-stage has one extra boundary file for `2001-12-31`
- QA Terra does not include that extra boundary day
- there is no missing in-range Terra `2002-2024` date in the completed QA archive

Practical lesson retained:

- long Earthdata runs should not be judged by `tmux` session existence alone
- explicit file/date coverage checks are required
- authentication expiry and detached-session failures can leave silent gaps

## Movie Workflow

Script:

- [make_aqua_l3_daod_movies.py](/home/ec2-user/Research/Codex/make_aqua_l3_daod_movies.py)

Directory:

- [/home/ec2-user/Research/Codex/aqua_l3_daod_movies](/home/ec2-user/Research/Codex/aqua_l3_daod_movies)

Current state:

- one Li-Ginoux Aqua GIF was produced successfully
- the later movie run failed on a truncated image frame
- movie production was then postponed until the data-generation jobs finish

## Song2021 Monthly Comparison

Reference product copied locally:

- [/home/ec2-user/Research/Codex/Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/AquaModis_Ocean_ncountGE10_Land_AOD_20032019_Monthly_1degX1deg.nc)

New monthly products built from our Aqua daily L3 files:

- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)

Generated files:

- [/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_XGB_2003_2019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_XGB_2003_2019_Monthly_1degX1deg.nc)
- [/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_LiGinoux_2003_2019_Monthly_1degX1deg.nc](/home/ec2-user/Research/Codex/Song2021_data/MYD08_D3_DustAOD_LiGinoux_2003_2019_Monthly_1degX1deg.nc)

Averaging rule used in the monthly builder:

- if daily total AOD is missing, ignore that day
- if total AOD exists but daily dust AOD is missing, count the day and set dust AOD to `0`

Comparison script:

- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)
- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)

## Two-Stage Branch and QA Workflow

The two-stage branch was tested, tuned, and then merged into `main`.

Main two-stage documents and scripts:

- [TWO_STAGE_METHOD.md](/home/ec2-user/Research/Codex/TWO_STAGE_METHOD.md)
- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)
- [apply_two_stage_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_two_stage_dust_model_to_modis_db.py)
- [tune_two_stage_dust_model.py](/home/ec2-user/Research/Codex/tune_two_stage_dust_model.py)
- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)

Current Terra 2024 two-stage daily archive:

- [/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage_2024](/home/ec2-user/Research/Codex/modis_l3_daod_backfill/MOD08_D3_TwoStage_2024)

Coverage:

- `366` daily files for `2024`

Current QA interpretation:

- `QA>=1`
  - broad product
  - intentionally reduced to the direct XGBoost result
- `QA>=2`
  - moderate-confidence two-stage dust
- `QA>=3`
  - strict/high-confidence two-stage dust

Important 2024 QA outputs:

- [/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024](/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024)
- [QA climatology maps](/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024/MOD08_D3_TwoStage_QA_climatology_maps_2024.png)
- [QA difference maps](/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024/MOD08_D3_TwoStage_QA_difference_maps_2024.png)
- [QA detection frequency](/home/ec2-user/Research/Codex/modis_l3_two_stage_qa_2024/MOD08_D3_TwoStage_QA_detection_frequency_2024.png)

Tracked branch figure directory:

- [comparison_to_two_stage_QA_2024](/home/ec2-user/Research/Codex/comparison_to_two_stage_QA_2024)

Annual global means:

- Li-Ginoux: `0.0602`
- direct XGBoost: `0.0670`
- `QA>=1`: `0.0670`
- `QA>=2`: `0.0491`
- `QA>=3`: `0.0420`

Key interpretation:

- `QA>=1` preserves weak dust regions by design
- `QA>=2` and `QA>=3` are better for smoke-sensitive applications
- this QA setup is now the current recommended structure for broad/moderate/strict use

Comparison output directory:

- [/home/ec2-user/Research/Codex/modis_l3_daod_comparison_2024](/home/ec2-user/Research/Codex/modis_l3_daod_comparison_2024)

## Full L3 QA Backfills and Monthly Products

New backfill/orchestration scripts:

- [run_modis_l3_comparison_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_comparison_backfill.py)
- [build_modis_l3_two_stage_qa.py](/home/ec2-user/Research/Codex/build_modis_l3_two_stage_qa.py)

Monthly product builder:

- [build_l3_monthly_products.py](/home/ec2-user/Research/Codex/build_l3_monthly_products.py)

Output directories:

- [/home/ec2-user/Research/Codex/modis_l3_monthly_products](/home/ec2-user/Research/Codex/modis_l3_monthly_products)
- [/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024)

Key monthly files:

- [MOD08_D3_LiGinoux_monthly_2002_2024.nc](/home/ec2-user/Research/Codex/modis_l3_monthly_products/MOD08_D3_LiGinoux_monthly_2002_2024.nc)
- [MOD08_D3_TwoStageQA_monthly_2002_2024.nc](/home/ec2-user/Research/Codex/modis_l3_monthly_products/MOD08_D3_TwoStageQA_monthly_2002_2024.nc)
- [MYD08_D3_LiGinoux_monthly_2002_2024.nc](/home/ec2-user/Research/Codex/modis_l3_monthly_products/MYD08_D3_LiGinoux_monthly_2002_2024.nc)
- [MYD08_D3_TwoStageQA_monthly_2002_2024.nc](/home/ec2-user/Research/Codex/modis_l3_monthly_products/MYD08_D3_TwoStageQA_monthly_2002_2024.nc)

Important file structure:

- monthly QA files use `(year, month, lat, lon)`
- they include `dust_aod_qa1`, `dust_aod_qa2`, `dust_aod_qa3`, `total_aod`, and `n_days`
- Aqua early-2002 months remain `NaN` where `MYD08_D3` did not yet exist

## Trend Analysis and Significance

Trend script:

- [compute_l3_monthly_trends.py](/home/ec2-user/Research/Codex/compute_l3_monthly_trends.py)

Trend methodology:

- deseasonalize each grid box by subtracting its monthly climatology
- fit a linear trend to the monthly anomalies
- report slope as dust AOD per decade
- also compute `p_value` and a significance flag for `p < 0.05`

Important outputs:

- [MYD08_D3_TwoStageQA_monthly_trends.nc](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_monthly_trends.nc)
- [MYD08_D3_TwoStageQA_monthly_trend_maps.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_monthly_trend_maps.png)
- [MYD08_D3_TwoStageQA_monthly_significance_maps.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_monthly_significance_maps.png)
- [MOD08_D3_TwoStageQA_monthly_trends.nc](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MOD08_D3_TwoStageQA_monthly_trends.nc)
- [MYD08_D3_LiGinoux_monthly_significance_maps.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_LiGinoux_monthly_significance_maps.png)

Trend files now include total-AOD trends too:

- `total_aod_trend_per_decade`
- `total_aod_p_value`
- `total_aod_significant_p_lt_0p05`

## Sahel and North-Band Regional Diagnostics

Regional analysis scripts:

- [analyze_sahel_trends.py](/home/ec2-user/Research/Codex/analyze_sahel_trends.py)
- [analyze_regime_shift_regions.py](/home/ec2-user/Research/Codex/analyze_regime_shift_regions.py)

Sahel positive-trend region:

- annotated map: [MYD08_D3_TwoStageQA_monthly_trend_maps_sahel_box.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_monthly_trend_maps_sahel_box.png)
- selected points: [MYD08_D3_TwoStageQA_Sahel_selected_points.csv](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_Sahel_selected_points.csv)
- point time series for `QA1`, `QA2`, `QA3` were generated

North-band negative-trend region:

- annotated map: [MYD08_D3_TwoStageQA_NorthBand_monthly_trend_maps_box.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_NorthBand_monthly_trend_maps_box.png)
- selected points: [MYD08_D3_TwoStageQA_NorthBand_selected_points.csv](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_NorthBand_selected_points.csv)
- seasonal-cycle figure for `15.5N, 17.5E`: [MYD08_D3_TwoStageQA_point_15p5N_17p5E_seasonal_cycle_QA2.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/MYD08_D3_TwoStageQA_point_15p5N_17p5E_seasonal_cycle_QA2.png)

Regime-shift test results for Aqua `QA2`:

- southern band shows an upward shift centered around `2015`
- northern band shows a marked decrease concentrated in `2020-2024`
- summary table: [dust_aod_qa2_regime_shift_summary.csv](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/regime_shift_analysis/dust_aod_qa2_regime_shift_summary.csv)

## Niamey Duststorm Context

Historical weather extractor:

- [extract_niamey_duststorm_dates.py](/home/ec2-user/Research/Codex/extract_niamey_duststorm_dates.py)

Outputs:

- [/home/ec2-user/Research/Codex/niamey_duststorm_dates](/home/ec2-user/Research/Codex/niamey_duststorm_dates)
- [niamey_duststorm_vs_sahel_point1_yearly.png](/home/ec2-user/Research/Codex/modis_l3_trends_2002_2024/niamey_duststorm_vs_sahel_point1_yearly.png)

Important finding:

- the Timeanddate Niamey archive only starts in `2010-10`
- within the available archive, `2017-04-17` is indeed listed as a duststorm day in Niamey

## MODIS DB Aerosol-Type QA Flag Test

QA-flag scripts:

- [plot_modis_db_aerosol_type_flag.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_flag.py)
- [compare_db_flag_vs_two_stage.py](/home/ec2-user/Research/Codex/compare_db_flag_vs_two_stage.py)

Purpose:

- decode the `Deep Blue Aerosol Type` flag from `Quality_Assurance_Land`
- compare it against the tuned two-stage dust detector on:
  - heavy dust case `2025-03-14`
  - heavy smoke case `2024-09-22`

Important comparison result:

- the DB aerosol-type flag identifies the smoke case as mostly smoke
- but it is too noisy/inconsistent to use as a standalone dust mask
- the two-stage detector is much more conservative in the smoke case and therefore more useful for dust-screening

Key output directory:

- [/home/ec2-user/Research/Codex/db_flag_vs_two_stage_comparison](/home/ec2-user/Research/Codex/db_flag_vs_two_stage_comparison)

- [/home/ec2-user/Research/Codex/song2021_comparison](/home/ec2-user/Research/Codex/song2021_comparison)

Important output figures:

- [global monthly mean DAOD time series](/home/ec2-user/Research/Codex/song2021_comparison/aqua_2003_2019_global_land_monthly_mean_daod_timeseries.png)
- [global monthly-cycle figure](/home/ec2-user/Research/Codex/song2021_comparison/aqua_2003_2019_global_land_monthly_cycle_daod.png)
- [DAOD climatology maps](/home/ec2-user/Research/Codex/song2021_comparison/aqua_2003_2019_daod_climatology_maps.png)
- [DAOD difference maps](/home/ec2-user/Research/Codex/song2021_comparison/aqua_2003_2019_daod_difference_maps.png)

Important plotting fixes:

- Song stores latitude ascending from `-89.5` to `89.5`
- our monthly files were descending from `89.5` to `-89.5`
- the comparison script now reorders our fields to Song orientation before masking and plotting
- climatology and difference maps now use Robinson projection with coastlines, borders, and labeled lat-lon grids

Published-value comparison:

- the relevant published benchmark is Song et al. 2021 Table 4, not Table 2
- Song et al. report MODIS land DAOD over `60S-60N` for `2007-2019` as about `0.103`

Our area-weighted land means over `60S-60N` for `2007-2019`:

- Song2021 monthly product: `0.1021`
- XGBoost: `0.0694`
- Li-Ginoux: `0.0708`

Interpretation:

- the copied Song monthly product reproduces the published value closely
- both our current Aqua products are about `31-33 %` lower than the Song reference on that metric

## Key Operational Caveats

1. Earthdata authentication appears to expire during long runs, so month-level
   retries and re-login logic are necessary.
2. tmux-based supervision alone was not reliable enough to guarantee completion.
3. L3 date coverage should always be validated after a long run.
4. For L2 publication-style plots, swath rendering is preferable to scatter.

## Best Current Entry Points

Use these scripts for future work:

- [train_modis_db_dust_aod550_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_dust_aod550_xgb.py)
- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)
- [DAOD_from_DB_L3_LiGinoux.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_LiGinoux.py)
- [DAOD_from_DB_L3_XGBoost.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_XGBoost.py)
- [monthly_mean_l3_daod_compare.py](/home/ec2-user/Research/Codex/monthly_mean_l3_daod_compare.py)
- [run_modis_l3_daod_backfill.py](/home/ec2-user/Research/Codex/run_modis_l3_daod_backfill.py)
- [build_song_style_monthly_daod.py](/home/ec2-user/Research/Codex/build_song_style_monthly_daod.py)
- [compare_song2021_monthly_daod.py](/home/ec2-user/Research/Codex/compare_song2021_monthly_daod.py)
