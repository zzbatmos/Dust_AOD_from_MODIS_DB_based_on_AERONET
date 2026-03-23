# Session Notes

## Scope

This session extended the project from single-case notebook extraction into a
more complete operational workflow for L2 and L3 MODIS Deep Blue dust-AOD
production, comparison, and visualization.

The major additions were:

- direct `550 nm` XGBoost dust-AOD training and application
- L2 and L3 production-style application scripts
- September 2024 monthly L3 Terra comparison
- long backfill jobs for Terra and Aqua daily L3 products
- improved L2 swath plotting for local case studies

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

These were used for:

- daily L3 application over long time ranges
- monthly averaging with `NaN`-ignoring means
- large-scale monitoring and resumable reruns

### L2 local-application plotting improvements

- [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

Important visualization change:

- L2 plots were switched from point-based scatter plots to swath-style
  `pcolormesh` rendering

This improves:

- continuity at the granule edge
- readability of dense swath products
- side-by-side comparison of Li-Ginoux and XGBoost maps

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

Status at the latest validation:

- Aqua (`MYD08_D3`) is complete from `2002-07-04` through `2024-12-31`
- Terra (`MOD08_D3`) is not yet complete
- within each platform, Li-Ginoux and XGBoost have matching date coverage

Coverage check performed:

- Terra expected dates `2002-01-01` through `2024-12-31`
- Aqua expected dates `2002-07-04` through `2024-12-31`

Result:

- Terra is still missing `39` dates
- Aqua is still missing `25` dates relative to the ideal continuous range

Practical lesson:

- long Earthdata runs should not be judged by tmux session existence alone
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

