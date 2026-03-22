# Session Notes

## Scope

This session focused on extracting notebook workflows from
[`AERONET_DPR_analysis.ipynb`](/home/ec2-user/Research/Codex/AERONET_DPR_analysis.ipynb)
into standalone scripts and applying the trained dust-AOD models to a real
MODIS Deep Blue case.

The notebook copy in this directory has 109 cells. The user-requested
"cells 123-157" corresponded to notebook execution counts, not physical cell
indices.

## Main Scripts

### 1. Training / extraction

[`extract_modis_db_dust_model.py`](/home/ec2-user/Research/Codex/extract_modis_db_dust_model.py)

Purpose:
- Load collocated MOD04 and MYD04 pickle files from the parent directory
- Rebuild the AERONET/MODIS DB training dataframe
- Train XGBoost dust-fraction models
- Save models, metadata, metrics, and feature importance

Important defaults:
- `--mod04` defaults to `../AERONET_MOD04_L2_collocation_with_SDA_data.pkl`
- `--myd04` defaults to `../AERONET_MYD04_L2_collocation_with_SDA_data.pkl`
- `--lr-dust-675` defaults to `56.0`

Model variants currently implemented:
- `log1p`
- `logit`
- `log1p_weighted_bins`

Outputs:
- `dust_aod_xgb_log1p.json`
- `dust_aod_xgb_log1p_meta.json`
- `dust_aod_xgb_logit.json`
- `dust_aod_xgb_logit_meta.json`
- `dust_aod_xgb_log1p_weighted_bins.json`
- `dust_aod_xgb_log1p_weighted_bins_meta.json`
- `dust_aod_xgb_model_comparison.csv`

Legacy aliases:
- `dust_aod_xgb.json`
- `dust_aod_xgb_meta.json`

These legacy files currently point to the best-performing model, which is
still `log1p`.

### 2. MODIS application / inference

[`apply_dust_model_to_modis_db.py`](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

Purpose:
- Search/download a MOD04 or MYD04 granule via `earthaccess`, or use a local HDF
- Read MODIS Deep Blue fields via `../MODIS_Lib.py`
- Apply trained dust-fraction models
- Derive dust AOD from predicted dust fraction
- Save PNG maps and summary files
- Apply the Li and Ginoux empirical FMF parameterization for comparison

Supports:
- local granule input via `--input-file`
- Earthdata search via `--short-name`, `--start`, `--end`, `--lon`, `--lat`

## Important Model Context

### What the current ML model predicts

The current trained XGBoost models predict:
- `dust_fraction_675`

They do **not** directly predict dust AOD.

Dust AOD maps are derived after inference as:
- `predicted_dust_fraction * MODIS_DB_AOD660`

This was chosen because predicting dust fraction keeps the result physically
bounded and avoids dust AOD exceeding total AOD.

### Why the ML maps have fewer valid pixels than raw DB AOD

The model requires all 8 predictor fields to be finite at the same pixel:
- `MODIS_DB_AOD550`
- `MODIS_DB_AOD412`
- `MODIS_DB_AOD470`
- `MODIS_DB_AOD660`
- `MODIS_DB_AE`
- `MODIS_DB_SSA412`
- `MODIS_DB_SSA470`
- `MODIS_DB_SSA660`

For the March 14, 2025 Terra granule:
- AOD and AE valid pixels: `6905`
- `SSA412` valid pixels: `2253`
- `SSA470` valid pixels: `3489`
- `SSA660` valid pixels: `3981`
- all 8 valid together: `1768`

So the ML dust-fraction and dust-AOD maps have `1768` valid pixels.

## Training Results

From [`dust_aod_xgb_model_comparison.csv`](/home/ec2-user/Research/Codex/dust_aod_xgb_model_comparison.csv):

- `log1p`: RMSE `0.2485`, MAE `0.1870`, R² `0.3789`, bias `-0.0236`
- `log1p_weighted_bins`: RMSE `0.2505`, MAE `0.1902`, R² `0.3687`, bias `+0.0235`
- `logit`: RMSE `0.2955`, MAE `0.1774`, R² `0.1217`, bias `-0.1166`

Conclusion:
- `log1p` remains the best overall model on the current split
- `logit` did not improve the result
- inverse-frequency bin weighting did not improve the result

## Li and Ginoux Comparison

Paper:
- Li and Ginoux, 2025, GRL
- DOI: https://doi.org/10.1029/2024GL114397

Implemented parameterization:
- Eq. 7: `FMF = 0.085 * AE^2 + 0.336 * AE + 0.051`

Applied method:
1. derive FMF from MODIS DB `AE`
2. derive coarse-mode fraction as `1 - FMF`
3. approximate dust AOD as `(1 - FMF) * MODIS_DB_AOD550`
4. require `SSA412 < SSA470`; otherwise set dust AOD to `NaN`

This method is implemented in:
- [`apply_dust_model_to_modis_db.py`](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)

## March 14, 2025 Terra MODIS Case

Granule used:
- [`MOD04_L2.A2025073.1705.061.2025074013638.hdf`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638.hdf)

Output directory:
- [`dust_model_application_2025-03-14`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14)

### Core summaries

ML summary:
- [`MOD04_L2.A2025073.1705.061.2025074013638_model_application_summary.csv`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_model_application_summary.csv)

MODIS DB reference summary:
- [`MOD04_L2.A2025073.1705.061.2025074013638_reference_products_summary.csv`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_reference_products_summary.csv)

Li-Ginoux summary:
- [`MOD04_L2.A2025073.1705.061.2025074013638_li_ginoux_summary.json`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_li_ginoux_summary.json)

Run metadata:
- [`MOD04_L2.A2025073.1705.061.2025074013638_run_metadata.json`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_run_metadata.json)

### Key numbers for this case

ML `log1p`:
- dust fraction median: `0.0901`
- derived dust AOD median: `0.0077`

Li-Ginoux:
- FMF median: `0.0510`
- dust AOD proxy median: `0.1072`

Interpretation:
- Li-Ginoux is much more aggressive than the ML estimate for this dust event
- the two methods still have strong spatial agreement

Scatter comparison:
- valid paired pixels: `1766`
- Pearson r: `0.958`
- Bias (Li-Ginoux - ML): `0.074`
- MAE: `0.074`
- RMSE: `0.097`

## Important Figures

### ML products

- [`log1p dust fraction`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_log1p_dust_fraction.png)
- [`log1p dust AOD`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_log1p_dust_aod_660.png)

### MODIS DB reference products

- [`MODIS DB AOD550`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_MODIS_DB_AOD550.png)
- [`MODIS DB AOD660`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_MODIS_DB_AOD660.png)
- [`MODIS DB AE`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_MODIS_DB_AE.png)
- [`MODIS DB SSA412`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_MODIS_DB_SSA412.png)
- [`MODIS DB SSA470`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_MODIS_DB_SSA470.png)

### Li-Ginoux products

- [`Li-Ginoux FMF`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_li_ginoux_fmf_ssa_constraint.png)
- [`Li-Ginoux dust AOD`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_li_ginoux_dust_aod_ssa_constraint.png)

### Comparison figures

- [`comparison panel`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_comparison_panel.png)
- [`ML vs Li-Ginoux scatter`](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_ml_vs_li_ginoux_dust_aod_scatter.png)

## Resume Suggestions

Most useful next steps if continuing this work:
- build a fallback model without SSA features to increase spatial coverage
- compare ML and Li-Ginoux on more dust cases and non-dust cases
- harmonize comparison color scales across ML and Li-Ginoux dust-AOD maps
- test a two-stage ML model: dust detection first, dust-fraction regression second
