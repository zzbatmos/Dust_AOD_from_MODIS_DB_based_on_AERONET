# Production Freeze L2 DAOD V1

This document freezes the MODIS Deep Blue Level-2 dust aerosol optical depth
(DAOD) production definition used for mass processing and manuscript results.

## Scope

The freeze version processes MODIS Deep Blue Level-2 land aerosol products:

- Terra: `MOD04_L2`
- Aqua: `MYD04_L2`

The workflow downloads one day of Level-2 granules, derives dust products at
native DB pixel resolution, deletes the downloaded HDF scratch files, and keeps
compressed per-granule NPZ outputs. The NPZ outputs are then aggregated to
daily and monthly `0.5 degree x 0.5 degree` products.

## Included Methods

### Two-stage XGBoost method

The two-stage method uses:

- first-stage classifier probability: `P(dust)`
- second-stage raw regressor dust fraction: `two_stage_fraction_raw`

The raw regressor fraction is saved before applying QA thresholds. This is
required so the QA tiers remain distinct.

QA thresholds are fixed as:

| Product | Threshold | Interpretation |
| --- | --- | --- |
| `dust_aod_two_stage_qa1` | `P(dust) >= 0.4` | Broad/aggressive dust screening |
| `dust_aod_two_stage_qa2` | `P(dust) >= 0.6` | Recommended balanced product |
| `dust_aod_two_stage_qa3` | `P(dust) >= 0.7` | Strict/high-confidence dust screening |

For any valid total-AOD pixel below the selected QA threshold, dust AOD is set
to zero. Pixels with invalid total AOD are excluded from averaging.

### Li-Ginoux method

The Li-Ginoux parameterization is:

```text
FMF = 0.085 * AE^2 + 0.336 * AE + 0.051
DAOD = (1 - FMF) * AOD550
```

The raw diagnostic field uses the existing spectral SSA condition:

```text
SSA412 < SSA470
```

The recommended Li-Ginoux comparison field additionally applies:

```text
FMF <= 0.7
```

This is equivalent to approximately `AE <= 1.42` for the Li-Ginoux
parameterization. The screened field is the default Li-Ginoux field in the
production NetCDF output; the unscreened field is retained as a diagnostic.

## Excluded Experimental Branches

The following methods are intentionally excluded from this production freeze:

- DB-aerosol-type-aware branch
- suspect-veto branch
- high-AOD-rescue branch

These branches remain useful as sensitivity experiments, but they are not part
of the manuscript/mass-production product definition.

## L2 NPZ Variables

Each processed granule contains:

- `lat`
- `lon`
- `aod550`
- `li_dust_aod`
- `li_ginoux_fmf`
- `li_ginoux_ae_screen_dust_aod`
- `two_stage_probability`
- `two_stage_fraction_raw`
- `two_stage_qa1_dust_aod`
- `two_stage_qa2_dust_aod`
- `two_stage_qa3_dust_aod`

## L3 NetCDF Variables

Daily output:

- `total_aod`
- `dust_aod_li_ginoux`
- `dust_aod_li_ginoux_raw`
- `dust_aod_two_stage_qa1`
- `dust_aod_two_stage_qa2`
- `dust_aod_two_stage_qa3`
- `n_pixels`

Monthly output:

- `total_aod`
- `dust_aod_li_ginoux`
- `dust_aod_li_ginoux_raw`
- `dust_aod_two_stage_qa1`
- `dust_aod_two_stage_qa2`
- `dust_aod_two_stage_qa3`
- corresponding `*_valid_days` variables

## Production Scripts

Primary scripts:

- `run_l2_year_two_methods_production.py`
- `build_l2_daily_monthly_0p5deg_production.py`

Example L2 production command:

```bash
python run_l2_year_two_methods_production.py \
  --year 2017 \
  --short-name MYD04_L2 \
  --shard-index 0 \
  --num-shards 5 \
  --output-root modis_l2_production_freeze_v1 \
  --scratch-root modis_l2_production_freeze_v1_scratch
```

Example L3 aggregation command:

```bash
python build_l2_daily_monthly_0p5deg_production.py \
  --year 2017 \
  --input-root modis_l2_production_freeze_v1/MYD04_L2_2017-01-01_2017-12-31 \
  --daily-output modis_l3_production_freeze_v1/MYD04_L2_daily_0p5deg_2017.nc \
  --monthly-output modis_l3_production_freeze_v1/MYD04_L2_monthly_0p5deg_2017.nc
```

## Version

Freeze name:

```text
production-freeze-l2-daod-v1
```
