# 2023 MODIS Dust AOD Production Plan

## Scope

Process `2023` MODIS Deep Blue L2 data for:

- `MOD04_L2` (Terra)
- `MYD04_L2` (Aqua)

Dust AOD methods:

- `two-stage`
- `two-stage QA2 + high-AOD dust rescue` as a flagged enhanced-completeness companion
- `Li-Ginoux`

Products to generate:

1. L2 per-granule derived outputs at native Deep Blue resolution (`~10 km`)
2. Daily `0.5° x 0.5°` L3 grids
3. Monthly `0.5° x 0.5°` L3 grids
4. A small figure set after production completes

Current decision:

- keep the derived L2 outputs
- do not archive the raw downloaded MODIS HDF files beyond the day-level scratch workflow

## Existing code base to reuse

The project already has the right patterns:

- year-scale L2 day-by-day driver:
  - [run_l2_year_three_methods_noplot.py](/home/ec2-user/Research/Codex/run_l2_year_three_methods_noplot.py)
- per-granule two-stage and Li-Ginoux logic:
  - [apply_two_stage_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_two_stage_dust_model_to_modis_db.py)
  - [apply_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_dust_model_to_modis_db.py)
- daily/monthly gridded aggregation pattern:
  - [build_l2_daily_monthly_1deg.py](/home/ec2-user/Research/Codex/build_l2_daily_monthly_1deg.py)

## Production workflow

### Phase 1. Validation run

Run both sensors for a short period first:

- `2023-04-01` to `2023-04-07`

Validate:

- output schema
- day-by-day resume behavior
- disk growth
- throughput
- `0.5°` aggregation correctness

### Phase 2. L2 production

Run a day-by-day yearly driver separately for:

- Aqua `MYD04_L2`
- Terra `MOD04_L2`

For each day:

1. search and download that day’s granules
2. compute:
   - total AOD
   - Li-Ginoux dust AOD
   - two-stage dust probability
   - two-stage dust fraction
   - two-stage dust AOD
3. save compressed per-granule arrays
4. write day-level summary CSV/JSON
5. delete downloaded HDF files

### Phase 3. L3 aggregation

For each sensor, aggregate the saved L2 outputs to:

- daily `0.5° x 0.5°`
- monthly `0.5° x 0.5°`

Averaging rule:

- if total AOD is invalid, exclude the pixel
- Li-Ginoux daily dust AOD uses the saved L2 Li-Ginoux output directly
- two-stage daily dust AOD is computed from the saved two-stage fraction and total AOD
- if total AOD is valid and the two-stage probability is below the QA threshold, dust AOD is assigned `0`
- monthly means are simple averages of the daily grid means

Primary recommended product:

- `two-stage QA2`

Companion baseline:

- `Li-Ginoux`

Enhanced-completeness companion:

- `two-stage QA2 + high-AOD dust rescue`
- this product keeps the standard two-stage QA2 output unchanged and only
  raises QA2 dust AOD for a restricted high-AOD DB-dust regime
- default rescue settings are `AOD550 >= 2.0`, AE-screened Li-Ginoux
  `DAOD >= 0.8`, two-stage dust probability `>= 0.25`, DB aerosol type
  `Dust`, and a cap of `0.95 * AOD550`
- the rescue should remain separately flagged because it uses DB dust
  classification and Li-Ginoux information as a conditional completeness
  correction

### Phase 4. Final figures

After production:

- annual mean Terra maps
- annual mean Aqua maps
- Terra+Aqua mean maps
- regional zooms as needed
- method-comparison figures

## Output layout

Recommended layout:

- `modis_l2_2023_dust_aod/terra/...`
- `modis_l2_2023_dust_aod/aqua/...`
- `modis_l3_2023_dust_aod_0p5/terra_daily.nc`
- `modis_l3_2023_dust_aod_0p5/aqua_daily.nc`
- `modis_l3_2023_dust_aod_0p5/terra_monthly.nc`
- `modis_l3_2023_dust_aod_0p5/aqua_monthly.nc`

Each L2 day folder should contain:

- `run_metadata.json`
- `two_method_summary.csv`
- `two_method_skips.csv`
- `granules/*.npz`

Key per-granule NPZ fields:

- `aod550`
- `li_dust_aod`
- `two_stage_probability`
- `two_stage_fraction`
- `two_stage_dust_aod`
- `two_stage_qa2_dust_aod`
- `two_stage_veto_qa2_dust_aod`
- `two_stage_high_aod_rescue_qa2_dust_aod`
- `high_aod_rescue_flag`
- `high_aod_rescue_candidate`
- `high_aod_rescue_increment`
- `aerosol_type_code`

## Disk-space estimate

Reference point already available in this repo:

- Aqua 2017 three-method derived L2 archive:
  - [aqua_l2_2017_three_methods_noplot_by_day](/home/ec2-user/Research/Codex/aqua_l2_2017_three_methods_noplot_by_day)
  - size: about `6.5 GB`
  - per-granule files: `28,221`

Expected 2023 derived-output footprint:

- one sensor, two methods: about `4–5 GB`
- both sensors combined: about `8–10 GB`

Expected `0.5°` L3 outputs:

- daily + monthly NetCDFs for both sensors: about `1–3 GB`

Expected scratch during production:

- roughly `0.5–2 GB` active day-download scratch per worker
- with `4–5` workers, reserve `10–20 GB` scratch comfortably

If raw MODIS HDF files were archived permanently, the footprint would likely
increase to roughly `200–300 GB` for both sensors, which is not recommended for
this run.

## Processing-time estimate

The dominant uncertainty is Earthdata download latency and retry behavior.

Operational estimate:

- one sensor, `5` workers: about `8–16 hours`
- both sensors sequentially: about `16–32 hours`
- both sensors with overlap and stable network: about `10–20 hours`

Planning assumption:

- allocate `1–2` wall-clock days for the full 2023 production

## Parallelization plan

Recommended:

- run `4–5` day-shard workers per sensor
- keep Terra and Aqua as separate runs
- keep daily work resumable
- do not make plots during production

## Immediate implementation tasks

1. Add a two-method year driver:
   - save only Li-Ginoux and two-stage outputs
2. Add a `0.5°` daily/monthly aggregation script
3. Run a 1-week validation test
4. If outputs are clean, launch the full 2023 production
