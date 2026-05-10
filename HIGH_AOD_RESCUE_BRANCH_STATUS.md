# High-AOD Dust Rescue Branch Status

Branch: `two-stage-high-aod-dust-rescue`

## Purpose

This branch tests an optional high-AOD rescue layer for the two-stage XGBoost
MODIS Deep Blue dust-AOD method.

The goal is not to replace the default two-stage `QA2` product. The goal is to
recover intense dust plume cores where the default two-stage classifier is too
conservative.

## Production Definition

The high-AOD rescue can only increase standard two-stage `QA2` dust AOD when all
conditions are satisfied:

- DB `AOD550 >= 2.0`
- AE-screened Li-Ginoux dust AOD `>= 0.8`
- two-stage dust probability `>= 0.25`
- DB aerosol type is `Dust`
- inferred Li-Ginoux fine-mode fraction `<= 0.7`

The rescued dust AOD is capped:

```text
rescued_DAOD = max(two_stage_QA2_DAOD, min(Li_Ginoux_AE_screened_DAOD, 0.95 * AOD550))
```

The `0.95` cap is consistent with the Li-Ginoux parameterization at `AE = 0`,
where the fine-mode fraction is about `0.051` and the coarse-mode fraction is
about `0.949`.

## Output Variables

The production path keeps the standard two-stage output unchanged and adds
separate companion variables:

- `li_ginoux_ae_screen_dust_aod`
- `two_stage_qa2_dust_aod`
- `two_stage_high_aod_rescue_qa2_dust_aod`
- `high_aod_rescue_flag`
- `high_aod_rescue_candidate`
- `high_aod_rescue_increment`
- `aerosol_type_code`

The `0.5°` aggregation adds:

- `dust_aod_two_stage_high_aod_rescue_qa2`
- `high_aod_rescued_pixel_fraction`

## Validation Summary

### December 2017 Aqua Dust Test

The full Aqua `MYD04_L2` December 2017 month was processed and aggregated.

Status:

- `31/31` days completed
- `1905` derived L2 NPZ granules

Key Sahel high-AOD plume result:

- `2017-12-07`
- Li-Ginoux: `0.602`
- standard two-stage `QA2`: `0.294`
- high-AOD rescue `QA2`: `0.563`

Interpretation:

- the rescue recovers the intense dust plume that standard `QA2` missed
- the strongest effect is localized to North Africa/Sahel high-AOD events

### August 2017 Aqua Smoke Stress Test

The full Aqua `MYD04_L2` August 2017 month was processed and aggregated.

Status:

- `31/31` days completed
- `2896` derived L2 NPZ granules

Known smoke case:

- `2017-08-29`
- North America mean rescue increment: `0.000000`
- North America smoke-core mean rescue increment: `0.000000`
- rescued-pixel fraction in both regions: `0.000000`

Interpretation:

- the rescue passed the targeted North America smoke stress test
- August monthly impact is very small globally and regionally

## Current Recommendation

Use standard two-stage `QA2` as the default product.

Keep the high-AOD rescue as a separate enhanced-completeness companion product:

- useful for intense dust plume recovery
- not appropriate as the default product because it depends partly on DB aerosol
  type and Li-Ginoux support
- must be flagged separately in L2 and L3 outputs

## Suggested Branch Action

This branch is scientifically useful enough to keep.

Recommended merge strategy:

- merge the production code only if the main branch is ready to expose multiple
  companion products
- keep output names explicit so users do not confuse standard `QA2` with rescue
  `QA2`
- do not change the default method recommendation in manuscript text without a
  clear sensitivity section
