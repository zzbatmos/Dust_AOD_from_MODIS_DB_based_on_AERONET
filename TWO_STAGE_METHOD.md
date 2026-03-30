# Two-Stage Dust AOD Method

## Purpose

This document explains the two-stage MODIS Deep Blue dust-AOD method that was
added after the original single-stage XGBoost experiments.

The main motivation was scientific, not purely statistical:

- a single regression model has to solve two different problems at once
- first, decide whether a pixel is dusty at all
- second, estimate how much dust is present if the pixel is dusty

Those two tasks are related, but they are not the same task.

In practice, this matters because MODIS DB pixels include:

- true dust scenes
- smoke-dominated scenes
- mixed aerosol scenes
- weak-background aerosol scenes

When a single regressor is trained across all of them, it often blurs the
regimes together. That can cause:

- spurious weak dust retrievals in smoke or non-dust scenes
- underestimation in real dust events
- unstable behavior near zero

The two-stage method separates those roles explicitly.

## Core Idea

The method is:

1. `Dust detection`
   - classify whether the scene is dust-relevant
2. `Dust amount estimation`
   - only after a pixel is considered dust-relevant, estimate dust fraction
3. `Dust AOD derivation`
   - compute dust AOD from:
     `dust_AOD_550 = predicted_dust_fraction * MODIS_DB_AOD550`

This design preserves physical consistency:

- predicted dust fraction stays in `[0, 1]`
- dust AOD cannot exceed the total AOD at `550 nm`
- non-dust pixels can be forced to `0`

## Why We Did It This Way

### Problem with direct dust-AOD regression

If a model predicts dust AOD directly, it can produce:

- negative dust AOD
- dust AOD larger than total AOD

That is physically impossible unless extra constraints are added.

### Problem with a single dust-fraction regressor

Predicting dust fraction is physically cleaner, but a single regressor still
has to span:

- near-zero dust fraction
- moderate mixed cases
- strong dust cases

The mapping from MODIS DB features to dust fraction is different across these
regimes.

### Why a classifier first

The first-stage classifier is meant to answer:

- does this pixel look dust-like at all?

That is especially useful when trying to reject smoke.

The second-stage regressor is only responsible for:

- how much dust fraction should be assigned once the pixel is already considered
  dust-relevant

That division is much closer to the physical reasoning a human would use.

## Training Target and Features

### Target

The two-stage method uses the same DPR/LR-derived AERONET dust fraction target
used in the earlier MODIS training workflow:

- `dust_fraction_675`

This target comes from:

1. deriving AERONET dust fraction using DPR and LR
2. using the collocated MODIS DB features as predictors

### Predictor set

The method uses the same MODIS-compatible features as the earlier single-stage
models:

- `MODIS_DB_AOD550`
- `MODIS_DB_AOD412`
- `MODIS_DB_AOD470`
- `MODIS_DB_AOD660`
- `MODIS_DB_AE`
- `MODIS_DB_SSA412`
- `MODIS_DB_SSA470`
- `MODIS_DB_SSA660`

The fact that these variables already exist in the previous workflow is why the
two-stage method could be added cleanly without redefining the MODIS feature
space.

## Stage 1: Dust Detection

### Label definition

For the classifier, the training label is:

- dusty if `dust_fraction_675 >= threshold`

This threshold is tunable.

The first implementation started with:

- `dust_threshold = 0.2`

This was chosen as a moderate label:

- not too weak, so tiny ambiguous dust traces are not over-labeled as dust
- not too strict, so the classifier can still learn a useful dust population

### Model

The classifier is:

- `XGBClassifier`

Implemented in:

- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)

### Why classifier probability matters

The classifier outputs a probability:

- `P(dust)`

That probability then has to be thresholded during inference:

- if `P(dust) >= probability_threshold`, run the dusty branch
- otherwise set predicted dust fraction to `0`

This probability threshold is also tunable and strongly controls the smoke
rejection behavior.

## Stage 2: Dust Fraction Estimation

### Regress only on dusty samples

The regressor is trained only on the subset where:

- `dust_label = 1`

That means it is not burdened with learning the full near-zero population.

### Output variable

The regressor predicts:

- `dust_fraction_675`

To preserve the bounded nature of dust fraction, the regressor is trained on a
logit-transformed target:

```text
z = log(f / (1 - f))
```

and inverted at inference time with the logistic function.

### Model

The regressor is:

- `XGBRegressor`

also implemented in:

- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)

## Final Inference Logic

At prediction time:

1. compute `P(dust)` from the classifier
2. compute a dust-fraction estimate from the regressor
3. apply the classifier probability threshold:
   - if below threshold, set fraction to `0`
   - if above threshold, keep the regressed fraction
4. derive dust AOD:
   - `dust_AOD_550 = dust_fraction * MODIS_DB_AOD550`

This guarantees:

- non-dust scenes are forced toward zero
- dusty scenes get a quantitative estimate
- dust AOD remains physically bounded by total AOD

## QA Extension

After the initial two-stage implementation, the method was extended with a
user-facing QA concept so the product can support both:

- conservative smoke-resistant dust applications
- broader weak-dust applications

This is important because the strict tuned two-stage mask worked well over
Amazon smoke and India, but it also screened out weaker dust regions such as:

- the southeastern United States
- Australia
- South Africa

### Current QA interpretation

For the 2024 Terra L3 experiments, the QA tiers were defined operationally as:

- `QA>=1`
  - broad dust product
  - intentionally reduced to the direct single-stage XGBoost product
- `QA>=2`
  - moderate-confidence two-stage dust
  - requires stronger dust probability and AOD support
- `QA>=3`
  - strict/high-confidence two-stage dust
  - most conservative option for smoke-sensitive applications

This design was chosen because it keeps the useful broad-coverage behavior of
the direct XGBoost model in weak dust regions, while preserving the smoke
rejection advantage of the two-stage method at higher QA.

### Why `QA>=1` was mapped to direct XGBoost

The first attempt at a low-confidence tier still used only the two-stage
framework with looser probability thresholds. That recovered some dust, but it
did not fully restore weak dust regions. The updated QA design therefore uses:

- `QA>=1`: direct XGBoost behavior
- `QA>=2` and `QA>=3`: two-stage filtered behavior

This gives users a clearer progression:

- if broad spatial coverage is more important, use `QA>=1`
- if dust type purity is more important, use `QA>=2` or `QA>=3`

### 2024 Terra L3 QA processing

The yearly L3 processing and plotting code for this QA workflow is:

- [DAOD_from_DB_L3_TwoStage.py](/home/ec2-user/Research/Codex/DAOD_from_DB_L3_TwoStage.py)
- [compare_l3_two_stage_qa_year.py](/home/ec2-user/Research/Codex/compare_l3_two_stage_qa_year.py)

The 2024 Terra annual tests showed:

- `QA>=1` reproduces the direct XGBoost annual mean
- `QA>=2` provides an intermediate product
- `QA>=3` remains the strict smoke-resistant product

This QA structure is therefore the current recommended way to expose the
two-stage method to users.

## Code Structure

### Training

- [train_modis_db_two_stage_xgb.py](/home/ec2-user/Research/Codex/train_modis_db_two_stage_xgb.py)

This script:

- rebuilds the collocated training dataframe
- defines the dust/no-dust label
- trains the classifier
- trains the regressor on dusty samples only
- writes metrics, feature importance, and metadata

Output directory:

- [/home/ec2-user/Research/Codex/modis_db_two_stage_xgb](/home/ec2-user/Research/Codex/modis_db_two_stage_xgb)

Important files:

- [two_stage_dust_classifier.json](/home/ec2-user/Research/Codex/modis_db_two_stage_xgb/two_stage_dust_classifier.json)
- [two_stage_dust_fraction_regressor.json](/home/ec2-user/Research/Codex/modis_db_two_stage_xgb/two_stage_dust_fraction_regressor.json)
- [two_stage_metadata.json](/home/ec2-user/Research/Codex/modis_db_two_stage_xgb/two_stage_metadata.json)
- [two_stage_metrics.json](/home/ec2-user/Research/Codex/modis_db_two_stage_xgb/two_stage_metrics.json)

### Granule application

- [apply_two_stage_dust_model_to_modis_db.py](/home/ec2-user/Research/Codex/apply_two_stage_dust_model_to_modis_db.py)

This script:

- reads a local MODIS Level-2 granule
- applies the classifier and regressor
- computes:
  - dust probability
  - dust fraction
  - dust AOD at `550 nm`
- compares the result against Li-Ginoux
- saves maps, difference maps, and a scatter comparison

### Parameter tuning

- [tune_two_stage_dust_model.py](/home/ec2-user/Research/Codex/tune_two_stage_dust_model.py)

This script was written to tune:

- the dust-label threshold for the classifier target
- the probability cutoff used during inference

It evaluates candidate settings using:

- held-out test metrics
- the heavy-dust granule on `2025-03-14`
- the heavy-smoke granule on `2024-09-22`

Outputs:

- [/home/ec2-user/Research/Codex/two_stage_tuning](/home/ec2-user/Research/Codex/two_stage_tuning)
- [full sweep table](/home/ec2-user/Research/Codex/two_stage_tuning/two_stage_parameter_sweep.csv)
- [top 5 settings](/home/ec2-user/Research/Codex/two_stage_tuning/two_stage_parameter_sweep_top5.csv)
- [best model bundle](/home/ec2-user/Research/Codex/two_stage_tuning/best_model)

## How We Tuned It

### What was tuned

The tested dust-label thresholds were:

- `0.1`
- `0.2`
- `0.3`
- `0.4`

The tested classifier probability thresholds were:

- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`

### Why those parameters matter

#### Dust-label threshold

This changes the definition of what counts as dust in the training labels.

Lower threshold:

- more pixels are labeled dusty
- classifier is more permissive
- easier to detect weak/ambiguous dust
- higher risk of confusing smoke or mixed aerosol with dust

Higher threshold:

- only clearer dust cases are labeled dusty
- cleaner positive class
- more conservative detection
- risk of missing moderate dust

#### Probability threshold

This controls how cautious inference is.

Lower probability cutoff:

- more pixels pass through to the regressor
- more dust retained in real dust cases
- more false dust in smoke/non-dust scenes

Higher probability cutoff:

- stronger smoke rejection
- cleaner dust maps
- risk of suppressing moderate dust

## How We Chose the Best Setting

The tuning was not based only on held-out regression metrics.

That is important because the real scientific goal here was:

- preserve a heavy dust case
- strongly suppress a heavy smoke case

So the tuning script used a case-aware score:

- maximize dust/smoke separation
- while retaining a reasonable fraction of the heavy-dust mean DAOD

In practice, that meant optimizing a balance between:

- `dust_case_daod_mean`
- `smoke_case_daod_mean`
- their ratio
- dust retention relative to Li-Ginoux in the dust case

## Best Setting Found

The recommended setting from the sweep was:

- dust label threshold: `0.2`
- probability threshold: `0.7`

Saved summary:

- [best_selection_summary.json](/home/ec2-user/Research/Codex/two_stage_tuning/best_model/best_selection_summary.json)

Main numbers for that setting:

- classifier ROC-AUC: `0.7267`
- precision: `0.7791`
- recall: `0.2648`
- F1: `0.3953`

Case-based performance:

- heavy dust case mean DAOD: `0.0373`
- Li-Ginoux dust-case mean DAOD: `0.0473`
- dust retention relative to Li-Ginoux: `0.7876`

- heavy smoke case mean DAOD: `0.000734`
- Li-Ginoux smoke-case mean DAOD: `0.04475`
- smoke fraction relative to Li-Ginoux: `0.0164`

- dust/smoke mean ratio: `44.69`

This is why that setting was chosen:

- it keeps most of the dust signal in the heavy dust case
- it suppresses the smoke case very strongly

## What the Method Did on the Two Test Granules

### 1. Heavy dust case: 2025-03-14

Tuned output directory:

- [/home/ec2-user/Research/Codex/dust_model_application_2025-03-14_two_stage_tuned](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14_two_stage_tuned)

Useful files:

- [two-stage vs Li-Ginoux panel](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14_two_stage_tuned/MOD04_L2.A2025073.1705.061.2025074013638_two_stage_vs_li_ginoux_panel.png)
- [difference map](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14_two_stage_tuned/MOD04_L2.A2025073.1705.061.2025074013638_two_stage_minus_li_ginoux.png)
- [summary JSON](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14_two_stage_tuned/MOD04_L2.A2025073.1705.061.2025074013638_two_stage_summary.json)

Behavior:

- two-stage mean dust AOD: `0.0373`
- Li-Ginoux mean dust AOD: `0.0473`

Interpretation:

- the two-stage method kept a strong dust signal
- it is somewhat more conservative than Li-Ginoux
- the spatial correspondence remained high

### 2. Heavy smoke case: 2024-09-22

Tuned output directory:

- [/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon_two_stage_tuned](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon_two_stage_tuned)

Useful files:

- [two-stage vs Li-Ginoux panel](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon_two_stage_tuned/MOD04_L2.A2024266.1345.061.2024268021127_two_stage_vs_li_ginoux_panel.png)
- [difference map](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon_two_stage_tuned/MOD04_L2.A2024266.1345.061.2024268021127_two_stage_minus_li_ginoux.png)
- [summary JSON](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon_two_stage_tuned/MOD04_L2.A2024266.1345.061.2024268021127_two_stage_summary.json)

Behavior:

- two-stage mean dust AOD: `0.000734`
- Li-Ginoux mean dust AOD: `0.04475`

Interpretation:

- the two-stage detection stage strongly suppressed dust in the smoke scene
- this is exactly the kind of behavior the two-stage design was intended to
  encourage

## Strengths of the Two-Stage Design

1. It is physically cleaner than direct dust-AOD regression.
   - dust AOD is derived from dust fraction times total AOD
   - dust AOD cannot exceed total AOD

2. It separates detection from quantification.
   - this is more consistent with aerosol typing logic

3. It appears to help with smoke rejection.
   - this was the main practical success of the first implementation

4. It gives a clear place to tune conservativeness.
   - through the dust-label threshold
   - and the classifier probability cutoff

## Current Weaknesses and Caveats

1. Held-out regression skill is still weak.
   - the two-stage design improved smoke-vs-dust behavior in the case studies
   - but the overall DAOD regression metrics are still not strong

2. The classifier and regressor use the same limited MODIS DB feature set.
   - if the features do not cleanly separate smoke and dust everywhere, the
     model will still struggle

3. The tuning used only two case studies.
   - one heavy dust case
   - one heavy smoke case
   - this is useful, but still limited

4. The current label definition is based on `dust_fraction_675`.
   - future work may want to test other label definitions
   - for example, combining dust fraction with an SSA condition

## Recommended Next Steps

1. Test alternative dust-label definitions.
   - for example:
     - `dust_fraction_675 >= 0.3`
     - `dust_AOD_550 >= threshold`
     - combined dust-fraction and SSA logic

2. Compare models with and without SSA features.
   - SSA improves typing information
   - but reduces spatial coverage

3. Evaluate more case studies.
   - Saharan transport
   - Arabian dust
   - biomass-burning smoke
   - mixed aerosol scenes

4. Consider a soft version of the final estimate.
   - e.g. `P(dust) * predicted_fraction * AOD550`
   - this may reduce hard threshold artifacts

5. Add region-aware or season-aware features if available.

## Bottom Line

The two-stage method was introduced because a single regression model is not a
good representation of the dust-retrieval problem.

It separates:

- `Is this dust-like?`

from:

- `How much dust is there?`

In the first implementation, this did not solve every regression problem, but
it did produce the behavior we specifically wanted for the smoke-vs-dust test:

- retain dust in the March 14, 2025 heavy dust case
- strongly suppress false dust in the September 22, 2024 Amazon smoke case

That makes the two-stage framework a useful direction for further development,
especially if the next goal is robust aerosol-type discrimination rather than
only global regression skill.
