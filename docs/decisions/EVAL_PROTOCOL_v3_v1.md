# External evaluation protocol for estimator v3 (reference C5)

Fixed 22 November 2026 by the lead author (Zhibo Zhang); written by Claude.
File: `01_instructions/EVAL_PROTOCOL_v3_v1.md`. Commit and hash this file **before any output of the
final v3 model exists**. A change after that needs a dated amendment with its reason. The report
must list every amendment as a deviation and also give the results under the original protocol.

## 0. Principles

1. **Descriptive, not tuning.** Nothing here changes the reference, the model or any threshold.
2. **Identical samples.** Every comparison uses the same cells, months or records for all products.
3. **Comparators applied as their authors used them.**
   - Song et al. (2021): the archived product (DB confidence flag 3; their SSA screen and
     averaging).
   - LG25: as defined in manuscript Text S4 ("screened LG25").
   - Where a split that the archived Song product lacks is needed (by DB type or by AE regime),
     recompute it from the record's L2IN layer with the verified Song reconstruction (flag 3, SSA
     screen). First show that it matches the archived product for the same cells, as in SongCompare
     A0.
4. **Report where the comparators do better.** For example, Song's year-to-year agreement with
   CALIOP.
5. **Claims only by the criteria of Section 4.** A result that misses a criterion is reported
   without the claim.
6. **Intervals.** 400 resamples; seed 20261122. Whole sites for AERONET quantities; whole years for
   gridded time series.

## 1. Products and operators

**Products:**
- v3 (EV and QA2) from the final model;
- v2.0.0 (EV and QA2) from the frozen record;
- Song (archived);
- LG25 (from L2IN);
- CALIOP Version 2 dust climatology (Song et al., 2021).

**Operators.** Those of `11_song_compare_v1`, unchanged:
- the P3 operator and common cells on the 2 x 5 degree grid;
- seasonal means, and ratios to CALIOP as the mean of four seasonal ratios;
- the frozen land mask and region boxes:
  - Land 60S-60N, Sahara, Sahel, Middle East, Central Asia, East Asian deserts, India, East China,
    SW North America, Australia;
  - Amazonia_Cerrado, southern_Africa, mainland_SE_Asia;
  - EAS-5 and NWP for the Song-style items.

**Period.** 2007 to 2019 (Song ends 2019). Annual series use the 11 complete CALIOP years (2008 to
2015, 2017 to 2019). 2007 and 2016 enter the seasonal climatology only where complete.

## 2. Evaluation items

**E1. Climatology, 2007 to 2019.**
- Seasonal and annual land and regional means for every product.
- Ratios to CALIOP with the range over complete years.
- The Table 4 analog, and seasonal maps and difference maps against CALIOP.

**E2. Year-to-year variability.**
- Annual anomaly correlation with CALIOP by region (n = 11), with p.
- Sensitivity: 2008 to 2014 (n = 7, before the 2015 to 2018 DB input shift).

**E3. Trends.**
- 2007 to 2019, in %/yr with p, by the SongCompare B3 method on fixed common cells.
- Sensitivity: 2007 to 2014.

**E4. AERONET (independent of CALIOP).**
- Site-month comparison, 2007 to 2019, of every product against the C5 reference and against C0, at
  held-out sites. Use out-of-fold predictions for the models.
- Against the SDA coarse AOD as in SongCompare C1. Statistics: R, bias, RMSE, N, by continent.

**M. Map year (fixed rule, CALIOP only, computed before any multi-year v3 result).**
- Candidates: 2008 to 2014. These years have complete CALIOP data, lie before the DB input shift
  and fall within Song's record.
- Score of a year: the sum over Land 60S-60N, Sahara, Sahel, Middle East, India, East China and the
  East Asian deserts of z^2. Here z is that year's CALIOP annual DAOD minus the 2007 to 2019 mean,
  divided by the standard deviation over the 11 complete years.
- The map year is the year with the smallest score; a tie goes to the earlier year.
- Maps of the paper use this year. The multi-year statistics of E1 to E4 remain primary.

## 3. Prior-products audit

**P1. Dust in non-dust aerosol (AERONET only).**
- Analytic values of the Song and LG25 dust fractions at AE 1.0, 1.3, 1.5, 1.8 and 2.0.
- On AERONET Level 2 records, for each region:
  - Populations:
    - the PLDR-free non-dust selections of R1b (CLEAN, SMK fresh smoke, URB);
    - the R1c trajectory dust-free CLEAN, URB and SMK;
    - all records with PLDR1020 < 0.02.
  - Quantities: the mean and 90th percentile of the dust fraction and the DAOD.
  - Relations: Song and LG25 applied to the ground AE (440 to 870 nm), and at collocations to the
    DB AE. Beside them, C5 and v3.

**P2. Smoke regions in absolute values.**
- Regions and fire seasons:
  - Amazonia_Cerrado, fire season August to October;
  - southern_Africa, July to September;
  - mainland_SE_Asia, February to April.
  - All other months form the non-fire season.
- Report the seasonal mean DAOD (absolute) of Song, LG25, v3, v2.0.0 and CALIOP; each product
  minus CALIOP; the fire-season minus non-fire-season difference of each product; and the DB
  dust-typed and smoke-typed shares.

**P3. Inheritance of the DB aerosol type.**
- (a) The share of Song and LG25 DAOD that comes from DB dust-typed retrievals, by region and
  season (from L2IN).
- (b) At collocations, 2007 to 2019: the share of DB dust-typed retrievals that C5 labels as dust
  (f550 >= 0.2), by region.
- (c) The type sensitivity at a fixed ground state, by the DiffTests T3 method, for v3, Song and
  LG25. Report the number of finite bootstrap draws.

**P4. India and East China: split by depolarization.**
- Population: AERONET Level 2 records at sites inside the India and East China boxes.
- Split each quantity by PLDR1020 class (< 0.05, 0.05 to 0.15, >= 0.15) and by season.
  - Quantities: the Song and LG25 DAOD on the ground AE, the C5 DAOD, and the SDA coarse AOD.
  - For each class, report the share of each quantity's total.
- Report the CALIOP against C5 and v3 difference in these regions as it is, with no claim.

**P5. East Asia trend split.**
- Regions and period: East China and EAS-5, 2007 to 2019.
- Song DAOD (recomputed from L2IN, flag 3, SSA screen):
  - split by DB AE regime (< 0.6, 0.6 to 1.0, >= 1.0);
  - the trend of each regime's contribution.
- AERONET sites in these boxes with at least 8 years of Level 2 data: trends of the C5 DAOD, SDA
  coarse AOD, SDA fine AOD, and the Song relation on the ground AE.

## 4. Claim criteria (fixed now)

| Item | A claim is made only if |
|---|---|
| P1 | The mean Song or LG25 dust fraction on a PLDR-free non-dust selection is at least 0.10, with the lower interval bound above 0.05 |
| P2 | In the fire season, Song / CALIOP >= 1.5, and Song minus CALIOP >= 0.02 with the interval excluding 0 |
| P3 | (c) The type sensitivity of Song or LG25 minus that of v3 is > 0, with the interval excluding 0 and at least 300 finite draws |
| P4 | At least 30% of the Song or LG25 DAOD comes from PLDR1020 < 0.05, with the lower interval bound at least 20% |
| P5 | All three hold: the AE >= 0.6 regimes carry at least 30% of Song's 2007 to 2019 trend; that regime trend has p < 0.05; and the AERONET fine-mode AOD declines with p < 0.05 at the sites |
| E1 to E4 | No claims of superiority. Report the statistics. Any statement of better or worse agreement needs a paired interval that excludes zero |
