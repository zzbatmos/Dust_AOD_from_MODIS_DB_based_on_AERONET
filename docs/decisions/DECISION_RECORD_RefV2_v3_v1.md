# Decision record: AERONET dust reference v2 (C5) and estimator design v3

Frozen 22 November 2026 by the lead author (Zhibo Zhang); written by Claude.
File: `01_instructions/DECISION_RECORD_RefV2_v3_v1.md`. Commit this file unchanged to the new Git
branch (see `CODEX_TASK_FinalModel_v3_v1.md`, Section 1). Any later change needs a new, dated
version of this record, with the reason.

## 1. Reference: C5

**Definition.** At 870 nm, for each AERONET Version 3 Level 2.0 almucantar inversion, with retrieved
particle linear depolarization ratio d, retrieved lidar ratio S_p and AOD tau:

1. Dust backscatter fraction:
   `b = (d - d_nd)(1 + d_d) / ((d_d - d_nd)(1 + d))`, limited to [0, 1].
2. Implied non-dust lidar ratio:
   `S_nd* = (S_p - b S_d) / (1 - b)` for b < 1, limited to [LR05, LR95].
3. Dust extinction fraction at 870 nm:
   `f870 = b S_d / (b S_d + (1 - b) S_nd*)`; `f870 = 1` if b = 1.
4. Transfer to 550 nm:
   `DAOD550 = f870 tau870 (870/550)^a_d`; `f550 = min(1, DAOD550 / tau550)`, where tau550 is
   interpolated log-log between 440 and 675 nm.
5. Uncertainty: 200 Monte Carlo draws per record. They combine the site-bootstrap draws of the
   constants with measurement noise (sigma_tau = 0.01; sigma_d and sigma_S by level and AOD bin, as
   in R1 item A1.6). Store the median and the 16th and 84th percentiles.

**Constants (870 nm).** Global, with no regional values (0 of 64 regional sets passed the rule).
Measured at the reserved dust lidar-ratio sites from the strict PLDR-free selections:

| Quantity | Value |
|---|---|
| d_d (dust PLDR) | 0.296755 |
| d_nd (non-dust PLDR) | 0.0104275 |
| S_d (dust lidar ratio) | 44.4485 sr |
| LR05, LR95 (clean non-dust lidar ratio, 5th and 95th percentiles) | 28.9736 sr, 51.34145 sr |
| a_d (dust Angstrom exponent) | 0.094088 |

The full-precision values are those of `17_refv2_R1/C/SHORTLIST_v1.json`.

**How it was chosen.**
- C5 ranked first among the eligible candidates under the R1 shortlist rule, which was fixed in
  advance (shortlist hash beginning 132e55ea).
- R2a could not confirm the scale with independent lidar. Under the fixed rule of R2a Section 6, the
  R1 ranking therefore stands (`REFERENCE_DECISION_v1.json`, hash beginning 26db05e4).
- R1b and R1c found no admissible correction (`R1B_DECISIONS_v1.json`, `R1C_DECISIONS_v2.json`).

**Use.**
- Level 2.0 records only, for training and AERONET evaluation.
- Level 1.5 values (AOD440 0.1 to 0.4) remain provisional. They are not used for training.

**Alternative and secondary references.**
- C0 (the frozen D1; estimator v2.0.0) is kept as the alternative, so the dependence of results on
  the dust definition can be shown.
- C6 (the mean of the 870 and 1020 nm F3 forms) stays the declared secondary. It is not run unless
  the lead author asks.

## 2. Estimator design v3

- **Unchanged from v2.0.0:**
  - features, branches (full8, else noSSA5) and hyperparameters;
  - seeds, inner calibration and the two-part and direct forms;
  - QA tiers (0.4, 0.6, 0.7 on the raw probability), the QA2 and EV definitions, and the gridded
    operators.
- **Changed: the target only.**
  - Detection label: C5 f550 >= 0.2.
  - Amount target: C5 f550, using the Monte Carlo median.
- **Not adopted** (R3a gate, fixed in advance):
  - soft labels (S);
  - month inputs (M); kept as a candidate for a separate confirmation test only;
  - Level 1.5 clean-day records (L).
- **Private version label:** `3.0.0-rc1`. The public label is decided by the authors before any
  release.

## 3. Known limitations (stated, not corrected)

1. No independent confirmation of the reference scale:
   - ACTRIS particle depolarization is not quality assured by EARLINET;
   - ATLID gave no Level 2 case before its effort cap.
2. Some selections depolarize under every reference, C0 included, without recent transport from the
   13 desert boxes:
   - remote marine, aged smoke, and spring coarse continental aerosol;
   - whether this is local or regional dust or non-dust is undetermined.
3. Clean days (AOD440 < 0.4) have no Level 2 reference.
4. AERONET's radius limit (about 15 micrometres) may undercount super-coarse dust near sources.
5. The margin of C5 over C7 in the shortlist ranking is small (low-AOD gap 0.037 against 0.038).
   C7 has the lower uncertainty (0.14 against 0.25).
6. At dust-source sites the estimator reaches 0.87 [0.76, 0.96] of its own reference (R3a,
   out of fold).

## 4. Not decided by this record

- The external evaluation. It is fixed separately in `EVAL_PROTOCOL_v3_v1.md`, before any result of
  the final model exists.
- The public version label, any release, and any manuscript number.

## 5. Public-repository rule

The Git repository is public.
- **The commit contains:**
  - this record;
  - the protocol;
  - the training configuration and code changes;
  - the SHA-256 of the model artifacts and of the result files named above.
- **The commit does not contain:**
  - result tables;
  - record-level data;
  - model binaries;
  - credentials.
