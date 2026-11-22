# Amendment 1 to `CODEX_TASK_FinalModel_v3_v1.md`

Issued 22 November 2026 by Claude for the lead author (Zhibo Zhang). It answers the two questions of
`FINALV3_STAGE2_OPENING_QUESTION_20261122_v1`, before any v3 model or output exists.

1. **Configuration selector: approved.**
   - Apply `CONFIG_SELECTION_PROPOSAL_v2.patch` on the branch `refv2/v3.0.0-rc1` only, never on
     `release/v2.0.0`. It adds an explicit `DBDUST_CONFIG` selector in `dbdust/config.py` and keeps
     `configs/dbdust_v2.0.0.yaml` as the default.
   - Conditions, checked before any training:
     - (a) the predictor, the MODIS reader and the manifest verification are unchanged;
     - (b) with no selector set, the frozen test suite passes as at the freeze, and the golden
       granule outputs are bit-identical to the stored golden case;
     - (c) the v3 configuration is a new file, `configs/dbdust_v3.0.0-rc1.yaml`, with its SHA-256
       recorded.
   - Commit the patch, the new configuration and the check results to the branch, and push.

2. **Target: the C5 point value.**
   - `DECISION_RECORD_RefV2_v3_v2.md` replaces version 1. The final v3 models are trained on the
     point value `C5_f` (detection label `C5_f >= 0.2`; amount target `C5_f`), as in R3a.
   - The Monte Carlo percentiles remain the per-record uncertainty.
   - Opening check (ii) stays as written: the C5 out-of-fold predictions must reproduce R3a B/C5
     within 1e-6.
   - Commit version 2 of the decision record to `docs/decisions/` beside version 1 (keep version 1),
     and push, before Stage 2 training.

Everything else in the task is unchanged, including the protocol (SHA-256 `084bb635...`) and the
Stage 3 stop.
