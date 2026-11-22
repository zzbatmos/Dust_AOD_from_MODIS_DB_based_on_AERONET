# Amendment 1 to `EVAL_PROTOCOL_v3_v1.md`: Song splits computed from L2IN

Issued 22 November 2026 by Claude for the lead author (Zhibo Zhang), before any P3(a) or P5 result
exists. Commit this file to the branch `refv2/v3.0.0-rc1` and push it before computing these
items.

**Reason.**
- The accepted Song reconstruction uses full-swath footprints.
- L2IN keeps only valid-pixel locations.
- So the reconstruction cannot be reproduced from L2IN as Section 0, item 3, assumed.

**Replacement for the Song splits of P3(a) and P5 only.**

1. **Song-L2IN.**
   - For every L2IN retrieval with DB confidence flag 3, compute the Song relation fraction
     `0.98 - 0.5089 a + 0.0512 a^2` on the DB AE, limited to [0, 1].
   - Apply the SSA screen: fraction 0 where DB SSA470 >= 0.99, as in the reconstruction.
   - Multiply by the DB AOD at 550 nm.
   - Grid with the frozen retrieval-count-weighted operator on the 2 x 5 degree grid, then the P3
     operator.

2. **Acceptance against the archived Song product (fixed now).** Region by region, Song-L2IN is used
   only if both hold:
   - (i) its 2007 to 2019 seasonal regional means are within 10% of the archived product in every
     season;
   - (ii) the monthly cell correlation with the archived product is at least 0.8.

   The regions tested are Land 60S-60N, East China, EAS-5 and the P2 and P3 regions. Where a region
   fails, its split is marked unavailable and its acceptance statistics are reported.

3. **The splits are shares within Song-L2IN.** Report them next to the archived-Song totals:
   - P3(a): the share of Song-L2IN DAOD from DB dust-typed retrievals;
   - P5: the shares and trends by DB AE regime.
   - For P5, report the total trend of Song-L2IN beside the archived Song trend. The P5 claim
     criterion is applied to Song-L2IN only if its total trend has the same sign as the archived trend
     and lies within its interval.

All other items, criteria and the archived-Song comparisons are unchanged.
