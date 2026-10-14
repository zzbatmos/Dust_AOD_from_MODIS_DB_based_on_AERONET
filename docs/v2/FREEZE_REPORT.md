# DBDust 2.0.0 freeze report

## State

Local implementation and verification complete on `release/v2.0.0`, based on spec commit
`41a791e6a91f2d55f9e58e6bee848ba9f1647359`. This is not approval to publish.
No model refitting, calibration, estimator selection or thresholds changed.
Existing v1 scripts, experimental artifacts and the dirty original worktree
were preserved. Quantitative unpublished results stay in private author storage.

## Input consistency

The training and application use the all-quality AOD SDS and no additional
confidence cutoff. Every eligible cached training row passes the application
AOD/geolocation screen. Legacy/application reader masks and values agree on
three retained case granules. Detailed row counts, hashes and tests are in the
private `input_consistency_audit.json` delivered to the authors.

This is not a reconstruction of every historical training raw granule: those
files were not retained, and the original pickle did not pin its reader hash.
Collocation independently averages available fields inside a site neighborhood;
the pixel product requires a co-valid feature vector. Document that support
difference rather than claiming identical joint sampling.

## Resolved specification discrepancies

1. Reference code clips the backscatter fraction before multiplying by the
   lidar-ratio ratio. Retain this validated operation; the initial spec omitted it.
2. Feature eligibility requires physical AOD/SSA ranges, not only finiteness.
3. The actual validated environment is Linux x86_64 on EC2, not macOS arm64.
4. The frozen comparison baseline E6 uses an exact FMF cutoff and SSA ordering;
   a rounded AE cutoff is not substituted. Baselines are not product variables.
5. Experimental implementation files were not all committed. The preregistration
   commit `e40f71a` anchors the experiment, not every later source byte. Use the
   existing source/artifact hash manifests; do not claim that commit contains
   the complete experimental implementation.

## Reproduction checks (13 October 2026, US Eastern)

- Complete configured test suite after independent review: 238 passed, no skips. Includes target
  reconstruction, all cached collocation inputs, three raw case readers and
  golden files, product semantics and scientific-constant checks.
- Fold-specific inference and metrics: 81 primary fold/population combinations,
  with original classifiers and postprocessors, not deployment fits scored on
  held-out data. Saved metric and bootstrap recipes reproduce at their original
  precision. A test ordering error affecting a float32 Brier reduction was
  corrected by retaining the original event order; no tolerance was relaxed.
- Fixed raw samples: 300 granules from each of January and July 2017. Every raw
  checksum matches the original monthly ledger. Pixel predictions and counts
  agree exactly; the largest grid-sum absolute difference is below 3e-12.
  Both grids and the low-loading contribution sums were checked. Samples compare
  paired implementations, not scaled estimates of complete-month sums.
- IOOS compliance-checker 6.1.0, CF-1.8 strict: synthetic and raw-sample monthly
  output passed. Internal counts remain int64; CF output uses int32 with an
  explicit overflow rejection. Deterministic title/history metadata is included.
- Wheel builds and loads its configuration and hash-verified models in an
  isolated environment outside the source directory.
- Independent review added fail-closed validation: acceptance evidence must
  match the executing package/configuration/converter; source grid coordinates
  and dimension order must match before remapping; the standalone verifier
  rejects nonfinite supported values, malformed counts, invalid probabilities,
  fraction/support inconsistencies and incorrect QA ordering. No inference,
  target or aggregation formula changed.

The sample manifest ranks original ledger filenames by SHA-256 of
`DBDust-v2-freeze-sample-20261014|` plus filename and takes the first 300 per
month. This reduces redundant raw downloads while exceeding one million
supported pixel checks. Derived verification caches are private. Temporary
operational HDFs were deleted. The complete product conversion separately
checks every original monthly file's hash, fingerprint and support counts.

Primary model-manifest SHA-256:
`70bb03adcf5d5ab8bb79446895d4d49ceb331500a10e41c74a869f936a6b109c`.
Primary/sensitivity binary hashes, sanitized source metadata hashes and fit
configuration are retained under `models/`. Membership inventories, raw
collocations and unpublished performance tables remain private for archive
review; the freeze does not authorize their public redistribution.

All 26 product files (December 2016 through December 2017, both grids) were
repackaged using code commit `2bb442ab12bd9db380589e6c78919efd83dc83be`.
The earlier candidate is retained privately; every scientific array is identical
between the two candidates. Only provenance metadata changed. The final configured
suite passed all 238 tests without skips, and installed-wheel verification passed.
Every mapped field agrees exactly with the original sufficient statistics;
every product passes value/count/missing invariants. The actual monthly
product passed strict CF-1.8 (524/524). All source grids have equal valid-AOD
and ML support; conversion explicitly verifies this before reusing AOD sums.
Source coordinate variables are read from the parent NetCDF group, not
subgroup dimension indexes. Product manifests and verification reports stay
in private author storage rather than adding unpublished datasets to Git.

## Handoff status (14 October 2026, US Eastern)

Release-candidate [pull request #1](https://github.com/zzbatmos/Dust_AOD_from_MODIS_DB_based_on_AERONET/pull/1)
is open. It is not merged, tagged or published. Subsequent reporting work on
`paper/resubmission-v2` does not change the verified product-generating commit
listed above. Numbers and figure delivery are tracked in private author storage.

Remaining gates are final author review and archive preparation. A linked
Zenodo draft and DOI reservation are authorized if authenticated editing access
is available; neither has been completed. This is not publication approval.
Archive preparation must resolve exact third-party data scope, attribution,
component licenses and author metadata. Do not merge, tag, create a release
or publish the archive without explicit author approval.
