# Changelog

## 2.0.0 - unreleased

- Replaces the primary 675-nm/56-sr reference with a reserved-site-calibrated
  1020-nm reference and explicit transfer to 550 nm.
- Adds the full8/noSSA5 hierarchy, selected only from input availability.
- Adds expected-value DAOD for averaging, separate from raw-probability QA fields.
- Preserves missing ML outputs as missing rather than treating them as dust-free.
- Retains QA probability boundaries in the single versioned configuration.
- Documents site-grouped and leave-region-out evaluation instead of a random split.
- Packages existing models without refitting; does not include veto or rescue.
- Uses retrieval-count-weighted monthly products, not equal-day averaging.

## 1.0.0

Submitted-method code and tagged history remain unchanged.
