# Manuscript reporting

These scripts read existing, private frozen results. They do not train models,
run inference, change the release configuration or edit the manuscript. Numerical
outputs must be written outside this public code worktree.

```bash
python -m paper.numbers --source-root /path/to/revision_v2_work \
    --output-dir /private/resubmission/06_numbers \
    --claims /private/claim_requests.json
python -m paper.tables \
    --registry /private/resubmission/06_numbers/paper_numbers.json \
    --output-dir /private/resubmission/04_figures/tables
python -m paper.figures.fig02_learning_space \
    --registry /private/resubmission/06_numbers/paper_numbers.json \
    --output-dir /private/resubmission/04_figures/main
```

Install `paper/requirements.txt` in an isolated plotting environment. Set
`MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable scratch directories on servers.
Inference environment files remain unchanged.

## Numbers contract

The registry preserves the exact numeric rows and column names of the original
analysis tables, with source SHA-256, row count, selection dimensions and statistic
semantics. `Registry.number(table, metric, **filters)` requires an unambiguous
selection and returns a stable key, value, interval (when available), sample
counts, units, reference, scheme, support and exact source filter. Intervals from
the grouped experiment are joined by all original identity columns, not position.
Missing intervals remain missing. Source hashes are checked before/after ingestion.

Figure and table scripts read numerical results exclusively from this registry;
their exports retain the scalar keys and source evidence. Claim checks compare
draft numbers at their stated precision, not with an arbitrary tolerance. A
registry with unresolved claims is explicitly preliminary, not a completed audit.

Original grouped evaluation pools repeated prediction events. Expected-value
evaluation averages repeat-specific metrics. Counts distinguish unique records,
site-months, sites and events. These are not interchangeable definitions.

The screened Li-Ginoux baseline has a smaller support; its global statistics
must be shown separately with its own CALIOP reference mean and count. The
DB aerosol-type baseline is a fitted type-conditional mean, not the binary dust
flag. The full-year map of branch availability and the sampled coverage audit
are also different populations and must not be described as the same census.

## Export and review

Figures are drawn at their final physical size with embedded TrueType fonts,
vector PDF, 300-dpi PNG previews, and 600-dpi RGB LZW TIFF for map figures.
Exports include plotted values, source hashes and script/style hashes. Generated
color-vision and grayscale previews still require visual review; their existence
alone does not constitute a passed accessibility check. Captions and private
figure-QA records accompany the deliverables.

`python -m paper.review_bundle --registry /private/paper_numbers.json
--output-dir /private/review_batch` renders the currently implemented subset,
compiles the tables, collects captions and assembles final-size contact PDFs.
It explicitly lists missing main figures and rejects mixed registry hashes.
Keep shared review versions immutable; render later changes into a new batch.

The `derive_*` modules extract plotting arrays from existing OOF or gridded
caches, validate them against saved summaries, and write CSV/provenance pairs.
Include these with `paper.numbers --derived-dir PATH --derive-claims` before
plotting. Unique collocations, repeated prediction events, site-months, and
area-weighted seasonal cells remain separate units of analysis.

`paper.fetch_case_rgb` downloads NASA GIBS context imagery for the prescribed
platform/date/box. Daily mosaics are not guaranteed single-granule imagery and
are never retrieval inputs. `paper.prepare_archive` creates a private candidate
and records public predecessor metadata; it neither creates a remote deposit
nor publishes, reserves a DOI, tags a release, or uploads collocation data.
