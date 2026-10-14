# DBDust v2.0.0 product data format

The product definitions are frozen by `FREEZE_V2_SPEC.md` and
`configs/dbdust_v2.0.0.yaml`. The configuration is the source of scientific
constants, grid definitions, branch codes, thresholds, and variable lists.
Writers package existing predictor outputs: they do not run inference, fit
models, or multiply fractions by AOD again.

## Pixel product (L2)

`dbdust.product_l2.write_l2(frame, path, provenance=None, **provenance_fields)`
writes NetCDF4 to the exact caller-supplied path. `write_pixels` and `write`
are aliases. The parent directory must already exist. Publication is atomic:
an unsuccessful write removes its temporary file and leaves any existing
destination untouched.

The input is a pandas DataFrame with scalar numeric columns `lat`, `lon`, and
all eight pixel output columns below. Extra columns and the DataFrame index
are ignored. Nullable numeric columns are accepted. Coordinates are floating
point degrees, not arrays within cells. Each row remains one output pixel;
there is no row removal or reconstruction of a two-dimensional MODIS swath.
Empty frames with the required columns produce an empty pixel dimension.

All variables have shape `(pixel,)`. `pixel` is an unlimited dimension;
`lat(pixel)` and `lon(pixel)` are auxiliary coordinates, with CF standard names
`latitude` and `longitude` and units `degrees_north` and `degrees_east`.
Continuous product values are float64. Flags are int8. All eight product
variables have units `1`.

| Variable | Definition |
| --- | --- |
| `branch` | Input-validity branch: 0 none, 1 `full8`, 2 `noSSA5`. |
| `dust_probability` | Raw classifier probability P, not the isotonic-calibrated probability. |
| `qa_tier` | 0 for P < 0.4; 1 for 0.4 <= P < 0.6; 2 for 0.6 <= P < 0.7; 3 for P >= 0.7. |
| `dust_fraction_conditional` | Unsmeared positive-regressor dust fraction. |
| `daod550_qa2` | Conditional fraction times DB AOD550 when P >= 0.6; zero for a supported retrieval below that gate. |
| `daod550_ev` | Selected expected-value fraction times DB AOD550: E3c for `full8`, E4 for `noSSA5`. |
| `aod550_db` | Echo of the input DB AOD550. |
| `db_aerosol_type` | Echo of decoded DB type: 0 mixed, 1 dust, 2 smoke, 3 sulfate. |

Flag variables include `flag_values` and `flag_meanings`. Optical-depth
variables carry `wavelength_nm`; probability and conditional fraction have
`valid_range = [0, 1]`.

### Screening and missing values

The writer uses configured finite, nonnegative AOD and valid geolocation:
latitude in [-90, 90), longitude in [-180, 180], without an additional DB
confidence screen. Invalid AOD or geolocation makes every product output
missing, even if stale predictions were supplied. Invalid coordinate values
are also written as missing.

Valid branch-0 pixels retain AOD and aerosol type but have missing probability,
QA tier, conditional fraction, and both DAOD fields. They are not zero-dust
retrievals. A supported pixel rejected by the QA2 gate has a real zero in
`daod550_qa2`. Missing aerosol type alone does not remove ML support.

The configured float fill is NaN. Missing flag values use `_FillValue = -1`,
including invalid-AOD branch flags and missing QA tiers. A normal CF-decoding
reader such as xarray masks these flag sentinels to NaN; raw netCDF4 flags are
int8. Branch 0 remains a valid flag, never a fill value.

Supported rows must have finite, physically bounded predictions and a QA tier
consistent with raw probability. Unknown valid-AOD branch flags, invalid type
flags, or nonzero DAOD for supported QA2 gate rejections raise `ValueError`.
This is input-contract validation, not an additional scientific support mask.
The writer does not repair or recompute predictions. Missing fields on a
supported branch are errors rather than silently reducing its support.

## Monthly product (L3M)

`dbdust.product_l3.MonthlyAccumulator(grid_name='1x1')` accepts the same pixel
frames through repeated `update(frame)` calls. The caller selects the month:
the accumulator has no retrieval-time column and does not sort or deduplicate
observations. Each supplied row contributes once. Use a new accumulator for
each month and grid.

The supported grids are:

| Grid | `(lat, lon)` size | Latitude centers | Longitude centers |
| --- | --- | --- | --- |
| `1x1` | `(180, 360)` | -89.5 to 89.5, step 1 | -179.5 to 179.5, step 1 |
| `2x5` | `(90, 72)` | -89 to 89, step 2 | -177.5 to 177.5, step 5 |

Cell intervals are lower-inclusive and upper-exclusive. Longitude +180 wraps
to -180; latitude +90 is excluded by the frozen screening configuration.

Science and count variables have shape `(time, lat, lon)`, with one monthly
time coordinate. The weighting is arithmetic mean over supported retrievals,
not equal-day, equal-granule, area, or cosine-latitude weighting. Supported
retrievals are valid-AOD/geolocation rows assigned `full8` or `noSSA5` with
complete predictions. AOD is averaged over that same ML support, not over all
valid AOD rows.

| Variable | Definition |
| --- | --- |
| `daod550_ev` | Sum of supported expected-value DAOD / `n_ml`. |
| `daod550_qa2` | Sum of supported QA2 DAOD, including gate-rejection zeros / `n_ml`. |
| `aod550_db` | Sum of DB AOD on ML support / `n_ml`. |
| `n_valid_aod` | Count of screened valid-AOD/geolocation retrievals, including branch 0. |
| `n_ml` | Count of ML-supported retrievals. |
| `n_full8` | Count of supported `full8` retrievals. |
| `n_noSSA5` | Count of supported `noSSA5` retrievals. |
| `frac_full8` | `n_full8 / n_ml`. |
| `frac_ml_missing` | `(n_valid_aod - n_ml) / n_valid_aod`. |
| `dust_probability_mean` | Sum of supported raw classifier probabilities / `n_ml`. |
| `frac_qa1` | Count of supported P >= 0.4 / `n_ml`. |
| `frac_qa2` | Count of supported P >= 0.6 / `n_ml`. |
| `frac_qa3` | Count of supported P >= 0.7 / `n_ml`. |
| `ev_contribution_aod_lt_0p2` | Sum of supported EV DAOD where AOD < 0.2 / full `n_ml`. |
| `ev_contribution_p_lt_0p3` | Sum of supported EV DAOD where raw P < 0.3 / full `n_ml`. |

The last two fields are contributions to the full supported mean, not means
over the low-loading subsets. No matching subset gives a true zero if the
cell has ML support.

All fields have units `1`. Count variables in dataset snapshots and NetCDF
files are int32 for CF 1.8 and retain real zeros without a fill sentinel.
Internal accumulator counts remain int64. Before narrowing a snapshot, every
count variable is checked against `np.iinfo(np.int32)`; an out-of-range count
raises `OverflowError` rather than silently wrapping. This also prevents a
failed write from replacing an existing output. Other fields are float64
with a NaN `_FillValue`. Every zero
denominator yields a missing statistic. Thus an empty cell has zero counts
and missing statistics; a valid-AOD-only cell has positive `n_valid_aod`,
zero `n_ml`, missing ML means/fractions, and `frac_ml_missing = 1`.
The identities `n_ml = n_full8 + n_noSSA5` and `n_ml <= n_valid_aod` hold.

### Coordinates, calendars, and metadata

Files declare `Conventions = "CF-1.8"`. Latitude and longitude have axis
attributes `Y` and `X`, respectively, and boundary variables
`lat_bounds(lat, bounds)` and `lon_bounds(lon, bounds)`. `bounds` has size 2.

`time(time)` is the interval midpoint, not an observation timestamp.
`time_bounds(time, bounds)` contains month start and next month start, defining
the half-open monthly interval. Both encode as float64 days since month start
under `proleptic_gregorian`, including leap years. Xarray may omit redundant
`units` and `calendar` attributes on `time_bounds`; those are inherited from
the parent time coordinate under CF bounds conventions. Decoding restores
both as datetime values. Coordinate and bounds variables have no fill value.

Global attributes include version, reference name and serialized definition,
branch rule, screening configuration, missing conventions, product level,
`source_code_commit`, and `model_manifest_hash`. L3M also includes grid name,
weighting, aggregation description, and time coverage. Provenance may be a
mapping or keyword arguments. `commit` is also recorded as
`source_code_commit`; omitted commit/hash values are explicitly `unknown`.
Release generation should always supply both. Caller provenance cannot
override the frozen scientific definition attributes.

Both L2 and L3M include nonempty, deterministic global `title` and `history`.
The title identifies the versioned DBDust product; history records packaging
from predictor outputs, the source commit, and the model manifest hash. No
wall-clock timestamp is inserted, so identical provenance produces identical
history for golden comparisons. These generated fields cannot be replaced
by caller-supplied title/history values.

Non-count fields identify their denominator and carry a retrieval-count
`cell_methods` description. Subset and QA fractions carry their configured
threshold metadata. Baselines and sensitivity estimates are not product
variables.

## APIs and memory bounds

```python
from dbdust.product_l2 import write_l2
from dbdust.product_l3 import MonthlyAccumulator, monthly_filename

provenance = {
    "commit": source_commit,
    "model_manifest_hash": predictor.manifest_sha256,
}
acc = MonthlyAccumulator(grid_name="1x1")
for inputs in granules_for_one_month:
    frame = predictor.predict_frame(inputs)  # inputs include scalar lat/lon
    acc.update(frame)
    # Optional case-study output, choosing a distinct path per granule:
    # write_l2(frame, pixel_path, provenance)

path = output_directory / monthly_filename("2017-01", "1x1")
acc.write(path, "2017-01", provenance)
```

`update` returns the accumulator. `to_dataset(month, provenance=None,
**provenance_fields)` returns an independent xarray snapshot;
`write(path, month, provenance=None, **provenance_fields)` writes atomically
and returns a `Path`. Month accepts `YYYYMM`, `YYYY-MM`, or a datetime-like
value. The suggested filename is
`DBDust_v2.0.0_Aqua_L3M_{1x1|2x5}_YYYYMM.nc`; `write` never changes the
supplied path or infers a different grid from its name.

The accumulator retains only fixed-size int64 counts and float64 sums. Its
state is O(grid cells), with scratch space O(grid cells + configured batch
size), independent of the number of granules. It never retains input frames.
L2 writing similarly processes `runtime.batch_size` rows at a time; the
caller's existing input frame is outside this working-memory bound.
`dbdust.product_l2.to_dataset(frame, provenance=None, **provenance_fields)`
is an optional in-memory convenience, not the bounded-memory writing path.

Validation does not mutate input frames. An L3 update that raises after
earlier chunks may have partially accumulated that frame: discard that
accumulator and restart the month after correcting the input. Writers clean
up incomplete output files, including failures in later L2 chunks.

### Example header excerpt

The following illustrates `ncdump -h` structure for January on `1x1`; repeated
variable attributes and global provenance are omitted for brevity.

```text
dimensions:
    time = 1 ;
    lat = 180 ;
    lon = 360 ;
    bounds = 2 ;
variables:
    double lat(lat) ;
        lat:standard_name = "latitude" ;
        lat:units = "degrees_north" ;
        lat:bounds = "lat_bounds" ;
    double lon(lon) ;
        lon:standard_name = "longitude" ;
        lon:units = "degrees_east" ;
        lon:bounds = "lon_bounds" ;
    double time(time) ;
        time:standard_name = "time" ;
        time:bounds = "time_bounds" ;
        time:units = "days since 2017-01-01" ;
        time:calendar = "proleptic_gregorian" ;
    double time_bounds(time, bounds) ;
    double lat_bounds(lat, bounds) ;
    double lon_bounds(lon, bounds) ;
    double daod550_ev(time, lat, lon) ;
        daod550_ev:_FillValue = NaN ;
        daod550_ev:units = "1" ;
        daod550_ev:denominator = "n_ml" ;
        daod550_ev:cell_methods = "time: mean (weighted by retrieval count)" ;
    int n_ml(time, lat, lon) ;
        n_ml:units = "1" ;
```

Synthetic tests in `tests/test_products.py` verify sums/counts, both grids,
threshold boundaries, invalid input handling, missing-versus-zero behavior,
NetCDF round trips, time bounds and deterministic metadata, guarded int32
count encoding with int64 internal state, atomic writes, and chunk/state
bounds. They do not substitute for an independent CF checker or the frozen
deployment/golden-case reproduction checks.
