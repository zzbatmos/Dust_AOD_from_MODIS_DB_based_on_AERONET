"""Derive private Fig. 4 arrays from cached OOF predictions, never from models.

Run ``python -m paper.derive_pixel_diagnostics --source-root REVISION_V2_WORK
--output-dir PRIVATE_DERIVED_DIR``. Outputs are pixel_pr_curves,
pixel_qa_operating_points, pixel_prevalences, pixel_baseline_points,
pixel_ev_scatter and pixel_overrides CSVs, each with a provenance JSON sidecar.
Existing outputs are not overwritten. Reliability already has saved Stage B
bins and is deliberately not recalculated here. No fitted module is imported.

PR/QA detection scores pool the two held-out site-CV repeats, matching grouped
validation. EV annotations are checked as means of repeat-specific scores.
No prediction is averaged over repeats or clipped to figure axis limits.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from dbdust.config import CONFIG, config_path
from paper.numbers import digest


REFERENCE = CONFIG['reference']['name']
SCHEME = 'site_cv'
MODELS = ('full8', 'noSSA5', 'hierarchy_prespecified')
BASELINES = ('AEfixedthreshold', 'LGscreened_coarseproxy')
KEYS = ['row_id', 'split_id']
CORE = ['E0', 'E1_QA1', 'E1', 'E1_QA3', 'E2', 'E3', 'E5', 'E7',
        'E3c', 'E4', 'E8', 'Eselected']
# Numerical CSV round-trip tolerance, not a scientific selection threshold.
CHECK_ATOL = 1e-12


def site_cv(frame):
    """Select held-out events without altering source precision or row order."""
    required = [*KEYS, 'site_key', 'repeat', 'reference', 'scheme']
    if set(required) - set(frame):
        raise ValueError('Missing OOF identity columns')
    if not frame.reference.eq(REFERENCE).all():
        raise ValueError('Unexpected reference in OOF source')
    q = frame.loc[frame.scheme.eq(SCHEME)].copy()
    if q.empty or q[required].isna().any().any():
        raise ValueError('Missing held-out site-CV events or identity values')
    if q.duplicated(KEYS).any() or q.duplicated(['row_id', 'repeat']).any():
        raise ValueError('Duplicate OOF prediction event')
    if set(q['repeat'].unique()) != {0, 1}:
        raise ValueError('Expected both saved site-CV repeats')
    return q


def detection_support(frame):
    """Explicit finite evaluation intersection shared by curve and QA points."""
    columns = ['probability', 'truth_fraction', 'truth_daod', 'positive_fraction', 'DB_AOD550']
    q = frame.loc[np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)].copy()
    if q.empty:
        raise ValueError('Empty classifier detection support')
    for name in ('probability', 'truth_fraction', 'positive_fraction'):
        if ((q[name] < 0) | (q[name] > 1)).any():
            raise ValueError(f'{name} outside [0, 1]')
    return q


def _truth(frame):
    return frame.truth_fraction.to_numpy() >= CONFIG['reference']['dust_label_threshold']


def _divide(a, b):
    return a / b if b else np.nan


def _counts(tp, fp, fn, tn):
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=_divide(tp, tp + fn), FPR=_divide(fp, fp + tn),
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def operating_point(frame, positive):
    truth = _truth(frame)
    positive = np.asarray(positive, dtype=bool)
    if positive.shape != truth.shape:
        raise ValueError('Classification mask is not aligned with events')
    return _counts(np.sum(truth & positive), np.sum(~truth & positive),
                   np.sum(truth & ~positive), np.sum(~truth & ~positive))


def support_counts(frame):
    truth = _truth(frame)
    return dict(prediction_events=len(frame), unique_rows=frame.row_id.nunique(),
                sites=frame.site_key.nunique(), repeats=frame['repeat'].nunique(),
                positives=int(truth.sum()), negatives=int((~truth).sum()),
                prevalence=_divide(int(truth.sum()), len(frame)))


def pr_curve(frame):
    """All distinct raw-score thresholds, ties together; no pruning/smoothing.

    Rows follow descending thresholds. The first point is the conventional
    no-predictions PR endpoint (precision=1, recall=0, undefined threshold),
    not a measured QA score. With no true positives, measured recall is NaN.
    """
    scores, truth = frame.probability.to_numpy(), _truth(frame)
    if not len(scores):
        raise ValueError('Empty PR support')
    order = np.argsort(-scores, kind='stable')
    scores, truth = scores[order], truth[order]
    last = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(truth, dtype=np.int64)[last]
    fp = last + 1 - tp
    positives, negatives = int(truth.sum()), int((~truth).sum())
    endpoint = _counts(0, 0, positives, negatives)
    endpoint.update(threshold=np.nan, precision=1., recall=0., event='pr_endpoint', point_index=0)
    rows = [endpoint]
    for i, (at, a, b) in enumerate(zip(last, tp, fp), start=1):
        rows.append(dict(_counts(a, b, positives - a, negatives - b),
                         threshold=float(scores[at]), event='score_threshold', point_index=i))
    return pd.DataFrame(rows)


def _auc(frame, curve):
    if not _truth(frame).any() or _truth(frame).all():
        return np.nan
    # The sorted threshold path is also an exact tie-aware empirical ROC path.
    recall = curve.recall.to_numpy()
    return float(np.sum(np.diff(curve.FPR) * (recall[1:] + recall[:-1]) / 2))


def _select(saved, **filters):
    q = saved
    for name, value in filters.items():
        q = q.loc[q[name].eq(value)]
    if len(q) != 1:
        raise ValueError(f'Missing or ambiguous saved row: {filters}')
    return q.iloc[0]


def _check(actual, expected, label, checks, exact=False):
    good = actual == expected if exact else np.isclose(actual, expected, rtol=0,
                                                      atol=CHECK_ATOL, equal_nan=True)
    if not good:
        raise ValueError(f'Saved comparison mismatch: {label}: {actual!r} != {expected!r}')
    checks.append(dict(check=label, status='PASS'))


def _check_detection(frame, point, row, label, checks):
    observed = support_counts(frame)
    for field, saved_field in (('prediction_events', 'n'), ('unique_rows', 'unique_rows'), ('sites', 'sites')):
        _check(observed[field], row[saved_field], f'{label}/{field}', checks, exact=True)
    for field in ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'FPR', 'F1'):
        _check(point[field], row[field], f'{label}/{field}', checks,
               exact=field in ('TP', 'FP', 'FN', 'TN'))


def _metadata(model):
    return dict(reference=REFERENCE, scheme=SCHEME, population='own', model=model,
                branch='hierarchy_own' if model == 'hierarchy_prespecified' else model + '_own',
                support=model + '_own_finite_detection')


def baseline_points(parent, baselines, model, saved, checks):
    """Deterministic decisions, preserving each baseline's actual support."""
    meta = _metadata(model)
    records = []
    known = parent.db_type.isin(['Dust', 'Mixed', 'Smoke', 'Sulfate'])
    q = parent.loc[known].copy()
    candidates = [('DB_dust_flag', q, q.db_type.eq('Dust').to_numpy())]
    for name, baseline in baselines.items():
        columns = KEYS + ['probability', 'truth_fraction']
        merged = parent.drop(columns='probability').merge(
            baseline[columns].rename(columns={'truth_fraction': 'baseline_truth'}),
            on=KEYS, how='left', validate='one_to_one', indicator=True)
        matched = merged['_merge'].eq('both')
        if name != 'LGscreened_coarseproxy' and not matched.all():
            raise ValueError(f'Missing baseline event keys: {name}')
        if not np.array_equal(merged.loc[matched, 'truth_fraction'],
                              merged.loc[matched, 'baseline_truth']):
            raise ValueError(f'Baseline truth mismatch: {name}')
        if name == 'AEfixedthreshold' and not np.isfinite(merged.probability).all():
            raise ValueError('Missing AE baseline scores')
        q = merged.loc[np.isfinite(merged.probability)].copy()
        if not q.probability.isin([0, 1]).all():
            raise ValueError(f'Expected cached binary baseline score: {name}')
        candidates.append((name, q, q.probability.eq(1).to_numpy()))
    for name, q, positive in candidates:
        point = operating_point(q, positive)
        record = dict(meta, estimator=name, comparison_support=meta['support'],
                      support=meta['support'] + '__' + name + '_available',
                      **support_counts(q), **point,
                      parent_events=len(parent), excluded_events=len(parent) - len(q),
                      same_support=len(q) == len(parent))
        records.append(record)
        if name in BASELINES:
            population = 'common' if model == 'full8' else 'own'
            row = _select(saved, reference=REFERENCE, population=population, model=name,
                          scheme=SCHEME, scope='overall', group='all',
                          QA=CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']])
            _check_detection(q, point, row, f'{model}/{name}', checks)
    return records


def overrides(frame, saved, checks):
    """Regional rejection fractions use eligible DB-dust events, not vetoes."""
    rows = []
    groups = [('overall', 'all', frame),
              *(('region', name, q) for name, q in frame.groupby('region', sort=True))]
    gate = CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']]
    for scope, group, q in groups:
        eligible = q.loc[q.db_type.eq('Dust') & np.isfinite(q.probability)]
        if not np.isfinite(eligible.truth_fraction).all():
            raise ValueError('Missing reference on eligible DB-dust events')
        rejected = eligible.loc[eligible.probability < gate]
        correct = int((rejected.truth_fraction < CONFIG['reference']['dust_label_threshold']).sum())
        den, n = len(eligible), len(rejected)
        record = dict(_metadata('hierarchy_prespecified'), scope=scope, group=group,
                      event='veto_DB_dust', eligible=den,
                      eligible_unique_rows=eligible.row_id.nunique(),
                      eligible_sites=eligible.site_key.nunique(),
                      overrides=n, overridden_sites=rejected.site_key.nunique(),
                      correct_overrides=correct, incorrect_overrides=n - correct,
                      supported_rejection_fraction=_divide(correct, den),
                      unsupported_rejection_fraction=_divide(n - correct, den),
                      override_rate=_divide(n, den), agreement=_divide(correct, n))
        row = _select(saved, reference=REFERENCE, population='own', model='hierarchy_prespecified',
                      scheme=SCHEME, scope=scope, group=group, event='veto_DB_dust')
        for field in ('eligible', 'overrides', 'correct_overrides'):
            _check(record[field], row[field], f'override/{scope}/{group}/{field}', checks, exact=True)
        _check(record['overridden_sites'], row.unique_sites,
               f'override/{scope}/{group}/overridden_sites', checks, exact=True)
        for field in ('override_rate', 'agreement'):
            _check(record[field], row[field], f'override/{scope}/{group}/{field}', checks)
        rows.append(record)
    return pd.DataFrame(rows)


def ev_scatter(frame, saved, checks):
    columns = ['truth_daod', 'truth_fraction', 'DB_AOD550', *CORE]
    q = frame.loc[np.isfinite(frame[columns].to_numpy(float)).all(axis=1)].copy()
    if q.empty:
        raise ValueError('Empty hierarchy core support')
    q['Eselected_daod'] = q.Eselected * q.DB_AOD550
    expected = _select(saved, stage='b', reference=REFERENCE, branch='hierarchy_own',
                       scheme=SCHEME, pairing='core', stratum='all', estimator='Eselected',
                       repeat_summary='mean_over_repeats')
    observed = dict(prediction_events=len(q), sites=q.site_key.nunique(),
                    unique_rows=q.row_id.nunique(), repeats=q['repeat'].nunique(),
                    unique_site_months=len(q[['site_key', 'year', 'month']].drop_duplicates()))
    for name, value in observed.items():
        _check(value, expected[name], f'EV/{name}', checks, exact=True)
    scores = []
    for _, group in q.groupby('repeat', sort=True):
        x, y = group.truth_daod.to_numpy(), group.Eselected_daod.to_numpy()
        r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
        scores.append(dict(mean_ref=np.mean(x), mean_pred=np.mean(y), bias=np.mean(y - x),
                           mae=np.mean(abs(y - x)), rmse=np.sqrt(np.mean((y - x) ** 2)), r=r))
    for name in scores[0]:
        value = np.mean([s[name] for s in scores])
        _check(value, expected[name], f'EV/mean_over_repeats/{name}', checks)
    columns = ['reference', 'scheme', *KEYS, 'repeat', 'site_key', 'region', 'year', 'month',
               'branch', 'selected_estimator', 'truth_fraction', 'truth_daod',
               'DB_AOD550', 'AER_AOD550', 'Eselected', 'Eselected_daod']
    result = q[columns].copy()
    result['pairing'] = 'core'
    result['estimator'] = 'Eselected'
    return result


SEMANTICS = {
    'pixel_pr_curves': 'Raw classifier PR curves on each model own finite detection support '
        '(probability, truth_fraction, truth_daod, positive_fraction, DB_AOD550 all finite); '
        'site-CV repeats pooled as separate prediction events. All unique score thresholds, '
        'descending, ties together; pr_endpoint has undefined threshold and conventional precision=1, '
        'recall=0. No fitting, interpolation, smoothing, or confidence intervals.',
    'pixel_qa_operating_points': 'Raw P at configured QA thresholds on the same support as each PR curve. '
        'Source score dtype is retained for threshold comparisons. Counts, precision/recall/FPR/F1, '
        'AUC and Brier checked against saved grouped summaries. Undefined precision uses the saved '
        'zero convention; recall with no positives is missing.',
    'pixel_prevalences': 'Separate empirical truth_fraction>=configured-label prevalence for every '
        'classifier/baseline support. No universal prevalence line across differing populations. '
        'Prediction events include repeats; unique_rows and sites are separate counts.',
    'pixel_baseline_points': 'Binary DB_dust_flag means db_type==Dust, NOT E5 training-type conditional '
        'mean/probability. AEfixedthreshold and LGscreened_coarseproxy use saved binary decisions. '
        'Each baseline is intersected with its parent curve support; absent screened-LG events '
        'and nonfinite scores remain excluded, never negatives. AE requires complete event keys '
        'and finite scores. Parent/excluded counts and actual prevalence are explicit. '
        'The noSSA5 own population is not fallback-only.',
    'pixel_ev_scatter': 'Held-out hierarchy Stage B core intersection; one row per row_id/split_id, '
        'finite truth_daod, truth_fraction, DB_AOD550 and E0/E1_QA1/E1/E1_QA3/E2/E3/E5/E7/'
        'E3c/E4/E8/Eselected required; E6 availability does not restrict support. '
        'Both repeats retained. x=truth_daod from AERONET, y=Eselected_daod=stored Eselected fraction '
        'times DB_AOD550 once. No axis-limit truncation or repeat averaging. Eselected is nested '
        'fold selection, not a replacement by final deployment branch choices. Saved annotation '
        'metrics are means of repeat-specific metrics, not scores of averaged predictions.',
    'pixel_overrides': 'Hierarchy site-CV raw-P QA2 vetoes among eligible DB-dust events. Both '
        'eligibility (db_type==Dust and finite raw probability) and rejection (raw probability '
        'strictly below configured product gate) preserve event identity. Missing reference '
        'on eligible events is an error, not a denominator exclusion. '
        'Reference-supported and reference-unsupported rejection fractions divide by eligible, '
        'not overrides. Zero eligible gives missing rates. eligible_sites is the denominator-site '
        'count for sparse-region labeling; overridden_sites reproduces saved unique_sites. Regions '
        'are frozen first-match labels, not new fire boxes; repeats are not independent pixels.',
}


def derive(source_root):
    """Return six checked tables, immutable-input hashes, and comparison checks."""
    source_root = Path(source_root).resolve()
    sources, checks = {}, []

    def read(relative):
        path = source_root / relative
        sources[str(path)] = digest(path)
        return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)

    config = Path(config_path()).resolve()
    sources[str(config)] = digest(config)
    saved = read('grouped_validation/metrics.csv')
    override_saved = read('grouped_validation/type_overrides.csv')
    ev_saved = read('expected_value_test/stage_b/collocation_metrics.csv')
    primary = {name: site_cv(read(f'grouped_validation/oof/{REFERENCE}__own__{name}.parquet'))
               for name in MODELS}
    baselines = {name: site_cv(read(f'grouped_validation/oof/{REFERENCE}__own__{name}.parquet'))
                 for name in BASELINES}
    curves, qa, prevalence, baseline = [], [], [], []
    for model, frame in primary.items():
        q = detection_support(frame)
        meta, counts = _metadata(model), support_counts(q)
        curve = pr_curve(q)
        auc = _auc(q, curve)
        brier = np.mean((q.probability.to_numpy() - _truth(q)) ** 2)
        curves.append(curve.assign(**meta, estimator=model, **counts))
        prevalence.append(dict(meta, estimator=model, scope='classifier', **counts))
        for tier, threshold in enumerate(CONFIG['qa']['thresholds'], start=1):
            point = operating_point(q, q.probability.to_numpy() >= threshold)
            row = _select(saved, reference=REFERENCE, population='own', model=model,
                          scheme=SCHEME, scope='overall', group='all', QA=threshold)
            label = f'{model}/QA{tier}'
            _check_detection(q, point, row, label, checks)
            _check(auc, row.AUC, label + '/AUC', checks)
            _check(brier, row.Brier, label + '/Brier', checks)
            qa.append(dict(meta, estimator=model, QA=threshold, qa_tier=tier,
                           **counts, **point, AUC=auc, Brier=float(brier)))
        records = baseline_points(q, baselines, model, saved, checks)
        baseline.extend(records)
        for record in records:
            prevalence.append({**{k: record[k] for k in [*meta, 'estimator', *counts]}, 'scope': 'baseline'})
    ev = site_cv(read(f'expected_value_test/predictions/stage_b_{REFERENCE}_hierarchy_own.parquet'))
    tables = dict(pixel_pr_curves=pd.concat(curves, ignore_index=True),
                  pixel_qa_operating_points=pd.DataFrame(qa),
                  pixel_prevalences=pd.DataFrame(prevalence),
                  pixel_baseline_points=pd.DataFrame(baseline),
                  pixel_ev_scatter=ev_scatter(ev, ev_saved, checks),
                  pixel_overrides=overrides(primary['hierarchy_prespecified'], override_saved, checks))
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f'Frozen source changed during derivation: {path}')
    return tables, sources, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output, source = args.output_dir.resolve(), args.source_root.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.is_relative_to(source):
        raise ValueError('Write derived arrays outside the public worktree and frozen source tree')
    targets = [output / (name + suffix) for name in SEMANTICS for suffix in ('.csv', '.provenance.json')]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError('Derived pixel outputs already exist; choose a fresh output directory')
    tables, sources, checks = derive(source)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pixel-', dir=output) as temporary:
        for name, table in tables.items():
            path = Path(temporary) / (name + '.csv')
            table.to_csv(path, index=False)
            provenance = dict(sources=sources, output_sha256=digest(path),
                              script_sha256=digest(Path(__file__)), semantics=SEMANTICS[name],
                              reference=REFERENCE, scheme=SCHEME, rows=len(table), checks=checks,
                              numerical_check_atol=CHECK_ATOL)
            path.with_suffix('.provenance.json').write_text(
                json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        for path in sorted(Path(temporary).iterdir()):
            os.replace(path, output / path.name)
    print(f'Wrote {len(tables)} pixel diagnostic CSVs to {output}; {len(checks)} saved comparisons passed')


if __name__ == '__main__':
    main()
