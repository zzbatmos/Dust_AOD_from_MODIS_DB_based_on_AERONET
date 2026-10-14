"""Synthetic cached-event tests; no models, fits, or source data are invoked."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper import derive_pixel_diagnostics as diagnostics
from paper.numbers import digest, ingest


def event_frame():
    """Two held-out repeats with deliberately different errors and supports."""
    frames = []
    for repeat in (0, 1):
        p = np.array([.1, .4, .6, .8, .7, .3, .9], dtype=np.float32)
        if repeat:
            p[0] = .8
        frame = pd.DataFrame({
            'row_id': np.arange(7), 'split_id': f'site_cv_r{repeat}_fold0',
            'repeat': repeat, 'site_key': [f'S{i}' for i in range(7)],
            'reference': diagnostics.REFERENCE, 'scheme': diagnostics.SCHEME,
            'population': 'own', 'model': 'hierarchy_prespecified',
            'region': ['dust_region'] * 5 + ['no_dust_region'] * 2,
            'db_type': ['Dust'] * 5 + ['Mixed', 'Smoke'],
            'probability': p, 'probability_calibrated': 1 - p,
            'positive_fraction': .5,
            'truth_fraction': [.1, .3, .3, .1, .3, .1, .3],
            'DB_AOD550': np.arange(1, 8) / 5,
            'AER_AOD550': np.arange(1, 8) / 4,
            'year': 2017, 'month': 1,
            'branch': ['full8'] * 3 + ['noSSA5'] * 4,
            'selected_estimator': 'E4' if repeat else 'E3c',
        })
        frame['truth_daod'] = frame.truth_fraction * frame.AER_AOD550
        for name in diagnostics.CORE:
            frame[name] = .25
        frame['E6'] = np.where(frame.row_id < 3, .25, np.nan)
        frame['Eselected'] = np.arange(1, 8) / (10 if repeat else 20)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_row(frame, model, population, threshold):
    """Independent scalar confusion and pairwise-rank AUC oracle."""
    q = frame.loc[frame.probability.notna()]
    p = q.probability.to_numpy()
    truth = q.truth_fraction.to_numpy() >= diagnostics.CONFIG['reference']['dust_label_threshold']
    decision = p >= threshold
    tp, fp, fn, tn = (int(np.sum(mask)) for mask in (
        truth & decision, ~truth & decision, truth & ~decision, ~truth & ~decision))
    pairwise = p[truth, None] - p[~truth][None, :]
    auc = np.mean((pairwise > 0) + (pairwise == 0) / 2)
    return dict(reference=diagnostics.REFERENCE, population=population, model=model,
                scheme=diagnostics.SCHEME, scope='overall', group='all', QA=threshold,
                n=len(q), unique_rows=q.row_id.nunique(), sites=q.site_key.nunique(),
                TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn), FPR=fp / (fp + tn),
                F1=2 * tp / (2 * tp + fp + fn), AUC=auc,
                Brier=float(np.mean((p - truth) ** 2)))


def override_rows(frame):
    gate = diagnostics.CONFIG['qa']['thresholds'][diagnostics.CONFIG['qa']['product_gate_index']]
    rows = []
    for scope, group, q in [('overall', 'all', frame),
                            *[('region', name, f) for name, f in frame.groupby('region')]]:
        eligible = q.loc[q.db_type.eq('Dust') & q.probability.notna()]
        rejected = eligible.loc[eligible.probability < gate]
        correct = int((rejected.truth_fraction < diagnostics.CONFIG['reference']['dust_label_threshold']).sum())
        rows.append(dict(reference=diagnostics.REFERENCE, population='own',
                         model='hierarchy_prespecified', scheme=diagnostics.SCHEME,
                         scope=scope, group=group, event='veto_DB_dust',
                         eligible=len(eligible), overrides=len(rejected), correct_overrides=correct,
                         override_rate=len(rejected) / len(eligible) if len(eligible) else np.nan,
                         agreement=correct / len(rejected) if len(rejected) else np.nan,
                         unique_sites=rejected.site_key.nunique()))
    return pd.DataFrame(rows)


def ev_scores(frame):
    # E6 is intentionally irrelevant to the frozen core intersection.
    q = frame.loc[frame[diagnostics.CORE].notna().all(axis=1)]
    scores = []
    for _, f in q.groupby('repeat'):
        x = f.truth_daod.to_numpy()
        y = (f.Eselected * f.DB_AOD550).to_numpy()
        residual = y - x
        centered_x, centered_y = x - x.mean(), y - y.mean()
        scores.append(dict(mean_ref=x.mean(), mean_pred=y.mean(), bias=residual.mean(),
                           mae=np.abs(residual).mean(), rmse=np.sqrt(np.mean(residual ** 2)),
                           r=np.sum(centered_x * centered_y) /
                           np.sqrt(np.sum(centered_x ** 2) * np.sum(centered_y ** 2))))
    return pd.DataFrame([dict(stage='b', reference=diagnostics.REFERENCE,
                              branch='hierarchy_own', scheme=diagnostics.SCHEME,
                              pairing='core', stratum='all', estimator='Eselected',
                              repeat_summary='mean_over_repeats', prediction_events=len(q),
                              sites=q.site_key.nunique(), unique_rows=q.row_id.nunique(),
                              repeats=q['repeat'].nunique(),
                              unique_site_months=len(q[['site_key', 'year', 'month']].drop_duplicates()),
                              **pd.DataFrame(scores).mean().to_dict())])


@pytest.fixture
def cached(tmp_path):
    root = tmp_path / 'frozen'
    (root / 'grouped_validation/oof').mkdir(parents=True)
    (root / 'expected_value_test/predictions').mkdir(parents=True)
    (root / 'expected_value_test/stage_b').mkdir(parents=True)
    frame = event_frame()
    metrics = []
    for model in diagnostics.MODELS:
        q = frame.loc[frame.row_id < 3].copy() if model == 'full8' else frame.copy()
        q['model'] = model
        q.to_parquet(root / f'grouped_validation/oof/{diagnostics.REFERENCE}__own__{model}.parquet')
        metrics.extend(score_row(q, model, 'own', threshold)
                       for threshold in diagnostics.CONFIG['qa']['thresholds'])
    baselines = {}
    gate = diagnostics.CONFIG['qa']['thresholds'][diagnostics.CONFIG['qa']['product_gate_index']]
    for name in diagnostics.BASELINES:
        q = frame.copy()
        q['model'] = name
        q['probability'] = q.row_id.mod(2).to_numpy(dtype=np.float32)
        if name == 'LGscreened_coarseproxy':
            q.loc[q.row_id >= 3, 'probability'] = np.nan
        q.to_parquet(root / f'grouped_validation/oof/{diagnostics.REFERENCE}__own__{name}.parquet')
        baselines[name] = q
        metrics.extend([score_row(q, name, 'own', gate),
                        score_row(q.loc[q.row_id < 3], name, 'common', gate)])
    pd.DataFrame(metrics).to_csv(root / 'grouped_validation/metrics.csv', index=False)
    override_rows(frame).to_csv(root / 'grouped_validation/type_overrides.csv', index=False)
    frame.to_parquet(root / f'expected_value_test/predictions/stage_b_{diagnostics.REFERENCE}_hierarchy_own.parquet')
    ev_scores(frame).to_csv(root / 'expected_value_test/stage_b/collocation_metrics.csv', index=False)
    return root, frame, baselines


def test_pr_ties_endpoint_and_raw_probability():
    frame = event_frame().iloc[:4].copy()
    frame['probability'] = np.array([.8, .8, .3, .1], dtype=np.float32)
    curve = diagnostics.pr_curve(frame)
    assert len(curve) == 4
    assert curve.event.iloc[0] == 'pr_endpoint'
    assert np.isnan(curve.threshold.iloc[0])
    assert tuple(curve.loc[0, ['precision', 'recall']]) == (1, 0)
    assert tuple(curve.loc[1, ['TP', 'FP', 'FN', 'TN']]) == (1, 1, 1, 1)
    assert tuple(curve.loc[3, ['precision', 'recall']]) == (.5, 1)
    assert diagnostics._auc(frame, curve) == pytest.approx(.625)


def test_pr_empty_and_single_class():
    frame = event_frame()
    frame['truth_fraction'] = 0.
    curve = diagnostics.pr_curve(frame)
    assert curve.recall.iloc[1:].isna().all()
    assert np.isnan(diagnostics._auc(frame, curve))
    with pytest.raises(ValueError, match='Empty PR'):
        diagnostics.pr_curve(frame.iloc[:0])


def test_branch_supports_repeats_qa_precision_and_prevalences(cached):
    root, frame, _ = cached
    tables, sources, checks = diagnostics.derive(root)
    assert set(tables) == set(diagnostics.SEMANTICS)
    assert all(c['status'] == 'PASS' for c in checks)
    assert len(checks) > 100
    assert all(Path(p).is_absolute() and digest(Path(p)) == h for p, h in sources.items())
    qa = tables['pixel_qa_operating_points']
    assert len(qa) == 9
    assert set(qa.prediction_events) == {6, 14}
    assert set(qa.repeats) == {2}
    for row in qa.itertuples():
        support = frame.loc[frame.row_id < 3] if row.model == 'full8' else frame
        expected = score_row(support, row.model, 'own', row.QA)
        assert (row.TP, row.FP) == (expected['TP'], expected['FP'])
        curve = tables['pixel_pr_curves']
        curve = curve.loc[curve.model.eq(row.model)]
        assert len(curve) == support.probability.nunique() + 1
    prevalence = tables['pixel_prevalences'].query("scope == 'classifier'")
    assert prevalence.set_index('model').loc['full8', 'prevalence'] == pytest.approx(2 / 3)
    assert prevalence.set_index('model').loc['noSSA5', 'prevalence'] == pytest.approx(4 / 7)


def test_lg_absent_keys_equivalent_to_nan_ae_fail_closed(cached):
    root, frame, baselines = cached
    saved = pd.read_csv(root / 'grouped_validation/metrics.csv')
    a = diagnostics.baseline_points(frame, baselines, 'noSSA5', saved, [])
    sparse = {**baselines, 'LGscreened_coarseproxy': baselines['LGscreened_coarseproxy'].dropna(subset=['probability'])}
    b = diagnostics.baseline_points(frame, sparse, 'noSSA5', saved, [])
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))
    points = pd.DataFrame(b).set_index('estimator')
    lg = points.loc['LGscreened_coarseproxy']
    assert (lg.prediction_events, lg.excluded_events, lg.sites) == (6, 8, 3)
    assert not lg.same_support
    # The binary DB flag is not E5 (constant .25 in this fixture).
    assert points.loc['DB_dust_flag', 'TP'] == 6
    assert points.loc['DB_dust_flag', 'FP'] == 4
    sparse['AEfixedthreshold'] = baselines['AEfixedthreshold'].iloc[1:]
    with pytest.raises(ValueError, match='Missing baseline event keys: AE'):
        diagnostics.baseline_points(frame, sparse, 'noSSA5', saved, [])
    sparse['AEfixedthreshold'] = baselines['AEfixedthreshold'].copy()
    sparse['AEfixedthreshold'].loc[0, 'probability'] = np.nan
    with pytest.raises(ValueError, match='Missing AE baseline scores'):
        diagnostics.baseline_points(frame, sparse, 'noSSA5', saved, [])


def test_baseline_matched_truth_and_binary_score_validation(cached):
    root, frame, baselines = cached
    saved = pd.read_csv(root / 'grouped_validation/metrics.csv')
    lg = baselines['LGscreened_coarseproxy'].dropna(subset=['probability']).copy()
    baselines['LGscreened_coarseproxy'] = lg
    lg.loc[0, 'truth_fraction'] = 1.
    with pytest.raises(ValueError, match='Baseline truth mismatch'):
        diagnostics.baseline_points(frame, baselines, 'noSSA5', saved, [])
    lg.loc[0, 'truth_fraction'] = frame.loc[0, 'truth_fraction']
    lg.loc[0, 'probability'] = .5
    with pytest.raises(ValueError, match='binary baseline'):
        diagnostics.baseline_points(frame, baselines, 'noSSA5', saved, [])


def test_sparse_lg_cache_has_identical_derived_tables(cached):
    root, _, baselines = cached
    original, _, _ = diagnostics.derive(root)
    path = root / f'grouped_validation/oof/{diagnostics.REFERENCE}__own__LGscreened_coarseproxy.parquet'
    baselines['LGscreened_coarseproxy'].dropna(subset=['probability']).to_parquet(path)
    sparse, _, _ = diagnostics.derive(root)
    for name in original:
        pd.testing.assert_frame_equal(original[name], sparse[name])


def test_override_denominator_sites_components_and_zero_support():
    frame = event_frame()
    table = diagnostics.overrides(frame, override_rows(frame), []).set_index('group')
    overall = table.loc['all']
    assert (overall.eligible, overall.overrides, overall.correct_overrides) == (10, 3, 1)
    assert (overall.eligible_sites, overall.overridden_sites) == (5, 2)
    assert overall.supported_rejection_fraction == pytest.approx(1 / 10)
    assert overall.unsupported_rejection_fraction == pytest.approx(2 / 10)
    assert overall.override_rate == pytest.approx(overall.supported_rejection_fraction + overall.unsupported_rejection_fraction)
    empty = table.loc['no_dust_region']
    assert empty.eligible == empty.eligible_sites == empty.overrides == 0
    assert np.isnan(empty.override_rate) and np.isnan(empty.agreement)
    assert np.isnan(empty.supported_rejection_fraction)
    frame.loc[0, 'truth_fraction'] = np.nan
    with pytest.raises(ValueError, match='Missing reference'):
        diagnostics.overrides(frame, override_rows(frame), [])


def test_ev_core_retains_no_ssa_and_repeats_and_checks_repeat_metrics():
    frame = event_frame()
    expected = ev_scores(frame)
    scatter = diagnostics.ev_scatter(frame, expected, [])
    assert len(scatter) == 14 and scatter.row_id.nunique() == 7
    assert not scatter.duplicated(['row_id', 'repeat']).any()
    assert set(scatter.branch) == {'full8', 'noSSA5'}
    np.testing.assert_array_equal(scatter.Eselected_daod, frame.Eselected * frame.DB_AOD550)
    assert scatter.Eselected_daod.max() > 0.5  # No figure-axis clipping.
    averaged = scatter.groupby('row_id')[['truth_daod', 'Eselected_daod']].mean()
    expected.loc[0, 'rmse'] = np.sqrt(np.mean((averaged.Eselected_daod - averaged.truth_daod) ** 2))
    with pytest.raises(ValueError, match='EV/mean_over_repeats/rmse'):
        diagnostics.ev_scatter(frame, expected, [])
    frame.loc[0, 'E2'] = np.nan
    assert len(diagnostics.ev_scatter(frame, ev_scores(frame), [])) == 13


@pytest.mark.parametrize('problem', ['event', 'repeat', 'reference', 'missing_identity'])
def test_invalid_oof_identity_refused(problem):
    frame = event_frame()
    if problem == 'event':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == 'repeat':
        frame = frame.loc[frame['repeat'].eq(0)]
    elif problem == 'reference':
        frame.loc[0, 'reference'] = 'other'
    else:
        frame.loc[0, 'site_key'] = None
    with pytest.raises(ValueError):
        diagnostics.site_cv(frame)


def test_detection_finite_support_and_invalid_range():
    frame = event_frame()
    frame.loc[0, 'positive_fraction'] = np.nan
    frame.loc[1, 'truth_daod'] = np.nan
    assert len(diagnostics.detection_support(frame)) == len(frame) - 2
    frame.loc[2, 'probability'] = 2.
    with pytest.raises(ValueError, match='outside'):
        diagnostics.detection_support(frame)


def test_cli_provenance_registry_ingestion_determinism_and_no_overwrite(cached, tmp_path):
    root, _, _ = cached
    outputs = [tmp_path / 'derived1', tmp_path / 'derived2']
    for output in outputs:
        diagnostics.main(['--source-root', str(root), '--output-dir', str(output)])
        assert len(list(output.iterdir())) == 2 * len(diagnostics.SEMANTICS)
        for name, semantics in diagnostics.SEMANTICS.items():
            path = output / (name + '.csv')
            metadata = json.loads(path.with_suffix('.provenance.json').read_text())
            table = ingest(path, 'derived/' + name)
            assert metadata['output_sha256'] == digest(path) == table['source_sha256']
            assert metadata['semantics'] == semantics
            assert metadata['rows'] == table['source_rows']
            assert metadata['sources'] and metadata['checks']
            assert metadata['script_sha256'] == digest(Path(diagnostics.__file__))
    for path in outputs[0].iterdir():
        assert path.read_bytes() == (outputs[1] / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        diagnostics.main(['--source-root', str(root), '--output-dir', str(outputs[0])])
    with pytest.raises(ValueError, match='outside'):
        diagnostics.main(['--source-root', str(root), '--output-dir', str(root / 'derived')])
    with pytest.raises(ValueError, match='outside'):
        diagnostics.main(['--source-root', str(root), '--output-dir', str(Path(diagnostics.__file__).parent / 'derived')])


@pytest.mark.parametrize('table,column', [
    ('grouped_validation/metrics.csv', 'TP'),
    ('grouped_validation/type_overrides.csv', 'eligible'),
    ('expected_value_test/stage_b/collocation_metrics.csv', 'rmse'),
])
def test_saved_disagreement_prevents_publication(cached, tmp_path, table, column):
    root, _, _ = cached
    path = root / table
    saved = pd.read_csv(path)
    saved.loc[0, column] += 1
    saved.to_csv(path, index=False)
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError, match='Saved comparison mismatch'):
        diagnostics.main(['--source-root', str(root), '--output-dir', str(output)])
    assert not output.exists()


def test_input_hash_change_prevents_publication(cached, tmp_path, monkeypatch):
    root, _, _ = cached
    target = root / 'grouped_validation/metrics.csv'
    calls = 0

    def changing_digest(path):
        nonlocal calls
        if Path(path) == target:
            calls += 1
            if calls > 1:
                return 'changed-input'
        return digest(path)

    monkeypatch.setattr(diagnostics, 'digest', changing_digest)
    output = tmp_path / 'must_not_exist'
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        diagnostics.main(['--source-root', str(root), '--output-dir', str(output)])
    assert not output.exists()
