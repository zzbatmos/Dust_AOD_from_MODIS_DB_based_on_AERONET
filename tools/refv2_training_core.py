"""Honest site-grouped nested EV fitting, independent of the real-data loader."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

CONFIG = dict(n_estimators=192, max_depth=4, learning_rate=.05,
              min_child_weight=5, subsample=.8, colsample_bytree=.8,
              tree_method="hist", n_jobs=2, random_state=20261012)
SEED = 20261013
EPS = 1e-4


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def row_hash(ids):
    return hashlib.sha256("|".join(sorted(map(str, ids))).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def check_rows(frame):
    if frame.row_id.duplicated().any():
        raise ValueError("Duplicate row memberships")
    if frame.calibration_reserved.any():
        raise ValueError("Reserved calibration rows entered a fit/holdout")
    if frame[["row_id", "site_key", "granule"]].isna().any().any():
        raise ValueError("Missing grouping identifiers")


def audit(folder, train, heldout, reference, *, parent=None, purged=None, role="split"):
    """Persist explicit memberships, including positive-only regression membership."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    check_rows(train)
    check_rows(heldout)
    for key in ("row_id", "site_key", "granule"):
        if set(train[key]) & set(heldout[key]):
            raise ValueError(f"{role}: {key} leakage")
    if parent is not None:
        allowed = set(parent.row_id)
        if not (set(train.row_id) | set(heldout.row_id)) <= allowed:
            raise ValueError("Nested membership outside parent training population")
    if purged is None:
        purged = train.iloc[:0]
    roles = {"train": train, "heldout": heldout, "purged": purged,
             "positive_regression": train.loc[train[reference] >= .2],
             "negative_mean": train.loc[train[reference] < .2]}
    records = {}
    for name, rows in roles.items():
        with gzip.open(folder / f"{name}_rows.json.gz", "wt") as stream:
            json.dump(rows.row_id.tolist(), stream)
        records[name] = dict(rows=len(rows), sites=int(rows.site_key.nunique()),
                             row_sha256=row_hash(rows.row_id),
                             site_sha256=row_hash(rows.site_key.unique()),
                             granule_sha256=row_hash(rows.granule.unique()))
    write_json(folder / "audit.json", dict(role=role, memberships=records,
               parent_row_sha256=None if parent is None else row_hash(parent.row_id),
               reserved_rows=0, row_overlap=0, site_overlap=0, granule_overlap=0,
               inner_seed=SEED, inner_groups=5))


def grouped_splits(frame):
    """Exactly five deterministic site groups, with shared-granule purging."""
    check_rows(frame)
    sites = np.sort(frame.site_key.unique())
    if len(sites) < 5:
        raise ValueError("Five site groups required; refusing a reduced-fold fallback")
    shuffled = np.random.default_rng(SEED).permutation(sites)
    mapping = {site: i % 5 for i, site in enumerate(shuffled)}
    assignments = frame.site_key.map(mapping)
    for fold in range(5):
        heldout = frame.loc[assignments.eq(fold)].copy()
        candidate = frame.loc[~assignments.eq(fold)]
        shared = candidate.granule.isin(heldout.granule)
        train = candidate.loc[~shared].copy()
        if train.empty:
            raise ValueError("Granule purge emptied a training fold")
        yield fold, train, heldout, candidate.loc[shared].copy()


@dataclass
class Pair:
    classifier: object = None
    regressor: object = None
    constant_probability: float = None
    constant_fraction: float = None

    def predict(self, x):
        with threadpool_limits(limits=2):
            p = (self.classifier.predict_proba(x)[:, 1] if self.classifier is not None
                 else np.full(len(x), self.constant_probability))
            if self.regressor is not None:
                raw_z = self.regressor.predict(x)
                # Preserve the cached float32 expit recipe for ungated/QA f_pos.
                f = expit(raw_z)
                z = raw_z.astype(float)
            else:
                f = np.full(len(x), self.constant_fraction)
                z = np.full(len(x), logit(np.clip(self.constant_fraction, EPS, 1 - EPS)))
        return np.asarray(p, dtype=float), z, f

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, model in (("classifier", self.classifier), ("positive_fraction", self.regressor)):
            if model is not None:
                model.save_model(folder / f"{name}.ubj")
        write_json(folder / "pair.json", dict(constant_probability=self.constant_probability,
                   constant_fraction=self.constant_fraction))

    @classmethod
    def load(cls, folder, cached=False):
        folder = Path(folder)
        meta = json.loads((folder / ("metadata.json" if cached else "pair.json")).read_text())
        pair = cls(constant_probability=meta.get("constant_probability"),
                   constant_fraction=meta.get("constant_fraction"))
        if (folder / "classifier.ubj").exists():
            pair.classifier = XGBClassifier(n_jobs=2)
            pair.classifier.load_model(folder / "classifier.ubj")
        if (folder / "positive_fraction.ubj").exists():
            pair.regressor = XGBRegressor(n_jobs=2)
            pair.regressor.load_model(folder / "positive_fraction.ubj")
        if pair.classifier is None and pair.constant_probability is None:
            raise ValueError("Missing classifier and constant probability")
        if pair.regressor is None and pair.constant_fraction is None:
            raise ValueError("Missing positive regressor and constant fraction")
        return pair


class XGBBackend:
    """No search, early stopping, class weights, or parallel fit dispatch."""
    def pair(self, train, features, reference):
        y = train[reference].to_numpy(float)
        if not len(y) or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError("Training fractions must be finite in [0, 1]")
        x = train[features].to_numpy(float)
        label = (y >= .2).astype(int)
        pair = Pair()
        with threadpool_limits(limits=2):
            if len(np.unique(label)) == 1:
                pair.constant_probability = float(label[0])
            else:
                pair.classifier = XGBClassifier(**CONFIG, objective="binary:logistic", eval_metric="logloss")
                pair.classifier.fit(x, label)
            pos = label == 1
            if pos.sum() < 2:
                pair.constant_fraction = float(y[pos].mean()) if pos.any() else 0.
            else:
                pair.regressor = XGBRegressor(**CONFIG, objective="reg:squarederror")
                pair.regressor.fit(x[pos], logit(np.clip(y[pos], EPS, 1 - EPS)))
        return pair

    def direct(self, train, features, reference):
        model = XGBRegressor(**CONFIG, objective="reg:squarederror")
        with threadpool_limits(limits=2):
            model.fit(train[features].to_numpy(float), train[reference].to_numpy(float))
        return model


def negative_mean(train, reference):
    negative = train.loc[train[reference] < .2, reference]
    # Matches the established Stage A convention if a fold has no negative rows.
    return float(negative.mean()) if len(negative) else 0.


def smear(logits, residuals, query_chunk=256, residual_chunk=4096):
    """Exact empirical Duan integral; bounded memory, no residual subsampling."""
    z = np.asarray(logits, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Smearing needs nonempty finite logit residuals")
    result = np.full(z.shape, np.nan)
    unique, counts = np.unique(residuals, return_counts=True)
    weights = counts.astype(float) / len(residuals)
    good = np.flatnonzero(np.isfinite(z))
    for start in range(0, len(good), query_chunk):
        take = good[start:start + query_chunk]
        total = np.zeros(len(take))
        for r in range(0, len(unique), residual_chunk):
            total += np.sum(expit(z[take, None] + unique[None, r:r + residual_chunk])
                            * weights[None, r:r + residual_chunk], axis=1)
        result[take] = total
    return result


@dataclass
class Postprocessor:
    iso_x: np.ndarray
    iso_y: np.ndarray
    residuals: np.ndarray
    f_neg: float
    no_positive_rows: bool = False
    curve_x: np.ndarray = None
    curve_y: np.ndarray = None

    def calibrated(self, probability):
        p = np.asarray(probability, dtype=float)
        return np.where(np.isfinite(p), np.interp(p, self.iso_x, self.iso_y), np.nan)

    def corrected(self, z):
        if self.no_positive_rows:
            return np.where(np.isfinite(z), 0., np.nan)
        if self.curve_x is not None:
            z = np.asarray(z, dtype=float)
            inside = np.isfinite(z) & (z >= self.curve_x[0]) & (z <= self.curve_x[-1])
            result = np.full(z.shape, np.nan)
            result[inside] = np.interp(z[inside], self.curve_x, self.curve_y)
            outside = np.isfinite(z) & ~inside
            result[outside] = smear(z[outside], self.residuals)
            return result
        return smear(z, self.residuals)

    def predict(self, probability, z):
        fstar = self.corrected(z)
        pc = self.calibrated(probability)
        return dict(positive_fraction_star=fstar, probability_calibrated=pc,
                    E3=probability * fstar + (1 - probability) * self.f_neg,
                    E3c=pc * fstar + (1 - pc) * self.f_neg)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "postprocessor.npz", iso_x=self.iso_x, iso_y=self.iso_y,
                            residuals=self.residuals, f_neg=self.f_neg,
                            no_positive_rows=self.no_positive_rows)

    @classmethod
    def load(cls, folder):
        with np.load(Path(folder) / "postprocessor.npz", allow_pickle=False) as a:
            return cls(a["iso_x"], a["iso_y"], a["residuals"], float(a["f_neg"]),
                       bool(a["no_positive_rows"]))


def oof_record(test, reference, pair, features):
    p, z, f = pair.predict(test[features].to_numpy(float))
    out = test[["row_id", "site_key", "granule", "time", "DB_AOD550", "AER_AOD550"]].copy()
    out["truth_fraction"] = test[reference].to_numpy(float)
    out["truth_daod"] = out.truth_fraction * out.AER_AOD550
    out["probability"] = p
    out["logit_prediction"] = z
    out["positive_fraction"] = f
    pos = out.truth_fraction >= .2
    out["residual_logit"] = np.where(pos, logit(np.clip(out.truth_fraction, EPS, 1 - EPS)) - z, np.nan)
    out["residual_fraction"] = np.where(pos, out.truth_fraction - f, np.nan)
    return out


def fit_postprocessor(train, oof, reference, folder):
    if set(train.row_id) != set(oof.row_id) or oof.row_id.duplicated().any():
        raise ValueError("Postprocessor must use exactly one OOF event per training row")
    # Each OOF event was generated with its entire site and shared granules excluded.
    p = oof.probability.to_numpy(float)
    labels = (oof.truth_fraction >= .2).to_numpy(int)
    if not np.isfinite(p).all():
        raise ValueError("Missing OOF probabilities")
    iso = IsotonicRegression(y_min=0., y_max=1., out_of_bounds="clip").fit(p, labels)
    positive = oof.loc[labels == 1]
    residuals = positive.residual_logit.to_numpy(float)
    post = Postprocessor(iso.X_thresholds_, iso.y_thresholds_,
                         residuals if len(residuals) else np.array([0.]),
                         negative_mean(train, reference), not len(positive))
    post.save(folder)
    oof.to_parquet(Path(folder) / "oof.parquet", index=False)
    write_json(Path(folder) / "postprocessor_diagnostics.json", dict(
        calibration_rows=len(oof), calibration_row_sha256=row_hash(oof.row_id),
        residual_rows=len(positive), residual_row_sha256=row_hash(positive.row_id),
        positive_sites=int(positive.site_key.nunique()),
        sparse_positive_sites_warning=bool(positive.site_key.nunique() < 5),
        sparse_positive_policy="retain five all-row site groups; positive-only regressor; no regrouping",
        negative_rows=int((train[reference] < .2).sum()), f_neg=post.f_neg,
        positive_observed_mean=finite_mean(positive.truth_fraction),
        positive_oof_predicted_mean=finite_mean(positive.positive_fraction),
        residual_fraction_mean=finite_mean(positive.residual_fraction),
        residual_logit_mean=finite_mean(residuals),
        residual_logit_std=float(residuals.std()) if len(residuals) else None,
        no_positive_rows=post.no_positive_rows,
        no_positive_policy="zero positive component, unit zero residual placeholder",
        isotonic_x=post.iso_x.tolist(), isotonic_y=post.iso_y.tolist(),
        calibration_brier_before=float(np.mean((p - labels) ** 2)),
        calibration_brier_apparent_after=float(np.mean((post.calibrated(p) - labels) ** 2)),
        apparent_after_warning="In-sample calibrator diagnostic, never a selection score",
        residual_scale="logit(clip(y,1e-4,1-1e-4)) minus raw logit prediction",
        smearing="exact empirical inverse-logit integration; no fixed bias sign"))
    return post


def build_deploy_curve(post, folder, lower=-24., upper=24., step=.002):
    """Certified numerical acceleration, not a new estimator or fitted model."""
    if step <= 0 or step > .002 or upper <= lower:
        raise ValueError("Curve requires 0 < step <= .002 and ordered bounds")
    x = np.linspace(lower, upper, int(np.ceil((upper - lower) / step)) + 1)
    y = np.zeros(len(x)) if post.no_positive_rows else smear(x, post.residuals)
    width = float(np.diff(x).max())
    bound = width ** 2 / (48 * np.sqrt(3))
    midpoint = (x[:-1] + x[1:]) / 2
    # A deterministic spread checks arithmetic; the derivative bound covers all
    # intervals, not just these checks, for any empirical residual distribution.
    sample = midpoint[np.linspace(0, len(midpoint) - 1, min(257, len(midpoint))).astype(int)]
    truth = np.zeros(len(sample)) if post.no_positive_rows else smear(sample, post.residuals)
    error = float(np.max(np.abs(np.interp(sample, x, y) - truth)))
    if error > bound + 1e-12:
        raise ValueError("Smearing curve exceeded its interpolation certificate")
    np.savez_compressed(Path(folder) / "smear_curve.npz", logit=x, fraction=y)
    write_json(Path(folder) / "smear_curve_diagnostics.json", dict(
        lower=lower, upper=upper, max_step=width, knots=len(x),
        sigmoid_second_derivative_bound=float(1 / (6 * np.sqrt(3))),
        uniform_interpolation_bound=bound, floating_point_tolerance=1e-12,
        midpoint_checks=len(sample), max_checked_absolute_error=error,
        outside_range="exact empirical smearing", OOF="always exact; no curve",
        approximation="linear interpolation of exact empirical smearing; not a new model"))
    return x, y


def crossfit_pair(train, features, reference, folder, backend):
    frames = []
    for fold, subtrain, test, purged in grouped_splits(train):
        child = Path(folder) / f"subinner_{fold}"
        audit(child, subtrain, test, reference, parent=train, purged=purged, role="subinner")
        pair = backend.pair(subtrain, features, reference)
        out = oof_record(test, reference, pair, features)
        out["subinner_fold"] = fold
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def choose_estimator(oof):
    """Strict pairing first; pool honest inner OOF cells, never outer results."""
    cols = ["E3c", "E4", "truth_daod", "DB_AOD550"]
    data = oof.loc[np.isfinite(oof[cols].to_numpy(float)).all(axis=1)].copy()
    if data.row_id.duplicated().any():
        raise ValueError("Duplicate selection OOF row")
    dates = pd.to_datetime(data.time)
    data["year"], data["month"] = dates.dt.year, dates.dt.month
    for name in ("E3c", "E4"):
        data[name + "_daod"] = data[name] * data.DB_AOD550
    g = data.groupby(["site_key", "year", "month"], observed=True)
    cells = g[["truth_daod", "E3c_daod", "E4_daod"]].mean()
    cells["unique_rows"] = g.row_id.nunique()
    cells = cells.loc[cells.unique_rows >= 3].reset_index()
    sites = int(cells.site_key.nunique())
    use_month = sites >= 5
    evaluation = cells if use_month else data
    if evaluation.empty:
        raise ValueError("No paired inner data for selection")
    scores = {name: float(np.sqrt(np.mean((evaluation[name + "_daod"] - evaluation.truth_daod) ** 2)))
              for name in ("E3c", "E4")}
    winner = "E3c" if scores["E3c"] < scores["E4"] else "E4"
    return dict(selected=winner, scores=scores, criterion="site_month_RMSE" if use_month else "collocation_RMSE",
                minimum_cell_rows=3, qualifying_sites=sites, qualifying_cells=len(cells),
                input_rows=len(oof), paired_rows=len(data), dropped_rows=len(oof) - len(data),
                paired_row_sha256=row_hash(data.row_id), exact_tie="E4",
                selection_source="honest inner held-out only; E3c postprocessing from sub-inner OOF")


def nested_training(train, features, reference, folder, backend=None):
    """Fit only on an outer-training population; has no outer-test argument."""
    backend = backend or XGBBackend()
    folder = Path(folder)
    frames = []
    for fold, inner_train, validation, purged in grouped_splits(train):
        child = folder / f"inner_{fold}"
        audit(child, inner_train, validation, reference, parent=train, purged=purged, role="inner")
        suboof = crossfit_pair(inner_train, features, reference, child, backend)
        post = fit_postprocessor(inner_train, suboof, reference, child)
        pair = backend.pair(inner_train, features, reference)
        out = oof_record(validation, reference, pair, features)
        pred = post.predict(out.probability.to_numpy(), out.logit_prediction.to_numpy())
        for key, value in pred.items():
            out[key] = value
        direct = backend.direct(inner_train, features, reference)
        with threadpool_limits(limits=2):
            out["E4"] = np.clip(direct.predict(validation[features].to_numpy(float)), 0, 1)
        out["inner_fold"] = fold
        out.to_parquet(child / "honest_validation.parquet", index=False)
        frames.append(out)
    inner_oof = pd.concat(frames, ignore_index=True)
    decision = choose_estimator(inner_oof)
    # The decision is durable before any model is applied to outer evaluation rows.
    write_json(folder / "selection.json", decision)
    post = fit_postprocessor(train, inner_oof, reference, folder)
    return post, decision


def quadratic_fit(train, reference):
    ae = train.DB_AE.to_numpy(float)
    design = np.column_stack([np.ones(len(ae)), ae, ae ** 2])
    coef, _, rank, singular = np.linalg.lstsq(design, train[reference].to_numpy(float), rcond=None)
    return coef, dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]),
                      rank=int(rank), singular_values=singular.tolist(),
                      training_rows=len(train), training_row_sha256=row_hash(train.row_id),
                      target="dust fraction (NOT FMF)", coefficient_order="a + b*DB_AE + c*DB_AE^2",
                      published_FMF=dict(constant=.051, linear=.336, quadratic=.085),
                      weighting="unweighted ordinary least squares; clip only at inference")


def quadratic_predict(ae, coefficients):
    ae = np.asarray(ae, dtype=float)
    a, b, c = coefficients
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.clip(a + b * ae + c * ae ** 2, 0, 1)
    return np.where(np.isfinite(ae), result, np.nan)
