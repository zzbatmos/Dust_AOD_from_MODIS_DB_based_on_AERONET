"""Inference-only implementation of the frozen full8/noSSA5 hierarchy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

from .branches import assign_branches, qa_tiers, valid_features
from .config import CONFIG


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify_manifest(root):
    root = Path(root)
    manifest = root / CONFIG['artifacts']['manifest']
    if file_hash(manifest) != CONFIG['artifacts']['manifest_sha256']:
        raise ValueError('Model manifest does not match frozen configuration')
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return file_hash(manifest)


def empirical_smear(z, residuals):
    """Same empirical integral and summation order as frozen deployment."""
    z = np.asarray(z, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError('Smearing needs finite residuals')
    unique, counts = np.unique(residuals, return_counts=True)
    weights = counts.astype(float)/len(residuals)
    result = np.full(z.shape, np.nan)
    good = np.flatnonzero(np.isfinite(z))
    qs = CONFIG['runtime']['smear_query_chunk']
    rs = CONFIG['runtime']['smear_residual_chunk']
    for start in range(0, len(good), qs):
        take = good[start:start+qs]
        total = np.zeros(len(take))
        for r in range(0, len(unique), rs):
            total += np.sum(expit(z[take, None]+unique[None, r:r+rs])*weights[None, r:r+rs], axis=1)
        result[take] = total
    return result


class BranchModel:
    """Loads fitted files only. Also usable with held-out fold artifacts."""
    def __init__(self, folder, branch):
        self.folder = Path(folder)
        self.spec = CONFIG['branches'][branch]
        self.pair = json.loads((self.folder/'pair.json').read_text())
        self.classifier = self._model('classifier.ubj', XGBClassifier)
        self.regressor = self._model('positive_fraction.ubj', XGBRegressor)
        self.direct = self._model('direct_fraction.ubj', XGBRegressor)
        with np.load(self.folder/'postprocessor.npz', allow_pickle=False) as p:
            self.post = {name: p[name].copy() for name in p.files}
        self.curve = None
        if (self.folder/'smear_curve.npz').exists():
            with np.load(self.folder/'smear_curve.npz', allow_pickle=False) as c:
                self.curve = (c['logit'].copy(), c['fraction'].copy())
        if self.classifier is None and self.pair.get('constant_probability') is None:
            raise ValueError('Missing classifier')
        if self.regressor is None and self.pair.get('constant_fraction') is None:
            raise ValueError('Missing positive regressor')

    def _model(self, name, cls):
        path = self.folder/name
        if not path.exists():
            return None
        model = cls(n_jobs=CONFIG['runtime']['n_jobs'])
        model.load_model(path)
        return model

    def predict(self, frame):
        x = frame[self.spec['features']].to_numpy(float)
        with threadpool_limits(limits=CONFIG['runtime']['n_jobs']):
            p = (self.classifier.predict_proba(x)[:, 1] if self.classifier is not None
                 else np.full(len(x), self.pair['constant_probability']))
            if self.regressor is not None:
                raw_z = self.regressor.predict(x)
                f = expit(raw_z)  # Preserve float32 recipe for the QA field.
                z = raw_z.astype(float)
            else:
                f = np.full(len(x), self.pair['constant_fraction'])
                eps = CONFIG['reference']['logit_epsilon']
                z = np.full(len(x), logit(np.clip(self.pair['constant_fraction'], eps, 1-eps)))
            direct = np.clip(self.direct.predict(x), 0, 1) if self.direct is not None else np.full(len(x), np.nan)
        p = np.asarray(p, dtype=float)
        pc = np.interp(p, self.post['iso_x'], self.post['iso_y'])
        if bool(self.post['no_positive_rows']):
            star = np.zeros(len(x))
        elif self.curve is None:
            star = empirical_smear(z, self.post['residuals'])
        else:
            cx, cy = self.curve
            inside = (z >= cx[0]) & (z <= cx[-1])
            star = np.full(len(x), np.nan)
            star[inside] = np.interp(z[inside], cx, cy)
            star[~inside] = empirical_smear(z[~inside], self.post['residuals'])
        mixture = pc*star+(1-pc)*float(self.post['f_neg'])
        return dict(probability=p, positive_fraction=f, logit_prediction=z,
                    probability_calibrated=pc, positive_fraction_star=star,
                    E3c=mixture, E4=direct)


class EVPredictor:
    def __init__(self, artifact_root=None):
        root = Path(artifact_root) if artifact_root else Path(__file__).resolve().parents[1]/CONFIG['artifacts']['root']
        if artifact_root is None and not root.exists():
            root = Path(__file__).resolve().parent/CONFIG['artifacts']['root']
        self.manifest_sha256 = verify_manifest(root)
        self.models = {}
        for name, spec in CONFIG['branches'].items():
            meta = json.loads((root/name/'metadata.json').read_text())
            if meta['feature_names'] != spec['features'] or meta['selected'] != spec['selected']:
                raise ValueError(f'Configuration/artifact mismatch for {name}')
            if meta['reference'] != CONFIG['reference']['name']:
                raise ValueError('Unexpected reference')
            self.models[name] = BranchModel(root/name, name)

    def predict_frame(self, frame, batch_size=None):
        frame = pd.DataFrame(frame)
        if 'DB_AOD550' not in frame:
            raise ValueError('DB_AOD550 is required')
        batch_size = batch_size or CONFIG['runtime']['batch_size']
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        codes = assign_branches(frame)
        supported_aod = valid_features(frame, ['DB_AOD550'])
        out = pd.DataFrame(np.nan, index=frame.index, columns=CONFIG['outputs']['pixel'])
        out['branch'] = np.where(supported_aod, codes, -1).astype(np.int8)
        out['aod550_db'] = np.where(supported_aod, frame.DB_AOD550, np.nan)
        if 'db_aerosol_type' in frame:
            out['db_aerosol_type'] = np.where(supported_aod, frame.db_aerosol_type, np.nan)
        for key in ('lat', 'lon'):
            if key in frame:
                out[key] = frame[key].to_numpy()
        gate = CONFIG['qa']['thresholds'][CONFIG['qa']['product_gate_index']]
        for name, spec in CONFIG['branches'].items():
            indices = np.flatnonzero(codes == spec['code'])
            for start in range(0, len(indices), batch_size):
                positions = indices[start:start+batch_size]
                v = self.models[name].predict(frame.iloc[positions])
                aod = frame.DB_AOD550.to_numpy(float)[positions]
                fields = dict(dust_probability=v['probability'],
                    dust_fraction_conditional=v['positive_fraction'],
                    daod550_qa2=np.where(v['probability'] >= gate, v['positive_fraction'], 0)*aod,
                    daod550_ev=v[spec['selected']]*aod)
                for field, values in fields.items():
                    out.iloc[positions, out.columns.get_loc(field)] = values
        out['qa_tier'] = qa_tiers(out.dust_probability)
        return out
