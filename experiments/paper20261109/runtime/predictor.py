"""Portable NumPy-only inference for a two-head reliability calibrator."""
from pathlib import Path
import json
import numpy as np


def stable_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    expz = np.exp(z[~positive])
    out[~positive] = expz / (1.0 + expz)
    return out


class ReliabilityCalibrator:
    head_names = ('text_correct', 'teacher_correct')

    def __init__(self, mean, scale, coef, intercept, feature_indices,
                 input_dim, feature_names=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.input_dim = int(input_dim)
        self.feature_names = list(feature_names) if feature_names is not None else []
        k = len(self.feature_indices)
        assert self.mean.shape == self.scale.shape == (k,)
        assert self.coef.shape == (k, 2) and self.intercept.shape == (2,)
        assert np.all(self.scale > 0) and np.isfinite(self.scale).all()
        assert len(set(self.feature_indices.tolist())) == k
        assert np.all((self.feature_indices >= 0) & (self.feature_indices < self.input_dim))
        assert all(np.isfinite(x).all() for x in [self.mean, self.coef, self.intercept])

    def logits_np(self, X, batch_size=65536):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f'Expected X[N,{self.input_dim}], got {X.shape}')
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite input features')
        out = np.empty((len(X), 2), dtype=np.float64)
        for start in range(0, len(X), int(batch_size)):
            z = (X[start:start + batch_size, self.feature_indices].astype(np.float64) - self.mean) / self.scale
            out[start:start + batch_size] = z @ self.coef + self.intercept
        return out

    def predict_np(self, X, batch_size=65536):
        """Return float32[N,2]: P(text correct), P(teacher correct)."""
        return stable_sigmoid(self.logits_np(X, batch_size=batch_size)).astype(np.float32)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            return cls(z['mean'], z['scale'], z['coef'], z['intercept'],
                       z['feature_indices'], int(z['input_dim']), z['feature_names'].tolist())

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, mean=self.mean, scale=self.scale, coef=self.coef,
                                intercept=self.intercept, feature_indices=self.feature_indices,
                                input_dim=np.asarray(self.input_dim, np.int64),
                                feature_names=np.asarray(self.feature_names, dtype=str),
                                head_names=np.asarray(self.head_names, dtype=str))
        tmp.replace(path)
        info = {'format': 'numpy_dual_sigmoid_linear_v1', 'head_names': list(self.head_names),
                'input_dim': self.input_dim, 'feature_names': self.feature_names,
                'feature_indices': self.feature_indices.tolist(), 'fit_mean': self.mean.tolist(),
                'fit_scale': self.scale.tolist(), 'coef': self.coef.tolist(),
                'intercept': self.intercept.tolist(), 'metadata': metadata or {}}
        tmp_json = path.with_suffix('.json.tmp')
        tmp_json.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
        tmp_json.replace(path.with_suffix('.json'))


def load_calibrator(path):
    return ReliabilityCalibrator.load(path)
