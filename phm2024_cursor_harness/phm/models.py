import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .data import ENV
from .features import FeatureBuilder


class JointModel:
    """Physics target reconstruction is a supervised target, never an inference feature.

    Classifier stacking uses group-aware OOF margin predictions on fitting rows.
    Features and auxiliary regression are refitted inside every OOF fold.
    """
    def __init__(self, mode, architecture, cfg, seed=42, keep=None):
        self.mode, self.architecture, self.cfg, self.seed, self.keep = mode, architecture, cfg, seed, keep

    def _new_reg(self):
        if self.architecture == 'linear':
            return make_pipeline(StandardScaler(), Ridge(alpha=10.))
        if self.architecture == 'target_torque':
            return make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                                 StandardScaler(), Ridge(alpha=1e-4))
        return HistGradientBoostingRegressor(max_iter=self.cfg['iterations'],
                max_leaf_nodes=self.cfg['leaves'], learning_rate=.08,
                l2_regularization=1., early_stopping=False, random_state=self.seed)

    def _reg_x(self, x, builder):
        if self.architecture == 'target_torque':
            # Keep physical demand modelling independent of measured torque/engine output.
            return x[ENV]
        f = builder.transform(x)
        return f if self.keep is None else f[self.keep]

    def _target(self, x, y):
        if self.architecture == 'target_torque':
            return np.log(x.trq_measured.to_numpy() / (1 + y.trq_margin.to_numpy() / 100))
        return y.trq_margin.to_numpy()

    def _margin(self, reg, rx, x):
        p = reg.predict(rx)
        if self.architecture == 'target_torque':
            # Positive design torque via log link; exponent clips prevent overflow.
            return 100 * (x.trq_measured.to_numpy() / np.exp(np.clip(p, -10, 10)) - 1)
        return p

    def fit(self, x, y, groups):
        self.builder = FeatureBuilder(self.mode, self.cfg['physics'], self.seed).fit(x)
        nfold = min(self.cfg['stack_folds'], len(np.unique(groups)))
        if nfold < 2:
            raise ValueError('At least two groups required for stacking')
        oof = np.empty(len(x))
        for a, b in GroupKFold(n_splits=nfold).split(x, groups=groups):
            xb, yb = x.iloc[a], y.iloc[a]
            builder = FeatureBuilder(self.mode, self.cfg['physics'], self.seed).fit(xb)
            reg = self._new_reg().fit(self._reg_x(xb, builder), self._target(xb, yb))
            oof[b] = self._margin(reg, self._reg_x(x.iloc[b], builder), x.iloc[b])
        self.oof_residual = y.trq_margin.to_numpy() - oof
        f = self.builder.transform(x)
        if self.keep is not None:
            f = f[self.keep]
        self.base_features = list(f.columns)
        f = f.assign(predicted_margin=oof)
        self.clf = HistGradientBoostingClassifier(max_iter=self.cfg['iterations'],
                max_leaf_nodes=self.cfg['leaves'], learning_rate=.08,
                l2_regularization=1., early_stopping=False, random_state=self.seed)
        self.clf.fit(f, y.faulty)
        self.reg = self._new_reg().fit(self._reg_x(x, self.builder), self._target(x, y))
        # Heteroscedastic Gaussian scale: learn OOF residual magnitude, not fit residuals.
        # Fitting rows receive OOF margins here too.
        self.scale = HistGradientBoostingRegressor(max_iter=max(30, self.cfg['iterations']//2),
                max_leaf_nodes=15, learning_rate=.08, l2_regularization=5.,
                early_stopping=False, random_state=self.seed)
        self.scale.fit(f, np.log(np.maximum(np.abs(self.oof_residual), self.cfg['scale_floor'])))
        return self

    def inference_features(self, x):
        f = self.builder.transform(x)[self.base_features]
        mu = self._margin(self.reg, self._reg_x(x, self.builder), x)
        return f.assign(predicted_margin=mu)

    def predict(self, x):
        f = self.inference_features(x)
        p = self.clf.predict_proba(f)[:, 1]
        sigma = np.maximum(np.exp(np.clip(self.scale.predict(f), -10, 10)) * 1.253314,
                           self.cfg['scale_floor'])
        return p, f.predicted_margin.to_numpy(), sigma


class Calibrator:
    def fit(self, model, x, y):
        p, mu, sigma = model.predict(x)
        # Sigmoid calibration; this data is neither model fitting nor model selection data.
        logits = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))[:, None]
        self.platt = LogisticRegression(C=1e3, max_iter=1000).fit(logits, y.faulty)
        self.scale_factor = max(.05, float(np.sqrt(np.mean(((y.trq_margin.to_numpy()-mu)/sigma)**2))))
        pc = self.platt.predict_proba(logits)[:, 1]
        thresholds = np.linspace(.05, .95, 181)
        # Select F2 threshold for conventional labels; submission action uses expected score.
        from sklearn.metrics import fbeta_score
        self.threshold = float(max(thresholds, key=lambda t: fbeta_score(y.faulty, pc >= t, beta=2)))
        return self

    def predict(self, model, x):
        p, mu, sigma = model.predict(x)
        logits = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))[:, None]
        return self.platt.predict_proba(logits)[:, 1], mu, sigma * self.scale_factor
