"""Negative Binomial count model (GLM) with a road-length exposure offset.

The classic road-safety approach (Highway Safety Manual): predict the expected
NUMBER of accidents per segment. Accident counts are overdispersed (variance >> mean),
which is why Negative Binomial is used instead of Poisson.

Saved as plain JSON (coefficients + scaling), so loading needs neither pickle nor statsmodels.
"""

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from . import register


class NegBinomialRiskModel:
    """A fitted NB GLM: log(mu) = intercept + coef . scaled(x) + log(offset / offset_unit)."""

    def __init__(self, features, intercept, coef, mean, scale, alpha, offset_col, offset_unit):
        self.features = list(features)          # regressors (offset column excluded)
        self.intercept = float(intercept)
        self.coef = np.asarray(coef, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.alpha = float(alpha)               # NB dispersion: Var = mu + alpha * mu^2
        self.offset_col = offset_col
        self.offset_unit = float(offset_unit)

    def _scaled(self, X):
        return (X[self.features].to_numpy(dtype=float) - self.mean) / self.scale

    def _log_offset(self, X):
        if self.offset_col is None:
            return 0.0
        return np.log(X[self.offset_col].to_numpy(dtype=float).clip(min=1) / self.offset_unit)

    def predict_count(self, X):
        """Expected number of events per segment."""
        return np.exp(self.intercept + self._scaled(X) @ self.coef + self._log_offset(X))

    def predict_risk(self, X):
        """Probability of at least one event: P(Y >= 1) = 1 - (1 + alpha*mu)^(-1/alpha)."""
        mu = self.predict_count(X)
        return 1.0 - (1.0 + self.alpha * mu) ** (-1.0 / self.alpha)

    def explain(self, X):
        """Per-feature contributions to log(expected count), relative to an average segment.

        For a linear model, coef * (x - mean) is the exact SHAP value. Columns follow X;
        the exposure offset (road length) is not a regressor, so its column is 0.
        """
        contrib = dict(zip(self.features, (self._scaled(X) * self.coef).T))
        zeros = np.zeros(len(X))
        return np.column_stack([contrib.get(c, zeros) for c in X.columns])

    def to_dict(self):
        return {
            "features": self.features,
            "intercept": self.intercept,
            "coef": self.coef.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "alpha": self.alpha,
            "offset_col": self.offset_col,
            "offset_unit": self.offset_unit,
        }


def estimate_alpha(y, mu):
    """Cameron & Trivedi auxiliary regression: ((y - mu)^2 - y) / mu = alpha * mu."""
    z = ((y - mu) ** 2 - y) / mu
    return float(max((z * mu).sum() / (mu ** 2).sum(), 1e-6))


@register("neg_binomial")
class NegBinomialPlugin:
    @staticmethod
    def fit(model_cfg, X_train, y_train, X_cal, y_cal):
        # X_cal / y_cal are not needed: a count model gives probabilities directly.
        params = model_cfg.get("params", {})
        offset_col = params.get("offset_col")
        offset_unit = params.get("offset_unit", 100.0)
        features = [c for c in X_train.columns if c != offset_col]

        # 1. Standardize regressors (numerical stability; also makes coefficients comparable)
        Xf = X_train[features].to_numpy(dtype=float)
        mean, scale = Xf.mean(axis=0), Xf.std(axis=0)
        scale[scale == 0] = 1.0
        exog = sm.add_constant((Xf - mean) / scale, has_constant="add")
        y = np.asarray(y_train, dtype=float)
        offset = (np.log(X_train[offset_col].to_numpy(dtype=float).clip(min=1) / offset_unit)
                  if offset_col else None)

        # 2. Dispersion alpha: from a Poisson fit (or fixed in the config)
        alpha = params.get("alpha", "auto")
        if alpha == "auto":
            poisson = sm.GLM(y, exog, family=sm.families.Poisson(), offset=offset).fit()
            alpha = estimate_alpha(y, poisson.mu)

        # 3. Negative Binomial GLM
        res = sm.GLM(y, exog, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
        return NegBinomialRiskModel(features, res.params[0], res.params[1:], mean, scale,
                                    alpha, offset_col, offset_unit)

    @staticmethod
    def save(model, out_dir):
        (Path(out_dir) / "glm.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(model_dir):
        d = json.loads((Path(model_dir) / "glm.json").read_text(encoding="utf-8"))
        return NegBinomialRiskModel(**d)
