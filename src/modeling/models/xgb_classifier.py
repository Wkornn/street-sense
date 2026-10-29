"""XGBoost binary classifier with optional isotonic calibration."""

from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from . import register


class XGBRiskModel:
    """A fitted XGBoost model plus its calibrator, kept as two separate parts.

    booster:    the XGBClassifier (SHAP explains this part)
    calibrator: maps the booster's raw score to a real probability (or None)
    """

    def __init__(self, booster, calibrator=None):
        self.booster = booster
        self.calibrator = calibrator

    def predict_raw(self, X):
        """Uncalibrated score from XGBoost."""
        return self.booster.predict_proba(X)[:, 1]

    def predict_risk(self, X):
        """Calibrated probability of a fatal/serious accident (0-1)."""
        raw = self.predict_raw(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def explain(self, X):
        """Per-feature contributions (SHAP values, log-odds), one row per sample.

        Computed by XGBoost itself (exact TreeSHAP). They explain the booster;
        calibration only re-scales its output, so the ranking of factors is the same.
        """
        contrib = self.booster.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, :-1]  # last column is the bias (base value)


@register("xgb_classifier")
class XGBClassifierPlugin:
    @staticmethod
    def fit(model_cfg, X_train, y_train, X_cal, y_cal):
        # 1. Imbalance: weight rare positives by the negative/positive ratio of the training split
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # 2. Train XGBoost on the training split
        booster = XGBClassifier(
            **model_cfg.get("params", {}),
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
        )
        booster.fit(X_train, y_train)

        # 3. Calibrate on the held-out calibration split (booster stays fixed)
        method = model_cfg.get("calibration")
        if method is None:
            calibrator = None
        elif method == "isotonic":
            raw_cal = booster.predict_proba(X_cal)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)
        else:
            raise ValueError(f"Unsupported calibration '{method}' (use isotonic or null)")

        return XGBRiskModel(booster, calibrator)

    @staticmethod
    def save(model, out_dir):
        out_dir = Path(out_dir)
        # XGBoost's own format: stable across versions, no pickle
        model.booster.save_model(out_dir / "booster.json")
        # The calibrator is a small scikit-learn object -> joblib
        if model.calibrator is not None:
            joblib.dump(model.calibrator, out_dir / "calibrator.joblib")

    @staticmethod
    def load(model_dir):
        model_dir = Path(model_dir)
        booster = XGBClassifier()
        booster.load_model(model_dir / "booster.json")
        cal_path = model_dir / "calibrator.joblib"
        calibrator = joblib.load(cal_path) if cal_path.exists() else None
        return XGBRiskModel(booster, calibrator)
