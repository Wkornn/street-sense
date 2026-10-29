"""EXPERIMENTAL - V4 "spatial pattern inference", ported from the korn-backup branch.

Idea: learn what risky roads look like ONLY from roads that had accidents, then
apply that pattern to every road.
  1. Training pool = training-split segments with y > 0 (they had at least one event)
  2. Fuzzy C-means groups the pool into k risk grades using surrogate traffic metrics
     (low speed, congestion) and orders the grades from safest to riskiest
  3. XGBoost learns to predict the grade from road features
  4. Risk score for any road = expected grade / (k - 1), a 0-1 score

Measured on the shared test set it ranks roads worse than random (ROC-AUC < 0.5) and its
score is not a probability. It is kept to document the approach, not for the app.

Differences from the original script:
  - No accident-history columns (acc_*) as inputs or clustering metrics (they leaked the answer)
  - RFECV feature selection dropped (slow); RENN and Optuna are optional and off by default
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import register


class FCMGradeModel:
    """XGBoost multiclass model predicting the FCM risk grade (0 = safest ... k-1 = riskiest)."""

    def __init__(self, booster, n_grades):
        self.booster = booster
        self.n_grades = int(n_grades)

    def predict_grade_proba(self, X):
        return self.booster.predict_proba(X)

    def predict_risk(self, X):
        """Expected grade scaled to 0-1. NOT a probability of an accident."""
        proba = self.predict_grade_proba(X)
        return proba @ np.arange(self.n_grades) / (self.n_grades - 1)

    def explain(self, X):
        """SHAP contributions (log-odds) toward the riskiest grade, computed by XGBoost."""
        contrib = self.booster.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, -1, :-1]  # (rows, grades, features + bias) -> riskiest grade, no bias


def fcm_grades(Z, n_grades, seed):
    """Fuzzy C-means on standardized metrics; grades ordered by centroid riskiness."""
    import skfuzzy as fuzz  # imported here so the registry works without scikit-fuzzy

    centers, membership, *_ = fuzz.cluster.cmeans(
        Z.T, c=n_grades, m=2.0, error=0.005, maxiter=1000, seed=seed
    )
    labels = membership.argmax(axis=0)
    order = np.argsort(centers.sum(axis=1))            # higher sum = riskier (speed is inverted)
    rank = {cluster: grade for grade, cluster in enumerate(order)}
    return np.array([rank[c] for c in labels])


@register("fcm_xgb")
class FCMXGBPlugin:
    @staticmethod
    def fit(model_cfg, X_train, y_train, X_cal, y_cal):
        params = dict(model_cfg.get("params", {}))
        n_grades = params.pop("n_grades", 4)
        surrogates = params.pop("surrogate_cols")
        invert = params.pop("invert_cols", [])         # columns where LOWER means riskier
        use_renn = params.pop("renn", False)
        n_trials = params.pop("optuna_trials", 0)
        seed = params.get("random_state", 42)

        # 1. Training pool: only segments that had at least one event
        pool = X_train[np.asarray(y_train) > 0]

        # 2. Unsupervised risk grades from surrogate metrics
        Z = StandardScaler().fit_transform(pool[surrogates])
        for col in invert:
            Z[:, surrogates.index(col)] *= -1
        grades = fcm_grades(Z, n_grades, seed)
        print(f"  FCM grades on {len(pool):,} event segments: {np.bincount(grades).tolist()}")

        X_fit, y_fit = pool, grades

        # 3. Optional: RENN under-sampling (cleans class boundaries)
        if use_renn:
            from imblearn.under_sampling import RepeatedEditedNearestNeighbours
            X_fit, y_fit = RepeatedEditedNearestNeighbours().fit_resample(X_fit, y_fit)

        # 4. Optional: Optuna tuning, scored on a hold-out slice of the pool
        if n_trials:
            params.update(_tune(X_fit, y_fit, n_trials, seed))

        booster = XGBClassifier(**params, objective="multi:softprob", eval_metric="mlogloss")
        booster.fit(X_fit, y_fit)
        return FCMGradeModel(booster, n_grades)

    @staticmethod
    def save(model, out_dir):
        out_dir = Path(out_dir)
        model.booster.save_model(out_dir / "booster.json")
        (out_dir / "grades.json").write_text(json.dumps({"n_grades": model.n_grades}), encoding="utf-8")

    @staticmethod
    def load(model_dir):
        model_dir = Path(model_dir)
        booster = XGBClassifier()
        booster.load_model(model_dir / "booster.json")
        n_grades = json.loads((model_dir / "grades.json").read_text(encoding="utf-8"))["n_grades"]
        return FCMGradeModel(booster, n_grades)


def _tune(X, y, n_trials, seed):
    """Optuna search over XGBoost settings (accuracy on a 25% hold-out of the pool)."""
    import optuna
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        m = XGBClassifier(**p, random_state=seed, objective="multi:softprob").fit(X_tr, y_tr)
        return accuracy_score(y_va, m.predict(X_va))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
