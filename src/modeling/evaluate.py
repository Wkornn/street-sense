"""Shared evaluation. Every approach is scored with the same functions on the same test split."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    roc_auc_score,
)


def top_decile_lift(y_true, y_score):
    """Positive rate among the 10% highest-scored segments divided by the overall rate."""
    y_true = np.asarray(y_true)
    k = max(1, int(len(y_true) * 0.1))
    top = np.argsort(-np.asarray(y_score), kind="stable")[:k]
    return float(y_true[top].mean() / y_true.mean())


def classification_metrics(y_true, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "top_decile_lift": top_decile_lift(y_true, y_prob),
    }


def count_metrics(y_true, y_count):
    return {
        "mae": float(mean_absolute_error(y_true, y_count)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_count))),
        "poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_count, 1e-9, None))),
    }


def evaluate(target_type, y_true, model, X):
    """Metrics for one model on one split.

    Every model is scored on "any event" (y > 0) vs predict_risk(), so approaches with
    different targets stay comparable. Count models also get count metrics.
    """
    metrics = classification_metrics((np.asarray(y_true) > 0).astype(int), model.predict_risk(X))
    if target_type == "count":
        metrics.update(count_metrics(y_true, model.predict_count(X)))
    return metrics
