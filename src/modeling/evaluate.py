"""Shared evaluation. Every approach is scored with the same functions on the same test split."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


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
