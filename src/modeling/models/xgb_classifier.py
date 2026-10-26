"""XGBoost binary classifier."""

from xgboost import XGBClassifier

from . import register


@register("xgb_classifier")
def build(params, scale_pos_weight=1.0):
    # scale_pos_weight balances rare positives (computed from the training split)
    return XGBClassifier(**params, scale_pos_weight=scale_pos_weight, eval_metric="logloss")
