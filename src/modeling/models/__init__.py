"""Model registry.

Each approach lives in its own module and registers a plugin class under a name.
The plugin owns everything specific to its model type (imbalance handling,
calibration, file format). train.py only calls these three methods:

    @register("xgb_classifier")
    class XGBClassifierPlugin:
        @staticmethod
        def fit(model_cfg, X_train, y_train, X_cal, y_cal): ...  # return a fitted model
        @staticmethod
        def save(model, out_dir): ...                            # write model files into out_dir
        @staticmethod
        def load(model_dir): ...                                 # return a fitted model

Every fitted model must provide:
    predict_risk(X)  -> probability of >= 1 event per row (0-1); used for the map and metrics
    explain(X)       -> per-feature contributions, shape (rows, features); used for "Top Factors"
Count models also provide:
    predict_count(X) -> expected number of events per row

A training config then picks it with `model.type: xgb_classifier`.
Anything that uses a trained model (app, explainer) calls `load_model(model_dir)`
and never needs to know which plugin made it.
To add an approach: create a module here and import it at the bottom of this file.
"""

import json
from pathlib import Path

MODELS = {}


def register(name):
    def wrap(plugin):
        if name in MODELS:
            raise ValueError(f"Model '{name}' is already registered")
        MODELS[name] = plugin
        return plugin
    return wrap


def get_plugin(name):
    """Look up a registered plugin by the name used in configs and meta.json."""
    if name not in MODELS:
        raise KeyError(f"Unknown model type '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name]


def load_model(model_dir):
    """Load a trained model folder, using the plugin named in its meta.json."""
    meta = json.loads((Path(model_dir) / "meta.json").read_text(encoding="utf-8"))
    return get_plugin(meta["model_type"]).load(model_dir)


# Importing the modules runs their @register decorators.
from . import fcm_xgb  # noqa: E402,F401  (experimental)
from . import neg_binomial  # noqa: E402,F401
from . import xgb_classifier  # noqa: E402,F401
