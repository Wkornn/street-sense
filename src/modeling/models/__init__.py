"""Model registry.

Each approach lives in its own module and registers a plugin class under a name.
The plugin owns everything specific to its model type (imbalance handling,
calibration, file format). train.py only calls these three methods:

    @register("xgb_classifier")
    class XGBClassifierPlugin:
        @staticmethod
        def fit(model_cfg, X_train, y_train, X_cal, y_cal): ...  # return a fitted model with predict_risk(X)
        @staticmethod
        def save(model, out_dir): ...                            # write model files into out_dir
        @staticmethod
        def load(model_dir): ...                                 # return a model with predict_risk(X)

A training config then picks it with `model.type: xgb_classifier`.
To add an approach: create a module here and import it at the bottom of this file.
"""

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


# Importing the modules runs their @register decorators.
from . import xgb_classifier  # noqa: E402,F401
