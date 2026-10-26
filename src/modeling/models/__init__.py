"""Model registry.

Each approach lives in its own module and registers a builder under a name:

    @register("xgb_classifier")
    def build(params, **extra): ...

A training config then picks it with `model.type: xgb_classifier`.
To add an approach: create a module here and import it at the bottom of this file.
"""

MODELS = {}


def register(name):
    def wrap(builder):
        if name in MODELS:
            raise ValueError(f"Model '{name}' is already registered")
        MODELS[name] = builder
        return builder
    return wrap


def build_model(model_cfg, **extra):
    """Build an unfitted model from the `model:` section of a training config."""
    name = model_cfg["type"]
    if name not in MODELS:
        raise KeyError(f"Unknown model type '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name](model_cfg.get("params", {}), **extra)


# Importing the modules runs their @register decorators.
from . import xgb_classifier  # noqa: E402,F401
