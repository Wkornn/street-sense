"""Shared data steps for every training approach: load, target, exposure, features, split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path):
    return pd.read_parquet(path)


def make_target(df, target_cfg):
    """Build the target from the `target:` section of a training config.

    binary: 1 if any of the columns is > 0 (classifiers)
    count:  sum of the columns, e.g. number of severe accidents (count models)
    """
    kind = target_cfg.get("type", "binary")
    cols = df[target_cfg["columns"]]
    if kind == "binary":
        return (cols > 0).any(axis=1).astype(int)
    if kind == "count":
        return cols.sum(axis=1).astype(int)
    raise ValueError(f"Unknown target type '{kind}' (use binary or count)")


def add_exposure(df):
    """Traffic exposure = how much driving happens on the segment (probe volume x length)."""
    df = df.copy()
    df["exposure"] = df["probe_count"].fillna(0) * df["length_m"]
    df["log_exposure"] = np.log1p(df["exposure"])
    return df


def select_features(df, features):
    """Return the feature matrix. Fails loudly if a configured column is missing."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[features].fillna(0)


def split(X, y, split_cfg):
    """Random split into train / calibration / test.

    Stratified on "any event" (y > 0), so rare positives are spread evenly.
    For a binary target this is the same as stratifying on y itself.
    """
    if split_cfg.get("method", "random") != "random":
        raise NotImplementedError(f"Split method '{split_cfg['method']}' not implemented yet")

    cal, test, seed = split_cfg["cal_size"], split_cfg["test_size"], split_cfg["seed"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=cal + test, stratify=(y > 0), random_state=seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=test / (cal + test), stratify=(y_rest > 0), random_state=seed
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test
