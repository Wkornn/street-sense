"""Shared data steps for every training approach: load, target, exposure, features, split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


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


def spatial_groups(segment_ids, segments_path, cell_size_m):
    """Grid cell of each segment, from its centroid: a label like '331_6851' per segment.

    Segments in the same cell end up in the same part of a spatial split.
    """
    import geopandas as gpd  # only needed for spatial splits

    segments = gpd.read_file(segments_path, columns=["segment_id"])
    centroids = segments.geometry.centroid  # projected CRS (metres)
    cells = (
        (centroids.x // cell_size_m).astype(int).astype(str)
        + "_"
        + (centroids.y // cell_size_m).astype(int).astype(str)
    )
    lookup = pd.Series(cells.to_numpy(), index=segments["segment_id"])
    missing = ~pd.Index(segment_ids).isin(lookup.index)
    if missing.any():
        raise KeyError(f"{missing.sum()} segment IDs not found in {segments_path}")
    return pd.Series(lookup.loc[segment_ids].to_numpy(), index=pd.Index(segment_ids).rename(None))


def split(X, y, split_cfg, groups=None):
    """Split into train / calibration / test.

    random:  segments are assigned individually, stratified on "any event" (y > 0).
    spatial: whole grid cells (`groups`, one label per row) are assigned, so a test area
             has no neighbours in training. StratifiedGroupKFold keeps the share of
             event segments similar in every part.
    """
    method = split_cfg.get("method", "random")
    cal, test, seed = split_cfg["cal_size"], split_cfg["test_size"], split_cfg["seed"]

    if method == "random":
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=cal + test, stratify=(y > 0), random_state=seed
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_rest, y_rest, test_size=test / (cal + test), stratify=(y_rest > 0), random_state=seed
        )
        return X_train, X_cal, X_test, y_train, y_cal, y_test

    if method == "spatial":
        if groups is None:
            raise ValueError("A spatial split needs `groups` (the grid cell of each row)")
        n_folds = round(1 / test)
        if abs(cal - test) > 1e-9 or abs(n_folds * test - 1) > 1e-6:
            raise ValueError("Spatial split needs cal_size == test_size == 1/k (e.g. 0.2)")
        folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        parts = list(folds.split(X, (y > 0).astype(int), np.asarray(groups)))
        test_pos, cal_pos = parts[0][1], parts[1][1]
        is_train = np.ones(len(X), dtype=bool)
        is_train[np.r_[test_pos, cal_pos]] = False
        return (
            X.iloc[is_train], X.iloc[cal_pos], X.iloc[test_pos],
            y.iloc[is_train], y.iloc[cal_pos], y.iloc[test_pos],
        )

    raise ValueError(f"Unknown split method '{method}' (use random or spatial)")


def describe_split(parts, groups=None):
    """Size, event count and event rate of each part (and grid cells for spatial splits).

    parts: {"train": (X_train, y_train), "cal": (...), "test": (...)}
    """
    summary = {}
    for name, (X_part, y_part) in parts.items():
        row = {
            "segments": len(y_part),
            "events": int((y_part > 0).sum()),
            "event_rate": round(float((y_part > 0).mean()), 5),
        }
        if groups is not None:
            row["cells"] = int(groups.loc[X_part.index].nunique())
        summary[name] = row
    return summary
