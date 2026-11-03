"""Shared test fixtures: small synthetic data, so tests run in seconds without real data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_dataset():
    """2,000 fake road segments with a learnable risk pattern (fast, major roads are riskier)."""
    rng = np.random.default_rng(0)
    n = 2000
    highway_rank = rng.integers(1, 6, n)
    speed = rng.normal(40, 10, n) + 5 * highway_rank
    df = pd.DataFrame({
        "segment_id": np.arange(n),
        "highway_rank": highway_rank,
        "length_m": rng.uniform(20, 100, n),
        "speed_mean": speed,
        "congestion_score": rng.uniform(0, 50, n),
        "speed_drop_morning": rng.normal(5, 3, n),
        "lanes": rng.integers(1, 5, n),
        "probe_count": rng.integers(0, 500, n).astype(float),
    })
    rate = np.exp(-6 + 0.6 * highway_rank + 0.03 * (speed - 50))
    df["acc_fatal"] = rng.poisson(rate * 0.3)
    df["acc_serious"] = rng.poisson(rate)
    return df


@pytest.fixture
def toy_features():
    return ["highway_rank", "length_m", "speed_mean", "congestion_score", "speed_drop_morning", "lanes"]
