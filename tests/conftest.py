"""Shared fixtures: a small synthetic dataset with the same schema as Kaggle's."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FEATURE_COLUMNS, TARGET_COLUMN, load_params


def make_fake_dataset(n: int = 3000, fraud_ratio: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_ratio).astype(int)
    x = rng.normal(size=(n, len(FEATURE_COLUMNS))).astype("float32")
    x[y == 1, 1:5] += 2.5  # make frauds separable on a few V features
    df = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    df["Time"] = rng.uniform(0, 172_800, n)
    df["Amount"] = rng.exponential(80, n)
    df[TARGET_COLUMN] = y
    return df


@pytest.fixture
def fake_df() -> pd.DataFrame:
    return make_fake_dataset()


@pytest.fixture
def params(tmp_path, fake_df):
    """Project params pointing to a temporary raw CSV / processed dir."""
    p = copy.deepcopy(load_params())
    raw = tmp_path / "raw" / "creditcard.csv"
    raw.parent.mkdir()
    fake_df.to_csv(raw, index=False)
    p["data"]["raw_path"] = str(raw)
    p["data"]["processed_dir"] = str(tmp_path / "processed")
    p["train"]["epochs"] = 3
    p["train"]["batch_size"] = 256
    p["metrics_path"] = str(tmp_path / "metrics.json")
    return p
