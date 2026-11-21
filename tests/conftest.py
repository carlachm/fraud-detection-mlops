"""Shared fixtures: a small synthetic dataset with the same schema as Kaggle's."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from fraud_detection.config import load_params
from fraud_detection.data import synthetic_dataset


@pytest.fixture
def fake_df() -> pd.DataFrame:
    return synthetic_dataset()


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
