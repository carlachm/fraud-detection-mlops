import json
from pathlib import Path

import pytest

from fraud_detection.data import DataValidationError, load_split, prepare, split_data, validate


def test_validate_accepts_valid_data(fake_df):
    validate(fake_df)


def test_validate_rejects_missing_column(fake_df):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(fake_df.drop(columns=["V1"]))


def test_validate_rejects_nulls(fake_df):
    fake_df.loc[0, "Amount"] = None
    with pytest.raises(DataValidationError, match="null"):
        validate(fake_df)


def test_validate_rejects_non_binary_target(fake_df):
    fake_df.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="binary"):
        validate(fake_df)


def test_split_is_stratified_and_disjoint(fake_df):
    train, val, test = split_data(fake_df, test_size=0.15, val_size=0.15, seed=42)
    assert len(train) + len(val) + len(test) == len(fake_df)
    assert set(train.index).isdisjoint(test.index)
    ratio = fake_df["Class"].mean()
    for part in (train, val, test):
        assert abs(part["Class"].mean() - ratio) < 0.02


def test_split_is_reproducible(fake_df):
    a = split_data(fake_df, 0.15, 0.15, seed=1)[2]
    b = split_data(fake_df, 0.15, 0.15, seed=1)[2]
    assert a.index.equals(b.index)


def test_prepare_writes_splits_and_stats(params):
    stats = prepare(params)
    out = Path(params["data"]["processed_dir"])
    for name in ("train", "val", "test"):
        assert (out / f"{name}.parquet").exists()
    assert json.loads((out / "data_stats.json").read_text()) == stats
    x, y = load_split("train", params)
    assert x.shape[1] == 30 and len(x) == len(y)
