"""Data extraction, validation and preparation.

Usage:
    python -m fraud_detection.data download   # Kaggle -> data/raw/creditcard.csv
    python -m fraud_detection.data prepare    # raw -> train/val/test parquet
    python -m fraud_detection.data sample     # synthetic data (CI, smoke tests, demos)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_params,
)

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def download_raw_data(dest: Path) -> Path:
    """Download the Kaggle dataset with ``kagglehub``.

    Needs Kaggle credentials: either ``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.
    """
    import kagglehub  # imported lazily: only needed for this step

    dest.parent.mkdir(parents=True, exist_ok=True)
    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    shutil.copy(download_dir / "creditcard.csv", dest)
    logger.info("Raw data saved to %s", dest)
    return dest


def synthetic_dataset(n_rows: int = 3000, fraud_ratio: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Fake dataset with the exact Kaggle schema.

    Used by the tests and the CI smoke test, so they never need Kaggle
    credentials nor the real 150 MB file.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < fraud_ratio).astype(int)
    x = rng.normal(size=(n_rows, len(FEATURE_COLUMNS))).astype("float32")
    x[y == 1, 1:5] += 2.5  # frauds are separable on a few V features
    df = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    df["Time"] = rng.uniform(0, 172_800, n_rows)
    df["Amount"] = rng.exponential(80, n_rows)
    df[TARGET_COLUMN] = y
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate(df: pd.DataFrame) -> None:
    """Basic data quality checks (schema, nulls, label values)."""
    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = set(expected) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")
    if df[expected].isnull().any().any():
        raise DataValidationError("Dataset contains null values")
    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary (0/1)")
    if (df["Amount"] < 0).any():
        raise DataValidationError("Amount must be >= 0")


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (keeps the fraud ratio)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return train, val, test


def prepare(params: dict) -> dict:
    """Load raw CSV, validate, deduplicate, split and save as parquet."""
    cfg = params["data"]
    raw_path = PROJECT_ROOT / cfg["raw_path"]
    out_dir = PROJECT_ROOT / cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    validate(df)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    train, val, test = split_data(df, cfg["test_size"], cfg["val_size"], params["seed"])
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)

    stats = {
        "n_rows_raw": n_before,
        "n_duplicates_removed": n_before - len(df),
        "fraud_ratio": float(df[TARGET_COLUMN].mean()),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Prepared data: %s", stats)
    return stats


def load_split(name: str, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load one prepared split and return (X, y)."""
    path = PROJECT_ROOT / params["data"]["processed_dir"] / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["download", "prepare", "sample"])
    parser.add_argument("--rows", type=int, default=3000, help="number of rows for `sample`")
    args = parser.parse_args()

    params = load_params()
    raw_path = PROJECT_ROOT / params["data"]["raw_path"]
    if args.step == "download":
        download_raw_data(raw_path)
    elif args.step == "sample":
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_dataset(args.rows).to_csv(raw_path, index=False)
        logger.info("Synthetic dataset (%d rows) written to %s", args.rows, raw_path)
    else:
        prepare(params)


if __name__ == "__main__":
    main()
