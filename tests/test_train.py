"""Integration test: full training run against a throw-away local MLflow store."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.data import prepare
from fraud_detection.train import run


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


def test_training_run_logs_and_registers_model(params, local_mlflow):
    prepare(params)
    summary = run(params)

    assert Path(params["metrics_path"]).exists()
    assert json.loads(Path(params["metrics_path"]).read_text())["run_id"] == summary["run_id"]
    assert summary["test_pr_auc"] > 0.5  # synthetic data is easy
    assert summary["promoted"] is True  # first model always becomes champion

    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]
    champion = client.get_model_version_by_alias(name, params["mlflow"]["champion_alias"])
    assert str(champion.version) == summary["model_version"]
    assert "threshold" in champion.tags

    # The registered model can be loaded with the generic pyfunc API (what the API will do)
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    x = pd.DataFrame(np.random.randn(3, 30).astype(np.float32), columns=FEATURE_COLUMNS)
    preds = np.asarray(model.predict(x)).ravel()
    assert preds.shape == (3,) and ((preds >= 0) & (preds <= 1)).all()


def test_worse_model_is_not_promoted(params, local_mlflow):
    prepare(params)
    first = run(params)
    params["train"]["epochs"] = 1
    params["train"]["learning_rate"] = 1e-7  # basically untrained
    second = run(params)
    if second["test_pr_auc"] <= first["test_pr_auc"]:
        assert second["promoted"] is False


def test_quality_gate_blocks_bad_model(params, local_mlflow):
    prepare(params)
    params["quality_gate"] = {"min_test_pr_auc": 1.01}  # impossible bar
    summary = run(params)
    assert summary["quality_gate_passed"] is False
    assert summary["promoted"] is False
    client = MlflowClient()
    version = client.get_model_version(params["mlflow"]["registered_model_name"], summary["model_version"])
    assert version.tags["promotion"] == "gate_failed"
