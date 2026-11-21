"""PLACEHOLDER model with the final interface, so the API / web app / K8s can be built before training.

It is NOT a trained classifier: a fixed random linear map of the per-channel standard deviations,
followed by a softmax. It is registered like the real model will be (same name, same alias,
same input and output), so replacing it is just a matter of registering a better version.

    uv run python -m fraud_detection.api.dummy_model      # needs MLFLOW_TRACKING_URI
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from fraud_detection.api.features import ride_summary
from fraud_detection.api.model_store import MODEL_ALIAS, MODEL_NAME
from fraud_detection.api.schemas import CHANNELS, CLASSES

EXPERIMENT = "ride-classifier-placeholder"


class DummyRideClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, seed: int = 0):
        self.weights = np.random.default_rng(seed).normal(size=(len(CLASSES), len(CHANNELS)))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        x = model_input[CHANNELS].to_numpy(dtype=float)
        summary = ride_summary(x)
        stds = np.array([summary[f"{c}_std"] for c in CHANNELS])
        logits = self.weights @ np.log1p(stds)
        proba = np.exp(logits - logits.max())
        return (proba / proba.sum())[None, :]


def register() -> str:
    """Log the placeholder, register it and give it the serving alias. Returns the version."""
    example = pd.DataFrame(np.random.default_rng(1).normal(size=(100, len(CHANNELS))), columns=CHANNELS)
    mlflow.set_experiment(EXPERIMENT)  # explicit: do not depend on an experiment left active elsewhere
    with mlflow.start_run(run_name="dummy-ride-classifier"):
        mlflow.set_tag("dummy", "true")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DummyRideClassifier(),
            input_example=example,
            registered_model_name=MODEL_NAME,
        )
    version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    return version


if __name__ == "__main__":
    print(f"registered {MODEL_NAME} v{register()} @{MODEL_ALIAS}")
