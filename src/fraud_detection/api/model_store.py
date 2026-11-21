"""Load the model to serve, from the MLflow registry or from a local directory.

    MODEL_URI=models:/ride-classifier@champion   (default: straight from the registry)
    MODEL_URI=model                              (directory made by `python -m fraud_detection.api.model_store`)

The model is an MLflow pyfunc: it takes a DataFrame (one row per IMU sample, columns ax..rz)
and returns an array of shape (1, 5): the probability of each class of ``schemas.CLASSES``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

META_FILE = "model_meta.json"
MODEL_NAME = os.getenv("MODEL_NAME", "ride-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")


@dataclass
class LoadedModel:
    model: Any  # mlflow.pyfunc.PyFuncModel
    version: str


def default_model_uri() -> str:
    return f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def _registry_version(uri: str) -> str:
    """Version number behind ``models:/name@alias`` or ``models:/name/version``."""
    client = MlflowClient()
    name, _, alias = uri.removeprefix("models:/").partition("@")
    if alias:
        return str(client.get_model_version_by_alias(name, alias).version)
    return name.partition("/")[2]


def load(uri: str) -> LoadedModel:
    model = mlflow.pyfunc.load_model(uri)
    meta_path = Path(uri) / META_FILE
    if uri.startswith("models:/"):
        version = _registry_version(uri)
    elif meta_path.exists():
        version = str(json.loads(meta_path.read_text())["version"])
    else:
        version = uri
    return LoadedModel(model=model, version=version)


def export(uri: str, dest: Path) -> LoadedModel:
    """Download a registered model to ``dest`` so it can be baked into a Docker image."""
    loaded = load(uri)
    if dest.exists():
        shutil.rmtree(dest)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dest))
    (dest / META_FILE).write_text(json.dumps({"version": loaded.version}))
    return loaded


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
    info = export(os.getenv("MODEL_URI", default_model_uri()), out)
    print(f"exported version {info.version} to {out}")
