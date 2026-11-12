"""Train the fraud detection model and track everything with MLflow.

Usage:
    python -m fraud_detection.train

Env vars (see .env.example):
    MLFLOW_TRACKING_URI   e.g. http://localhost:5000 (default: local ./mlruns)
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_detection.config import PROJECT_ROOT, load_params, set_seed
from fraud_detection.data import load_split
from fraud_detection.evaluate import best_f1_threshold, compute_metrics, save_plots
from fraud_detection.model import FraudMLP, FraudProbaModel
from fraud_detection.registry import promote_if_better

logger = logging.getLogger(__name__)


def _flatten(d: dict, prefix: str = "") -> dict:
    """{'train': {'lr': 1}} -> {'train.lr': 1} (for mlflow.log_params)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _compute_pos_weight(y: np.ndarray, value) -> float:
    if value == "auto":
        return float((y == 0).sum() / max((y == 1).sum(), 1))
    return float(value)


def train_model(
    model: FraudMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> FraudMLP:
    """Training loop with early stopping on validation PR-AUC."""
    pos_weight = _compute_pos_weight(y_train, cfg["pos_weight"])
    mlflow.log_param("train.pos_weight_value", round(pos_weight, 2))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )

    best_score, best_state, patience = -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        val_metrics = compute_metrics(y_val, model.predict_proba(x_val))
        mlflow.log_metrics(
            {
                "train_loss": total_loss / len(x_train),
                "val_pr_auc_epoch": val_metrics["pr_auc"],
                "val_roc_auc_epoch": val_metrics["roc_auc"],
            },
            step=epoch,
        )
        logger.info("epoch %d - loss %.4f - val_pr_auc %.4f", epoch, total_loss / len(x_train), val_metrics["pr_auc"])

        if val_metrics["pr_auc"] > best_score:
            best_score, patience = val_metrics["pr_auc"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    mlflow.log_metric("best_epoch_val_pr_auc", best_score)
    return model


def run(params: dict | None = None) -> dict:
    """Full training run: data -> train -> evaluate -> log -> register."""
    params = params or load_params()
    set_seed(params["seed"])
    mlf = params["mlflow"]
    mlflow.set_experiment(mlf["experiment_name"])

    x_train_df, y_train = load_split("train", params)
    x_val_df, y_val = load_split("val", params)
    x_test_df, y_test = load_split("test", params)
    x_train, x_val, x_test = (d.to_numpy(np.float32) for d in (x_train_df, x_val_df, x_test_df))
    y_train, y_val, y_test = (s.to_numpy() for s in (y_train, y_val, y_test))

    with mlflow.start_run() as active_run:
        mlflow.set_tags({"model_type": "pytorch-mlp", "task": "binary-classification"})
        mlflow.log_params(_flatten({k: params[k] for k in ("seed", "model", "train", "data")}))
        mlflow.log_params({"n_train": len(x_train), "n_features": x_train.shape[1]})

        stats_path = PROJECT_ROOT / params["data"]["processed_dir"] / "data_stats.json"
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path), artifact_path="data")

        model = FraudMLP(x_train.shape[1], params["model"]["hidden_dims"], params["model"]["dropout"])
        model.fit_normalizer(x_train)
        model = train_model(model, x_train, y_train, x_val, y_val, params["train"])

        # Threshold chosen on validation, metrics reported on the untouched test set
        threshold = best_f1_threshold(y_val, model.predict_proba(x_val))
        val_metrics = compute_metrics(y_val, model.predict_proba(x_val), threshold)
        test_proba = model.predict_proba(x_test)
        test_metrics = compute_metrics(y_test, test_proba, threshold)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            for p in save_plots(y_test, test_proba, threshold, Path(tmp)):
                mlflow.log_artifact(str(p), artifact_path="plots")

        # Log + register the model (probability output, normalisation included)
        export = FraudProbaModel(model).eval()
        example = x_test_df.head(5).astype(np.float32)
        signature = infer_signature(example, export(torch.tensor(example.to_numpy())).detach().numpy())
        model_info = mlflow.pytorch.log_model(
            export,
            name="model",
            signature=signature,
            input_example=example,
            # pickle + code_paths: the model ships with the fraud_detection code it needs,
            # so it can be loaded anywhere (API container, K8s pod) without installing this repo.
            serialization_format="pickle",
            code_paths=[str(Path(__file__).parent)],
            registered_model_name=mlf["registered_model_name"],
        )

        client = MlflowClient()
        version = str(model_info.registered_model_version)
        client.set_model_version_tag(mlf["registered_model_name"], version, "threshold", f"{threshold:.6f}")
        promoted = promote_if_better(
            client, mlf["registered_model_name"], version, test_metrics["pr_auc"], alias=mlf["champion_alias"]
        )

        summary = {
            "run_id": active_run.info.run_id,
            "model_version": version,
            "promoted": promoted,
            "threshold": threshold,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

    # metrics.json is tracked by DVC (dvc metrics show / dvc metrics diff)
    (PROJECT_ROOT / params.get("metrics_path", "metrics.json")).write_text(json.dumps(summary, indent=2))
    logger.info("Run summary: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
