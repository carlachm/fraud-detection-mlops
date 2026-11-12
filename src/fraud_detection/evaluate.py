"""Evaluation metrics adapted to a highly imbalanced problem (~0.17% fraud).

Accuracy is meaningless here (predicting "never fraud" gives 99.8%), so we
focus on PR-AUC, recall, precision and F1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Decision threshold maximising F1 (computed on the validation set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more element than thresholds
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.5


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_plots(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, out_dir: Path) -> list[Path]:
    """Save PR curve + confusion matrix (logged as MLflow artifacts)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test)")
    ax.grid(alpha=0.3)
    pr_path = out_dir / "pr_curve.png"
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)

    y_pred = (y_proba >= threshold).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]), display_labels=["legit", "fraud"])
    disp.plot(values_format="d")
    disp.ax_.set_title(f"Confusion matrix (threshold={threshold:.3f})")
    cm_path = out_dir / "confusion_matrix.png"
    disp.figure_.savefig(cm_path, bbox_inches="tight")
    plt.close(disp.figure_)

    return [pr_path, cm_path]
