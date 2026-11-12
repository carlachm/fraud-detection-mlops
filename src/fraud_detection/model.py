"""PyTorch model: a simple MLP for tabular binary classification.

Feature standardisation is stored *inside* the model (as buffers), so the
saved model is self-contained: the API only has to send raw features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FraudMLP(nn.Module):
    """Multi-layer perceptron returning logits (use ``predict_proba`` for probas)."""

    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def fit_normalizer(self, x: np.ndarray) -> None:
        """Compute mean / std on the TRAINING set only (no data leakage)."""
        self.mean.copy_(torch.as_tensor(x.mean(axis=0), dtype=torch.float32))
        std = torch.as_tensor(x.std(axis=0), dtype=torch.float32)
        self.std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return torch.sigmoid(self(x)).cpu().numpy()


class FraudProbaModel(nn.Module):
    """Wrapper logged to MLflow: outputs the fraud *probability* directly."""

    def __init__(self, model: FraudMLP):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))
