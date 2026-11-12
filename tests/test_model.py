import numpy as np
import torch

from fraud_detection.evaluate import best_f1_threshold, compute_metrics
from fraud_detection.model import FraudMLP, FraudProbaModel


def test_forward_shape():
    model = FraudMLP(30, [16, 8])
    model.eval()
    out = model(torch.randn(10, 30))
    assert out.shape == (10,)


def test_predict_proba_in_unit_interval():
    model = FraudMLP(30, [16])
    proba = model.predict_proba(np.random.randn(20, 30))
    assert proba.shape == (20,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_normalizer_is_stored_in_state_dict():
    model = FraudMLP(3, [4])
    x = np.array([[1.0, 10.0, 5.0], [3.0, 30.0, 5.0]], dtype=np.float32)
    model.fit_normalizer(x)
    state = model.state_dict()
    assert torch.allclose(state["mean"], torch.tensor([2.0, 20.0, 5.0]))
    # constant column -> std replaced by 1 (no division by zero)
    assert state["std"][2] == 1.0


def test_proba_wrapper_matches_sigmoid():
    model = FraudMLP(30, [8]).eval()
    x = torch.randn(4, 30)
    assert torch.allclose(FraudProbaModel(model)(x), torch.sigmoid(model(x)))


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["pr_auc"] == 1.0 and m["recall"] == 1.0 and m["fp"] == 0


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.7
