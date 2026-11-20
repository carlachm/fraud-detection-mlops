"""Load test of the API (S4 optional, S5 sizing report).

    uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 1m \
        --csv reports/load_local

RIDE_SAMPLES (default 3000, like a real ride of ~5 min) sets the payload size.
"""

import os

import numpy as np
from locust import HttpUser, between, task

from fraud_detection.api.schemas import CHANNELS

N = int(os.getenv("RIDE_SAMPLES", "3000"))
_rng = np.random.default_rng(1)
_RIDES = [
    {
        "ride_id": f"load-{i}",
        "samples": [dict(zip(CHANNELS, map(float, row), strict=True)) for row in _rng.normal(size=(N, 6))],
    }
    for i in range(10)
]


class Client(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def predict(self) -> None:
        self.client.post("/v1/predict", json=_RIDES[int(_rng.integers(len(_RIDES)))], name="/v1/predict")
