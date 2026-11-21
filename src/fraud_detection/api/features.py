"""Per-ride summary of the IMU signal, shared by the dummy model, the API log and the monitoring."""

from __future__ import annotations

import numpy as np

from fraud_detection.api.schemas import CHANNELS


def ride_summary(x: np.ndarray) -> dict[str, float]:
    """Mean and standard deviation of each channel; NaN (sensor dropouts) are ignored.

    ``x`` has shape (n_samples, 6), columns in the order of ``CHANNELS``.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(CHANNELS):
        col = x[:, i]
        out[f"{name}_mean"] = float(np.nanmean(col)) if not np.isnan(col).all() else 0.0
        out[f"{name}_std"] = float(np.nanstd(col)) if not np.isnan(col).all() else 0.0
    return out
