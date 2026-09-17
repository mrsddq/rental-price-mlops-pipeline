from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


def regression_metrics(actual, predicted) -> RegressionMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted arrays must have the same shape")
    if actual_values.size == 0:
        raise ValueError("metrics require at least one prediction")
    if actual_values.ndim != 1:
        raise ValueError("metrics require one-dimensional arrays")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("metrics require finite values")

    errors = actual_values - predicted_values
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    total_variance = float(np.sum((actual_values - actual_values.mean()) ** 2))
    residual_variance = float(np.sum(errors**2))
    # Match the finite constant-target convention: perfect predictions are 1,
    # otherwise 0. An undefined division must never reach a quality gate.
    if total_variance == 0:
        r2 = 1.0 if residual_variance == 0 else 0.0
    else:
        r2 = 1 - residual_variance / total_variance

    return RegressionMetrics(rmse=rmse, mae=mae, r2=float(r2))
