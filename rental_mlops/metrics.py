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

    errors = actual_values - predicted_values
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    total_variance = float(np.sum((actual_values - actual_values.mean()) ** 2))
    residual_variance = float(np.sum(errors**2))
    r2 = 1.0 if total_variance == 0 and residual_variance == 0 else 1 - residual_variance / total_variance

    return RegressionMetrics(rmse=rmse, mae=mae, r2=float(r2))
