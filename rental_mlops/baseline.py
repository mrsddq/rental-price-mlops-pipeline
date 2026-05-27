from dataclasses import dataclass

import numpy as np

from .metrics import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class BaselineResult:
    name: str
    metrics: RegressionMetrics


def mean_price_baseline(train_target, test_target) -> BaselineResult:
    train_values = np.asarray(train_target, dtype=float)
    if train_values.size == 0:
        raise ValueError("baseline requires at least one training value")

    predictions = np.full_like(np.asarray(test_target, dtype=float), train_values.mean())
    return BaselineResult(
        name="mean_price",
        metrics=regression_metrics(test_target, predictions),
    )
