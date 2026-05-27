from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG, TrainingConfig
from .data import load_housing_data, split_feature_target
from .metrics import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class TrainingResult:
    metrics: RegressionMetrics
    train_rows: int
    test_rows: int


def train_and_evaluate(data_path: str | Path, config: TrainingConfig = DEFAULT_CONFIG) -> TrainingResult:
    from sklearn.linear_model import LinearRegression

    frame = load_housing_data(data_path)
    x_train, x_test, y_train, y_test = split_feature_target(frame, config)

    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = regression_metrics(y_test, predictions)
    return TrainingResult(metrics=metrics, train_rows=len(x_train), test_rows=len(x_test))
