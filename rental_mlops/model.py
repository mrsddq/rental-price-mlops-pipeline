from dataclasses import dataclass
from pathlib import Path

from .data import build_feature_target, load_housing_data
from .metrics import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class TrainingResult:
    metrics: RegressionMetrics
    train_rows: int
    test_rows: int


def train_and_evaluate(data_path: str | Path, test_size: float = 0.2, random_state: int = 42) -> TrainingResult:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    frame = load_housing_data(data_path)
    features, target = build_feature_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = regression_metrics(y_test, predictions)
    return TrainingResult(metrics=metrics, train_rows=len(x_train), test_rows=len(x_test))
