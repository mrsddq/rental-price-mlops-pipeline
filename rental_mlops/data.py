from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, TrainingConfig

REQUIRED_COLUMNS = ("rooms", "sqft", "price")


@dataclass(frozen=True)
class DatasetReport:
    row_count: int
    column_count: int
    min_price: float
    max_price: float
    average_price: float
    average_sqft: float


def load_housing_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_housing_data(frame)
    return frame


def validate_housing_data(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    if frame.empty:
        raise ValueError("housing dataset must contain at least one row")

    numeric_columns = frame.loc[:, REQUIRED_COLUMNS]
    if numeric_columns.isna().any().any():
        raise ValueError("housing dataset contains missing numeric values")

    invalid_rooms = frame["rooms"] <= 0
    invalid_sqft = frame["sqft"] <= 0
    invalid_price = frame["price"] <= 0
    if invalid_rooms.any() or invalid_sqft.any() or invalid_price.any():
        raise ValueError("rooms, sqft, and price must all be positive")


def summarize_housing_data(frame: pd.DataFrame) -> DatasetReport:
    validate_housing_data(frame)
    return DatasetReport(
        row_count=len(frame),
        column_count=len(frame.columns),
        min_price=float(frame["price"].min()),
        max_price=float(frame["price"].max()),
        average_price=float(frame["price"].mean()),
        average_sqft=float(frame["sqft"].mean()),
    )


def build_feature_target(frame: pd.DataFrame, config: TrainingConfig = DEFAULT_CONFIG):
    validate_housing_data(frame)
    return frame[list(config.feature_columns)].to_numpy(), frame[config.target_column].to_numpy()


def split_feature_target(frame: pd.DataFrame, config: TrainingConfig = DEFAULT_CONFIG):
    from sklearn.model_selection import train_test_split

    features, target = build_feature_target(frame, config)
    return train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
