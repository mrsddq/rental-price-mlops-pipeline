from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG, TrainingConfig
from .data import build_feature_target, load_housing_data


@dataclass(frozen=True)
class RentalInput:
    rooms: float
    sqft: float

    def __post_init__(self):
        if self.rooms <= 0:
            raise ValueError("rooms must be positive")
        if self.sqft <= 0:
            raise ValueError("sqft must be positive")

    def as_feature_row(self, config: TrainingConfig = DEFAULT_CONFIG):
        values = {
            "rooms": self.rooms,
            "sqft": self.sqft,
        }
        return [values[column] for column in config.feature_columns]


def fit_price_model(data_path: str | Path, config: TrainingConfig = DEFAULT_CONFIG):
    from sklearn.linear_model import LinearRegression

    frame = load_housing_data(data_path)
    features, target = build_feature_target(frame, config)
    return LinearRegression().fit(features, target)


def predict_price(model, rental_input: RentalInput, config: TrainingConfig = DEFAULT_CONFIG) -> float:
    row = np.asarray([rental_input.as_feature_row(config)], dtype=float)
    return float(model.predict(row)[0])
