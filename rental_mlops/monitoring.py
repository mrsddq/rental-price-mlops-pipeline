from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from .data import validate_housing_data
from .predict import RentalInput


@dataclass(frozen=True)
class PredictionEvent:
    event_id: str
    created_at: str
    rooms: float
    sqft: float
    predicted_price: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def input_range_warnings(training_frame: pd.DataFrame, rental_input: RentalInput) -> tuple[str, ...]:
    validate_housing_data(training_frame)
    warnings: list[str] = []
    for column, value in (("rooms", rental_input.rooms), ("sqft", rental_input.sqft)):
        minimum = float(training_frame[column].min())
        maximum = float(training_frame[column].max())
        if value < minimum:
            warnings.append(f"{column} {value:.2f} is below training minimum {minimum:.2f}")
        if value > maximum:
            warnings.append(f"{column} {value:.2f} is above training maximum {maximum:.2f}")
    return tuple(warnings)


def build_prediction_event(
    rental_input: RentalInput,
    predicted_price: float,
    training_frame: pd.DataFrame,
) -> PredictionEvent:
    return PredictionEvent(
        event_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        rooms=float(rental_input.rooms),
        sqft=float(rental_input.sqft),
        predicted_price=float(predicted_price),
        warnings=input_range_warnings(training_frame, rental_input),
    )
