from pathlib import Path

from .data import load_housing_data, summarize_housing_data
from .monitoring import build_prediction_event
from .predict import RentalInput, fit_price_model, predict_price


DATA_PATH = Path("data/housing_1000.csv")


def build_health_payload(data_path: str | Path = DATA_PATH) -> dict[str, object]:
    frame = load_housing_data(data_path)
    report = summarize_housing_data(frame)
    return {
        "status": "ok",
        "model": "linear_regression",
        "dataset_rows": report.row_count,
        "feature_columns": ["rooms", "sqft"],
        "target_column": "price",
    }


def build_prediction_payload(model, training_frame, rooms: float, sqft: float) -> dict[str, object]:
    rental_input = RentalInput(rooms=rooms, sqft=sqft)
    prediction = predict_price(model, rental_input)
    event = build_prediction_event(rental_input, prediction, training_frame)
    return event.to_dict()


def create_app(data_path: str | Path = DATA_PATH):
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class PredictionRequest(BaseModel):
        rooms: float = Field(gt=0)
        sqft: float = Field(gt=0)

    app = FastAPI(
        title="Rental Price Prediction API",
        version="0.1.0",
        description="Small serving layer for the rental-price MLOps case study.",
    )
    training_frame = load_housing_data(data_path)
    model = fit_price_model(data_path)

    @app.get("/health")
    def health():
        return build_health_payload(data_path)

    @app.post("/predict")
    def predict(request: PredictionRequest):
        return build_prediction_payload(model, training_frame, request.rooms, request.sqft)

    return app
