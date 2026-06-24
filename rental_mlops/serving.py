from pathlib import Path

from .data import load_housing_data, summarize_housing_data
from .monitoring import build_prediction_event
from .predict import RentalInput, fit_price_model, predict_price


DATA_PATH = Path("data/housing_1000.csv")


class ServingMetrics:
    def __init__(self) -> None:
        self.requests = 0
        self.warnings = 0

    def observe_prediction(self, warning_count: int) -> None:
        self.requests += 1
        self.warnings += warning_count

    def prometheus_text(self) -> str:
        return "\n".join(
            [
                "# HELP rental_price_predictions_total Total prediction requests.",
                "# TYPE rental_price_predictions_total counter",
                f"rental_price_predictions_total {self.requests}",
                "# HELP rental_price_prediction_warnings_total Total prediction warnings.",
                "# TYPE rental_price_prediction_warnings_total counter",
                f"rental_price_prediction_warnings_total {self.warnings}",
                "",
            ]
        )


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
    from fastapi import FastAPI, Response
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
    metrics = ServingMetrics()

    @app.get("/health")
    def health():
        return build_health_payload(data_path)

    @app.post("/predict")
    def predict(request: PredictionRequest):
        payload = build_prediction_payload(model, training_frame, request.rooms, request.sqft)
        metrics.observe_prediction(len(payload["warnings"]))
        return payload

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(metrics.prometheus_text(), media_type="text/plain; version=0.0.4")

    return app
