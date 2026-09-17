from pathlib import Path
import hashlib
import math
import os
from threading import Lock

import pandas as pd

from .artifacts import load_model_artifact
from .config import TrainingConfig
from .data import load_housing_data, summarize_housing_data
from .monitoring import build_prediction_event
from .predict import RentalInput, fit_price_model, predict_price


DATA_PATH = Path("data/housing_1000.csv")


class ServingMetrics:
    def __init__(self) -> None:
        self.requests = 0
        self.warnings = 0
        self._lock = Lock()

    def observe_prediction(self, warning_count: int) -> None:
        with self._lock:
            self.requests += 1
            self.warnings += warning_count

    def prometheus_text(self) -> str:
        with self._lock:
            requests, warnings = self.requests, self.warnings
        return "\n".join(
            [
                "# HELP rental_price_predictions_total Total prediction requests.",
                "# TYPE rental_price_predictions_total counter",
                f"rental_price_predictions_total {requests}",
                "# HELP rental_price_prediction_warnings_total Total prediction warnings.",
                "# TYPE rental_price_prediction_warnings_total counter",
                f"rental_price_prediction_warnings_total {warnings}",
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


def build_prediction_payload(
    model, training_frame, rooms: float, sqft: float
) -> dict[str, object]:
    rental_input = RentalInput(rooms=rooms, sqft=sqft)
    prediction = predict_price(model, rental_input)
    event = build_prediction_event(rental_input, prediction, training_frame)
    return event.to_dict()


def create_app(
    data_path: str | Path = DATA_PATH, artifact_path: str | Path | None = None
):
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    class PredictionRequest(BaseModel):
        rooms: float = Field(gt=0, allow_inf_nan=False)
        sqft: float = Field(gt=0, allow_inf_nan=False)

    app = FastAPI(
        title="Rental Price Prediction API",
        version="0.1.0",
        description="Small serving layer for the rental-price MLOps case study.",
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Pydantic error inputs can themselves contain NaN/Infinity; echoing
        # those values through JSONResponse turns an invalid request into 500.
        errors = [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    selected_artifact = artifact_path or os.environ.get("MODEL_ARTIFACT_PATH")
    prediction_config = TrainingConfig()
    if selected_artifact:
        bundle = load_model_artifact(selected_artifact)
        columns = bundle["feature_columns"]
        if len(columns) != 2 or set(columns) != {"rooms", "sqft"}:
            raise ValueError("serving requires exactly rooms and sqft features")
        if bundle["target_column"] != "price":
            raise ValueError("serving requires a price target")
        ranges = bundle.get("feature_ranges", {})
        for column in columns:
            bounds = ranges.get(column)
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or not all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                    for value in bounds
                )
                or bounds[0] > bounds[1]
            ):
                raise ValueError(
                    "artifact needs valid feature_ranges; regenerate it before serving"
                )
        # Range monitoring needs extrema only, not the original training CSV.
        training_frame = pd.DataFrame({**ranges, "price": [1.0, 1.0]})
        model = bundle["model"]
        prediction_config = TrainingConfig(feature_columns=tuple(columns))
        health_payload = {
            "status": "ok",
            "model": bundle.get("model_type", type(model).__name__),
            "source": "artifact",
            "feature_columns": list(columns),
            "target_column": "price",
            "dataset_rows": bundle.get("dataset", {}).get("row_count"),
            "dataset_sha256": bundle.get("dataset_sha256"),
            "artifact_sha256": hashlib.sha256(
                Path(selected_artifact).read_bytes()
            ).hexdigest(),
        }
    else:
        training_frame = load_housing_data(data_path)
        model = fit_price_model(data_path)
        health_payload = build_health_payload(data_path)
        health_payload["source"] = "sample-training"
    metrics = ServingMetrics()

    @app.get("/health")
    def health():
        return health_payload

    @app.post("/predict")
    def predict(request: PredictionRequest):
        rental_input = RentalInput(rooms=request.rooms, sqft=request.sqft)
        try:
            prediction = predict_price(model, rental_input, prediction_config)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="features exceed the model's numeric range"
            ) from exc
        payload = build_prediction_event(
            rental_input, prediction, training_frame
        ).to_dict()
        metrics.observe_prediction(len(payload["warnings"]))
        return payload

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(
            metrics.prometheus_text(), media_type="text/plain; version=0.0.4"
        )

    return app
