from __future__ import annotations

import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, TrainingConfig
from .data import load_housing_data, summarize_housing_data
from .model import train_and_evaluate
from .predict import RentalInput, fit_price_model, predict_price
from .quality import DEFAULT_GATE, QualityGate, evaluate_quality


ARTIFACT_VERSION = "1"


def build_model_bundle(
    data_path: str | Path,
    config: TrainingConfig = DEFAULT_CONFIG,
    gate: QualityGate = DEFAULT_GATE,
) -> dict[str, Any]:
    frame = load_housing_data(data_path)
    model = fit_price_model(data_path, config)
    result = train_and_evaluate(data_path, config)
    quality = evaluate_quality(result, gate)
    dataset = summarize_housing_data(frame)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "feature_columns": list(config.feature_columns),
        "target_column": config.target_column,
        "dataset": asdict(dataset),
        "quality": quality.to_dict(),
        "model": model,
    }


def write_model_artifact(
    output_path: str | Path,
    data_path: str | Path,
    config: TrainingConfig = DEFAULT_CONFIG,
    gate: QualityGate = DEFAULT_GATE,
) -> dict[str, Any]:
    bundle = build_model_bundle(data_path, config, gate)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(bundle, file)
    return {key: value for key, value in bundle.items() if key != "model"}


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        bundle = pickle.load(file)
    if bundle.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported model artifact version")
    for key in ("model", "feature_columns", "target_column", "quality"):
        if key not in bundle:
            raise ValueError(f"model artifact missing {key}")
    return bundle


def predict_from_artifact(path: str | Path, rental_input: RentalInput) -> float:
    bundle = load_model_artifact(path)
    return predict_price(bundle["model"], rental_input)
