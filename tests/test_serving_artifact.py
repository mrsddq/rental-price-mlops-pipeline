import pickle
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rental_mlops.artifacts import write_model_artifact
from rental_mlops.config import TrainingConfig
from rental_mlops.predict import RentalInput, fit_price_model, predict_price
from rental_mlops.serving import DATA_PATH, create_app


def test_api_loads_selected_artifact_without_training_or_reading_csv(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "model.pkl"
    write_model_artifact(artifact, DATA_PATH)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact))
    with patch(
        "rental_mlops.serving.fit_price_model",
        side_effect=AssertionError("must not train"),
    ):
        app = create_app(tmp_path / "does-not-exist.csv")
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["source"] == "artifact"
        assert len(health["artifact_sha256"]) == 64
        assert len(health["dataset_sha256"]) == 64
        prediction = client.post("/predict", json={"rooms": 3, "sqft": 1100})
        assert prediction.status_code == 200
        assert prediction.json()["predicted_price"] > 0
        assert "rental_price_predictions_total 1" in client.get("/metrics").text


def test_api_uses_artifact_feature_order_and_changes_predictions(tmp_path):
    data = pd.read_csv(DATA_PATH)
    data["price"] *= 2
    changed = tmp_path / "changed.csv"
    data.to_csv(changed, index=False)
    config = TrainingConfig(feature_columns=("sqft", "rooms"))
    expected = predict_price(
        fit_price_model(changed, config), RentalInput(3, 1100), config
    )
    from rental_mlops.quality import QualityGate

    artifact = tmp_path / "v2.pkl"
    write_model_artifact(artifact, changed, config, QualityGate(3000, 3000, 0.1))
    with TestClient(create_app(artifact_path=artifact)) as client:
        assert client.post("/predict", json={"rooms": 3, "sqft": 1100}).json()[
            "predicted_price"
        ] == pytest.approx(expected)


@pytest.mark.parametrize(
    "rooms,sqft",
    [(0, 100), (-1, 100), ("Infinity", 100), ("NaN", 100), (2, "Infinity")],
)
def test_api_rejects_invalid_features(rooms, sqft, monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    with TestClient(create_app()) as client:
        assert (
            client.post("/predict", json={"rooms": rooms, "sqft": sqft}).status_code
            == 422
        )


def test_missing_or_rejected_artifact_fails_startup(tmp_path):
    artifact = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError):
        create_app(artifact_path=artifact)
    write_model_artifact(artifact, DATA_PATH)
    with artifact.open("rb") as stream:
        bundle = pickle.load(stream)
    bundle["quality"]["passed"] = False
    artifact.write_bytes(pickle.dumps(bundle))
    with pytest.raises(ValueError, match="quality gate"):
        create_app(artifact_path=artifact)


@pytest.mark.parametrize(
    "body",
    [
        '{"rooms": NaN, "sqft": 1000}',
        '{"rooms": Infinity, "sqft": 1000}',
        '{"rooms": 1e308, "sqft": 1e308}',
    ],
)
def test_numeric_edge_requests_return_client_errors_not_server_errors(
    body, monkeypatch
):
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/predict", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]
