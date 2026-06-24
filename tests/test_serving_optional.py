import importlib.util
import unittest
from pathlib import Path

from rental_mlops.serving import ServingMetrics, build_health_payload, build_prediction_payload


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "housing_1000.csv"


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
class ServingContractTests(unittest.TestCase):
    def test_health_payload_describes_model_contract(self):
        payload = build_health_payload(DATA_PATH)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["feature_columns"], ["rooms", "sqft"])
        self.assertGreater(payload["dataset_rows"], 0)

    def test_prediction_payload_contains_monitoring_metadata(self):
        from rental_mlops.data import load_housing_data
        from rental_mlops.predict import fit_price_model

        frame = load_housing_data(DATA_PATH)
        model = fit_price_model(DATA_PATH)
        payload = build_prediction_payload(model, frame, rooms=3, sqft=1100)

        self.assertGreater(payload["predicted_price"], 0)
        self.assertIn("event_id", payload)
        self.assertIn("warnings", payload)

    def test_serving_metrics_render_prometheus_text(self):
        metrics = ServingMetrics()
        metrics.observe_prediction(2)
        text = metrics.prometheus_text()

        self.assertIn("rental_price_predictions_total 1", text)
        self.assertIn("rental_price_prediction_warnings_total 2", text)


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI is not installed")
class ServingApiSchemaTests(unittest.TestCase):
    def test_predict_accepts_json_request_body(self):
        from rental_mlops.serving import create_app

        app = create_app(DATA_PATH)
        operation = app.openapi()["paths"]["/predict"]["post"]

        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)
        self.assertIn("/metrics", app.openapi()["paths"])


if __name__ == "__main__":
    unittest.main()
