import importlib.util
import tempfile
import unittest
from pathlib import Path

from rental_mlops.artifacts import load_model_artifact, predict_from_artifact, write_model_artifact
from rental_mlops.predict import RentalInput


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "housing_1000.csv"


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
class ArtifactTests(unittest.TestCase):
    def test_model_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            metadata = write_model_artifact(path, DATA_PATH)
            bundle = load_model_artifact(path)
            prediction = predict_from_artifact(path, RentalInput(rooms=3, sqft=1100))

        self.assertEqual(metadata["artifact_version"], "1")
        self.assertEqual(bundle["feature_columns"], ["rooms", "sqft"])
        self.assertGreater(prediction, 0)


if __name__ == "__main__":
    unittest.main()
