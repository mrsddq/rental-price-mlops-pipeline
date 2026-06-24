import importlib.util
import tempfile
import unittest
from pathlib import Path

from rental_mlops.registry import write_registry_record


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "housing_1000.csv"


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
class RegistryTests(unittest.TestCase):
    def test_registry_record_writes_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = write_registry_record(
                output_path=root / "registry.json",
                data_path=DATA_PATH,
                artifact_path=root / "model.pkl",
                version="2.0.0",
                image_tag="2.0.0",
                stage="staging",
            )
            text = (root / "registry.json").read_text(encoding="utf-8")

        self.assertEqual(record.version, "2.0.0")
        self.assertIn('"approval_status"', text)
        self.assertIn('"image_tag": "2.0.0"', text)


if __name__ == "__main__":
    unittest.main()
