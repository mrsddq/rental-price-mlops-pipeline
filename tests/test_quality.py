import tempfile
import unittest
from pathlib import Path

from rental_mlops.data import load_housing_data, summarize_housing_data
from rental_mlops.model import train_and_evaluate
from rental_mlops.model_card import render_model_card
from rental_mlops.quality import QualityGate, evaluate_quality, write_quality_report


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "housing_1000.csv"


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_passes_for_repository_dataset(self):
        result = train_and_evaluate(DATA_PATH)
        report = evaluate_quality(result, QualityGate(max_rmse=1300, max_mae=1100, min_r2=0.1))

        self.assertTrue(report.passed, report.failures)

    def test_quality_report_writes_json(self):
        result = train_and_evaluate(DATA_PATH)
        report = evaluate_quality(result, QualityGate(max_rmse=1300, max_mae=1100, min_r2=0.1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            write_quality_report(path, report)
            text = path.read_text(encoding="utf-8")

        self.assertIn('"passed"', text)

    def test_model_card_contains_operational_sections(self):
        frame = load_housing_data(DATA_PATH)
        result = train_and_evaluate(DATA_PATH)
        report = evaluate_quality(result, QualityGate(max_rmse=1300, max_mae=1100, min_r2=0.1))
        card = render_model_card(summarize_housing_data(frame), result, report)

        self.assertIn("Intended Use", card)
        self.assertIn("Quality Gate", card)


if __name__ == "__main__":
    unittest.main()
