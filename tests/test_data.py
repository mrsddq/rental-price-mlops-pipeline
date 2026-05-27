import unittest

import pandas as pd

from rental_mlops.config import TrainingConfig
from rental_mlops.data import build_feature_target, summarize_housing_data, validate_housing_data


class HousingDataTests(unittest.TestCase):
    def test_summarize_housing_data(self):
        frame = pd.DataFrame(
            {
                "rooms": [2, 3],
                "sqft": [800, 1200],
                "price": [1200, 1800],
            }
        )

        report = summarize_housing_data(frame)

        self.assertEqual(report.row_count, 2)
        self.assertEqual(report.column_count, 3)
        self.assertEqual(report.average_price, 1500)
        self.assertEqual(report.average_sqft, 1000)

    def test_validate_housing_data_rejects_missing_columns(self):
        frame = pd.DataFrame({"rooms": [2], "price": [1200]})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_housing_data(frame)

    def test_validate_housing_data_rejects_non_positive_values(self):
        frame = pd.DataFrame({"rooms": [0], "sqft": [800], "price": [1200]})

        with self.assertRaisesRegex(ValueError, "positive"):
            validate_housing_data(frame)

    def test_build_feature_target(self):
        frame = pd.DataFrame({"rooms": [2], "sqft": [800], "price": [1200]})

        features, target = build_feature_target(frame)

        self.assertEqual(features.tolist(), [[2, 800]])
        self.assertEqual(target.tolist(), [1200])

    def test_build_feature_target_uses_configured_columns(self):
        frame = pd.DataFrame(
            {
                "rooms": [2],
                "sqft": [800],
                "price": [1200],
                "neighborhood_score": [7],
            }
        )
        config = TrainingConfig(feature_columns=("rooms", "neighborhood_score"))

        features, target = build_feature_target(frame, config)

        self.assertEqual(features.tolist(), [[2, 7]])
        self.assertEqual(target.tolist(), [1200])


if __name__ == "__main__":
    unittest.main()
