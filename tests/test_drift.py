import unittest

import pandas as pd

from rental_mlops.drift import compare_dataset_profiles


class DriftReportTests(unittest.TestCase):
    def test_compare_dataset_profiles_passes_small_shift(self):
        baseline = pd.DataFrame({"rooms": [2, 3], "sqft": [800, 1200], "price": [1200, 1800]})
        candidate = pd.DataFrame({"rooms": [2, 3], "sqft": [820, 1180], "price": [1220, 1780]})

        report = compare_dataset_profiles(baseline, candidate, threshold=0.10)

        self.assertTrue(report.passed)
        self.assertEqual(report.exceeded_columns, ())

    def test_compare_dataset_profiles_flags_large_shift(self):
        baseline = pd.DataFrame({"rooms": [2, 3], "sqft": [800, 1200], "price": [1200, 1800]})
        candidate = pd.DataFrame({"rooms": [5, 6], "sqft": [2400, 2600], "price": [6000, 6500]})

        report = compare_dataset_profiles(baseline, candidate, threshold=0.25)

        self.assertFalse(report.passed)
        self.assertIn("price", report.exceeded_columns)

    def test_compare_dataset_profiles_rejects_negative_threshold(self):
        baseline = pd.DataFrame({"rooms": [2], "sqft": [800], "price": [1200]})

        with self.assertRaisesRegex(ValueError, "non-negative"):
            compare_dataset_profiles(baseline, baseline, threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
