import unittest

from rental_mlops.metrics import regression_metrics


class RegressionMetricsTests(unittest.TestCase):
    def test_regression_metrics(self):
        metrics = regression_metrics([100, 200, 300], [110, 190, 310])

        self.assertAlmostEqual(metrics.rmse, 10.0)
        self.assertAlmostEqual(metrics.mae, 10.0)
        self.assertGreater(metrics.r2, 0.9)

    def test_regression_metrics_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            regression_metrics([1, 2], [1])

    def test_regression_metrics_rejects_empty_arrays(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            regression_metrics([], [])


if __name__ == "__main__":
    unittest.main()
