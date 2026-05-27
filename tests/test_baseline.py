import unittest

from rental_mlops.baseline import mean_price_baseline


class BaselineTests(unittest.TestCase):
    def test_mean_price_baseline(self):
        result = mean_price_baseline([1000, 2000], [1200, 1800])

        self.assertEqual(result.name, "mean_price")
        self.assertGreaterEqual(result.metrics.rmse, 0)

    def test_mean_price_baseline_rejects_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, "training value"):
            mean_price_baseline([], [1200])


if __name__ == "__main__":
    unittest.main()
