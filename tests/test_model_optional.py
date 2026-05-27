import importlib.util
import unittest

from rental_mlops.model import train_and_evaluate


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
class LocalModelTests(unittest.TestCase):
    def test_train_and_evaluate_returns_metrics(self):
        result = train_and_evaluate("data/housing_1000.csv")

        self.assertGreater(result.train_rows, 0)
        self.assertGreater(result.test_rows, 0)
        self.assertGreaterEqual(result.metrics.rmse, 0)
        self.assertGreaterEqual(result.metrics.mae, 0)


if __name__ == "__main__":
    unittest.main()
