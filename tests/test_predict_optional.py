import importlib.util
import unittest

from rental_mlops.predict import RentalInput, fit_price_model, predict_price


class RentalInputTests(unittest.TestCase):
    def test_rental_input_as_feature_row(self):
        row = RentalInput(rooms=3, sqft=1200).as_feature_row()

        self.assertEqual(row, [3, 1200])


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
class PredictionTests(unittest.TestCase):
    def test_predict_price_returns_positive_value(self):
        model = fit_price_model("data/housing_1000.csv")
        prediction = predict_price(model, RentalInput(rooms=3, sqft=1100))

        self.assertGreater(prediction, 0)


if __name__ == "__main__":
    unittest.main()
