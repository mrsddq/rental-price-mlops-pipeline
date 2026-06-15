import unittest

import pandas as pd

from rental_mlops.monitoring import build_prediction_event, input_range_warnings
from rental_mlops.predict import RentalInput


class MonitoringTests(unittest.TestCase):
    def test_input_range_warnings_for_out_of_range_values(self):
        frame = pd.DataFrame(
            {
                "rooms": [1, 2, 3],
                "sqft": [500, 900, 1200],
                "price": [1000, 1600, 2100],
            }
        )

        warnings = input_range_warnings(frame, RentalInput(rooms=4, sqft=400))

        self.assertEqual(len(warnings), 2)
        self.assertIn("rooms", warnings[0])
        self.assertIn("sqft", warnings[1])

    def test_prediction_event_is_serializable(self):
        frame = pd.DataFrame(
            {
                "rooms": [1, 2, 3],
                "sqft": [500, 900, 1200],
                "price": [1000, 1600, 2100],
            }
        )

        event = build_prediction_event(RentalInput(rooms=2, sqft=900), 1750.0, frame)
        payload = event.to_dict()

        self.assertEqual(payload["predicted_price"], 1750.0)
        self.assertEqual(payload["warnings"], [])
        self.assertIn("event_id", payload)


if __name__ == "__main__":
    unittest.main()
