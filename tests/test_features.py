import unittest

import pandas as pd

from rental_mlops.features import enrich_housing_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_enrich_housing_features(self):
        frame = pd.DataFrame({"rooms": [2], "sqft": [800], "price": [1600]})

        enriched = enrich_housing_features(frame)

        self.assertEqual(enriched.loc[0, "price_per_sqft"], 2)
        self.assertEqual(enriched.loc[0, "sqft_per_room"], 400)
        self.assertNotIn("price_per_sqft", frame.columns)


if __name__ == "__main__":
    unittest.main()
