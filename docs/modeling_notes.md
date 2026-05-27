# Modeling Notes

## Baseline

The baseline predictor uses the mean training price as every test prediction. This gives the linear regression model a simple comparison point and makes local evaluation more meaningful than reporting one metric in isolation.

## Features

The default training features are:

- `rooms`
- `sqft`

The feature engineering helpers also compute:

- `price_per_sqft`
- `sqft_per_room`

These derived fields are useful for validation and exploratory analysis, while the baseline model intentionally remains simple.

## Metrics

Local evaluation reports:

- RMSE
- MAE
- R2
- mean-price baseline RMSE
