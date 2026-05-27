# Rental Housing Data Contract

The sample dataset is intentionally small, versioned, and human-readable so pipeline behavior can be reviewed without external services.

## Required Columns

| Column | Type | Rule |
| --- | --- | --- |
| `rooms` | numeric | Positive room count. |
| `sqft` | numeric | Positive square footage. |
| `price` | numeric | Positive monthly rental price. |

## Validation Rules

- The dataset must contain at least one row.
- Required columns must be present.
- Required values must not be missing.
- `rooms`, `sqft`, and `price` must be positive.

## Modeling Contract

The baseline model uses `rooms` and `sqft` as features and predicts `price`. Evaluation reports RMSE, MAE, and R2 on a deterministic train/test split.
