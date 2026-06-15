# API

The serving layer is intentionally small so the project can demonstrate model packaging and inference without requiring cloud infrastructure.

## Run Locally

```bash
uvicorn rental_mlops.serving:create_app --factory --host 0.0.0.0 --port 8000
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Health

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "model": "linear_regression",
  "dataset_rows": 1000,
  "feature_columns": ["rooms", "sqft"],
  "target_column": "price"
}
```

## Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"rooms\": 3, \"sqft\": 1100}"
```

The response includes a prediction event id, timestamp, predicted price, and range warnings when input values sit outside the training data range.
