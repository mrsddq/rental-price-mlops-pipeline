# API

The serving layer is intentionally small so the project can demonstrate model packaging and inference without requiring cloud infrastructure.

## Run Locally

```bash
python main.py --no-compile --write-artifact --artifact-path outputs/model/rental-price-model.pkl
MODEL_ARTIFACT_PATH=outputs/model/rental-price-model.pkl uvicorn rental_mlops.serving:create_app --factory --host 127.0.0.1 --port 8000
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

The example health payload above describes the local sample dataset. The live
response also identifies `source` (`artifact` or `sample-training`). Artifact mode
includes `artifact_sha256` and `dataset_sha256`, preserves the recorded feature
order, and works without access to the original CSV. It fails startup if the
artifact is missing, rejected, or incompatible. Only trusted local pickle files
are supported. Non-positive, NaN, and infinite inputs return HTTP 422. `/metrics`
exposes process-local successful prediction and warning counters.
