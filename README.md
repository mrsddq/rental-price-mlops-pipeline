# Rental Price MLOps Pipeline

Kubeflow Pipelines project for rental price prediction.

This repository demonstrates the first useful MLOps milestone: turn a local scikit-learn model workflow into a reproducible pipeline artifact that can be compiled and submitted to Kubeflow.

## Structure

```text
data/
  housing_1000.csv
rental_mlops/
  artifacts.py
  baseline.py
  config.py
  data.py
  features.py
  metrics.py
  model.py
  monitoring.py
  predict.py
  serving.py
tests/
main.py
Dockerfile
docker-compose.yml
requirements.txt
rental_price_prediction_pipeline.yaml
docs/
  API.md
  data_contract.md
  PORTFOLIO_EVIDENCE.md
  SYSTEM_DESIGN.md
  lifecycle.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Compile Pipeline

```bash
python main.py --compile-only
```

This writes `rental_price_prediction_pipeline.yaml`.

## Validate Data And Evaluate Locally

```bash
python main.py --validate-data --compile-only
python main.py --evaluate-local --compile-only
python main.py --predict --rooms 3 --sqft 1100 --compile-only
python main.py --validate-data --no-compile
python main.py --no-compile --write-reports --report-dir outputs/reports
python main.py --no-compile --write-artifact --artifact-path outputs/model/rental-price-model.pkl
```

The validation path checks the expected schema and summarizes the sample dataset. The local evaluation path trains the same linear regression workflow and prints RMSE, MAE, and R2.
The prediction path fits the local model and scores a single rental example.

## Serve Predictions Locally

```bash
uvicorn rental_mlops.serving:create_app --factory --host 0.0.0.0 --port 8000
```

Then call:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"rooms\": 3, \"sqft\": 1100}"
```

See [docs/API.md](docs/API.md) for the full API contract.

## Production Engineering Layer

- Dataset validation and summary reporting
- Train/test evaluation against a mean-price baseline
- Quality gate with RMSE, MAE, R2, and baseline-lift thresholds
- Generated model card and JSON quality report
- Exportable model artifact with metadata and quality report
- FastAPI serving layer with input-range monitoring warnings
- Docker Compose API runtime
- CI checks for tests, compile safety, model quality, and pipeline compilation
- Make targets for install, test, compile, evaluate, reports, artifact, serve, and clean

## Portfolio Evidence

Use [docs/PORTFOLIO_EVIDENCE.md](docs/PORTFOLIO_EVIDENCE.md) to generate the model card, quality report, and pipeline artifact before pinning or presenting this repo.

Use [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) to explain the data, training, quality, serving, and monitoring flow in interviews.

## Run Against Kubeflow

```bash
python main.py --host http://localhost:8080 --run
```

## Docker

```bash
docker build -t rental-price-pipeline:local .
docker run --rm rental-price-pipeline:local
docker compose up --build
```

## MLOps Concepts Demonstrated

- componentized model training
- reproducible pipeline compilation
- container-friendly entry point
- local dataset reference instead of remote data dependency
- experiment/run separation
- service contract around model inference
- prediction-time input monitoring

## Next Improvements

- split data preparation, training, evaluation, and registration into separate components
- publish a sample quality report generated from the current data snapshot
- add a model registry handoff instead of local pickle artifacts
- add request/response logging to durable storage
