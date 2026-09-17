# Rental Price MLOps Pipeline

[![CI](https://github.com/mrsddq/rental-price-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/rental-price-mlops-pipeline/actions/workflows/ci.yml)

Kubeflow Pipelines project for rental price prediction.

This repository demonstrates a local MLOps workflow: data validation, model training, quality gates, versioned artifact packaging, registry metadata, and FastAPI inference from an approved artifact. Docker and Helm configuration, monitoring examples, and rollback runbooks provide deployment references. The local registry is JSON metadata, not an integrated MLflow promotion service; cloud deployment and real-market rental accuracy are not claimed.

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
  registry.py
  serving.py
tests/
main.py
Dockerfile
docker-compose.yml
helm/
kubernetes/
mlflow/
requirements.txt
rental_price_prediction_pipeline.yaml
docs/
  API.md
  DEPLOYMENT.md
  data_contract.md
  PORTFOLIO_EVIDENCE.md
  RUNBOOK.md
  SYSTEM_DESIGN.md
  lifecycle.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python -m pytest -q
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
python main.py --no-compile --write-registry-record --registry-path outputs/model/registry-record.json
```

The validation path checks the expected schema and summarizes the sample dataset. The local evaluation path trains the same linear regression workflow and prints RMSE, MAE, and R2.
The prediction path fits the local model and scores a single rental example.

## Serve Predictions Locally

```bash
python main.py --no-compile --write-artifact --artifact-path outputs/model/rental-price-model.pkl
MODEL_ARTIFACT_PATH=outputs/model/rental-price-model.pkl uvicorn rental_mlops.serving:create_app --factory --host 127.0.0.1 --port 8000
```

Then call:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"rooms\": 3, \"sqft\": 1100}"
```

See [docs/API.md](docs/API.md) for the full API contract.

## Deploy With GitOps

```bash
helm template rental-price-api helm/rental-price-api
kubectl apply -f kubernetes/argocd/application.yaml
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) and [docs/RUNBOOK.md](docs/RUNBOOK.md) for model promotion, rollback, and incident handling.

## Engineering Features

- Dataset validation and summary reporting
- Train/test evaluation against a mean-price baseline
- Quality gate with RMSE, MAE, R2, and baseline-lift thresholds
- Generated model card and JSON quality report
- Exportable model artifact with metadata and quality report
- Local registry record with quality decision and image metadata
- FastAPI serving layer with input-range monitoring warnings
- Docker Compose API runtime
- Helm chart and Argo CD Application for Kubernetes deployment
- PrometheusRule, ServiceMonitor, and Grafana dashboard starters
- CI checks for tests, compile safety, model quality, and pipeline compilation
- Make targets for install, test, compile, evaluate, reports, artifact, registry, serve, and clean

## Portfolio Evidence

Use [docs/PORTFOLIO_EVIDENCE.md](docs/PORTFOLIO_EVIDENCE.md) to generate the model card, quality report, and pipeline artifact before pinning or presenting this repo.

Use [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) to explain the data, training, quality, serving, and monitoring flow in interviews.

## Run Against Kubeflow

```bash
python main.py --host http://localhost:8080 --run --dataset-uri '<cluster-accessible-dataset-uri>'
```

Upload your CSV to storage accessible to your Kubeflow installation first. The
pipeline imports it as a typed Dataset artifact; `--data` only selects a local
CSV and does not upload it to the cluster. Compilation is tested locally and in
CI; executing the compiled workflow requires your cluster and storage credentials.

## Artifact and quality contract

- NaN/Infinity are rejected in datasets, prediction inputs, and model metrics.
- Failed quality gates cannot create or replace a serving artifact.
- Artifacts contain the dataset SHA-256, training configuration, feature order,
  feature ranges, and quality report. Writes replace the destination atomically.
- `MODEL_ARTIFACT_PATH` selects the model actually loaded by the API. Missing,
  rejected, or incompatible artifacts fail startup; inference does not retrain.
- `/health` reports the selected artifact digest. Changing the artifact/image
  changes the served model; changing a Helm label alone does not.
- Without the environment variable, the API explicitly uses sample-training
  mode for local exploration. Docker Compose and Helm select the baked artifact.
- Only load pickle artifacts built by a trusted process; pickle deserialization
  can execute code. This demo does not authenticate or sign external model files.
- Metrics are process-local. Use a single worker per container and aggregate
  replicas in Prometheus; durable request logging is not implemented.

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
- add request/response logging to durable storage
