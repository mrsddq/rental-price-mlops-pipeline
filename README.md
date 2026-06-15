# MLOps Project

Kubeflow Pipelines project for rental price prediction.

This repository demonstrates the first useful MLOps milestone: turn a local scikit-learn model workflow into a reproducible pipeline artifact that can be compiled and submitted to Kubeflow.

## Structure

```text
data/
  housing_1000.csv
rental_mlops/
  baseline.py
  config.py
  data.py
  features.py
  metrics.py
  model.py
  predict.py
tests/
main.py
Dockerfile
requirements.txt
rental_price_prediction_pipeline.yaml
docs/
  data_contract.md
  PORTFOLIO_EVIDENCE.md
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
```

The validation path checks the expected schema and summarizes the sample dataset. The local evaluation path trains the same linear regression workflow and prints RMSE, MAE, and R2.
The prediction path fits the local model and scores a single rental example.

## Production Engineering Layer

- Dataset validation and summary reporting
- Train/test evaluation against a mean-price baseline
- Quality gate with RMSE, MAE, R2, and baseline-lift thresholds
- Generated model card and JSON quality report
- CI checks for tests, compile safety, model quality, and pipeline compilation
- Make targets for install, test, compile, evaluate, reports, and clean

## Portfolio Evidence

Use [docs/PORTFOLIO_EVIDENCE.md](docs/PORTFOLIO_EVIDENCE.md) to generate the model card, quality report, and pipeline artifact before pinning or presenting this repo.

## Run Against Kubeflow

```bash
python main.py --host http://localhost:8080 --run
```

## Docker

```bash
docker build -t rental-price-pipeline:local .
docker run --rm rental-price-pipeline:local python main.py --compile-only
```

## MLOps Concepts Demonstrated

- componentized model training
- reproducible pipeline compilation
- container-friendly entry point
- local dataset reference instead of remote data dependency
- experiment/run separation

## Next Improvements

- split data preparation, training, evaluation, and registration into separate components
- add model artifact output
- publish a sample quality report generated from the current data snapshot
- add a lightweight FastAPI serving layer
