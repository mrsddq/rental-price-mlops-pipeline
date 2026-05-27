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
```

The validation path checks the expected schema and summarizes the sample dataset. The local evaluation path trains the same linear regression workflow and prints RMSE, MAE, and R2.
The prediction path fits the local model and scores a single rental example.

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
- log RMSE/R2 metrics
- add CI to compile the pipeline on every PR
- add a lightweight FastAPI serving layer
