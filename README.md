# MLOps Project

Kubeflow Pipelines starter project for rental price prediction.

This repository demonstrates the first useful MLOps milestone: turn a local scikit-learn model workflow into a reproducible pipeline artifact that can be compiled and submitted to Kubeflow.

## Structure

```text
data/
  housing_1000.csv
main.py
Dockerfile
requirements.txt
rental_price_prediction_pipeline.yaml
docs/
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
- local dataset reference instead of hidden remote dependency
- experiment/run separation

## Next Improvements

- split data preparation, training, evaluation, and registration into separate components
- add model artifact output
- log RMSE/R2 metrics
- add CI to compile the pipeline on every PR
- add a lightweight FastAPI serving layer
