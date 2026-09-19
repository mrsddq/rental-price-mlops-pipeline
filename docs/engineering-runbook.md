# Rental Price MLOps Pipeline runbook

Run these commands from the repository root with Python 3.10 or newer.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python main.py --validate-data --no-compile
python main.py --no-compile --write-reports --report-dir outputs/reports
python main.py --compile-only --output outputs/rental_price_prediction_pipeline.yaml
```

The tests cover data validation, quality gates, artifact selection, API input
handling, and the compiled Dataset importer contract. Pipeline compilation does
not submit a training job. Inspect the generated quality report before using a
model; failed quality gates prevent artifact creation or replacement.

## Build and serve an approved artifact

```bash
python main.py --no-compile --write-artifact --artifact-path outputs/model/rental-price-model.pkl
python main.py --no-compile --write-registry-record --registry-path outputs/model/registry-record.json --artifact-path outputs/model/rental-price-model.pkl
MODEL_ARTIFACT_PATH=outputs/model/rental-price-model.pkl uvicorn rental_mlops.serving:create_app --factory --host 127.0.0.1 --port 8000
```

From another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"rooms":3,"sqft":1100}'
curl --fail http://127.0.0.1:8000/metrics
```

Check that health reports `source: artifact` and the expected artifact digest.
A missing, incompatible, or rejected artifact prevents startup. An unset
`MODEL_ARTIFACT_PATH` enables sample-training mode, so it is not equivalent to
serving a previously approved artifact. Stop the local server with Ctrl+C.

## Container verification

```bash
docker build -t rental-price-pipeline:local .
docker run --rm rental-price-pipeline:local
docker build --target test -t rental-price-pipeline:test .
docker run --rm rental-price-pipeline:test
```

The default image compiles the pipeline. The separate test target installs test
dependencies and runs the full pytest suite as the non-root runtime user.
For an API container, use the Compose configuration described in the README.

## Operational limits

- Load pickle artifacts only from a trusted build process. Their deserialization can execute code.
- Registry records are local JSON metadata; this is not an automated MLflow promotion service.
- To roll back serving, select the previous approved artifact and matching image, restart, and verify the health digest and a prediction.
- Kubernetes/Helm resources and monitoring dashboards are deployment references. A live cluster, storage access, and runtime verification are separate prerequisites.
- Kubeflow execution requires a cluster-accessible `--dataset-uri`; local `--data` is not uploaded automatically.
- Metrics are process-local. The sample dataset and tests do not establish real-market prediction accuracy.
