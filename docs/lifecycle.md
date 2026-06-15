# MLOps Lifecycle

## 1. Data

- Keep sample data small and versioned.
- Keep production datasets outside Git.
- Record schema and target column.

## 2. Training

- Use deterministic train/test splits.
- Log RMSE, MAE, and R2 metrics.
- Compare model performance with a mean-price baseline.
- Save model artifacts.
- Attach model metadata and quality-gate status to artifacts.

## 3. Pipeline

- Compile the pipeline to YAML.
- Keep pipeline code and compiled artifact in sync.
- Use separate components for data prep, training, evaluation, and registration.

## 4. Deployment

- Build a container image.
- Submit the pipeline to Kubeflow.
- Track experiments and runs.
- Serve local predictions through the FastAPI API for review/demo workflows.

## 5. Monitoring

- Compare live inputs to training data.
- Track prediction quality when labels arrive.
- Retrain when drift or performance decay is detected.
- Emit prediction events with range warnings for out-of-distribution inputs.
