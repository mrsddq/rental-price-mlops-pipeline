# MLOps Lifecycle

## 1. Data

- Keep sample data small and versioned.
- Keep production datasets outside Git.
- Record schema and target column.

## 2. Training

- Use deterministic train/test splits.
- Log metrics.
- Save model artifacts.

## 3. Pipeline

- Compile the pipeline to YAML.
- Keep pipeline code and compiled artifact in sync.
- Use separate components for data prep, training, evaluation, and registration.

## 4. Deployment

- Build a container image.
- Submit the pipeline to Kubeflow.
- Track experiments and runs.

## 5. Monitoring

- Compare live inputs to training data.
- Track prediction quality when labels arrive.
- Retrain when drift or performance decay is detected.
