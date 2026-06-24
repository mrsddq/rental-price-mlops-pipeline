# Rental Price MLOps Runbook

## Deploy New Model Version

1. Run `make reports` and confirm the quality gate passes.
2. Run `make registry` to write model registry metadata.
3. Build and scan the API image.
4. Update `helm/rental-price-api/values.yaml` with the new image tag and model version.
5. Merge after review and let Argo CD sync.
6. Watch API health, error rate, latency, and prediction warning trends.

## Rollback

1. Revert the Helm values change.
2. Confirm Argo CD sync reaches healthy state.
3. Compare post-rollback metrics with the incident window.
4. Record the root cause and next prevention action.

## Quality Gate Failure

- Do not promote the model.
- Review drift report and data contract changes.
- Compare against baseline RMSE and MAE.
- Add an issue with the failed metric and training data snapshot.
