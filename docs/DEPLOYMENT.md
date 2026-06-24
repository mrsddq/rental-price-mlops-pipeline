# Deployment

## Local MLflow

```bash
docker compose -f mlflow/docker-compose.yml up
```

MLflow is used here as the tracking and registry handoff concept. The repository also writes a local registry record so the flow can be demonstrated without running external services.

## Kubernetes

```bash
helm template rental-price-api helm/rental-price-api
kubectl apply -f kubernetes/argocd/application.yaml
```

Argo CD owns the Helm chart after bootstrap.

## Promotion Contract

- Model version changes happen through pull requests.
- CI writes quality reports and registry metadata.
- Helm values pin the model version and image tag.
- Rollback is a Git revert.
