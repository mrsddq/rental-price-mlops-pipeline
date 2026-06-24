from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "rental_mlops/registry.py",
    "helm/rental-price-api/Chart.yaml",
    "helm/rental-price-api/values.yaml",
    "helm/rental-price-api/templates/deployment.yaml",
    "helm/rental-price-api/templates/servicemonitor.yaml",
    "kubernetes/argocd/application.yaml",
    "kubernetes/monitoring/prometheus-rule.yaml",
    "kubernetes/monitoring/grafana-dashboard.json",
    "mlflow/docker-compose.yml",
    "docs/RUNBOOK.md",
    "docs/DEPLOYMENT.md",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"missing required files: {', '.join(missing)}")
    print("rental mlops platform layout validation passed")


if __name__ == "__main__":
    main()
