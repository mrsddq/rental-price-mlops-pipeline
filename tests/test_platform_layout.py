import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PlatformLayoutTests(unittest.TestCase):
    def test_helm_chart_has_runtime_controls(self):
        deployment = (ROOT / "helm" / "rental-price-api" / "templates" / "deployment.yaml").read_text(encoding="utf-8")
        self.assertIn("readinessProbe:", deployment)
        self.assertIn("livenessProbe:", deployment)
        self.assertIn("resources:", deployment)

    def test_argocd_points_to_helm_chart(self):
        app = (ROOT / "kubernetes" / "argocd" / "application.yaml").read_text(encoding="utf-8")
        self.assertIn("path: helm/rental-price-api", app)

    def test_runbook_mentions_rollback(self):
        runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn("Rollback", runbook)


if __name__ == "__main__":
    unittest.main()
