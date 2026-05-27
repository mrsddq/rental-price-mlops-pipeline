from __future__ import annotations

from dataclasses import asdict

from .data import DatasetReport
from .model import TrainingResult
from .quality import QualityReport


def render_model_card(dataset: DatasetReport, result: TrainingResult, quality: QualityReport) -> str:
    return "\n".join(
        [
            "# Rental Price Model Card",
            "",
            "## Intended Use",
            "",
            "Batch estimation of monthly rental prices from room count and square footage.",
            "",
            "## Dataset",
            "",
            f"- Rows: {dataset.row_count}",
            f"- Columns: {dataset.column_count}",
            f"- Average sqft: {dataset.average_sqft:.2f}",
            f"- Average price: {dataset.average_price:.2f}",
            "",
            "## Metrics",
            "",
            f"- RMSE: {result.metrics.rmse:.4f}",
            f"- MAE: {result.metrics.mae:.4f}",
            f"- R2: {result.metrics.r2:.4f}",
            f"- Baseline RMSE: {result.baseline.metrics.rmse:.4f}",
            "",
            "## Quality Gate",
            "",
            f"- Passed: {str(quality.passed).lower()}",
            f"- Gate: `{asdict(quality.gate)}`",
            f"- Failures: {', '.join(quality.failures) if quality.failures else 'none'}",
            "",
        ]
    )
