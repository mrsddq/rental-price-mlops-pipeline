from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import TrainingResult


@dataclass(frozen=True)
class QualityGate:
    max_rmse: float
    max_mae: float
    min_r2: float
    require_baseline_lift: bool = True


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    failures: tuple[str, ...]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    gate: QualityGate

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "gate": asdict(self.gate),
        }


DEFAULT_GATE = QualityGate(max_rmse=1300.0, max_mae=1100.0, min_r2=0.1)


def evaluate_quality(result: TrainingResult, gate: QualityGate = DEFAULT_GATE) -> QualityReport:
    failures: list[str] = []
    if result.metrics.rmse > gate.max_rmse:
        failures.append(f"rmse {result.metrics.rmse:.4f} exceeds {gate.max_rmse:.4f}")
    if result.metrics.mae > gate.max_mae:
        failures.append(f"mae {result.metrics.mae:.4f} exceeds {gate.max_mae:.4f}")
    if result.metrics.r2 < gate.min_r2:
        failures.append(f"r2 {result.metrics.r2:.4f} is below {gate.min_r2:.4f}")
    if gate.require_baseline_lift and result.metrics.rmse >= result.baseline.metrics.rmse:
        failures.append("model rmse does not improve over mean-price baseline")
    return QualityReport(
        passed=not failures,
        failures=tuple(failures),
        metrics=asdict(result.metrics),
        baseline_metrics=asdict(result.baseline.metrics),
        gate=gate,
    )


def write_quality_report(path: str | Path, report: QualityReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
