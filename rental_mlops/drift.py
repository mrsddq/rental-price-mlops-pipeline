from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import REQUIRED_COLUMNS, validate_housing_data


@dataclass(frozen=True)
class DriftReport:
    baseline_rows: int
    candidate_rows: int
    threshold: float
    relative_mean_deltas: dict[str, float]
    exceeded_columns: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.exceeded_columns

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exceeded_columns"] = list(self.exceeded_columns)
        payload["passed"] = self.passed
        return payload


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_housing_data(frame)
    return frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric)


def compare_dataset_profiles(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    threshold: float = 0.25,
) -> DriftReport:
    if threshold < 0:
        raise ValueError("drift threshold must be non-negative")

    baseline_numeric = _numeric_frame(baseline)
    candidate_numeric = _numeric_frame(candidate)
    deltas: dict[str, float] = {}
    exceeded: list[str] = []

    for column in REQUIRED_COLUMNS:
        baseline_mean = float(baseline_numeric[column].mean())
        candidate_mean = float(candidate_numeric[column].mean())
        if baseline_mean == 0:
            delta = 0.0 if candidate_mean == 0 else float("inf")
        else:
            delta = abs(candidate_mean - baseline_mean) / abs(baseline_mean)
        deltas[column] = delta
        if delta > threshold:
            exceeded.append(column)

    return DriftReport(
        baseline_rows=len(baseline_numeric),
        candidate_rows=len(candidate_numeric),
        threshold=threshold,
        relative_mean_deltas=deltas,
        exceeded_columns=tuple(exceeded),
    )


def write_drift_report(
    path: str | Path,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    threshold: float = 0.25,
) -> DriftReport:
    report = compare_dataset_profiles(baseline, candidate, threshold)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
