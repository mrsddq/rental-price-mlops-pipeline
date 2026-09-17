from dataclasses import replace
import json

import pandas as pd
import pytest

from rental_mlops.artifacts import write_model_artifact
from rental_mlops.data import validate_housing_data, summarize_housing_data
from rental_mlops.metrics import regression_metrics
from rental_mlops.model import train_and_evaluate
from rental_mlops.predict import RentalInput
from rental_mlops.quality import QualityGate, evaluate_quality
from rental_mlops.serving import DATA_PATH


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_fail_input_and_metric_contracts(value):
    with pytest.raises(ValueError):
        validate_housing_data(
            pd.DataFrame({"rooms": [2], "sqft": [value], "price": [1500]})
        )
    with pytest.raises(ValueError):
        RentalInput(value, 1000)
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, 2], [1, value])


def test_quality_gate_cannot_pass_nan_metrics():
    result = train_and_evaluate(DATA_PATH)
    result = replace(result, metrics=replace(result.metrics, rmse=float("nan")))
    report = evaluate_quality(result)
    assert not report.passed
    assert "model rmse must be finite" in report.failures
    json.dumps(report.to_dict(), allow_nan=False)


def test_failed_quality_cannot_replace_existing_artifact(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"previous approved artifact")
    with pytest.raises(ValueError, match="failed quality gate"):
        write_model_artifact(artifact, DATA_PATH, gate=QualityGate(0, 0, 1))
    assert artifact.read_bytes() == b"previous approved artifact"


def test_constant_targets_have_finite_r2():
    assert regression_metrics([3, 3], [3, 3]).r2 == 1.0
    assert regression_metrics([3, 3], [2, 2]).r2 == 0.0


def test_numeric_strings_are_summarized_numerically():
    report = summarize_housing_data(
        pd.DataFrame(
            {"rooms": ["2", "3"], "sqft": ["900", "1000"], "price": ["900", "1000"]}
        )
    )
    assert report.min_price == 900
    assert report.average_price == 950
