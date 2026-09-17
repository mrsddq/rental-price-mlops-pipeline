import yaml

from main import compile_pipeline


def test_compiled_pipeline_transfers_dataset_artifact_to_training(tmp_path):
    output = tmp_path / "pipeline.yaml"
    compile_pipeline(str(output))
    spec = yaml.safe_load(output.read_text())
    assert "dataset_uri" in spec["root"]["inputDefinitions"]["parameters"]
    tasks = spec["root"]["dag"]["tasks"]
    training = tasks["train-rental-price-model"]
    assert (
        training["inputs"]["artifacts"]["dataset"]["taskOutputArtifact"]["producerTask"]
        == "importer"
    )
    assert "importer" in training["dependentTasks"]
