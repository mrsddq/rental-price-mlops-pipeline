import argparse
from pathlib import Path

from rental_mlops.artifacts import write_model_artifact
from rental_mlops.data import load_housing_data, summarize_housing_data
from rental_mlops.model_card import render_model_card
from rental_mlops.model import train_and_evaluate
from rental_mlops.predict import RentalInput, fit_price_model, predict_price
from rental_mlops.quality import DEFAULT_GATE, evaluate_quality, write_quality_report


PIPELINE_FILE = "rental_price_prediction_pipeline.yaml"
DATA_PATH = "data/housing_1000.csv"


def build_pipeline():
    from kfp import dsl

    @dsl.component(
        base_image="python:3.10",
        packages_to_install=["pandas==2.2.2", "numpy==2.1.1", "scikit-learn==1.5.1"],
    )
    def train_rental_price_model(data_path: str = DATA_PATH) -> float:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split

        frame = pd.read_csv(data_path)
        features = frame[["rooms", "sqft"]].values
        target = frame["price"].values

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )

        model = LinearRegression().fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print(f"RMSE: {rmse:.2f}")
        return float(rmse)

    @dsl.pipeline(name="rental-price-prediction-pipeline")
    def rental_price_prediction_pipeline() -> float:
        training_task = train_rental_price_model()
        return training_task.output

    return rental_price_prediction_pipeline


def compile_pipeline(output_path=PIPELINE_FILE):
    from kfp import compiler

    compiler.Compiler().compile(build_pipeline(), output_path)
    print(f"Compiled pipeline: {Path(output_path).resolve()}")


def validate_dataset(data_path):
    frame = load_housing_data(data_path)
    report = summarize_housing_data(frame)
    print(f"Rows: {report.row_count}")
    print(f"Columns: {report.column_count}")
    print(f"Average sqft: {report.average_sqft:.2f}")
    print(f"Average price: {report.average_price:.2f}")
    print(f"Price range: {report.min_price:.2f} - {report.max_price:.2f}")


def evaluate_local_model(data_path):
    result = train_and_evaluate(data_path)
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"RMSE: {result.metrics.rmse:.2f}")
    print(f"MAE: {result.metrics.mae:.2f}")
    print(f"R2: {result.metrics.r2:.4f}")
    print(f"Baseline ({result.baseline.name}) RMSE: {result.baseline.metrics.rmse:.2f}")
    quality = evaluate_quality(result, DEFAULT_GATE)
    print(f"Quality gate passed: {quality.passed}")
    for failure in quality.failures:
        print(f"Gate failure: {failure}")


def predict_local_price(data_path, rooms, sqft):
    model = fit_price_model(data_path)
    prediction = predict_price(model, RentalInput(rooms=rooms, sqft=sqft))
    print(f"Predicted monthly rent: {prediction:.2f}")


def write_reports(data_path, output_dir):
    output = Path(output_dir)
    frame = load_housing_data(data_path)
    dataset = summarize_housing_data(frame)
    result = train_and_evaluate(data_path)
    quality = evaluate_quality(result)
    write_quality_report(output / "quality-report.json", quality)
    (output / "model-card.md").write_text(
        render_model_card(dataset, result, quality),
        encoding="utf-8",
    )
    print(f"Wrote reports to {output.resolve()}")
    if not quality.passed:
        raise SystemExit(2)


def write_artifact(data_path, artifact_path):
    metadata = write_model_artifact(artifact_path, data_path)
    print(f"Wrote model artifact to {Path(artifact_path).resolve()}")
    print(f"Artifact version: {metadata['artifact_version']}")
    print(f"Model type: {metadata['model_type']}")
    print(f"Quality gate passed: {metadata['quality']['passed']}")


def run_pipeline(host, experiment_name):
    import kfp

    client = kfp.Client(host=host)
    client.create_run_from_pipeline_func(
        pipeline_func=build_pipeline(),
        run_name="rental-price-prediction-run",
        experiment_name=experiment_name,
    )
    print(f"Submitted pipeline run to {host}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true", help="Compile the pipeline YAML and exit.")
    parser.add_argument("--no-compile", action="store_true", help="Run local checks without compiling the pipeline.")
    parser.add_argument("--run", action="store_true", help="Submit the pipeline to a Kubeflow endpoint.")
    parser.add_argument("--host", help="Kubeflow Pipelines endpoint, for example http://localhost:8080.")
    parser.add_argument("--experiment", default="Rental Price Prediction")
    parser.add_argument("--output", default=PIPELINE_FILE)
    parser.add_argument("--data", default=DATA_PATH, help="Path to the housing CSV file.")
    parser.add_argument("--validate-data", action="store_true", help="Validate and summarize the dataset.")
    parser.add_argument("--evaluate-local", action="store_true", help="Train and evaluate the local model.")
    parser.add_argument("--predict", action="store_true", help="Fit locally and predict one rental price.")
    parser.add_argument("--write-reports", action="store_true", help="Write model card and quality gate reports.")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--write-artifact", action="store_true", help="Train and write a local model artifact.")
    parser.add_argument("--artifact-path", default="outputs/model/rental-price-model.pkl")
    parser.add_argument("--rooms", type=float, help="Room count for --predict.")
    parser.add_argument("--sqft", type=float, help="Square footage for --predict.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.validate_data:
        validate_dataset(args.data)

    if args.evaluate_local:
        evaluate_local_model(args.data)

    if args.predict:
        if args.rooms is None or args.sqft is None:
            raise ValueError("--rooms and --sqft are required when using --predict")
        predict_local_price(args.data, args.rooms, args.sqft)

    if args.write_reports:
        write_reports(args.data, args.report_dir)

    if args.write_artifact:
        write_artifact(args.data, args.artifact_path)

    if not args.no_compile:
        compile_pipeline(args.output)

    if args.run:
        if not args.host:
            raise ValueError("--host is required when using --run")
        run_pipeline(args.host, args.experiment)
    elif args.no_compile:
        print("Local checks complete. Pipeline compilation was skipped.")
    elif not args.compile_only:
        print("Pipeline compiled. Use --run --host <endpoint> to submit it to Kubeflow.")
