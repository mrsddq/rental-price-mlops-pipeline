import argparse
from pathlib import Path

import kfp
from kfp import compiler, dsl

from rental_mlops.data import load_housing_data, summarize_housing_data
from rental_mlops.model import train_and_evaluate
from rental_mlops.predict import RentalInput, fit_price_model, predict_price


PIPELINE_FILE = "rental_price_prediction_pipeline.yaml"
DATA_PATH = "data/housing_1000.csv"


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


def compile_pipeline(output_path=PIPELINE_FILE):
    compiler.Compiler().compile(rental_price_prediction_pipeline, output_path)
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


def predict_local_price(data_path, rooms, sqft):
    model = fit_price_model(data_path)
    prediction = predict_price(model, RentalInput(rooms=rooms, sqft=sqft))
    print(f"Predicted monthly rent: {prediction:.2f}")


def run_pipeline(host, experiment_name):
    client = kfp.Client(host=host)
    client.create_run_from_pipeline_func(
        pipeline_func=rental_price_prediction_pipeline,
        run_name="rental-price-prediction-run",
        experiment_name=experiment_name,
    )
    print(f"Submitted pipeline run to {host}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true", help="Compile the pipeline YAML and exit.")
    parser.add_argument("--run", action="store_true", help="Submit the pipeline to a Kubeflow endpoint.")
    parser.add_argument("--host", help="Kubeflow Pipelines endpoint, for example http://localhost:8080.")
    parser.add_argument("--experiment", default="Rental Price Prediction")
    parser.add_argument("--output", default=PIPELINE_FILE)
    parser.add_argument("--data", default=DATA_PATH, help="Path to the housing CSV file.")
    parser.add_argument("--validate-data", action="store_true", help="Validate and summarize the dataset.")
    parser.add_argument("--evaluate-local", action="store_true", help="Train and evaluate the local model.")
    parser.add_argument("--predict", action="store_true", help="Fit locally and predict one rental price.")
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

    compile_pipeline(args.output)

    if args.run:
        if not args.host:
            raise ValueError("--host is required when using --run")
        run_pipeline(args.host, args.experiment)
    elif not args.compile_only:
        print("Pipeline compiled. Use --run --host <endpoint> to submit it to Kubeflow.")
