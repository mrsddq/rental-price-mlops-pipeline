import argparse
from pathlib import Path

import kfp
from kfp import compiler, dsl


PIPELINE_FILE = "rental_price_prediction_pipeline.yaml"
DATA_URL = "https://raw.githubusercontent.com/mrsddq/mlops-project/master/data/housing_1000.csv"


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas==2.2.2", "numpy==2.1.1", "scikit-learn==1.5.1"],
)
def train_rental_price_model(data_url: str = DATA_URL) -> float:
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(data_url)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compile_pipeline(args.output)

    if args.run:
        if not args.host:
            raise ValueError("--host is required when using --run")
        run_pipeline(args.host, args.experiment)
    elif not args.compile_only:
        print("Pipeline compiled. Use --run --host <endpoint> to submit it to Kubeflow.")
