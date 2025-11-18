import kfp
from kfp import dsl
from kfp import compiler
from kfp import components


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas == 1.2.4', 'numpy == 1.21.0', 'scikit-learn == 0.24.2']
)
def predict_rental_price_model() -> float:
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mrsddq/MLOps-Project/refs/heads/master/src/data/housing_1000.csv")
    
    # Prepare Data
    X = df[['rooms', 'sqft']].values # Features - rooms and sqft
    y = df['price' ].values
    
    # Split Data for Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training
    model = LinearRegression().fit(X_train, y_train) # Train the model
    
    # Actuals and Predictions from the dataset
    #predicted_rental = model.predict(np.array([[X_test[0][0], X_test[0][1]]]))
    predicted_rental = model.predict(X_test[0].reshape(1,-1))[0]
    
    print("Actual Rental Price for Property with rooms=",X_test[0][0],"and Area Sqft=",X_test[0][1],"is=",y_test[0])
    print("Predicted Rental Price for Property with rooms=",X_test[0][0],"and Area Sqft=",X_test[0][1],"is=",predicted_rental)
    
    return float(predicted_rental) # Return a single float value

@dsl.pipeline(name='rental-price-prediction-pipeline')

def rental_price_prediction_pipeline() -> float:
    prepare_data_task = predict_rental_price_model()
    return prepare_data_task.output

if __name__ == "__main__":
    # compile the pipeline
    kfp.compiler.Compiler().compile(rental_price_prediction_pipeline, 'rental_price_prediction_pipeline.yaml')
    
    # connect to kfp server
    # host = 'http://localhost:8080'
    kfp_endpoint = 'http://localhost:8080'
    client = kfp.Client(host=kfp_endpoint)
    
    # create an experiment
    experiment_name = 'Predict Rental Price Experiment'
    experiment = client.create_experiment(name=experiment_name)
    print(f'Experiment created: {experiment}')

    # run the pipeline using create_run from pipeline func
    run_name = f'Run of Rental Price Prediction Pipeline'
    run_result = client.create_run_from_pipeline_func(
        pipeline_func=rental_price_prediction_pipeline,
        run_name=run_name,
        experiment_name=experiment_name
    )
    print(f'Pipeline run initiated: {run_result}')