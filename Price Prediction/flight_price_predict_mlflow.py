import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def run_training():
    # MLflow Setup
    mlflow.set_tracking_uri("http://mlflow:5000")  # Container alias works inside Docker
    mlflow.set_experiment("Flight Price Prediction")

    with mlflow.start_run():
        # Load dataset
        df = pd.read_csv("/app/Mlflow/flights.csv")

        # Parse and engineer date
        df['date'] = pd.to_datetime(df['date'])
        df['week_day'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['week_no'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        df.rename(columns={"to": "destination"}, inplace=True)

        # Derived feature
        df['flight_speed'] = round(df['distance'] / df['time'], 2)

        # Encoding
        df = pd.get_dummies(df, columns=['from', 'destination', 'flightType', 'agency'])

        # Drop irrelevant columns
        df.drop(columns=['time', 'flight_speed', 'month', 'year', 'distance'], axis=1, inplace=True)

        # Features and labels
        X = df.drop(['price', 'date'], axis=1)

        y = df['price']

        # Clean column names
        X.columns = [col.replace(" ", "_").replace("(", "").replace(")", "") for col in X.columns]

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training with GridSearch
        param_grid = {
            'n_estimators': [300],
            'max_depth': [15],
            'min_samples_split': [10],
            'max_features': ['sqrt'],
            'n_jobs': [2]
        }

        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', verbose=2)
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Save the model
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        print("✅ MLflow tracking complete. Run logged.")


if __name__ == "__main__":
    run_training()
