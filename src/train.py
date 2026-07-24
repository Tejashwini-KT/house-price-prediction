"""
Train a simple linear regression model to predict house prices.

Usage:
    python src/train.py
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")

FEATURE_COLUMNS = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]
TARGET_COLUMN = "price"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> tuple[LinearRegression, dict]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics


def main():
    df = load_data()
    model, metrics = train_model(df)

    print("Model trained.")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R2:  {metrics['r2']:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
