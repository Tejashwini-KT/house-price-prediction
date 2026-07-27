"""
Load the trained model and predict a house price for given features.

Usage:
    python src/predict.py --area 2000 --bedrooms 3 --bathrooms 2 --age 5 --location 8
    python src/predict.py --area 2000 --bedrooms 3 --bathrooms 2 --age 5 --location 8 --log
"""
import os
import argparse
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
PREDICTIONS_LOG_PATH = "predictions_log.csv"
FEATURE_COLUMNS = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]
LOG_COLUMNS = ["timestamp", "area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "predicted_price"]


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def predict_price(area_sqft: float, bedrooms: int, bathrooms: int,
                   age_years: int, location_score: int, model=None) -> float:
    """Predict a house price from feature inputs. Loads the saved model if none is passed in."""
    if model is None:
        model = load_model()

    input_df = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "location_score": location_score,
    }])[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


def log_prediction(area_sqft: float, bedrooms: int, bathrooms: int,
                   age_years: int, location_score: int, predicted_price: float,
                   log_path: str = PREDICTIONS_LOG_PATH):
    """
    Log a prediction to a CSV file.
    
    Creates the file with headers if it doesn't exist.
    Appends a new record with timestamp and prediction data.
    """
    timestamp = datetime.now().isoformat()
    
    # Create a DataFrame with the new record
    new_record = pd.DataFrame([{
        "timestamp": timestamp,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "location_score": location_score,
        "predicted_price": predicted_price,
    }])
    
    # Check if file exists and has content
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    
    # Append to CSV: write header only if file is new
    new_record.to_csv(log_path, mode='a', header=not file_exists, index=False)


def main():
    parser = argparse.ArgumentParser(description="Predict a house price.")
    parser.add_argument("--area", type=float, required=True, help="Area in sq ft")
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--age", type=int, required=True, help="Age of house in years")
    parser.add_argument("--location", type=int, required=True, help="Location score 1-10")
    parser.add_argument("--log", action="store_true", help="Log prediction to CSV")
    args = parser.parse_args()

    price = predict_price(
        area_sqft=args.area,
        bedrooms=args.bedrooms,
        bathrooms=args.bathrooms,
        age_years=args.age,
        location_score=args.location,
    )
    print(f"Predicted price: ${price:,.2f}")
    
    if args.log:
        log_prediction(
            area_sqft=args.area,
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            age_years=args.age,
            location_score=args.location,
            predicted_price=price,
        )
        print(f"Prediction logged to {PREDICTIONS_LOG_PATH}")


if __name__ == "__main__":
    main()
