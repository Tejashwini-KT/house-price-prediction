import os
import sys
import tempfile
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict_price, log_prediction  # noqa: E402
from train import load_data, train_model  # noqa: E402


def test_predict_price_returns_positive_number():
    df = load_data()
    model, _ = train_model(df)

    price = predict_price(
        area_sqft=2000,
        bedrooms=3,
        bathrooms=2,
        age_years=5,
        location_score=8,
        model=model,
    )

    assert isinstance(price, float)
    assert price > 0


def test_predict_price_larger_house_costs_more():
    df = load_data()
    model, _ = train_model(df)

    small_house_price = predict_price(1000, 2, 1, 10, 5, model=model)
    large_house_price = predict_price(4000, 2, 1, 10, 5, model=model)

    assert large_house_price > small_house_price


def test_log_prediction_creates_csv_with_headers():
    """Test that log_prediction creates a CSV file with proper headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_log.csv")
        
        log_prediction(
            area_sqft=2000,
            bedrooms=3,
            bathrooms=2,
            age_years=5,
            location_score=8,
            predicted_price=350000.0,
            log_path=log_path
        )
        
        # Check file exists
        assert os.path.exists(log_path)
        
        # Read and verify
        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert list(df.columns) == [
            "timestamp", "area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "predicted_price"
        ]
        assert df["area_sqft"].iloc[0] == 2000
        assert df["bedrooms"].iloc[0] == 3
        assert df["predicted_price"].iloc[0] == 350000.0


def test_log_prediction_appends_records():
    """Test that log_prediction appends new records without duplicating headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_log.csv")
        
        # First prediction
        log_prediction(
            area_sqft=2000,
            bedrooms=3,
            bathrooms=2,
            age_years=5,
            location_score=8,
            predicted_price=350000.0,
            log_path=log_path
        )
        
        # Second prediction
        log_prediction(
            area_sqft=3000,
            bedrooms=4,
            bathrooms=3,
            age_years=10,
            location_score=9,
            predicted_price=450000.0,
            log_path=log_path
        )
        
        # Read and verify
        df = pd.read_csv(log_path)
        assert len(df) == 2
        assert df["area_sqft"].iloc[0] == 2000
        assert df["area_sqft"].iloc[1] == 3000
        assert df["predicted_price"].iloc[0] == 350000.0
        assert df["predicted_price"].iloc[1] == 450000.0


def test_log_prediction_contains_timestamp():
    """Test that logged records contain ISO-formatted timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_log.csv")
        
        log_prediction(
            area_sqft=2000,
            bedrooms=3,
            bathrooms=2,
            age_years=5,
            location_score=8,
            predicted_price=350000.0,
            log_path=log_path
        )
        
        df = pd.read_csv(log_path)
        timestamp = df["timestamp"].iloc[0]
        
        # Verify timestamp is ISO format
        assert "T" in timestamp  # ISO format has 'T' separator
        assert len(timestamp) > 0
