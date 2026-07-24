import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict_price  # noqa: E402
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
