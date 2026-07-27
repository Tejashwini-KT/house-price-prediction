# House Price Prediction

A small ML project used to demo **GitHub Copilot coding agent** and its
"implement incremental features" capability.

## What's here

- `data/house_prices.csv` — synthetic dataset (300 rows): area, bedrooms,
  bathrooms, age, location score → price
- `src/train.py` — trains a linear regression model, prints MAE/R2, saves
  `model.joblib`
- `src/predict.py` — loads the trained model and predicts a price from
  feature inputs
- `tests/test_predict.py` — basic unit tests
- `.github/workflows/copilot-setup-steps.yml` — environment setup steps that
  GitHub Copilot coding agent runs before starting any assigned task

## Run it locally

```bash
pip install -r requirements.txt

# train the model
python src/train.py

# predict a price
python src/predict.py --area 2200 --bedrooms 3 --bathrooms 2 --age 8 --location 7

# predict a price and log to CSV
python src/predict.py --area 2200 --bedrooms 3 --bathrooms 2 --age 8 --location 7 --log

# run tests
pytest tests/ -v
```

## Prediction Logging

When you use the `--log` flag, predictions are automatically logged to `predictions_log.csv` with the following fields:

- `timestamp` — ISO 8601 timestamp of when the prediction was made
- `area_sqft` — house area in square feet
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `age_years` — age of the house in years
- `location_score` — location score (1-10)
- `predicted_price` — the predicted house price

### Example

```bash
# Make a prediction and log it
python src/predict.py --area 2200 --bedrooms 3 --bathrooms 2 --age 8 --location 7 --log
# Output: Predicted price: $425,123.45
#         Prediction logged to predictions_log.csv

# Make another prediction and log it
python src/predict.py --area 1500 --bedrooms 2 --bathrooms 1 --age 15 --location 6 --log
# Output: Predicted price: $305,678.90
#         Prediction logged to predictions_log.csv
```

The log file will accumulate records over time. You can view it with:

```bash
cat predictions_log.csv
```

Note: The `predictions_log.csv` file is added to `.gitignore` to keep your repository clean.

## Demo: assigning an incremental feature to Copilot coding agent

1. Push this repo to GitHub.
2. Enable Copilot coding agent on the repo (Settings → Copilot → Coding agent).
3. Open a new issue using the template in `COPILOT_ISSUE.md`.
4. Assign the issue to **Copilot**.
5. Watch progress at github.com/copilot/agents.
6. Review the pull request Copilot opens, then comment to ask for a follow-up
   change and watch it iterate.
