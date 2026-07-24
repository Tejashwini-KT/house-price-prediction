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

# run tests
pytest tests/ -v
```

## Demo: assigning an incremental feature to Copilot coding agent

1. Push this repo to GitHub.
2. Enable Copilot coding agent on the repo (Settings → Copilot → Coding agent).
3. Open a new issue using the template in `COPILOT_ISSUE.md`.
4. Assign the issue to **Copilot**.
5. Watch progress at github.com/copilot/agents.
6. Review the pull request Copilot opens, then comment to ask for a follow-up
   change and watch it iterate.
