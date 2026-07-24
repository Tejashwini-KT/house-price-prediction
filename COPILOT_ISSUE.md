Copy everything below into a new GitHub issue, then assign it to Copilot.

---

Title: Add a /predict CLI option to save predictions to a CSV log

Description:
Right now `src/predict.py` only prints the predicted price to the console.
Add an optional `--log` flag so that each prediction (inputs + predicted
price + timestamp) is appended as a new row to `predictions_log.csv` in the
project root. If the file doesn't exist yet, create it with a header row.

Acceptance criteria:
- New `--log` flag on the existing argparse CLI in src/predict.py
- When passed, append a row to predictions_log.csv with columns:
  timestamp, area_sqft, bedrooms, bathrooms, age_years, location_score,
  predicted_price
- If --log is not passed, behavior is unchanged (just prints to console)
- Add a unit test in tests/ covering the logging behavior
- Do not change the existing predict_price() function signature

Hints: relevant files are src/predict.py and tests/test_predict.py

---

Follow-up comment to leave on the PR (to demo iteration):

"@copilot also make sure predictions_log.csv is added to .gitignore so we
don't accidentally commit generated logs"
