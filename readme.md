# Freight Rate Prediction Challenge

Machine learning project for predicting freight load rates using the provided Spotter assessment data.

## Approach

* Cleaned missing and invalid values.
* Converted negative `weight` values to missing values and used median imputation.
* Extracted `month` from the `date` column.
* Used an 80/20 train-test split with `random_state=42`.
* Used 5-fold cross-validation.
* Applied `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
* Trained an XGBoost regression model.

## Model Results

| Metric |  Score |
| ------ | -----: |
| MAE    | 120.18 |
| RMSE   | 548.98 |
| R²     | 0.8591 |

## Final Predictions

The trained model generated predictions for all **12,000 validation loads**.

The provided scorer confirmed:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
```

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the model:

```bash
python final.py
```

Run the scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
