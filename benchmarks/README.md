# Benchmarks & Leaderboard

## Two Evaluation Tracks

### Track 1: Age Prediction Accuracy

Evaluate how well your model predicts chronological age from blood biomarkers.

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (years) — primary ranking metric |
| **RMSE** | Root Mean Squared Error (years) |
| **R²** | Coefficient of determination |
| **Pearson r** | Pearson correlation coefficient |

### Track 2: Mortality Prediction (Survival Analysis)

Evaluate whether your biological age predictions are associated with mortality.

| Metric | Description |
|--------|-------------|
| **Concordance** | Cox PH concordance index |
| **KM Separation** | Kaplan-Meier curve separation between accelerated/decelerated aging groups |

## Current Baselines

| Model | Features | MAE | R² | Concordance |
|-------|----------|-----|-----|-------------|
| Healome Standard | 21 | 5.11 | 0.906 | 0.81 |
| Healome Extended | 35 | 6.07 | 0.873 | 0.83 |

## How to Submit

1. **Train your model** on NHANES data (any cycles, any features)
2. **Evaluate** using the standard test split: `test_size=0.3, random_state=3454`
3. **Generate** your submission JSON:

```python
from healome_clock.evaluation.leaderboard import create_submission, save_submission

submission = create_submission(
    model_name="Your Model Name",
    y_true=y_test,
    y_pred=your_predictions,
    authors="Your Name",
    description="Brief description of your approach",
    model_type="XGBoost",  # or whatever you used
    n_features=25,
    concordance=0.95,  # from Cox PH, if you ran it
)

save_submission(submission, "benchmarks/submissions/your_model.json")
```

4. **Open a PR** adding your JSON file to `benchmarks/submissions/`

See `notebooks/evaluation.ipynb` for a complete walkthrough.

## Rules

- Use NHANES data for training (any cycles from 1999-2020)
- Report metrics on a held-out test set (specify your split method)
- Include enough detail for reproducibility
- Be honest about limitations

## Submission Format

```json
{
  "model_name": "string",
  "authors": "string",
  "description": "string",
  "date": "YYYY-MM-DD",
  "model_type": "string",
  "n_features": 0,
  "training_data": "string",
  "track1_age_prediction": {
    "mae": 0.0,
    "rmse": 0.0,
    "r2": 0.0,
    "pearson_r": 0.0,
    "n_test_samples": 0,
    "test_split_method": "string"
  },
  "track2_survival": {
    "concordance": 0.0,
    "n_mortality_records": 0,
    "km_separation": "string"
  }
}
```
