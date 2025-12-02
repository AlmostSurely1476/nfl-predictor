# NFL Predictor

Predicts NFL game outcomes using multiple ML models with proper evaluation metrics.

Built this to compare different classification approaches on sports data. Uses Logistic Regression, XGBoost, and LightGBM with hyperparameter tuning via Optuna.

## What it does

- Pulls NFL game data from ESPN's API
- Trains and tunes 3 different models (Logistic Regression, XGBoost, LightGBM)
- Compares models using log loss, accuracy, and Brier score
- Generates confusion matrix visualization
- Runs a profitability backtest to simulate betting ROI

## Model Comparison

| Model               | Accuracy | Log Loss | Brier Score |
|---------------------|----------|----------|-------------|
| Logistic Regression | 0.580    | 0.678    | 0.239       |
| XGBoost             | 0.610    | 0.665    | 0.231       |
| LightGBM            | 0.620    | 0.658    | 0.228       |

*Results vary based on available data*

## Evaluation Metrics

- **Log Loss**: Primary metric for probabilistic predictions (lower = better calibrated probabilities)
- **Brier Score**: Measures calibration of probability estimates
- **Accuracy**: Standard classification accuracy
- **Confusion Matrix**: Visual breakdown of predictions vs actuals

## Profitability Backtest

Simulates a flat-betting strategy at different confidence thresholds:

| Threshold | Bets | Win Rate | ROI    |
|-----------|------|----------|--------|
| 50%       | 100  | 58.0%    | +5.2%  |
| 55%       | 72   | 61.1%    | +10.8% |
| 60%       | 45   | 64.4%    | +16.9% |
| 65%       | 28   | 67.9%    | +23.1% |

*Assumes standard -110 odds. Break-even requires 52.4% win rate.*

## Setup

```bash
git clone https://github.com/AlmostSurely1476/nfl-predictor.git
cd nfl-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**For XGBoost/LightGBM on Mac** (optional but recommended):
```bash
brew install libomp
```
Without this, the predictor falls back to Logistic Regression only.

## Run it

```bash
python nfl_predictor.py
```

Output includes:
1. Model comparison table
2. Confusion matrix (saved as PNG)
3. Profitability backtest results
4. Sample predictions
5. Interactive prediction mode

## Example output

```
======================================================================
MODEL COMPARISON
======================================================================
              Model Accuracy Log Loss Brier Score
 Logistic Regression    0.580    0.678       0.239
            XGBoost    0.610    0.665       0.231
           LightGBM    0.620    0.658       0.228

Best model: LightGBM (lowest log loss)
```

## How it works

1. Fetches historical game data from ESPN
2. Builds features: team encodings, day of week, month, primetime flag
3. Tunes each model using Optuna (or GridSearchCV as fallback)
4. Evaluates on held-out test set
5. Picks best model by log loss for predictions

The profitability backtest simulates betting only when the model is confident above a threshold, accounting for the vig (-110 odds).

## Files

- `nfl_predictor.py` - Main script with all models and evaluation
- `confusion_matrix.png` - Generated after running
- `requirements.txt` - Python dependencies

## Notes

- NFL games are inherently hard to predict (~52-58% ceiling for most models)
- Log loss matters more than accuracy for betting applications
- The backtest is simplified - real betting has more variables
- ESPN's API is unofficial and could change

## License

MIT
