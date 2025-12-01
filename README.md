# NFL Predictor

Predicts NFL game outcomes using data from ESPN's API and a RandomForest model.

Built this because I wanted to mess around with sports data and see if I could beat my friends at picking games. Spoiler: the model is decent but still loses to people who actually watch football.

## What it does

- Pulls live NFL game data from ESPN
- Trains a simple ML model on the results
- Spits out score predictions and win probabilities

## Setup

```bash
git clone https://github.com/AlmostSurely1476/nfl-predictor.git
cd nfl-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python nfl_predictor.py
```

You'll see some example predictions, then you can enter your own matchups.

## Example output

```
============================================================
GAME PREDICTION: Buffalo Bills @ Kansas City Chiefs
============================================================
Predicted Score:
  Kansas City Chiefs                           27.5
  Buffalo Bills                                24.3

Predicted Winner: Kansas City Chiefs
Home Win Probability: 68.2%
Away Win Probability: 31.8%
============================================================
```

## How it works

1. Grabs current season data from ESPN's public API
2. Builds features from team matchups and game dates
3. Trains a RandomForest regressor to predict scores
4. Uses a sigmoid function to convert score differential to win probability

If there's not enough real data (like early in the season), it falls back to synthetic training data so the model still runs.

## Notes

- The model isn't amazing - NFL games are hard to predict
- Works best mid-to-late season when there's more data
- ESPN's API is unofficial so it might break if they change things

## License

MIT
