"""
NFL game predictor with multiple ML models and proper evaluation
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# try importing optional deps
try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, Exception):
    HAS_LGB = False
    lgb = None

try:
    import optuna
    HAS_OPTUNA = True
except (ImportError, Exception):
    HAS_OPTUNA = False
    optuna = None

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


class NFLPredictor:
    def __init__(self):
        self.base_url = ESPN_URL
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        
    def fetch_season_data(self, year=2024):
        """Fetch full season data from ESPN"""
        all_games = []
        
        # fetch multiple weeks
        for week in range(1, 19):
            try:
                url = f"{self.base_url}/scoreboard?dates={year}&week={week}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                for event in data.get('events', []):
                    comp = event['competitions'][0]
                    if 'competitors' not in comp:
                        continue
                    
                    home = away = None
                    for team in comp['competitors']:
                        if team['homeAway'] == 'home':
                            home = team
                        else:
                            away = team
                    
                    if not home or not away:
                        continue
                    
                    status = event['status']['type']['name']
                    if status != 'STATUS_FINAL':
                        continue
                    
                    all_games.append({
                        'date': event['date'],
                        'week': week,
                        'home_team': home['team']['displayName'],
                        'away_team': away['team']['displayName'],
                        'home_score': int(home.get('score', 0)),
                        'away_score': int(away.get('score', 0)),
                        'home_id': home['team']['id'],
                        'away_id': away['team']['id'],
                    })
            except Exception as e:
                continue
        
        return all_games
    
    def fetch_current_week(self):
        """Get this week's games for prediction"""
        try:
            resp = requests.get(f"{self.base_url}/scoreboard", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            games = []
            for event in data.get('events', []):
                comp = event['competitions'][0]
                if 'competitors' not in comp:
                    continue
                
                home = away = None
                for team in comp['competitors']:
                    if team['homeAway'] == 'home':
                        home = team
                    else:
                        away = team
                
                if home and away:
                    games.append({
                        'home_team': home['team']['displayName'],
                        'away_team': away['team']['displayName'],
                        'status': event['status']['type']['name']
                    })
            
            return games
        except:
            return []
    
    def build_features(self, games):
        """Create feature matrix from games"""
        if not games:
            return None, None
        
        df = pd.DataFrame(games)
        
        # encode teams
        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        if 'team' not in self.label_encoders:
            self.label_encoders['team'] = LabelEncoder()
            self.label_encoders['team'].fit(all_teams)
        
        le = self.label_encoders['team']
        
        # handle unseen teams
        def safe_transform(teams):
            result = []
            for t in teams:
                if t in le.classes_:
                    result.append(le.transform([t])[0])
                else:
                    result.append(-1)
            return np.array(result)
        
        df['home_encoded'] = safe_transform(df['home_team'])
        df['away_encoded'] = safe_transform(df['away_team'])
        
        # parse dates
        df['datetime'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
        df['is_primetime'] = df['datetime'].dt.hour >= 19
        
        # target: did home team win?
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['point_diff'] = df['home_score'] - df['away_score']
        
        features = ['home_encoded', 'away_encoded', 'day_of_week', 'month', 
                    'week', 'is_sunday', 'is_primetime']
        
        X = df[features].values
        y = df['home_win'].values
        
        return X, y, df
    
    def tune_logistic_regression(self, X_train, y_train):
        """Tune logistic regression with grid search"""
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid, cv=5, scoring='neg_log_loss', n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f"  Logistic Regression best params: {grid.best_params_}")
        return grid.best_estimator_
    
    def tune_xgboost(self, X_train, y_train):
        """Tune XGBoost with optuna or grid search"""
        if not HAS_XGB:
            return None
        
        if HAS_OPTUNA:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_train)[:, 1]
                return log_loss(y_train, probs)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['use_label_encoder'] = False
            best_params['eval_metric'] = 'logloss'
            
            print(f"  XGBoost best params: {best_params}")
            return xgb.XGBClassifier(**best_params)
        else:
            # fallback to grid search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            
            grid = GridSearchCV(
                xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            print(f"  XGBoost best params: {grid.best_params_}")
            return grid.best_estimator_
    
    def tune_lightgbm(self, X_train, y_train):
        """Tune LightGBM with optuna or grid search"""
        if not HAS_LGB:
            return None
        
        if HAS_OPTUNA:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_train)[:, 1]
                return log_loss(y_train, probs)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbose'] = -1
            
            print(f"  LightGBM best params: {best_params}")
            return lgb.LGBMClassifier(**best_params)
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            
            grid = GridSearchCV(
                lgb.LGBMClassifier(random_state=42, verbose=-1),
                param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            print(f"  LightGBM best params: {grid.best_params_}")
            return grid.best_estimator_
    
    def train_all_models(self, X_train, y_train):
        """Train and tune all available models"""
        print("\nTuning models...")
        
        # logistic regression
        print("  Training Logistic Regression...")
        self.models['Logistic Regression'] = self.tune_logistic_regression(X_train, y_train)
        
        # xgboost
        if HAS_XGB:
            print("  Training XGBoost...")
            self.models['XGBoost'] = self.tune_xgboost(X_train, y_train)
            self.models['XGBoost'].fit(X_train, y_train)
        
        # lightgbm
        if HAS_LGB:
            print("  Training LightGBM...")
            self.models['LightGBM'] = self.tune_lightgbm(X_train, y_train)
            self.models['LightGBM'].fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and compare"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        results = []
        
        for name, model in self.models.items():
            if model is None:
                continue
            
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            ll = log_loss(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            
            results.append({
                'Model': name,
                'Accuracy': f"{acc:.3f}",
                'Log Loss': f"{ll:.3f}",
                'Brier Score': f"{brier:.3f}"
            })
            
            self.evaluation_results[name] = {
                'accuracy': acc,
                'log_loss': ll,
                'brier': brier,
                'predictions': preds,
                'probabilities': probs
            }
        
        # print comparison table
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        print()
        
        # find best model by log loss (lower is better)
        best_name = min(self.evaluation_results.keys(), 
                       key=lambda x: self.evaluation_results[x]['log_loss'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"Best model: {best_name} (lowest log loss)")
        
        return df_results
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='confusion_matrix.png'):
        """Generate confusion matrix for best model"""
        if self.best_model is None:
            print("No model trained yet")
            return
        
        preds = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Away Win', 'Home Win'],
                   yticklabels=['Away Win', 'Home Win'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
        
        # print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=['Away Win', 'Home Win']))
    
    def backtest_profitability(self, X_test, y_test, df_test):
        """
        Simulate betting strategy to show model's financial value.
        Strategy: bet on predicted winner when model confidence > threshold
        """
        print("\n" + "=" * 70)
        print("PROFITABILITY BACKTEST")
        print("=" * 70)
        
        if self.best_model is None:
            print("No model trained")
            return
        
        probs = self.best_model.predict_proba(X_test)[:, 1]
        
        # simulate flat betting with -110 odds (standard)
        # need to win 52.4% to break even
        results = []
        
        for threshold in [0.50, 0.55, 0.60, 0.65]:
            bankroll = 1000
            bet_size = 10
            bets_made = 0
            wins = 0
            
            for i, prob in enumerate(probs):
                # bet home if prob > threshold, away if (1-prob) > threshold
                if prob > threshold:
                    bet_home = True
                    bets_made += 1
                elif (1 - prob) > threshold:
                    bet_home = False
                    bets_made += 1
                else:
                    continue
                
                # check result
                actual_home_win = y_test[i] == 1
                
                if bet_home == actual_home_win:
                    bankroll += bet_size * 0.91  # -110 odds means win $91 on $100
                    wins += 1
                else:
                    bankroll -= bet_size
            
            if bets_made > 0:
                win_rate = wins / bets_made
                roi = (bankroll - 1000) / (bets_made * bet_size) * 100
            else:
                win_rate = 0
                roi = 0
            
            results.append({
                'Threshold': f"{threshold:.0%}",
                'Bets': bets_made,
                'Wins': wins,
                'Win Rate': f"{win_rate:.1%}",
                'Final Bankroll': f"${bankroll:.2f}",
                'ROI': f"{roi:.1f}%"
            })
        
        df_backtest = pd.DataFrame(results)
        print(df_backtest.to_string(index=False))
        print()
        print("Note: Assumes -110 odds (bet $110 to win $100). Break-even is 52.4% win rate.")
        
        return df_backtest
    
    def predict_game(self, home_team, away_team):
        """Predict a single game"""
        if self.best_model is None:
            print("Model not trained yet")
            return None
        
        le = self.label_encoders.get('team')
        if le is None:
            return None
        
        # encode teams
        def get_encoding(team):
            if team in le.classes_:
                return le.transform([team])[0]
            return -1
        
        now = datetime.now()
        features = np.array([[
            get_encoding(home_team),
            get_encoding(away_team),
            now.weekday(),
            now.month,
            14,  # assume mid-season
            1 if now.weekday() == 6 else 0,
            1 if now.hour >= 19 else 0
        ]])
        
        prob = self.best_model.predict_proba(features)[0][1]
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': round(prob * 100, 1),
            'away_win_prob': round((1 - prob) * 100, 1),
            'predicted_winner': home_team if prob > 0.5 else away_team,
            'confidence': round(abs(prob - 0.5) * 200, 1)
        }
    
    def print_prediction(self, pred):
        if not pred:
            return
        
        print("\n" + "=" * 60)
        print(f"  {pred['away_team']} @ {pred['home_team']}")
        print("=" * 60)
        print(f"  Predicted Winner: {pred['predicted_winner']}")
        print(f"  Home Win: {pred['home_win_prob']}% | Away Win: {pred['away_win_prob']}%")
        print(f"  Confidence: {pred['confidence']}%")
        print("=" * 60)


def generate_synthetic_data(n_games=500):
    """Generate realistic synthetic NFL data for demo/testing"""
    np.random.seed(42)
    
    teams = [
        'Kansas City Chiefs', 'Buffalo Bills', 'San Francisco 49ers',
        'Philadelphia Eagles', 'Dallas Cowboys', 'Detroit Lions',
        'Baltimore Ravens', 'Cincinnati Bengals', 'Miami Dolphins',
        'Green Bay Packers', 'Seattle Seahawks', 'Los Angeles Rams',
        'New York Giants', 'New York Jets', 'New England Patriots',
        'Pittsburgh Steelers', 'Cleveland Browns', 'Denver Broncos',
        'Las Vegas Raiders', 'Los Angeles Chargers', 'Minnesota Vikings',
        'Chicago Bears', 'Tampa Bay Buccaneers', 'New Orleans Saints',
        'Atlanta Falcons', 'Carolina Panthers', 'Arizona Cardinals',
        'Tennessee Titans', 'Indianapolis Colts', 'Jacksonville Jaguars',
        'Houston Texans', 'Washington Commanders'
    ]
    
    # team strength (some teams are just better)
    team_strength = {team: np.random.uniform(0.3, 0.7) for team in teams}
    
    games = []
    for i in range(n_games):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        
        # home advantage + team strength determines outcome
        home_strength = team_strength[home] + 0.05  # home field advantage
        away_strength = team_strength[away]
        
        home_win_prob = home_strength / (home_strength + away_strength)
        home_win = np.random.random() < home_win_prob
        
        # generate scores
        if home_win:
            home_score = np.random.randint(17, 35)
            away_score = np.random.randint(10, home_score)
        else:
            away_score = np.random.randint(17, 35)
            home_score = np.random.randint(10, away_score)
        
        week = (i // 16) + 1
        month = 9 + (week // 5)
        if month > 12:
            month = month - 12
        
        games.append({
            'date': f'2024-{month:02d}-{np.random.randint(1, 28):02d}T13:00:00Z',
            'week': min(week, 18),
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'home_id': str(hash(home) % 100),
            'away_id': str(hash(away) % 100)
        })
    
    return games


def main():
    print("\n" + "=" * 70)
    print("  NFL PREDICTOR - Multi-Model Comparison")
    print("=" * 70)
    
    predictor = NFLPredictor()
    
    # try to fetch real data
    print("\nFetching NFL data from ESPN...")
    games = predictor.fetch_season_data(2024)
    
    if len(games) < 50:
        print(f"Only got {len(games)} games, using synthetic data for demo...")
        games = generate_synthetic_data(500)
    
    print(f"Working with {len(games)} games")
    
    # build features
    X, y, df = predictor.build_features(games)
    
    if X is None:
        print("Failed to build features")
        return
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # train all models
    predictor.train_all_models(X_train_scaled, y_train)
    
    # evaluate and compare
    predictor.evaluate_models(X_test_scaled, y_test)
    
    # confusion matrix
    predictor.plot_confusion_matrix(X_test_scaled, y_test)
    
    # profitability backtest
    df_test = df.iloc[-len(y_test):]
    predictor.backtest_profitability(X_test_scaled, y_test, df_test)
    
    # example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    matchups = [
        ("Kansas City Chiefs", "Buffalo Bills"),
        ("San Francisco 49ers", "Dallas Cowboys"),
        ("Detroit Lions", "Green Bay Packers"),
        ("Philadelphia Eagles", "Washington Commanders"),
    ]
    
    for home, away in matchups:
        pred = predictor.predict_game(home, away)
        predictor.print_prediction(pred)
    
    # interactive mode
    print("\n--- Custom Predictions ---")
    print("Enter matchups (or 'quit' to exit)\n")
    
    while True:
        try:
            away = input("Away team: ").strip()
            if away.lower() == 'quit':
                break
            
            home = input("Home team: ").strip()
            if not away or not home:
                print("Need both teams\n")
                continue
            
            pred = predictor.predict_game(home, away)
            predictor.print_prediction(pred)
        except KeyboardInterrupt:
            break
    
    print("\nDone!")


if __name__ == "__main__":
    main()
