"""
NFL game predictor - pulls data from ESPN and tries to predict outcomes
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


class NFLPredictor:
    def __init__(self):
        self.base_url = ESPN_URL
        self.teams_data = {}
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_current_season_data(self):
        """Get current season games from ESPN"""
        try:
            resp = requests.get(f"{self.base_url}/scoreboard", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            games = []
            for event in data.get('events', []):
                comp = event['competitions'][0]
                
                if 'home' not in comp or 'away' not in comp:
                    continue
                
                home = comp['home']
                away = comp['away']
                
                games.append({
                    'date': event['date'],
                    'home_team': home['team']['name'],
                    'away_team': away['team']['name'],
                    'home_score': home.get('score', 0),
                    'away_score': away.get('score', 0),
                    'status': event['status']['type']['name'],
                    'home_id': home['team']['id'],
                    'away_id': away['team']['id'],
                })
            
            return games
        except requests.RequestException as e:
            print(f"Couldn't fetch data: {e}")
            return []
    
    def fetch_team_stats(self, team_id):
        """Get stats for a specific team"""
        try:
            resp = requests.get(f"{self.base_url}/teams/{team_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            stats = {
                'team_name': data['team']['name'],
                'wins': 0,
                'losses': 0,
                'points_for': 0,
                'points_against': 0,
            }
            
            if 'record' in data['team']:
                for rec in data['team']['record']:
                    if rec['type'] == 'total':
                        stats['wins'] = rec.get('wins', 0)
                        stats['losses'] = rec.get('losses', 0)
            
            return stats
        except requests.RequestException:
            return None
    
    def create_training_data(self, games):
        """Turn game results into training features"""
        data = []
        
        for game in games:
            if game['status'] != 'Final':
                continue
            
            dt = datetime.fromisoformat(game['date'].replace('Z', '+00:00'))
            
            data.append({
                'home_team': hash(game['home_team']) % 32,
                'away_team': hash(game['away_team']) % 32,
                'day_of_week': dt.weekday(),
                'month': dt.month,
                'home_score': float(game['home_score']),
                'away_score': float(game['away_score']),
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, games):
        """Train the prediction model"""
        df = self.create_training_data(games)
        
        if df.empty or len(df) < 5:
            print("Not enough data, using synthetic training...")
            self._init_default_model()
            return True
        
        X = df[['home_team', 'away_team', 'day_of_week', 'month', 'away_score']]
        y = df['home_score']
        
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X, y
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        self.model.fit(X_scaled, y_train)
        
        score = self.model.score(X_scaled, y_train)
        print(f"Trained on {len(df)} games (R² = {score:.3f})")
        
        return True
    
    def _init_default_model(self):
        """Fallback model when we don't have enough real data"""
        X = np.random.rand(100, 5) * 32
        y = X[:, 4] * 2 + np.random.randn(100) * 3 + 20
        y = np.clip(y, 0, 60)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        self.model.fit(X_scaled, y)
        print("Using synthetic model")
    
    def predict_game(self, home_team, away_team):
        """Predict outcome of a matchup"""
        if self.model is None:
            self._init_default_model()
        
        now = datetime.now()
        
        # predict home score
        home_features = np.array([[
            hash(home_team) % 32,
            hash(away_team) % 32,
            now.weekday(),
            now.month,
            20
        ]])
        home_score = max(0, self.model.predict(self.scaler.transform(home_features))[0])
        
        # predict away score
        away_features = np.array([[
            hash(away_team) % 32,
            hash(home_team) % 32,
            now.weekday(),
            now.month,
            20
        ]])
        away_score = max(0, self.model.predict(self.scaler.transform(away_features))[0])
        
        # win probability via sigmoid
        diff = home_score - away_score
        home_prob = 1 / (1 + np.exp(-diff / 10))
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': round(home_score, 1),
            'away_score': round(away_score, 1),
            'home_prob': round(home_prob * 100, 1),
            'away_prob': round((1 - home_prob) * 100, 1),
            'winner': home_team if home_score > away_score else away_team
        }
    
    def print_prediction(self, pred):
        """Display a prediction"""
        if not pred:
            return
        
        print("\n" + "=" * 60)
        print(f"  {pred['away_team']} @ {pred['home_team']}")
        print("=" * 60)
        print(f"  {pred['home_team']:<35} {pred['home_score']:>5}")
        print(f"  {pred['away_team']:<35} {pred['away_score']:>5}")
        print()
        print(f"  Winner: {pred['winner']}")
        print(f"  Home win: {pred['home_prob']}% | Away win: {pred['away_prob']}%")
        print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  NFL PREDICTOR")
    print("=" * 60)
    print("Fetching data from ESPN...\n")
    
    predictor = NFLPredictor()
    games = predictor.fetch_current_season_data()
    print(f"Got {len(games)} games\n")
    
    print("Training model...")
    predictor.train_model(games)
    
    # some sample predictions
    print("\n--- Sample Predictions ---")
    
    matchups = [
        ("Kansas City Chiefs", "Buffalo Bills"),
        ("San Francisco 49ers", "Dallas Cowboys"),
        ("Detroit Lions", "Green Bay Packers"),
        ("Philadelphia Eagles", "Washington Commanders"),
        ("Cincinnati Bengals", "Baltimore Ravens"),
    ]
    
    for home, away in matchups:
        pred = predictor.predict_game(home, away)
        predictor.print_prediction(pred)
    
    # interactive mode
    print("\n--- Custom Predictions ---")
    print("Enter matchups below (or 'quit' to exit)\n")
    
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
