import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "models/crash_predictor.pth"
FEATURES_FILE = "data/nyc_collisions_features.csv"
TRAIN_FILE = "data/train.csv"
GRID_SIZE = 0.004

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
RISK_COLORS = {0: 'green', 1: 'orange', 2: 'red'}

class CrashPredictor(nn.Module):
    def __init__(self, input_size):
        super(CrashPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        return self.network(x)

def load_model():
    checkpoint = torch.load(MODEL_FILE, weights_only=False)
    model = CrashPredictor(input_size=15)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    scaler = checkpoint['scaler']
    return model, scaler

def get_cell_stats(grid_lat, grid_lon, train_df):
    cell = train_df[(train_df['grid_lat'] == grid_lat) & (train_df['grid_lon'] == grid_lon)]
    if len(cell) == 0:
        return train_df['crash_count'].mean(), train_df['crash_count'].max(), 0.0
    return cell['crash_count'].mean(), cell['crash_count'].max(), cell['crash_count'].std()

def build_features(grid_lat, grid_lon, hour, day_of_week, month, train_df):
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in [7,8,9,16,17,18,19] else 0
    is_night = 1 if hour in [22,23,0,1,2,3,4] else 0
    
    cell_avg, cell_max, cell_std = get_cell_stats(grid_lat, grid_lon, train_df)
    
    hour_avg = train_df[train_df['hour'] == hour]['crash_count'].mean()
    dow_avg = train_df[train_df['day_of_week'] == day_of_week]['crash_count'].mean()
    
    hour_x_dow = hour * day_of_week
    cell_x_hour = cell_avg * hour_avg
    
    return np.array([[
        grid_lat, grid_lon, hour, day_of_week,
        month, is_weekend, is_rush_hour, is_night,
        cell_avg, cell_max, cell_std,
        hour_avg, dow_avg, hour_x_dow, cell_x_hour
    ]])

def predict_risk(latitude, longitude, hour, day_of_week, month):
    model, scaler = load_model()
    train_df = pd.read_csv(TRAIN_FILE)
    
    grid_lat = int(latitude / GRID_SIZE)
    grid_lon = int(longitude / GRID_SIZE)
    
    features = build_features(grid_lat, grid_lon, hour, day_of_week, month, train_df)
    features = scaler.transform(features)
    
    with torch.no_grad():
        output = model(torch.FloatTensor(features))
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)
    
    return {
        'grid_lat': grid_lat,
        'grid_lon': grid_lon,
        'risk_level': RISK_LABELS[predicted_class],
        'risk_color': RISK_COLORS[predicted_class],
        'confidence': f"{probabilities[predicted_class]*100:.1f}%",
        'probabilities': {
            'Low': f"{probabilities[0]*100:.1f}%",
            'Medium': f"{probabilities[1]*100:.1f}%",
            'High': f"{probabilities[2]*100:.1f}%"
        }
    }

def predict_manhattan_heatmap(hour, day_of_week, month):
    train_df = pd.read_csv(TRAIN_FILE)
    cells = train_df[['grid_lat', 'grid_lon']].drop_duplicates()
    
    model, scaler = load_model()
    results = []
    
    for _, row in cells.iterrows():
        grid_lat = row['grid_lat']
        grid_lon = row['grid_lon']
        
        features = build_features(grid_lat, grid_lon, hour, day_of_week, month, train_df)
        features = scaler.transform(features)
        
        with torch.no_grad():
            output = model(torch.FloatTensor(features))
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        lat = (grid_lat + 0.5) * GRID_SIZE
        lon = (grid_lon + 0.5) * GRID_SIZE
        
        results.append({
            'latitude': lat,
            'longitude': lon,
            'grid_lat': grid_lat,
            'grid_lon': grid_lon,
            'risk_level': predicted_class,
            'risk_label': RISK_LABELS[predicted_class],
            'high_risk_prob': float(probabilities[2]),
            'confidence': probabilities[predicted_class]
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Testing single prediction...")
    print("Location: Levain Bakery, Upper West Side")
    print("Time: Thursday 4pm, June\n")
    
    result = predict_risk(
        latitude=40.7809,
        longitude=-73.9806,
        hour=16,
        day_of_week=3,
        month=6
    )
    
    print(f"Grid Cell: ({result['grid_lat']}, {result['grid_lon']})")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probabilities: {result['probabilities']}")
    
    print("\nGenerating Manhattan heatmap for Friday 5pm, June...")
    heatmap = predict_manhattan_heatmap(hour=17, day_of_week=4, month=6)
    print(f"Total grid cells: {len(heatmap)}")
    print(f"\nRisk distribution:")
    print(heatmap['risk_label'].value_counts())