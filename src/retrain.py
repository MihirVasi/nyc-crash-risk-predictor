import requests
import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('src')

DATA_FILE = "data/nyc_collisions_manhattan.csv"
CLEAN_FILE = "data/nyc_collisions_manhattan_clean.csv"
MODEL_FILE = "models/crash_predictor.pth"
METADATA_FILE = "data/last_update.json"
TRAIN_FILE = "data/train.csv"

MANHATTAN_LAT_MIN = 40.700
MANHATTAN_LAT_MAX = 40.880
MANHATTAN_LON_MIN = -74.020
MANHATTAN_LON_MAX = -73.910
GRID_SIZE = 0.004

class CrashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.LongTensor(np.array(y))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CrashPredictor(nn.Module):
    def __init__(self, input_size):
        super(CrashPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.network(x)

def get_last_update():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f).get('last_update', '2026-01-01')
    return '2026-01-01'

def save_last_update(date_str):
    with open(METADATA_FILE, 'w') as f:
        json.dump({'last_update': date_str, 'updated_at': datetime.now().isoformat()}, f)

def check_for_new_data():
    last_update = get_last_update()
    url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    params = {
        "$limit": 1,
        "$where": f"crash_date > '{last_update}' AND borough='MANHATTAN'",
        "$select": "crash_date"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return len(response.json()) > 0
    except:
        pass
    return False

def download_new_records():
    last_update = get_last_update()
    url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    params = {
        "$limit": 10000,
        "$where": f"crash_date > '{last_update}' AND borough='MANHATTAN' AND latitude > {MANHATTAN_LAT_MIN} AND latitude < {MANHATTAN_LAT_MAX} AND longitude > {MANHATTAN_LON_MIN} AND longitude < {MANHATTAN_LON_MAX}",
        "$select": "crash_date,crash_time,latitude,longitude,number_of_persons_injured,number_of_pedestrians_injured,number_of_cyclist_injured,on_street_name,cross_street_name,number_of_persons_killed",
        "$order": "crash_date DESC"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            new_data = response.json()
            if len(new_data) == 0:
                return None
            new_df = pd.DataFrame(new_data)
            existing_df = pd.read_csv(DATA_FILE)
            combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
            combined.to_csv(DATA_FILE, index=False)
            save_last_update(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
            return new_df
    except Exception as e:
        print(f"Download error: {e}")
    return None

def process_new_records(new_df):
    new_df['latitude'] = pd.to_numeric(new_df['latitude'], errors='coerce')
    new_df['longitude'] = pd.to_numeric(new_df['longitude'], errors='coerce')
    new_df = new_df.dropna(subset=['latitude', 'longitude'])

    new_df['crash_date'] = pd.to_datetime(new_df['crash_date'])
    new_df['hour'] = pd.to_datetime(new_df['crash_time'], format='%H:%M', errors='coerce').dt.hour
    new_df['day_of_week'] = new_df['crash_date'].dt.dayofweek
    new_df['month'] = new_df['crash_date'].dt.month
    new_df['is_weekend'] = (new_df['day_of_week'] >= 5).astype(int)
    new_df['is_rush_hour'] = new_df['hour'].isin([7,8,9,16,17,18,19]).astype(int)
    new_df['is_night'] = new_df['hour'].isin([22,23,0,1,2,3,4]).astype(int)

    new_df['grid_lat'] = (new_df['latitude'] / GRID_SIZE).astype(int)
    new_df['grid_lon'] = (new_df['longitude'] / GRID_SIZE).astype(int)
    new_df['number_of_persons_injured'] = pd.to_numeric(new_df['number_of_persons_injured'], errors='coerce').fillna(0)
    new_df['number_of_persons_killed'] = pd.to_numeric(new_df['number_of_persons_killed'], errors='coerce').fillna(0)
    new_df['severity_score'] = new_df['number_of_persons_injured'] + (new_df['number_of_persons_killed'] * 5)

    agg = new_df.groupby(['grid_lat', 'grid_lon', 'hour', 'day_of_week']).agg(
        crash_count=('severity_score', 'count'),
        avg_month=('month', 'mean'),
        is_weekend=('is_weekend', 'first'),
        is_rush_hour=('is_rush_hour', 'first'),
        is_night=('is_night', 'first'),
        total_injured=('number_of_persons_injured', 'sum'),
        total_killed=('number_of_persons_killed', 'sum')
    ).reset_index()

    train_df = pd.read_csv(TRAIN_FILE)
    agg['cell_avg'] = agg.apply(
        lambda r: train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]['crash_count'].mean()
        if len(train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]) > 0
        else train_df['crash_count'].mean(), axis=1
    )
    agg['cell_max'] = agg.apply(
        lambda r: train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]['crash_count'].max()
        if len(train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]) > 0
        else train_df['crash_count'].max(), axis=1
    )
    agg['cell_std'] = agg.apply(
        lambda r: train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]['crash_count'].std()
        if len(train_df[(train_df['grid_lat']==r['grid_lat']) & (train_df['grid_lon']==r['grid_lon'])]) > 0
        else 0.0, axis=1
    ).fillna(0)

    agg['hour_avg'] = agg['hour'].map(train_df.groupby('hour')['crash_count'].mean())
    agg['dow_avg'] = agg['day_of_week'].map(train_df.groupby('day_of_week')['crash_count'].mean())
    agg['hour_x_dow'] = agg['hour'] * agg['day_of_week']
    agg['cell_x_hour'] = agg['cell_avg'] * agg['hour_avg']

    low = train_df['crash_count'].quantile(0.60)
    high = train_df['crash_count'].quantile(0.85)
    agg['risk_level'] = agg['crash_count'].apply(
        lambda x: 0 if x <= low else (1 if x <= high else 2)
    )

    return agg

def fine_tune_model(new_features_df, epochs=20):
    checkpoint = torch.load(MODEL_FILE, weights_only=False)
    model = CrashPredictor(input_size=15)
    model.load_state_dict(checkpoint['model_state'])
    scaler = checkpoint['scaler']

    features = [
        'grid_lat', 'grid_lon', 'hour', 'day_of_week',
        'avg_month', 'is_weekend', 'is_rush_hour', 'is_night',
        'cell_avg', 'cell_max', 'cell_std',
        'hour_avg', 'dow_avg', 'hour_x_dow', 'cell_x_hour'
    ]

    new_features_df = new_features_df.rename(columns={'avg_month': 'avg_month'})
    available = [f for f in features if f in new_features_df.columns]
    if len(available) < len(features):
        print("Missing features in new data, skipping fine-tune")
        return False

    X = scaler.transform(new_features_df[features].values)
    y = new_features_df['risk_level'].values

    dataset = CrashDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Fine-tune epoch {epoch}/{epochs}")

    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'features': features
    }, MODEL_FILE)

    print("Model fine-tuned and saved.")
    return True

def retrain_if_needed():
    print("Checking for new crash data...")
    if not check_for_new_data():
        print("No new data. Model unchanged.")
        return False

    print("New data found. Downloading...")
    new_df = download_new_records()
    if new_df is None or len(new_df) == 0:
        return False

    print(f"Processing {len(new_df)} new records...")
    new_features = process_new_records(new_df)

    print("Fine-tuning model on new data...")
    success = fine_tune_model(new_features, epochs=20)
    return success

if __name__ == "__main__":
    retrain_if_needed()