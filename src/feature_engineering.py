import pandas as pd
import numpy as np

DATA_FILE = "data/nyc_collisions_manhattan_clean.csv"
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

GRID_SIZE = 0.002

def create_features():
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_FILE)
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['hour'] = pd.to_datetime(df['crash_time'], format='%H:%M').dt.hour
    df['day_of_week'] = df['crash_date'].dt.dayofweek
    df['month'] = df['crash_date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = (df['hour'].isin([7,8,9,16,17,18,19])).astype(int)
    df['is_night'] = (df['hour'].isin([22,23,0,1,2,3,4])).astype(int)
    
    df['grid_lat'] = (df['latitude'] / GRID_SIZE).astype(int)
    df['grid_lon'] = (df['longitude'] / GRID_SIZE).astype(int)
    
    df['number_of_persons_injured'] = pd.to_numeric(df['number_of_persons_injured'], errors='coerce').fillna(0)
    df['number_of_persons_killed'] = pd.to_numeric(df['number_of_persons_killed'], errors='coerce').fillna(0)
    df['severity_score'] = df['number_of_persons_injured'] + (df['number_of_persons_killed'] * 5)
    
    print("Aggregating data...")
    agg = df.groupby(['grid_lat', 'grid_lon', 'hour', 'day_of_week']).agg(
        crash_count=('severity_score', 'count'),
        avg_severity=('severity_score', 'mean'),
        total_injured=('number_of_persons_injured', 'sum'),
        total_killed=('number_of_persons_killed', 'sum'),
        avg_month=('month', 'mean'),
        is_weekend=('is_weekend', 'first'),
        is_rush_hour=('is_rush_hour', 'first'),
        is_night=('is_night', 'first')
    ).reset_index()
    
    # Cell level features
    agg['cell_avg'] = agg.groupby(['grid_lat', 'grid_lon'])['crash_count'].transform('mean')
    agg['cell_max'] = agg.groupby(['grid_lat', 'grid_lon'])['crash_count'].transform('max')
    agg['cell_std'] = agg.groupby(['grid_lat', 'grid_lon'])['crash_count'].transform('std').fillna(0)
    
    # Hour level features
    agg['hour_avg'] = agg.groupby('hour')['crash_count'].transform('mean')
    
    # Day level features
    agg['dow_avg'] = agg.groupby('day_of_week')['crash_count'].transform('mean')
    
    # Interaction features
    agg['hour_x_dow'] = agg['hour'] * agg['day_of_week']
    agg['cell_x_hour'] = agg['cell_avg'] * agg['hour_avg']
    
    low_threshold = agg['crash_count'].quantile(0.60)
    high_threshold = agg['crash_count'].quantile(0.85)
    print(f"Risk thresholds — Low: <={low_threshold:.1f}, Medium: <={high_threshold:.1f}, High: >{high_threshold:.1f}")
    
    agg['risk_level'] = agg['crash_count'].apply(
        lambda x: 0 if x <= low_threshold else (1 if x <= high_threshold else 2)
    )
    
    print(f"\nTotal combinations: {len(agg)}")
    print(f"Risk distribution:\n{agg['risk_level'].value_counts().sort_index()}")
    
    agg = agg.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(agg))
    train = agg[:split]
    test = agg[split:]
    
    print(f"\nTraining combinations: {len(train)}")
    print(f"Risk distribution:\n{train['risk_level'].value_counts().sort_index()}")
    print(f"\nTest combinations: {len(test)}")
    print(f"Risk distribution:\n{test['risk_level'].value_counts().sort_index()}")
    
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    print(f"\nSaved train to {TRAIN_FILE}")
    print(f"Saved test to {TEST_FILE}")

if __name__ == "__main__":
    create_features()