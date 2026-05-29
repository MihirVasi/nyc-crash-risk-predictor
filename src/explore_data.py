import pandas as pd
import numpy as np
import os

DATA_FILE = "data/nyc_collisions_manhattan.csv"

def explore_data():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    print(f"\n--- Basic Info ---")
    print(f"Total records: {len(df)}")
    
    print(f"\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Parse datetime
    print("\nParsing datetime...")
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['hour'] = pd.to_datetime(df['crash_time'], format='%H:%M').dt.hour
    df['day_of_week'] = df['crash_date'].dt.dayofweek
    df['month'] = df['crash_date'].dt.month
    df['year'] = df['crash_date'].dt.year
    
    # Convert numeric columns
    df['number_of_persons_injured'] = pd.to_numeric(df['number_of_persons_injured'], errors='coerce').fillna(0)
    df['number_of_persons_killed'] = pd.to_numeric(df['number_of_persons_killed'], errors='coerce').fillna(0)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    print(f"\n--- Crashes by Hour ---")
    print(df['hour'].value_counts().sort_index())
    
    print(f"\n--- Crashes by Day of Week (0=Monday, 6=Sunday) ---")
    print(df['day_of_week'].value_counts().sort_index())
    
    print(f"\n--- Crashes by Year ---")
    print(df['year'].value_counts().sort_index())
    
    print(f"\n--- Severity Stats ---")
    print(f"Total injured: {df['number_of_persons_injured'].sum():.0f}")
    print(f"Total killed: {df['number_of_persons_killed'].sum():.0f}")
    print(f"Crashes with injuries: {(df['number_of_persons_injured'] > 0).sum()}")
    
    # Save cleaned data
    output_file = "data/nyc_collisions_manhattan_clean.csv"
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")

if __name__ == "__main__":
    explore_data()