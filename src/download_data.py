import requests
import pandas as pd
import os

# Manhattan bounding box
MIDTOWN_LAT_MIN = 40.7000
MIDTOWN_LAT_MAX = 40.8800
MIDTOWN_LON_MIN = -74.0200
MIDTOWN_LON_MAX = -73.9100

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "nyc_collisions_manhattan.csv")

def download_collision_data():
    print("Downloading NYC collision data...")
    
    url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    
    params = {
        "$limit": 500000,"$where": f"latitude > {MIDTOWN_LAT_MIN} AND latitude < {MIDTOWN_LAT_MAX} AND longitude > {MIDTOWN_LON_MIN} AND longitude < {MIDTOWN_LON_MAX} AND borough='MANHATTAN'",
        "$select": "crash_date,crash_time,latitude,longitude,number_of_persons_injured,number_of_persons_killed,number_of_pedestrians_injured,number_of_cyclist_injured,on_street_name,cross_street_name",
        "$order": "crash_date DESC"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print("Download successful!")
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} records to {OUTPUT_FILE}")
        print(df.head())
    else:
        print(f"Download failed: {response.status_code}")
        print(response.text[:500])

if __name__ == "__main__":
    download_collision_data()