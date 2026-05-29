# NYC Manhattan Crash Risk Predictor

A spatial machine learning system that predicts vehicle collision risk across Manhattan using a PyTorch neural network trained on 14 years of NYPD crash data.

<img width="1344" height="875" alt="Screenshot 2026-05-29 170558" src="https://github.com/user-attachments/assets/8ac3a329-4cfe-454e-a677-09288e7f203a" /> <img width="1454" height="864" alt="Screenshot 2026-05-29 170546" src="https://github.com/user-attachments/assets/19d354b5-db2d-4f80-82d4-8409e3b24a3b" />


## What it does

Given a date, time, and location in Manhattan, the model predicts whether crash risk is Low, Medium, or High for every city block simultaneously. The interactive dashboard renders a live risk map that updates as you change the time and date.

## How it works

1. **Data** — 336,000+ NYPD collision records from 2012-2026 downloaded via NYC Open Data API
2. **Grid system** — Manhattan divided into 422 city-block-sized zones (~200m per cell)
3. **Feature engineering** — 15 features including temporal patterns (hour, day, month), spatial encoding, rush hour flags, severity weighting, and interaction terms
4. **Model** — 5-layer PyTorch neural network classifier predicting Low/Medium/High risk per zone
5. **Auto-retraining** — fine-tunes on new NYPD data automatically when the app loads

## Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 78% |
| High Risk Recall | 79% |
| Medium Risk Recall | 68% |
| Training Records | 336,000+ |
| Features | 15 |

## Features

- Interactive Manhattan risk map with color-coded grid zones
- Location search — type any Manhattan street or landmark
- Date and time picker — see how risk changes by hour and day
- Click anywhere on the map for instant risk prediction
- Auto fine-tunes on new NYPD crash data
- Outside-Manhattan detection with error handling

## Tech Stack

- **ML:** PyTorch, scikit-learn, pandas, numpy
- **Data:** NYC Open Data API (NYPD Motor Vehicle Collisions)
- **Backend:** Python, Streamlit
- **Maps:** Folium, OpenStreetMap
- **Geocoding:** Nominatim

## Setup

```bash
git clone https://github.com/MihirVasi/nyc-crash-risk-predictor.git
cd nyc-crash-risk-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
python src/explore_data.py
python src/feature_engineering.py
python src/model.py
streamlit run app.py
```

## Project Structure

```
├── src/
│   ├── download_data.py       # NYC Open Data API pipeline
│   ├── explore_data.py        # Data cleaning and analysis
│   ├── feature_engineering.py # Grid system and feature creation
│   ├── model.py               # PyTorch model training
│   ├── predict.py             # Inference and heatmap generation
│   └── retrain.py             # Incremental fine-tuning pipeline
├── app.py                     # Streamlit dashboard
└── requirements.txt
```
