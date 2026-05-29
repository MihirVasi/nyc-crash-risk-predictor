import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import requests
import sys
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append('src')
from predict import predict_manhattan_heatmap, predict_risk
from retrain import retrain_if_needed

st.set_page_config(
    page_title="NYC Crash Risk Predictor",
    page_icon="🚨",
    layout="wide"
)

MANHATTAN_LAT_MIN = 40.700
MANHATTAN_LAT_MAX = 40.880
MANHATTAN_LON_MIN = -74.020
MANHATTAN_LON_MAX = -73.910
HALF = 0.001

def is_in_manhattan(lat, lon):
    return (MANHATTAN_LAT_MIN <= lat <= MANHATTAN_LAT_MAX and
            MANHATTAN_LON_MIN <= lon <= MANHATTAN_LON_MAX)

def geocode_location(query):
    headers = {"User-Agent": "NYC-Crash-Predictor/1.0"}
    for search_query in [
        f"{query}, Manhattan, New York City",
        f"{query}, New York City",
        f"{query}, Manhattan, NY"
    ]:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": search_query, "format": "json", "limit": 1},
                headers=headers,
                timeout=10
            )
            if response.status_code == 200 and len(response.json()) > 0:
                result = response.json()[0]
                return float(result['lat']), float(result['lon']), result['display_name']
        except:
            continue
    return None, None, None

def weight_to_color(weight):
    cmap = plt.get_cmap('RdYlGn_r')
    return mcolors.to_hex(cmap(min(max(weight, 0.0), 1.0)))

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f0f7ff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #bfdbfe;
    }
    
    [data-testid="stSidebar"] * {
        color: #1e3a5f !important;
    }

    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e40af !important;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
        text-align: center;
    }
    
    .sidebar-subtitle {
        font-size: 0.7rem;
        color: #64748b !important;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #1e40af !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 1rem 0 0.25rem 0;
        border-bottom: 1px solid #bfdbfe;
        padding-bottom: 0.25rem;
    }

    .clock-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .clock-time {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        letter-spacing: 4px;
    }
    
    .clock-date {
        font-size: 0.75rem;
        color: #64748b;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    .stat-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 0.75rem 0;
    }

    .stat-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .stat-label {
        font-size: 0.6rem;
        color: #64748b;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .risk-low { color: #16a34a; }
    .risk-medium { color: #d97706; }
    .risk-high { color: #dc2626; }

    .error-msg {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        color: #dc2626;
        font-size: 0.75rem;
        margin: 0.5rem 0;
    }

    .success-msg {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        color: #16a34a;
        font-size: 0.75rem;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 1px;
        padding: 0.4rem 1rem;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
    }

    .stTextInput > div > div > input {
        background-color: #f8fafc;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #1e3a5f;
    }

    .stSelectbox > div > div {
        background-color: #f8fafc !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
    }

    .stDateInput > div > div > input {
        background-color: #f8fafc;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        color: #1e3a5f;
    }

    .legend-bar {
        height: 12px;
        background: linear-gradient(to right, #22c55e, #84cc16, #eab308, #f97316, #ef4444, #991b1b);
        border-radius: 6px;
        border: 1px solid #bfdbfe;
        margin: 0.25rem 0;
    }

    .legend-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.65rem;
        color: #64748b;
    }

    div[data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def check_and_retrain():
    try:
        return retrain_if_needed()
    except:
        return False

data_updated = check_and_retrain()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">NYC Crash Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Manhattan · ML Predictor</div>', unsafe_allow_html=True)

    if data_updated:
        st.markdown('<div class="success-msg">✓ Model updated with new NYPD data</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Date & Time</div>', unsafe_allow_html=True)
    selected_date = st.date_input("", value=date.today(), label_visibility="collapsed")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        hour_12 = st.selectbox("Hour", list(range(1, 13)), index=4)
    with col2:
        minute = st.selectbox("Min", [0, 15, 30, 45], index=0)
    with col3:
        ampm = st.selectbox("AM/PM", ["AM", "PM"], index=1)

    hour = (hour_12 % 12) + (12 if ampm == "PM" else 0)
    day_of_week = selected_date.weekday()
    month = selected_date.month
    day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]

    st.markdown(f"""
    <div class="clock-box">
        <div class="clock-time">{hour:02d}:{minute:02d}</div>
        <div class="clock-date">{day_name} · {selected_date.strftime('%b %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Search Location</div>', unsafe_allow_html=True)
    location_query = st.text_input("", placeholder="Penn Station, W 34th St...", label_visibility="collapsed")
    search_clicked = st.button("🔍 Search & Predict")

    with st.spinner("Predicting..."):
        heatmap_df = predict_manhattan_heatmap(
            hour=hour,
            day_of_week=day_of_week,
            month=month
        )
    heatmap_df = heatmap_df.dropna(subset=['latitude', 'longitude', 'high_risk_prob'])

    high_count = (heatmap_df['risk_label'] == 'High').sum()
    medium_count = (heatmap_df['risk_label'] == 'Medium').sum()
    low_count = (heatmap_df['risk_label'] == 'Low').sum()
    avg_prob = heatmap_df['high_risk_prob'].mean()

    st.markdown('<div class="section-label">Current Risk Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value risk-high">{high_count}</div>
            <div class="stat-label">High Risk</div>
        </div>
        <div class="stat-card">
            <div class="stat-value risk-medium">{medium_count}</div>
            <div class="stat-label">Medium Risk</div>
        </div>
        <div class="stat-card">
            <div class="stat-value risk-low">{low_count}</div>
            <div class="stat-label">Low Risk</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#1e40af">{avg_prob:.0%}</div>
            <div class="stat-label">Avg Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Risk Scale</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-bar"></div>
    <div class="legend-labels">
        <span style="color:#16a34a">Low</span>
        <span>Medium</span>
        <span style="color:#dc2626">High</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#64748b; line-height:1.8;">
        336,000+ NYPD crash records<br>
        14 years · 15 features<br>
        78% accuracy · 79% High recall<br>
        Auto fine-tunes on new data
    </div>
    """, unsafe_allow_html=True)

# Handle search
searched_lat = None
searched_lon = None
search_result = None
outside_manhattan = False
search_error = None

if search_clicked and location_query:
    with st.spinner("Searching..."):
        lat, lon, display_name = geocode_location(location_query)
        if lat is None:
            search_error = "Location not found. Try a more specific address."
        elif not is_in_manhattan(lat, lon):
            outside_manhattan = True
        else:
            searched_lat = lat
            searched_lon = lon
            search_result = predict_risk(
                latitude=searched_lat,
                longitude=searched_lon,
                hour=hour,
                day_of_week=day_of_week,
                month=month
            )

# Build map
map_center = [searched_lat, searched_lon] if searched_lat else [40.7549, -73.9840]
zoom = 15 if searched_lat else 13

m = folium.Map(
    location=map_center,
    zoom_start=zoom,
    tiles='CartoDB positron',
    min_zoom=11,
    max_zoom=18,
    max_bounds=True
)

Fullscreen().add_to(m)
m.fit_bounds([[40.700, -74.020], [40.880, -73.910]])

# Filter to Manhattan only
manhattan_df = heatmap_df[
    (heatmap_df['latitude'] >= MANHATTAN_LAT_MIN) &
    (heatmap_df['latitude'] <= MANHATTAN_LAT_MAX) &
    (heatmap_df['longitude'] >= MANHATTAN_LON_MIN) &
    (heatmap_df['longitude'] <= MANHATTAN_LON_MAX)
].copy()

# Draw individual colored rectangles — no blending, colors stay accurate at all zoom levels
risk_weight_map = {'Low': 0.15, 'Medium': 0.55, 'High': 1.0}

for _, row in manhattan_df.iterrows():
    if np.isnan(row['latitude']) or np.isnan(row['longitude']):
        continue
    risk_label = row['risk_label']
    confidence = float(row['confidence'])
    weight = risk_weight_map[risk_label] * (0.8 + 0.2 * confidence)
    color = weight_to_color(weight)
    lat = float(row['latitude'])
    lon = float(row['longitude'])
    prob = float(row['high_risk_prob'])

    folium.Rectangle(
        bounds=[[lat - HALF, lon - HALF], [lat + HALF, lon + HALF]],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.65,
        weight=0,
        popup=folium.Popup(
            f"""<div style="font-family:Inter,sans-serif; padding:4px; min-width:160px;">
                <div style="font-weight:700; color:{color}; font-size:14px;">{risk_label.upper()} RISK</div>
                <div style="font-size:11px; color:#64748b; margin-top:4px;">
                    High prob: {prob:.1%}<br>
                    Confidence: {confidence:.1%}
                </div>
            </div>""",
            max_width=200
        )
    ).add_to(m)

# Searched location marker
if searched_lat and search_result:
    risk_color = {'Low': '#16a34a', 'Medium': '#d97706', 'High': '#dc2626'}[search_result['risk_level']]
    folium.CircleMarker(
        location=[searched_lat, searched_lon],
        radius=10,
        color=risk_color,
        fill=True,
        fill_color=risk_color,
        fill_opacity=0.9,
        popup=folium.Popup(
            f"""<div style="font-family:Inter,sans-serif; min-width:200px; padding:4px;">
                <div style="font-weight:700; font-size:13px; color:#1e3a5f; margin-bottom:6px;">{location_query}</div>
                <div style="font-size:16px; font-weight:700; color:{risk_color}; margin-bottom:4px;">{search_result['risk_level'].upper()} RISK</div>
                <div style="font-size:12px; color:#64748b;">Confidence: {search_result['confidence']}</div>
                <hr style="border-color:#e2e8f0; margin:6px 0;">
                <div style="font-size:11px; color:#64748b;">
                    Low: {search_result['probabilities']['Low']}<br>
                    Medium: {search_result['probabilities']['Medium']}<br>
                    High: {search_result['probabilities']['High']}
                </div>
            </div>""",
            max_width=250,
            show=True
        )
    ).add_to(m)

map_data = st_folium(m, width="100%", height=750, returned_objects=["last_clicked"])

# Errors
if search_error:
    st.error(f"⚠ {search_error}")
if outside_manhattan:
    st.error("⚠ Location is outside Manhattan. This model covers Manhattan only.")

# Click prediction
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    if not is_in_manhattan(clicked_lat, clicked_lon):
        st.warning("⚠ Clicked location is outside Manhattan.")
    else:
        click_result = predict_risk(
            latitude=clicked_lat,
            longitude=clicked_lon,
            hour=hour,
            day_of_week=day_of_week,
            month=month
        )
        risk_class = click_result['risk_level'].lower()
        risk_colors = {'low': '#16a34a', 'medium': '#d97706', 'high': '#dc2626'}
        color = risk_colors[risk_class]

        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:1rem 1.5rem; border-left:4px solid {color}; margin-top:0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size:0.75rem; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.5rem;">
                Clicked Location · {clicked_lat:.4f}, {clicked_lon:.4f}
            </div>
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <span style="font-size:1.2rem; font-weight:700; color:{color};">{click_result['risk_level'].upper()} RISK</span>
                <span style="font-size:0.85rem; color:#64748b;">Confidence: <b style="color:#1e40af">{click_result['confidence']}</b></span>
                <span style="font-size:0.85rem; color:#64748b;">Low: {click_result['probabilities']['Low']} · Med: {click_result['probabilities']['Medium']} · High: {click_result['probabilities']['High']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)