import os
from dotenv import load_dotenv
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import sqlite3
from holidays import Canada

# ─── Load environment ─────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))
AESO_API_KEY = os.getenv("AESO_API_KEY")
OWM_KEY      = os.getenv("OWM_KEY")
DB_PATH      = os.path.join(PROJECT_DIR, "data", "wholesale.db")

# Validate API keys
if not AESO_API_KEY:
    st.error("AESO_API_KEY not set. Please configure your .env file.")
    st.stop()
if not OWM_KEY:
    st.error("OWM_KEY not set. Please configure your .env file.")
    st.stop()

# ─── App title & model ─────────────────────────────────────────────────────────
st.title("AESO Pool Price Forecasting")
model = joblib.load("stacked_model.joblib")

# ─── Fetch live pool price (last 24h) ───────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_live_price():
    headers = {"API-KEY": AESO_API_KEY}
    today = pd.Timestamp.now().normalize()
    start = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    url = "https://apimgw.aeso.ca/public/poolprice-api/v1.1/price/poolPrice"
    r = requests.get(url, headers=headers, params={"startDate": start, "endDate": end})
    r.raise_for_status()
    records = r.json().get("return", {}).get("Pool Price Report", [])
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["begin_datetime_utc"])
    df = df.set_index("date")[['pool_price']].rename(columns={'pool_price':'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill()
    return df

# ─── Load historical data from SQLite ───────────────────────────────────────────
def load_historical_data():
    conn = sqlite3.connect(DB_PATH)
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    if 'wholesale' not in tables:
        st.warning("No 'wholesale' table found in database. Skipping historical data.")
        return pd.DataFrame()
    df = pd.read_sql(
        "SELECT date, value FROM wholesale ORDER BY date",
        sqlite3.connect(DB_PATH), parse_dates=['date']
    )
    df = df.set_index('date')
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill()
    return df

# ─── Feature engineering ─────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill()
    lags = [1,2,3,6,12,24,48,72,168]
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df['roll24_mean']  = df['value'].rolling(24).mean()
    df['roll24_std']   = df['value'].rolling(24).std()
    df['roll168_mean'] = df['value'].rolling(168).mean()
    df['roll168_std']  = df['value'].rolling(168).std()
    df['hour'] = df.index.hour
    df['dow']  = df.index.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['is_onpeak']  = df['hour'].between(7,17).astype(int)
    df['is_midpeak'] = df['hour'].isin([6,18,19,20]).astype(int)
    df['is_offpeak'] = (~df['hour'].between(6,20)).astype(int)
    df['sin_d'] = np.sin(2*np.pi*df['hour']/24)
    df['cos_d'] = np.cos(2*np.pi*df['hour']/24)
    df['sin_w'] = np.sin(2*np.pi*df['dow']/7)
    df['cos_w'] = np.cos(2*np.pi*df['dow']/7)
    df['sin_y'] = np.sin(2*np.pi*df.index.dayofyear/365.25)
    df['cos_y'] = np.cos(2*np.pi*df.index.dayofyear/365.25)
    holidays = Canada(prov='AB')
    df['is_holiday']     = df.index.to_series().isin(holidays).astype(int)
    df['is_pre_holiday'] = df['is_holiday'].shift(24).fillna(0).astype(int)
    df['hol_3d_before']  = df['is_holiday'].rolling(72,closed='left').sum().shift(1).fillna(0)
    df['hol_3d_after']   = df['is_holiday'].rolling(72).sum().shift(-72).fillna(0)
    feature_cols = [f'lag_{lag}' for lag in lags] + [
        'roll24_mean','roll24_std','roll168_mean','roll168_std',
        'hour','dow','is_weekend','is_onpeak','is_midpeak','is_offpeak',
        'sin_d','cos_d','sin_w','cos_w','sin_y','cos_y',
        'is_holiday','is_pre_holiday','hol_3d_before','hol_3d_after'
    ]
    return df.dropna(subset=feature_cols)[feature_cols]

# ─── Main pipeline ─────────────────────────────────────────────────────────────
# Load data
hist = load_historical_data()
live = fetch_live_price()
# Merge
price_data = pd.concat([hist, live]).sort_index().drop_duplicates() if not hist.empty else live.copy()

# Debug info
st.write(f"Historical rows: {len(hist)}")
st.write(f"Live rows: {len(live)}")
if not price_data.empty:
    st.write(f"Data from {price_data.index.min()} to {price_data.index.max()}")

# Plot actual
st.write("## Actual Wholesale Price (Historical + Live)")
st.line_chart(price_data['value'])

# Compute in-sample MAPE on last 24h if available
if len(price_data) >= 24:
    feats_insamp = prepare_features(price_data)
    X_insamp    = feats_insamp.iloc[-24:][model.feature_names_in_]
    preds_insamp = model.predict(X_insamp)
    actual_insamp = price_data['value'].iloc[-24:]
    mape = (np.abs(actual_insamp - preds_insamp) / actual_insamp).mean() * 100
    st.write(f"MAPE (last 24h in-sample): {mape:.2f}%")

# Forecast next 24h
df_fc = price_data[['value']].copy()
future = []
for _ in range(24):
    nxt = df_fc.index[-1] + pd.Timedelta(hours=1)
    df_fc.loc[nxt] = df_fc['value'].iloc[-1]
    feats = prepare_features(df_fc)
    Xn = feats.loc[[nxt], model.feature_names_in_]
    yhat = model.predict(Xn)[0]
    future.append((nxt, yhat))
    df_fc.loc[nxt, 'value'] = yhat
future_df = pd.DataFrame(future, columns=['date','forecast']).set_index('date')

# Plot forecast
st.write("## 24-Hour Forecast")
st.line_chart(future_df['forecast'])

