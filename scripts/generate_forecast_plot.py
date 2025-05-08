#!/usr/bin/env python3
# scripts/generate_forecast_plot.py

import os
import sqlite3
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

def load_database():
    load_dotenv()  # loads .env locally or from SECRET_ENV in CI
    db_url = os.getenv("DATABASE_URL", "sqlite:///data/wholesale.db")
    # strip sqlite prefix
    db_path = db_url.replace("sqlite:///", "") if db_url.startswith("sqlite:///") else db_url
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path)

def build_initial_df(conn):
    # Read tables
    df_price = pd.read_sql("SELECT * FROM pool_price", conn)
    df_load  = pd.read_sql("SELECT * FROM actual_load", conn)

    # Merge on UTC timestamp
    df = df_price.merge(
        df_load,
        on="begin_datetime_utc",
        how="left",
        suffixes=("_price", "_load")
    )

    # Parse dt column
    df["dt"] = pd.to_datetime(df["begin_datetime_utc"])

    # Calendar features
    df["hour"]       = df["dt"].dt.hour
    df["dayofweek"]  = df["dt"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    # Lag features
    for lag in (1, 24, 168):
        df[f"lag_{lag}h_price"] = df["pool_price"].shift(lag)

    # Rolling means
    df["rolling_24h_price"]  = df["pool_price"].rolling(24).mean()
    df["rolling_168h_price"] = df["pool_price"].rolling(168).mean()

    # Drop rows with any NaNs in the features your model needs
    required = [
        "hour","dayofweek","is_weekend",
        "lag_1h_price","lag_24h_price","lag_168h_price",
        "rolling_24h_price","rolling_168h_price",
        "alberta_internal_load","forecast_alberta_internal_load",
        "pool_price"
    ]
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    return df_clean

def iterative_forecast(df_feat, model, horizon=24):
    # Prepare history indexed by dt
    history = df_feat.set_index("dt").copy()

    # Also index the load-forecast table for future load features
    load_df = history[["alberta_internal_load","forecast_alberta_internal_load"]]

    preds = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    for i in range(1, horizon+1):
        t = now + timedelta(hours=i)
        # Build feature dict for time t
        row = {
            "hour":       t.hour,
            "dayofweek":  t.weekday(),
            "is_weekend": int(t.weekday() in [5,6])
        }
        # Lags
        for lag in (1, 24, 168):
            past_t = t - timedelta(hours=lag)
            row[f"lag_{lag}h_price"] = history.at[past_t, "pool_price"]

        # Rolling means (use the last n hours in history)
        window_24  = history["pool_price"].loc[t - timedelta(hours=24) : t - timedelta(hours=1)]
        window_168 = history["pool_price"].loc[t - timedelta(hours=168): t - timedelta(hours=1)]
        row["rolling_24h_price"]  = window_24.mean()
        row["rolling_168h_price"] = window_168.mean()

        # Load features (use forecast if available, else last known)
        if t in load_df.index:
            row["alberta_internal_load"]          = load_df.at[t, "alberta_internal_load"]
            row["forecast_alberta_internal_load"] = load_df.at[t, "forecast_alberta_internal_load"]
        else:
            # .asof will give last known <= t
            row["alberta_internal_load"]          = load_df["alberta_internal_load"].asof(t)
            row["forecast_alberta_internal_load"] = load_df["forecast_alberta_internal_load"].asof(t)

        # Convert to DataFrame and predict
        X_new = pd.DataFrame([row])
        y_hat = model.predict(X_new)[0]

        # Save prediction and append to history for next iteration
        preds.append((t, y_hat))
        history.loc[t, "pool_price"] = y_hat

    # Build DataFrame of forecasts
    fc = pd.DataFrame(preds, columns=["dt","forecast_pool_price"]).set_index("dt")
    return fc

def plot_and_save(fc, out_path="plots/forecast.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(fc.index, fc["forecast_pool_price"], marker="o", linestyle="-")
    plt.title("Next 24-Hour Pool-Price Forecast")
    plt.xlabel("UTC Time")
    plt.ylabel("Forecast Pool Price (CAD)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    # Load DB & model
    conn  = load_database()
    df0   = build_initial_df(conn)
    model = joblib.load("models/stacked_model.joblib")

    # Forecast & plot
    fc = iterative_forecast(df0, model, horizon=24)
    plot_and_save(fc)

if __name__ == "__main__":
    main()
