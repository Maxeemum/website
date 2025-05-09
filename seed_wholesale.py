import os
import sqlite3
import pandas as pd
import requests
from datetime import timedelta
from dotenv import load_dotenv

# ─── Setup ───────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(PROJECT_DIR, ".env")
load_dotenv(env_path)
AESO_API_KEY = os.getenv("AESO_API_KEY")
if not AESO_API_KEY:
    raise RuntimeError("AESO_API_KEY not found in .env file")

AESO_PRICE_URL = "https://apimgw.aeso.ca/public/poolprice-api/v1.1/price/poolPrice"
DB_PATH = os.path.join(PROJECT_DIR, "data", "wholesale.db")

# ─── Parameters ──────────────────────────────────────────────────────
HIST_START = pd.to_datetime("2024-01-01")
HIST_END   = pd.Timestamp.now().normalize()
CHUNK_SIZE = timedelta(days=30)  # 30-day fetch windows

# ─── Fetch in chunks ───────────────────────────────────────────────────
all_records = []
curr_start = HIST_START
while curr_start < HIST_END:
    curr_end = min(curr_start + CHUNK_SIZE, HIST_END)
    start_str = curr_start.strftime("%Y-%m-%d")
    end_str   = curr_end.strftime("%Y-%m-%d")
    print(f"Fetching {start_str} → {end_str}...")
    try:
        resp = requests.get(
            AESO_PRICE_URL,
            headers={"API-KEY": AESO_API_KEY},
            params={"startDate": start_str, "endDate": end_str}
        )
        resp.raise_for_status()
        data = resp.json().get("return", {}).get("Pool Price Report", [])
        if not data:
            print(f"No data returned for {start_str}–{end_str}")
        else:
            all_records.extend(data)
    except Exception as e:
        print(f"Error fetching {start_str} → {end_str}: {e}")
    curr_start = curr_end + timedelta(days=1)

if not all_records:
    raise RuntimeError("No pool price records fetched; aborting.")

# ─── Build full DataFrame ──────────────────────────────────────────────
df = pd.DataFrame(all_records)
df["date"] = pd.to_datetime(df["begin_datetime_utc"])
df = (
    df.set_index("date")[['pool_price']]
      .rename(columns={'pool_price':'value'})
)
# ensure numeric and fill
df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill()

# ─── Write to SQLite ──────────────────────────────────────────────────
print(f"Writing {len(df)} rows to {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
df.reset_index().to_sql('wholesale', conn, if_exists='replace', index=False)
conn.close()
print("Database seeding complete.")
