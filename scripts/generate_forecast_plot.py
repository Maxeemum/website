import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1) Load your trained model
model = joblib.load('models/stacked_model.joblib')

# 2) Build next-24-hour feature DataFrame
now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
future = pd.date_range(now + timedelta(hours=1), periods=24, freq='H')
df = pd.DataFrame({
    'hour': future.hour,
    'dayofweek': future.dayofweek,
    # …add any lag/flag columns your model expects…
}, index=future)

# 3) Predict
y_pred = model.predict(df)

# 4) Plot & save
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(10,4))
plt.plot(future, y_pred, marker='o')
plt.title('Pool-Price Forecast')
plt.xlabel('UTC Time')
plt.ylabel('Forecast Pool Price')
plt.tight_layout()
plt.savefig('plots/forecast.png')
plt.close()
