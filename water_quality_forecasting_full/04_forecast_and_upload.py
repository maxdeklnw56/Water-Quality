import os
import joblib
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX

CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID", "YOUR_CHANNEL_ID")
READ_API_KEY = os.getenv("THINGSPEAK_READ_KEY", "YOUR_READ_API_KEY")
WRITE_API_KEY = os.getenv("THINGSPEAK_WRITE_KEY", "YOUR_WRITE_API_KEY")

def get_history(days=30):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    r = requests.get(url, params={"api_key": READ_API_KEY, "days": days, "results": 8000}, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["feeds"])[["created_at","field1","field2","field3"]].rename(columns={
        "created_at":"timestamp","field1":"temperature","field2":"ph","field3":"turbidity"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["temperature","ph","turbidity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df.temperature == -127, "temperature"] = np.nan
    df.loc[(df.ph < 0) | (df.ph > 14), "ph"] = np.nan
    df.loc[~df.turbidity.isin([0,1]), "turbidity"] = np.nan
    df = df.set_index("timestamp").sort_index()
    df["temperature"] = df["temperature"].interpolate(method="time", limit=6)
    df["ph"] = df["ph"].interpolate(method="time", limit=6)
    return df

df = get_history()
hourly = pd.DataFrame({
    "temperature": df.temperature.resample("1h").mean(),
    "ph": df.ph.resample("1h").mean(),
    "turbidity": df.turbidity.resample("1h").min()
}).dropna(subset=["temperature","ph"])

# Retrain SARIMA on latest historical data for a fresh next-hour forecast.
temp_model = SARIMAX(
    hourly.temperature, order=(1,1,1), seasonal_order=(1,1,1,24),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)
ph_model = SARIMAX(
    hourly.ph, order=(1,1,1), seasonal_order=(1,1,1,24),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)

temp_forecast = float(temp_model.forecast(1).iloc[0])
ph_forecast = float(ph_model.forecast(1).iloc[0])

# Logistic model predicts P(next hour is turbid), where target=1 means turbid.
pack = joblib.load("models/turbidity_logistic.joblib")
clf, features = pack["model"], pack["features"]

last = hourly.iloc[-1]
previous = hourly.iloc[-2] if len(hourly) >= 2 else last
X_next = pd.DataFrame([{
    "temperature": last.temperature,
    "ph": last.ph,
    "temperature_lag1": previous.temperature,
    "ph_lag1": previous.ph,
    "turbidity_lag1": int(last.turbidity == 0)
}])[features]

if clf is not None:
    turb_prob = float(clf.predict_proba(X_next)[:, 1][0])
    turb_pred = int(turb_prob >= 0.5)
else:
    turb_prob = 0.0
    turb_pred = 0

# ป้องกันค่า NaN ใน turbidity ก่อนแปลงเป็น int
turb_val = last.turbidity
if pd.isna(turb_val):
    turb_val = 1  # กำหนดค่าสำรองเป็น 1 (ปกติ) หากเป็นค่าว่าง

payload = {
    "api_key": WRITE_API_KEY,
    "field1": float(last.temperature),
    "field2": float(last.ph),
    "field3": int(turb_val),
    "field4": round(temp_forecast, 3),
    "field5": round(ph_forecast, 3),
    "field6": round(turb_prob, 4),
    "field7": turb_pred
}

r = requests.post("https://api.thingspeak.com/update", data=payload, timeout=30)
r.raise_for_status()
print("ThingSpeak entry:", r.text)
print(payload)
