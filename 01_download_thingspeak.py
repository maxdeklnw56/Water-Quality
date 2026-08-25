import os
import requests
import pandas as pd
from pathlib import Path

CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID", "YOUR_CHANNEL_ID")
READ_API_KEY = os.getenv("THINGSPEAK_READ_KEY", "YOUR_READ_API_KEY")
DAYS = int(os.getenv("THINGSPEAK_DAYS", "30"))

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
params = {"api_key": READ_API_KEY, "days": DAYS, "results": 8000}

r = requests.get(url, params=params, timeout=30)
r.raise_for_status()
data = r.json()

if "feeds" not in data:
    raise RuntimeError("ThingSpeak ไม่ส่ง feeds กลับมา ตรวจสอบ Channel ID / Read API Key")

df = pd.DataFrame(data["feeds"])
df = df[["created_at", "field1", "field2", "field3"]].rename(columns={
    "created_at": "timestamp",
    "field1": "temperature",
    "field2": "ph",
    "field3": "turbidity"
})

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
for c in ["temperature", "ph", "turbidity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("timestamp").drop_duplicates("timestamp")
Path("data").mkdir(exist_ok=True)
df.to_csv("data/raw_data.csv", index=False)
print(f"Saved {len(df)} rows -> data/raw_data.csv")
