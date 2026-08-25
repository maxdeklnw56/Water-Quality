import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
RAW = "data/raw_data.csv"
OUT = "data/hourly_clean.csv"

df = pd.read_csv(RAW, parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

# Source-specific cleaning: DS18B20 sentinel
df.loc[df["temperature"] == -127, "temperature"] = np.nan

# Plausibility check supported by the IOT.ino pH logic
df.loc[(df["ph"] < 0) | (df["ph"] > 14), "ph"] = np.nan

# Turbidity in IOT.ino is digital: 0 = turbid, 1 = normal.
df.loc[~df["turbidity"].isin([0, 1]), "turbidity"] = np.nan

# Time interpolation for continuous variables
df["temperature"] = df["temperature"].interpolate(method="time", limit=6)
df["ph"] = df["ph"].interpolate(method="time", limit=6)

# Hourly aggregation:
# continuous variables = mean
# turbidity = min, so any 0 in the hour means "turbid" (0)
hourly = pd.DataFrame({
    "temperature": df["temperature"].resample("1h").mean(),
    "ph": df["ph"].resample("1h").mean(),
    "turbidity": df["turbidity"].resample("1h").min()
})

hourly["turbidity"] = hourly["turbidity"].fillna(1).astype(int)
hourly["turbidity_target"] = (hourly["turbidity"] == 0).astype(int)

hourly.to_csv(OUT)
print(f"Saved {len(hourly)} hourly rows -> {OUT}")

def adf(name):
    s = hourly[name].dropna()
    stat, p, *_ = adfuller(s)
    print(f"{name}: ADF={stat:.4f}, p-value={p:.6f}, stationary={p < 0.05}")

adf("temperature")
adf("ph")

# EDA plots
Path("data/figures").mkdir(parents=True, exist_ok=True)
for col, title, unit in [
    ("temperature", "Temperature", "°C"),
    ("ph", "pH", "pH")
]:
    plt.figure(figsize=(13, 4))
    plt.plot(hourly.index, hourly[col])
    plt.title(title)
    plt.ylabel(unit)
    plt.tight_layout()
    plt.savefig(f"data/figures/{col}_trend.png", dpi=150)
    plt.close()

plt.figure(figsize=(10,4))
hourly.assign(hour=hourly.index.hour).groupby("hour")["temperature"].mean().plot(marker="o")
plt.title("Average Temperature by Hour")
plt.xlabel("Hour")
plt.ylabel("°C")
plt.tight_layout()
plt.savefig("data/figures/temperature_daily_pattern.png", dpi=150)
plt.close()

print("EDA figures saved in data/figures/")
