import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA = "data/hourly_clean.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA, index_col=0, parse_dates=True)

def regression_metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE_percent": float(mean_absolute_percentage_error(y, p) * 100)
    }

def moving_average_forecast(train, test, window=24):
    history = list(train)
    out = []
    for actual in test:
        out.append(np.mean(history[-window:]))
        history.append(actual)
    return np.array(out)

results = []

# Time split: 80/20, no random shuffle
split = int(len(df) * 0.8)

for target in ["temperature", "ph"]:
    train = df[target].iloc[:split]
    test = df[target].iloc[split:]

    # Baseline
    p_ma = moving_average_forecast(train, test, 24)
    m = regression_metrics(test, p_ma)
    results.append({"target": target, "model": "MovingAverage", **m})

    # SARIMA, daily seasonality for hourly data
    sarima = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    p_sarima = sarima.forecast(len(test))
    m = regression_metrics(test, p_sarima)
    results.append({"target": target, "model": "SARIMA", **m})
    sarima.save(MODEL_DIR / f"{target}_sarima.pkl")

# Turbidity Logistic Regression
feature_df = df.copy()
feature_df["temperature_lag1"] = feature_df["temperature"].shift(1)
feature_df["ph_lag1"] = feature_df["ph"].shift(1)
feature_df["turbidity_lag1"] = feature_df["turbidity_target"].shift(1)
feature_df["target_next_turbidity"] = feature_df["turbidity_target"].shift(-1)
feature_df = feature_df.dropna()

features = ["temperature", "ph", "temperature_lag1", "ph_lag1", "turbidity_lag1"]
X = feature_df[features]
y = feature_df["target_next_turbidity"].astype(int)

s2 = int(len(feature_df) * 0.8)
X_train, X_test = X.iloc[:s2], X.iloc[s2:]
y_train, y_test = y.iloc[:s2], y.iloc[s2:]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, pred).tolist()
classification = {
    "Accuracy": float(accuracy_score(y_test, pred)),
    "Precision": float(precision_score(y_test, pred, zero_division=0)),
    "Recall": float(recall_score(y_test, pred, zero_division=0)),
    "F1": float(f1_score(y_test, pred, zero_division=0)),
    "ConfusionMatrix": cm
}

joblib.dump({"model": clf, "features": features}, MODEL_DIR / "turbidity_logistic.joblib")

Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("results/regression_results.csv", index=False)
with open("results/turbidity_results.json", "w", encoding="utf-8") as f:
    json.dump(classification, f, ensure_ascii=False, indent=2)

print(pd.DataFrame(results))
print(json.dumps(classification, ensure_ascii=False, indent=2))
