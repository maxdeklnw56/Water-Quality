"""
Full training/evaluation pipeline.
Source-aligned:
field1 Temperature, field2 pH, field3 Turbidity (0=turbid, 1=normal).
Hourly data, time-based 80/20 split.
Models:
- Moving Average baseline
- SARIMA
- LSTM (if TensorFlow is installed)
- Logistic Regression for next-hour turbidity probability
Outputs metrics, predictions, plots and confusion matrix.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
DATA="data/hourly_clean.csv"
Path("models").mkdir(exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

df=pd.read_csv(DATA,index_col=0,parse_dates=True)

def reg_metrics(y,p):
    return {
        "MAE":float(mean_absolute_error(y,p)),
        "RMSE":float(np.sqrt(mean_squared_error(y,p))),
        "MAPE_percent":float(mean_absolute_percentage_error(y,p)*100)
    }

def ma_forecast(train,test,window=24):
    hist=list(train); out=[]
    for actual in test:
        out.append(np.mean(hist[-window:]))
        hist.append(actual)
    return np.array(out)

split=int(len(df)*0.8)
all_results=[]
prediction_frames={}

for target in ["temperature","ph"]:
    tr=df[target].iloc[:split].dropna()
    te=df[target].iloc[split:].dropna()
    idx=te.index

    p=ma_forecast(tr,te,24)
    all_results.append({"target":target,"model":"MovingAverage","split":"80/20",**reg_metrics(te,p)})
    prediction_frames[(target,"MovingAverage")]=pd.Series(p,index=idx)

    model=SARIMAX(tr,order=(1,1,1),seasonal_order=(1,1,1,24),
                  enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    p=np.asarray(model.forecast(len(te)))
    all_results.append({"target":target,"model":"SARIMA","split":"80/20",**reg_metrics(te,p)})
    prediction_frames[(target,"SARIMA")]=pd.Series(p,index=idx)
    model.save(f"models/{target}_sarima.pkl")

    # LSTM
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        series=df[target].astype(float).interpolate(limit=6).values.reshape(-1,1)
        scaler=MinMaxScaler()
        train_scaled=scaler.fit_transform(series[:split])
        lookback=24

        Xtr=[]; ytr=[]
        for i in range(lookback,len(train_scaled)):
            Xtr.append(train_scaled[i-lookback:i,0]); ytr.append(train_scaled[i,0])
        Xtr=np.array(Xtr).reshape(-1,lookback,1); ytr=np.array(ytr)

        model_lstm=Sequential([
            LSTM(64,input_shape=(lookback,1),return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model_lstm.compile(optimizer="adam",loss="mse")
        model_lstm.fit(Xtr,ytr,epochs=30,batch_size=32,validation_split=0.1,
                       callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
                       verbose=0)

        history=list(train_scaled[:,0])
        preds=[]
        for _ in range(len(te)):
            x=np.array(history[-lookback:]).reshape(1,lookback,1)
            pred=float(model_lstm.predict(x,verbose=0)[0,0])
            preds.append(pred)
            history.append(pred)
        p=scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
        all_results.append({"target":target,"model":"LSTM","split":"80/20",**reg_metrics(te,p)})
        prediction_frames[(target,"LSTM")]=pd.Series(p,index=idx)
        model_lstm.save(f"models/{target}_lstm.keras")
        import joblib
        joblib.dump(scaler,f"models/{target}_lstm_scaler.joblib")
    except Exception as e:
        all_results.append({"target":target,"model":"LSTM","split":"80/20",
                            "MAE":np.nan,"RMSE":np.nan,"MAPE_percent":np.nan,
                            "note":f"Skipped: {type(e).__name__}: {e}"})

    plt.figure(figsize=(14,4))
    plt.plot(te.index,te.values,label="Actual")
    for name in ["MovingAverage","SARIMA","LSTM"]:
        s=prediction_frames.get((target,name))
        if s is not None: plt.plot(s.index,s.values,label=name)
    plt.title(f"{target}: Actual vs Forecast")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"results/figures/{target}_model_comparison.png",dpi=160)
    plt.close()

# Logistic Regression
x=df.copy()
x["temperature_lag1"]=x.temperature.shift(1)
x["ph_lag1"]=x.ph.shift(1)
x["turbidity_lag1"]=(x.turbidity_target).shift(1)
x["target_next_turbidity"]=x.turbidity_target.shift(-1)
x=x.dropna()

features=["temperature","ph","temperature_lag1","ph_lag1","turbidity_lag1"]
s=int(len(x)*0.8)
Xtr,Xte=x[features].iloc[:s],x[features].iloc[s:]
ytr,yte=x.target_next_turbidity.iloc[:s].astype(int),x.target_next_turbidity.iloc[s:].astype(int)

clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
pred=clf.predict(Xte); prob=clf.predict_proba(Xte)[:,1]
cm=confusion_matrix(yte,pred)

cls={
"Accuracy":float(accuracy_score(yte,pred)),
"Precision":float(precision_score(yte,pred,zero_division=0)),
"Recall":float(recall_score(yte,pred,zero_division=0)),
"F1":float(f1_score(yte,pred,zero_division=0)),
"ConfusionMatrix":cm.tolist()
}
import joblib
joblib.dump({"model":clf,"features":features},"models/turbidity_logistic.joblib")

plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Turbidity Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig("results/figures/turbidity_confusion_matrix.png",dpi=160); plt.close()

pd.DataFrame(all_results).to_csv("results/regression_results.csv",index=False)
with open("results/turbidity_results.json","w",encoding="utf-8") as f: json.dump(cls,f,ensure_ascii=False,indent=2)
print(pd.DataFrame(all_results).to_string(index=False))
print(json.dumps(cls,ensure_ascii=False,indent=2))
