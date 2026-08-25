import os
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
# เปิดให้หน้าเว็บ (Frontend) ที่อยู่คนละโดเมนสามารถเรียกใช้งาน API นี้ได้
CORS(app)

# โหลดโมเดล Classification (ความขุ่น) เตรียมไว้
MODEL_PATH = "models/turbidity_logistic.joblib"
try:
    pack = joblib.load(MODEL_PATH)
    clf, features = pack["model"], pack["features"]
except Exception as e:
    clf, features = None, []
    print(f"Warning: Cannot load turbidity model: {e}")

@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.json
    if not data or not data.get('channel_id'):
        return jsonify({'status': 'error', 'error': 'Missing channel_id'}), 400

    channel_id = data.get('channel_id')
    read_key = data.get('read_key', '')

    try:
        # 1. ดึงข้อมูล 5 วันล่าสุดจาก ThingSpeak ของผู้ใช้ (ใช้เวลาน้อยลง ไม่ต้องดึงทั้งหมด)
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
        r = requests.get(url, params={"api_key": read_key, "days": 5, "results": 8000}, timeout=15)
        if r.status_code != 200:
            return jsonify({'status': 'error', 'error': 'ไม่สามารถดึงข้อมูลจาก ThingSpeak ได้ (ตรวจสอบ ID/Key)'}), 400
        
        feeds = r.json().get('feeds', [])
        if not feeds:
            return jsonify({'status': 'error', 'error': 'ไม่มีข้อมูลในช่อง ThingSpeak นี้เลย'}), 404

        # 2. จัดเตรียมข้อมูล (Cleaning)
        df = pd.DataFrame(feeds)[["created_at", "field1", "field2", "field3"]].rename(columns={
            "created_at":"timestamp", "field1":"temperature", "field2":"ph", "field3":"turbidity"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for c in ["temperature", "ph", "turbidity"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.loc[df.temperature == -127, "temperature"] = np.nan
        df.loc[(df.ph < 0) | (df.ph > 14), "ph"] = np.nan
        df.loc[~df.turbidity.isin([0,1]), "turbidity"] = np.nan
        df = df.set_index("timestamp").sort_index()

        df["temperature"] = df["temperature"].interpolate(method="time", limit=6)
        df["ph"] = df["ph"].interpolate(method="time", limit=6)

        # ทำเป็นข้อมูลรายชั่วโมง
        hourly = pd.DataFrame({
            "temperature": df.temperature.resample("1h").mean(),
            "ph": df.ph.resample("1h").mean(),
            "turbidity": df.turbidity.resample("1h").min()
        }).dropna(subset=["temperature", "ph"])

        # ปิดการแจ้งเตือนข้อมูลไม่พอชั่วคราว
        # if len(hourly) < 24:
        #     return jsonify({'status': 'error', 'error': 'ข้อมูลรายชั่วโมงมีไม่เพียงพอต่อการพยากรณ์ (ต้องการอย่างน้อย 24 ชม.)'}), 400

        # ปรับ seasonal_order ให้เป็น 0 ทั้งหมด เพื่อให้รันได้แม้ชั่วโมงข้อมูลจะไม่ต่อเนื่องกัน
        temp_model = SARIMAX(hourly.temperature, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        ph_model = SARIMAX(hourly.ph, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

        temp_forecast = float(temp_model.forecast(1).iloc[0])
        ph_forecast = float(ph_model.forecast(1).iloc[0])

        # 4. พยากรณ์ความขุ่นด้วย Logistic Regression
        hourly["temperature_lag1"] = hourly["temperature"].shift(1)
        hourly["ph_lag1"] = hourly["ph"].shift(1)
        hourly["turbidity_lag1"] = (hourly["turbidity"] == 0).astype(int).shift(1)

        last_row = hourly.iloc[-1]
        history_probs = []
        
        if clf is not None:
            # หาค่าพยากรณ์ล่วงหน้า 1 ชั่วโมง
            X_next = pd.DataFrame([last_row])[features]
            turb_prob = float(clf.predict_proba(X_next)[:, 1][0])
            turb_pred = int(turb_prob >= 0.5)

            # คำนวณความน่าจะเป็นย้อนหลัง 24 ชม. สำหรับวาดกราฟ
            valid_hourly = hourly.dropna(subset=features).tail(24)
            if not valid_hourly.empty:
                probs = clf.predict_proba(valid_hourly[features])[:, 1]
                for ts, p in zip(valid_hourly.index, probs):
                    history_probs.append({
                        "time": ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "prob": float(p)
                    })
        else:
            turb_prob = 0.0; turb_pred = 0

        # 5. ส่งผลลัพธ์กลับไปยังหน้าเว็บ
        timestamp_str = hourly.index[-1].strftime('%Y-%m-%dT%H:%M:%SZ')
        return jsonify({
            'status': 'success',
            'data': {
                'temperature': round(temp_forecast, 3),
                'ph': round(ph_forecast, 3),
                'turbidity_prob': round(turb_prob, 4),
                'turbidity_pred': turb_pred,
                'timestamp': timestamp_str,
                'history_probs': history_probs
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': 'เกิดข้อผิดพลาดในการประมวลผลโมเดล'}), 500

if __name__ == '__main__':
    # รันบนพอร์ต 5000
    app.run(host='0.0.0.0', port=5000, debug=True)