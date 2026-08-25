# Water Quality Forecasting — Full Version

## Source mapping
อิงตามไฟล์ระบบที่ให้มา:
- IOT.ino: field1=Temperature, field2=pH, field3=Turbidity
- ส่งข้อมูล ThingSpeak ทุก 10 นาที
- Turbidity digital: 0=พบความขุ่น, 1=ปกติ
- index.html: อ่าน feeds.json ย้อนหลังและแสดง Temperature/pH/Turbidity

## Pipeline
1. 01_download_thingspeak.py — ดึงย้อนหลัง
2. 02_clean_and_eda.py — sentinel -127, pH range, interpolation, hourly resample, EDA, ADF
3. 03_train_models_full.py — Moving Average, SARIMA, LSTM, Logistic Regression, metrics
4. 04_forecast_and_upload.py — forecast แล้วส่งกลับ ThingSpeak
5. 05_make_report.py — สร้าง PDF report
6. index_forecast.html — ตัวอย่างส่วนแสดง Forecast

## Time split
ใช้ 80/20 ตามเวลา ห้าม random shuffle

## Forecast fields
Field 4 = forecast temperature
Field 5 = forecast pH
Field 6 = probability that next hour is turbid
Field 7 = predicted turbidity (1=turbid, 0=normal)

## Install
pip install -r requirements_full.txt

ถ้าจะใช้ LSTM ต้องมี TensorFlow ที่ตรงกับ OS/Python ของเครื่อง
