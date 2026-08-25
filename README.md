# Water Quality Forecasting

โปรเจกต์นี้อิงโครงสร้างข้อมูลจาก IOT.ino และ index.html:
- field1 = Temperature
- field2 = pH
- field3 = Turbidity
- Sensor ส่ง ThingSpeak ทุก 10 นาที
- Turbidity: 0 = พบความขุ่น, 1 = ปกติ

## วิธีใช้

1. ตั้ง environment variables:
   - THINGSPEAK_CHANNEL_ID
   - THINGSPEAK_READ_KEY
   - THINGSPEAK_WRITE_KEY

2. ติดตั้ง:
   pip install -r requirements.txt

3. ดาวน์โหลดข้อมูล:
   python 01_download_thingspeak.py

4. Clean + EDA + ADF:
   python 02_clean_and_eda.py

5. Train/evaluate:
   python 03_train_models.py

6. Forecast + upload:
   python 04_forecast_and_upload.py

หมายเหตุ:
- Regression: Moving Average vs SARIMA. LSTM ควรเพิ่มหลังจาก baseline/SARIMA ผ่านการตรวจสอบแล้ว
- Classification: Logistic Regression; target=1 หมายถึงคาดว่าจะพบความขุ่น
- Train/test เป็น time-based split 80/20
- ห้ามใส่ API key จริงใน GitHub
