from pathlib import Path
import pandas as pd, json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

Path("results").mkdir(exist_ok=True)
doc=SimpleDocTemplate("results/forecasting_report.pdf",pagesize=A4,
                      rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.5*cm)
styles=getSampleStyleSheet()
story=[Paragraph("Water Quality Forecasting Report",styles["Title"]),
       Paragraph("อิงโครงสร้างจาก IOT.ino และ index.html: Field 1 Temperature, Field 2 pH, Field 3 Turbidity; ส่งข้อมูลทุก 10 นาที; Turbidity 0=ขุ่น, 1=ปกติ",styles["BodyText"]),Spacer(1,10)]
res=pd.read_csv("results/regression_results.csv")
data=[list(res.columns)]+res.fillna("").astype(str).values.tolist()
t=Table(data,repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.5,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
story += [Paragraph("Regression results",styles["Heading2"]),t,Spacer(1,10)]
with open("results/turbidity_results.json",encoding="utf-8") as f: cls=json.load(f)
story += [Paragraph("Turbidity classification",styles["Heading2"]),
          Paragraph(json.dumps(cls,ensure_ascii=False),styles["BodyText"]),Spacer(1,10)]
for p in ["temperature_model_comparison.png","ph_model_comparison.png","turbidity_confusion_matrix.png"]:
    path=Path("results/figures")/p
    if path.exists(): story += [Image(str(path),width=16*cm,height=5.2*cm),Spacer(1,6)]
doc.build(story)
print("Created results/forecasting_report.pdf")
