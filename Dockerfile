# استخدم صورة Python أساسية خفيفة مبنية على Alpine Linux (مشهورة بخفتها)
FROM python:3.10.18-alpine

WORKDIR /app



# نسخ باقي ملفات المشروع للـ image
COPY . .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
