# استخدم صورة Python أساسية خفيفة مبنية على Alpine Linux (مشهورة بخفتها)
FROM python:3.10.18-alpine

# ضبط متغيرات البيئة لـ Python عشان يكون الأداء أفضل
ENV PYTHONUNBUFFERED 1

# تحديث pip وتثبيت التبعيات الأساسية
RUN pip install --no-cache-dir --upgrade pip

# تثبيت PyTorch (CPU) بالإصدار المحدد (2.2.0)
# بنستخدم --extra-index-url عشان نجيبها من موقع PyTorch الرسمي
# وبنحدد --index-url عشان pip مايدورش في أماكن تانية ويلاقي إصدارات تانية
# بنحدد --no-cache-dir عشان نقلل حجم الـ image
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# إنشاء مجلد داخل الـ image للمشروع بتاعك
WORKDIR /app

# نسخ ملف requirements.txt (لو عندك)
# ده بيخلي Docker يعيد بناء الطبقة دي بس لو الـ requirements اتغيرت، وبيسرع البناء
COPY requirements.txt .

# تثبيت أي حزم تانية مطلوبة في ملف requirements.txt
# استخدم أمر pip install -r requirements.txt لو عندك حزم تانية
# لو ماعندكش requirements.txt، ممكن تشيل السطرين دول
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع للـ image
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
