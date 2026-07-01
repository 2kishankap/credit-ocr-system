# Single image shared by both the FastAPI service and the Celery worker
# (see docker-compose.yml -- the `worker` service overrides CMD). Keeping
# one image means the code that runs OCR/LLM logic in the background is
# byte-for-byte identical to what the API imports, so there's no drift
# between "what the API thinks it enqueued" and "what the worker runs."
FROM python:3.11-slim

# poppler-utils: required by pdf2image to rasterize PDF pages
# libgl1/libglib2.0-0: required by EasyOCR's OpenCV dependency
# fonts-dejavu-core: used to label bounding boxes in the annotated overlays
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
