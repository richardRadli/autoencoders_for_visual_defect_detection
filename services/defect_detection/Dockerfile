FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/defect_detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ /app/services/
COPY shared/ /app/shared/
COPY utils/ /app/utils/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.defect_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]