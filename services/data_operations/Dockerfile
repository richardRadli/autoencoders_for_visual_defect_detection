FROM python:3.12-slim

WORKDIR /app

COPY services/data_operations/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ /app/services/
COPY shared/ /app/shared/
COPY utils/ /app/utils/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.data_operations.main:app", "--host", "0.0.0.0", "--port", "8000"]