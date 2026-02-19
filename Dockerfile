FROM python:3.10-slim

WORKDIR /app

# Prevent pip cache
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Install CPU-only torch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
