FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MEDIAPIPE_DISABLE_GPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Do NOT hardcode 8501 — Railway injects $PORT at runtime (currently 8080)
# The shell form of CMD evaluates $PORT correctly at runtime
CMD streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false \
    --client.toolbarMode=minimal
