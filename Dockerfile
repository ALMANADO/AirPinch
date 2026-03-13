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
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

CMD streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false \
    --client.toolbarMode=minimal
