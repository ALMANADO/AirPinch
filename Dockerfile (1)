# python:3.11-slim is Debian Bookworm — all packages exist and resolve cleanly
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MEDIAPIPE_DISABLE_GPU=1

# Install ONLY packages that actually exist in Debian Bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libglib2.0-dev \
    libgl1 \
    libgl1-mesa-glx \
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

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

CMD streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8501} \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false \
    --client.toolbarMode=minimal
