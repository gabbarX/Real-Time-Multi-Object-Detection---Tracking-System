# =============================================================================
# Dockerfile — Real-Time Multi-Object Detection & Tracking System
# Base: NVIDIA CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# =============================================================================
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgstreamer1.0-0 gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-libav \
        ffmpeg wget curl git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# --- Python deps ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY . .

# --- Default entry-point ---
ENTRYPOINT ["python", "tools/run_pipeline.py"]
CMD ["--config", "config/default.yaml"]
