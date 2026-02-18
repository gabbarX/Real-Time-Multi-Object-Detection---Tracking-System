# =============================================================================
# Dockerfile — RTMODT (Compatible with Hugging Face Spaces + local Docker)
# Base: Python 3.11-slim for fast builds; use CUDA base for GPU inference
# =============================================================================
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg wget curl && \
    rm -rf /var/lib/apt/lists/*

# --- Non-root user (required by HF Spaces) ---
RUN useradd -m -u 1000 appuser

# --- Python deps ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY . .

# --- Download sample images at build time ---
RUN python tools/download_samples.py

# --- Pre-download the YOLOv8s model weights ---
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# --- Own the app directory ---
RUN chown -R appuser:appuser /app
USER appuser

# --- HF Spaces expects port 7860; overridable via env ---
ENV PORT=7860
EXPOSE 7860

CMD ["python", "start.py"]
