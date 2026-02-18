---
title: RTMODT — Real-Time Object Detection & Tracking
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Real-Time Multi-Object Detection & Tracking System (RTMODT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8s-orange)](#)
[![FastAPI](https://img.shields.io/badge/web-FastAPI-009688)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#)

Production-ready, low-latency pipeline for **RTSP stream ingestion**,
**YOLOv8-based multi-class object detection**, **persistent-ID multi-object
tracking** (ByteTrack), **zone-intrusion event triggering**, and **latency
profiling** — with an interactive **web frontend** and reproducible
**Docker deployment** on CUDA GPUs.

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd rtmodt
pip install -r requirements.txt

# 2. Download sample images for the web demo
python tools/download_samples.py

# 3. Launch the web app
python web/server.py

# 4. Open in browser
#    http://localhost:8000
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Web UI** | Interactive frontend — sample dataset, image/video upload, live webcam |
| **YOLOv8 Detection** | Pretrained YOLOv8s with 80 COCO classes, auto-downloads on first run |
| **ByteTrack Tracking** | Persistent ID tracking with two-stage IoU association |
| **Zone Events** | Polygon intrusion detection, dwell time, cooldown, JSON alerts |
| **Latency Profiler** | Per-stage GPU-sync profiling with p95/p99 stats |
| **RTSP Ingestion** | Threaded reader with GStreamer, single-frame buffer, auto-reconnect |
| **Config-Driven** | All behavior controlled via `config/default.yaml` |
| **Docker Ready** | CUDA 12.1 Dockerfile + Docker Compose |

---

## Prerequisites

- **Python** 3.10+
- **CUDA GPU** with drivers installed (CUDA 12.x recommended)
- **GStreamer** (optional, for HW-accelerated RTSP decode on Linux/edge)

---

## Installation

```bash
# Standard install (all core dependencies)
pip install -r requirements.txt

# Or install as a package with optional dependency groups
pip install -e ".[all]"           # everything
pip install -e ".[tracking,viz]"  # just tracking + visualization
```

### Optional Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `tracking` | lap, filterpy | ByteTrack / DeepSORT |
| `eval` | pycocotools, motmetrics | mAP, IDF1 evaluation |
| `viz` | supervision, Pillow | Annotation rendering |
| `tensorrt` | onnxruntime-gpu | GPU-accelerated ONNX inference |
| `profiling` | GPUtil, py-cpuinfo, tensorboard | Hardware profiling + TensorBoard |
| `dev` | pytest, ruff, pre-commit | Development & testing |

---

## Web Application

The interactive web UI provides three detection modes:

| Mode | Description |
|------|-------------|
| 🏠 **Sample Dataset** | Click pre-loaded images to run detection instantly |
| 📁 **Upload** | Drag & drop images or videos for detection |
| 📷 **Webcam** | Live camera stream with real-time detection at configurable intervals |

```bash
# Download sample images (COCO validation set)
python tools/download_samples.py

# Start the server
python web/server.py                     # http://localhost:8000
python web/server.py --port 9000         # custom port
python web/server.py --reload            # dev mode with auto-reload
```

The YOLOv8s model auto-downloads (~22 MB) on the first detection request.

---

## CLI Pipeline (RTSP / Video)

```bash
# Run on an RTSP camera stream
python tools/run_pipeline.py -c config/default.yaml \
    -s "rtsp://admin:pass@192.168.1.64:554/stream1"

# Run on a local video file
python tools/run_pipeline.py -c config/default.yaml -s path/to/video.mp4

# Run on a USB webcam
python tools/run_pipeline.py -c config/default.yaml -s 0

# Run headless (no display window)
python tools/run_pipeline.py -c config/default.yaml -s video.mp4 --no-display
```

Press **`q`** in the live window to stop.

---

## Training

```bash
# Download training dataset (COCO128 for testing, full COCO for production)
python tools/download_dataset.py --dataset coco128
python tools/download_dataset.py --dataset coco

# Train YOLOv8s
python tools/train.py --cfg config/training.yaml

# Monitor
tensorboard --logdir runs/train
```

---

## Model Export

```bash
# Export to TensorRT FP16 (fastest)
python tools/export_model.py -w runs/train/rtmodt_v1/weights/best.pt -f engine --half

# Export to ONNX
python tools/export_model.py -w runs/train/rtmodt_v1/weights/best.pt -f onnx --half
```

---

## Docker Deployment

```bash
# Build & run
docker build -t rtmodt .
docker run --gpus all --network host \
    -v ./config:/app/config:ro \
    -v ./weights:/app/weights:ro \
    -v ./output:/app/output \
    rtmodt

# Or with Docker Compose
docker compose up --build
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Project Structure

```
rtmodt/
├── config/
│   ├── default.yaml              # Master runtime config
│   ├── training.yaml             # YOLOv8 training hyperparameters
│   ├── dataset.yaml              # Dataset specification (COCO format)
│   └── logging.yaml              # Logging configuration
├── src/
│   ├── ingestion/
│   │   └── rtsp_reader.py        # Threaded RTSP reader (GStreamer/OpenCV)
│   ├── detection/
│   │   └── detector.py           # YOLOv8 inference wrapper (TRT/ONNX/PT)
│   ├── tracking/
│   │   └── tracker.py            # ByteTrack multi-object tracker
│   ├── events/
│   │   └── zone_engine.py        # Polygon intrusion & dwell-time engine
│   ├── profiling/
│   │   └── latency_profiler.py   # Per-stage GPU-sync latency profiler
│   ├── evaluation/
│   │   └── metrics.py            # mAP, IDF1, confusion matrix, drift
│   └── visualization/
│       └── renderer.py           # Annotated-frame renderer
├── tools/
│   ├── run_pipeline.py           # CLI entry-point for live pipeline
│   ├── train.py                  # Training launcher
│   ├── export_model.py           # ONNX / TensorRT export
│   ├── download_dataset.py       # Dataset downloader (COCO, MOT17)
│   └── download_samples.py       # Web demo sample image downloader
├── web/
│   ├── server.py                 # FastAPI backend (detection API)
│   ├── index.html                # Frontend SPA
│   └── static/
│       ├── style.css             # Dark-mode glassmorphism theme
│       ├── app.js                # Canvas rendering, webcam, uploads
│       └── samples/              # Pre-loaded demo images
├── tests/
│   └── test_web_api.py           # API smoke tests
├── Dockerfile                    # CUDA 12.1 production image
├── docker-compose.yml            # GPU-enabled compose
├── pyproject.toml                # Package metadata & CLI entry points
├── requirements.txt              # Pinned dependencies
├── .gitignore
├── .dockerignore
├── TECHNICAL_DESIGN_DOCUMENT.md  # Full architecture & design doc
└── README.md
```

---

## Configuration

All runtime behavior is driven by `config/default.yaml`:

| Section | Controls |
|---------|----------|
| `system` | Device, log level, output directory |
| `ingestion` | RTSP source, backend, buffer size, reconnect |
| `detection` | Model path, confidence, NMS IoU, target classes, FP16 |
| `tracking` | Algorithm, track buffer, match threshold |
| `events` | Polygon zones, dwell time, cooldown, alert logging |
| `profiling` | Enable/disable, GPU sync, log interval |
| `visualization` | Boxes, IDs, trails, zones, FPS HUD, recording |

---

## Documentation

See **[TECHNICAL_DESIGN_DOCUMENT.md](TECHNICAL_DESIGN_DOCUMENT.md)** for:
- System architecture and data flow diagrams
- RTSP ingestion design (OpenCV vs GStreamer)
- YOLOv8 training pipeline with concrete hyperparameters
- ByteTrack vs DeepSORT comparison
- Latency profiling methodology
- mAP / IDF1 evaluation framework
- Zone intrusion event engine
- Risk analysis (8 risks with mitigations)
- Benchmark report templates

---

## License

MIT
