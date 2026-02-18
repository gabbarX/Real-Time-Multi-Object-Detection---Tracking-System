# Real-Time Multi-Object Detection & Tracking System (RTMODT)

Production-ready, low-latency pipeline for RTSP stream ingestion, YOLOv8-based
multi-class object detection, persistent-ID multi-object tracking (ByteTrack),
zone-intrusion event triggering, and latency profiling — packaged for
reproducible Docker deployment on CUDA GPUs.

---

## Prerequisites

- **Python** 3.10+
- **CUDA GPU** with drivers installed (CUDA 12.x recommended)
- **GStreamer** (optional, for HW-accelerated RTSP decode on Linux/edge)

---

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd rtmodt

# Install Python dependencies
pip install -r requirements.txt
```

---

## 1. Download a Dataset

Use the built-in download script to fetch a supported dataset:

```bash
# COCO128 — 128 images, great for quick smoke-testing (~7 MB)
python tools/download_dataset.py --dataset coco128

# Full COCO 2017 — 118K train + 5K val images (~20 GB)
python tools/download_dataset.py --dataset coco

# MOT17 — tracking benchmark for IDF1/MOTA evaluation (~5 GB)
python tools/download_dataset.py --dataset mot17
```

After download, datasets are placed in `datasets/` and dataset YAMLs are auto-generated.

---

## 2. Train a Model

```bash
# Train YOLOv8s on the downloaded dataset
python tools/train.py --cfg config/training.yaml

# Override hyperparameters inline
python tools/train.py --cfg config/training.yaml --epochs 200 --batch 8

# Monitor training with TensorBoard
tensorboard --logdir runs/train --port 6006
```

Training configs live in `config/training.yaml` and `config/dataset.yaml`.

---

## 3. Export the Model

```bash
# Export to TensorRT FP16 engine (fastest inference)
python tools/export_model.py -w runs/train/rtmodt_v1/weights/best.pt -f engine --half

# Export to ONNX
python tools/export_model.py -w runs/train/rtmodt_v1/weights/best.pt -f onnx --half
```

Place the exported `.engine` or `.pt` file in `weights/` and update `config/default.yaml → detection.model_path`.

---

## 4. Run the Pipeline

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

Press **`q`** in the live window to stop the pipeline.

---

## 5. Docker Deployment

```bash
# Build the image
docker build -t rtmodt .

# Run with GPU access
docker run --gpus all --network host \
    -v ./config:/app/config:ro \
    -v ./weights:/app/weights:ro \
    -v ./output:/app/output \
    rtmodt

# Or use Docker Compose
docker compose up --build
```

---

## Project Layout

```
├── config/
│   ├── default.yaml          # master runtime config
│   ├── training.yaml         # YOLOv8 training hypers
│   └── dataset.yaml          # dataset spec (COCO format)
├── src/
│   ├── ingestion/            # RTSP reader (threaded, GStreamer)
│   ├── detection/            # YOLOv8 inference wrapper
│   ├── tracking/             # ByteTrack multi-object tracker
│   ├── events/               # Zone intrusion & dwell-time engine
│   ├── profiling/            # Per-stage latency profiler
│   ├── evaluation/           # mAP, IDF1, confusion matrix
│   └── visualization/        # Annotated-frame renderer
├── tools/
│   ├── run_pipeline.py       # CLI entry-point
│   ├── train.py              # Training launcher
│   ├── export_model.py       # ONNX / TensorRT export
│   └── download_dataset.py   # Dataset downloader (COCO, MOT17)
├── weights/                  # Model checkpoints (git-ignored)
├── datasets/                 # Downloaded datasets (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── TECHNICAL_DESIGN_DOCUMENT.md
```

## Key Commands

| Action | Command |
|--------|---------|
| Download COCO128 | `python tools/download_dataset.py -d coco128` |
| Train YOLOv8 | `python tools/train.py --cfg config/training.yaml` |
| Export to TensorRT | `python tools/export_model.py -w best.pt -f engine --half` |
| Run pipeline (RTSP) | `python tools/run_pipeline.py -c config/default.yaml -s "rtsp://..."` |
| Run pipeline (video) | `python tools/run_pipeline.py -c config/default.yaml -s video.mp4` |
| Docker run | `docker compose up --build` |

---

## Configuration

All runtime behaviour is controlled by `config/default.yaml`. Key sections:

- **`ingestion`** — RTSP source, backend (opencv/gstreamer), buffer size
- **`detection`** — model path, confidence threshold, NMS IoU, target classes
- **`tracking`** — algorithm (bytetrack/deepsort), track buffer, match threshold
- **`events`** — polygon zones, dwell time, cooldown, alert logging
- **`profiling`** — enable/disable, GPU sync, log interval
- **`visualization`** — boxes, IDs, trails, zone overlays, FPS HUD, video recording

---

## Documentation

See **[TECHNICAL_DESIGN_DOCUMENT.md](TECHNICAL_DESIGN_DOCUMENT.md)** for the
complete architecture, design rationale, profiling methodology, risk analysis,
and benchmark templates.
