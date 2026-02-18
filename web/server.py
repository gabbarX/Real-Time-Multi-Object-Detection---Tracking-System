#!/usr/bin/env python3
"""
FastAPI web server — serves the interactive frontend and exposes
detection + tracking endpoints.

Endpoints
─────────
  GET  /                      → Frontend SPA
  GET  /api/samples           → List of hardcoded sample image names
  POST /api/detect/image      → Detect objects in an uploaded image
  POST /api/detect/frame      → Detect objects in a base64 webcam frame
  GET  /api/detect/sample/{n} → Detect on a hardcoded sample image

Usage
─────
  python web/server.py                          # default port 8000
  python web/server.py --port 9000 --reload     # dev mode
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Project root on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"

# =========================================================================
# Lazy-loaded YOLOv8 model (auto-downloads pretrained weights on first use)
# =========================================================================
_model = None

# COCO class names for the 80-class pretrained YOLOv8 model
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


def _get_model():
    """Load YOLOv8s model — downloads automatically if not cached."""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    logger.info("Loading YOLOv8s model (will download on first run)…")
    _model = YOLO("yolov8s.pt")
    logger.info("✅ YOLOv8s model loaded.")
    return _model


# =========================================================================
# Detection logic
# =========================================================================

def _run_detection(img: np.ndarray) -> dict:
    """Run YOLOv8 detection on a BGR image. Returns JSON-friendly dict."""
    t0 = time.perf_counter()
    model = _get_model()

    # Run inference
    results = model(img, verbose=False, conf=0.35, iou=0.45)
    result = results[0]

    detections = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
            })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "detections": detections,
        "tracks": [],
        "inference_ms": round(elapsed, 1),
        "num_objects": len(detections),
        "image_size": [img.shape[1], img.shape[0]],
    }


# =========================================================================
# FastAPI app
# =========================================================================

app = FastAPI(title="RTMODT — Real-Time Detection & Tracking", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/samples")
async def list_samples():
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                samples.append({
                    "name": f.stem.replace("_", " ").title(),
                    "filename": f.name,
                    "url": f"/static/samples/{f.name}",
                })
    return {"samples": samples}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")
    return JSONResponse(_run_detection(img))


@app.post("/api/detect/frame")
async def detect_frame(payload: dict):
    """Accept a base64-encoded frame from the webcam."""
    data_url: str = payload.get("image", "")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(data_url)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode frame.")
    return JSONResponse(_run_detection(img))


@app.get("/api/detect/sample/{filename}")
async def detect_sample(filename: str):
    fpath = SAMPLES_DIR / filename
    if not fpath.exists() or not fpath.is_file():
        raise HTTPException(404, f"Sample '{filename}' not found.")
    img = cv2.imread(str(fpath))
    if img is None:
        raise HTTPException(500, "Could not read sample image.")
    return JSONResponse(_run_detection(img))


# =========================================================================
# CLI
# =========================================================================

@click.command()
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=8000, type=int, help="Port.")
@click.option("--reload", "do_reload", is_flag=True, help="Auto-reload on code changes.")
def serve(host: str, port: int, do_reload: bool):
    """Launch the RTMODT web application."""
    logger.info("Starting RTMODT Web UI on http://{}:{}", host, port)
    uvicorn.run(
        "web.server:app",
        host=host,
        port=port,
        reload=do_reload,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
