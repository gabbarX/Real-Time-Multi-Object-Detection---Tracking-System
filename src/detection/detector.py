"""
YOLOv8 / TensorRT detection wrapper.

Key design points
─────────────────
• Attempts to load a TensorRT engine first; falls back to the PyTorch .pt
  checkpoint via Ultralytics.
• Exposes a single `detect(frame) → Detections` interface consumed by the
  tracker and event engine.
• GPU warm-up at init prevents cold-start latency spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data container returned by the detector
# ---------------------------------------------------------------------------
@dataclass
class Detections:
    """Structured detection output for a single frame."""
    xyxy: np.ndarray             # (N, 4) float32 — x1 y1 x2 y2
    confidence: np.ndarray       # (N,)   float32
    class_id: np.ndarray         # (N,)   int32
    class_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.confidence)

    def filter_classes(self, keep: Sequence[int]) -> "Detections":
        """Return a new Detections keeping only specified class IDs."""
        mask = np.isin(self.class_id, keep)
        return Detections(
            xyxy=self.xyxy[mask],
            confidence=self.confidence[mask],
            class_id=self.class_id[mask],
            class_names=[n for n, m in zip(self.class_names, mask) if m],
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """Unified YOLOv8 detector (PyTorch / ONNX / TensorRT)."""

    _WARMUP_ITERATIONS = 10

    def __init__(
        self,
        model_path: str,
        fallback_model: Optional[str] = None,
        input_size: tuple[int, int] = (640, 640),
        confidence: float = 0.35,
        iou: float = 0.45,
        classes: Optional[list[int]] = None,
        half: bool = True,
        device: str = "cuda:0",
        max_det: int = 100,
        agnostic_nms: bool = False,
    ) -> None:
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.half = half and torch.cuda.is_available()
        self.device = device
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms

        # Load model — try primary, then fallback
        primary = Path(model_path)
        if primary.exists():
            self.model = YOLO(str(primary))
            logger.info("Loaded model: {}", primary.name)
        elif fallback_model and Path(fallback_model).exists():
            self.model = YOLO(fallback_model)
            logger.warning("Primary model missing; loaded fallback: {}", fallback_model)
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {fallback_model}")

        # Warm-up to stabilise CUDA graphs / TRT context
        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Detections:
        """Run inference on a single BGR frame and return `Detections`."""
        results = self.model.predict(
            source=frame,
            imgsz=self.input_size[0],
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            device=self.device,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        return self._parse(results[0])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse(self, result) -> Detections:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=np.int32),
            )
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        names = [result.names.get(int(c), str(c)) for c in cls]
        return Detections(xyxy=xyxy, confidence=conf, class_id=cls, class_names=names)

    def _warmup(self) -> None:
        dummy = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)  # HWC
        for _ in range(self._WARMUP_ITERATIONS):
            self.detect(dummy)
        logger.info("Detector warm-up complete ({} iterations).", self._WARMUP_ITERATIONS)
