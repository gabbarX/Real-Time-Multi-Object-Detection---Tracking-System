"""
RTSP / video stream reader with low-latency buffer management.

Design decisions
────────────────
• A dedicated reader thread decouples network jitter from the processing loop.
• Only the *latest* frame is kept (buffer_size=1) so the detector never works
  on stale data.
• GStreamer pipeline string is constructed for HW-accelerated decode when the
  backend is set to "gstreamer".  Falls back to OpenCV FFmpeg otherwise.
• Reconnection with exponential back-off keeps the pipeline alive across
  transient network failures.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class RTSPReader:
    """Thread-safe, low-latency RTSP/video frame provider."""

    # GStreamer pipeline for lowest-latency RTSP decode → BGR numpy array
    _GST_PIPELINE = (
        "rtspsrc location={uri} latency=0 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false"
    )

    def __init__(
        self,
        source: str,
        backend: str = "opencv",
        buffer_size: int = 1,
        target_fps: int = 30,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
        resolution: Optional[tuple[int, int]] = None,
    ) -> None:
        self.source = source
        self.backend = backend.lower()
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.resolution = resolution  # (width, height) or None

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> "RTSPReader":
        """Open the stream and start the reader thread."""
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info("RTSPReader started | source={} backend={}", self.source, self.backend)
        return self

    def read(self) -> tuple[bool, Optional[np.ndarray], int]:
        """Return (success, frame, frame_id).  Non-blocking."""
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_id
            return True, self._frame.copy(), self._frame_id

    def stop(self) -> None:
        """Signal the reader thread to stop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release()
        logger.info("RTSPReader stopped.")

    @property
    def is_alive(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _open(self) -> None:
        if self.backend == "gstreamer":
            pipeline = self._GST_PIPELINE.format(uri=self.source)
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            # Minimise internal buffer to 1 frame
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if self.resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if not self._cap.isOpened():
            raise ConnectionError(f"Cannot open stream: {self.source}")

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reader_loop(self) -> None:
        """Continuously grab frames; keep only the latest."""
        reconnects = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if reconnects >= self.max_reconnects:
                    logger.error("Max reconnect attempts reached. Stopping reader.")
                    self._running = False
                    break
                reconnects += 1
                wait = self.reconnect_delay * min(reconnects, 5)
                logger.warning("Reconnecting ({}/{}) in {:.1f}s …", reconnects, self.max_reconnects, wait)
                time.sleep(wait)
                try:
                    self._open()
                except ConnectionError:
                    continue
                continue

            grabbed = self._cap.grab()
            if not grabbed:
                self._release()
                continue

            ret, frame = self._cap.retrieve()
            if not ret or frame is None:
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1

            reconnects = 0  # reset on success

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "RTSPReader":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
