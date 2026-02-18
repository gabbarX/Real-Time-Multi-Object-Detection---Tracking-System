"""
Per-frame latency profiler with GPU synchronisation.

Usage
─────
    profiler = LatencyProfiler(gpu_sync=True, warmup=50)

    profiler.tick("decode")
    frame = reader.read()
    profiler.tock("decode")

    profiler.tick("inference")
    dets = detector.detect(frame)
    profiler.tock("inference")

    profiler.end_frame()           # accumulates & optionally logs
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np
from loguru import logger

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


class LatencyProfiler:
    """Hierarchical, GPU-sync-aware latency profiler."""

    STAGE_ORDER = ["decode", "preprocess", "inference", "nms", "tracking", "events", "visualization", "total"]

    def __init__(
        self,
        gpu_sync: bool = True,
        warmup_frames: int = 50,
        log_interval: int = 100,
    ) -> None:
        self.gpu_sync = gpu_sync and _CUDA_AVAILABLE
        self.warmup = warmup_frames
        self.log_interval = log_interval

        self._starts: dict[str, float] = {}
        self._frame_times: dict[str, float] = {}
        self._history: dict[str, list[float]] = defaultdict(list)
        self._frame_count: int = 0
        self._fps_t0: float = time.perf_counter()
        self._fps_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tick(self, stage: str) -> None:
        """Mark the start of a stage."""
        if self.gpu_sync:
            torch.cuda.synchronize()
        self._starts[stage] = time.perf_counter()

    def tock(self, stage: str) -> float:
        """Mark the end of a stage.  Returns elapsed ms."""
        if self.gpu_sync:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._starts[stage]) * 1000.0
        self._frame_times[stage] = elapsed
        return elapsed

    def end_frame(self) -> Optional[dict[str, float]]:
        """
        Finalise the current frame.

        Returns a summary dict every ``log_interval`` frames after warmup,
        otherwise ``None``.
        """
        self._frame_count += 1
        total = sum(self._frame_times.values())
        self._frame_times["total"] = total

        if self._frame_count > self.warmup:
            for k, v in self._frame_times.items():
                self._history[k].append(v)

            now = time.perf_counter()
            dt = now - self._fps_t0
            if dt > 0:
                self._fps_history.append(1.0 / dt)
            self._fps_t0 = now

        self._frame_times.clear()

        if (
            self._frame_count > self.warmup
            and (self._frame_count - self.warmup) % self.log_interval == 0
        ):
            summary = self.summary()
            self._log_summary(summary)
            return summary
        return None

    def summary(self) -> dict[str, float]:
        """Compute mean/p95/p99 for each stage + FPS."""
        result: dict[str, float] = {}
        for stage in self.STAGE_ORDER:
            arr = self._history.get(stage, [])
            if arr:
                a = np.array(arr)
                result[f"{stage}_mean_ms"] = float(np.mean(a))
                result[f"{stage}_p95_ms"] = float(np.percentile(a, 95))
                result[f"{stage}_p99_ms"] = float(np.percentile(a, 99))
        if self._fps_history:
            fps = np.array(self._fps_history)
            result["fps_mean"] = float(np.mean(fps))
            result["fps_p5"] = float(np.percentile(fps, 5))
        return result

    def reset(self) -> None:
        self._history.clear()
        self._fps_history.clear()
        self._frame_count = 0

    @property
    def current_fps(self) -> float:
        if len(self._fps_history) < 2:
            return 0.0
        return float(np.mean(self._fps_history[-30:]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _log_summary(self, s: dict[str, float]) -> None:
        parts = []
        for stage in self.STAGE_ORDER:
            key = f"{stage}_mean_ms"
            if key in s:
                parts.append(f"{stage}={s[key]:.1f}ms")
        fps_str = f"FPS={s.get('fps_mean', 0):.1f}"
        logger.info("PROFILE [{}] | {}", fps_str, " | ".join(parts))
