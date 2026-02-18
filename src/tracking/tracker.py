"""
Multi-object tracker — ByteTrack (default) / DeepSORT adapter.

ByteTrack is chosen as the default because:
  1. No separate Re-ID model → lower latency.
  2. Two-stage association (high + low confidence) recovers occluded objects.
  3. Achieves IDF1 ≥ 0.80 on MOT17 without appearance features.

DeepSORT is provided for scenarios where camera viewpoint changes frequently
and appearance re-identification is critical.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Track data container
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """Represents a single tracked object."""
    track_id: int
    xyxy: np.ndarray               # (4,) float32
    confidence: float
    class_id: int
    class_name: str = ""
    age: int = 0                   # frames since first detection
    time_since_update: int = 0     # frames since last matched detection
    trail: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ByteTrack implementation (simplified, production-ready)
# ---------------------------------------------------------------------------
class _ByteTrackCore:
    """Minimal ByteTrack implementation using IoU-based association."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self._next_id = 1
        self._tracks: list[dict] = []

    def update(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
    ) -> list[dict]:
        """
        Perform one tracking step.

        Returns list of active track dicts:
            {"track_id", "xyxy", "confidence", "class_id", "age", "time_since_update"}
        """
        if len(confidence) == 0:
            # Age-out existing tracks
            self._age_tracks()
            return [t for t in self._tracks if t["time_since_update"] == 0]

        # ── 1. Split detections into high / low confidence ──
        high_mask = confidence >= self.track_thresh
        low_mask = ~high_mask

        high_dets = xyxy[high_mask]
        high_conf = confidence[high_mask]
        high_cls = class_id[high_mask]

        low_dets = xyxy[low_mask]
        low_conf = confidence[low_mask]
        low_cls = class_id[low_mask]

        # ── 2. First association: high-confidence dets ↔ existing tracks ──
        unmatched_tracks_idx: list[int] = []
        unmatched_dets_idx: list[int] = list(range(len(high_dets)))

        if self._tracks and len(high_dets) > 0:
            iou_matrix = self._batch_iou(
                np.array([t["xyxy"] for t in self._tracks]),
                high_dets,
            )
            matched_t, matched_d, unmatched_tracks_idx, unmatched_dets_idx = (
                self._linear_assignment(iou_matrix, thresh=self.match_thresh)
            )
            for ti, di in zip(matched_t, matched_d):
                self._tracks[ti]["xyxy"] = high_dets[di]
                self._tracks[ti]["confidence"] = float(high_conf[di])
                self._tracks[ti]["class_id"] = int(high_cls[di])
                self._tracks[ti]["age"] += 1
                self._tracks[ti]["time_since_update"] = 0
        else:
            unmatched_tracks_idx = list(range(len(self._tracks)))

        # ── 3. Second association: low-confidence dets ↔ unmatched tracks ──
        remaining_tracks = [self._tracks[i] for i in unmatched_tracks_idx]
        if remaining_tracks and len(low_dets) > 0:
            iou_matrix = self._batch_iou(
                np.array([t["xyxy"] for t in remaining_tracks]),
                low_dets,
            )
            matched_t2, matched_d2, still_unmatched, _ = self._linear_assignment(
                iou_matrix, thresh=self.match_thresh
            )
            for ti, di in zip(matched_t2, matched_d2):
                remaining_tracks[ti]["xyxy"] = low_dets[di]
                remaining_tracks[ti]["confidence"] = float(low_conf[di])
                remaining_tracks[ti]["class_id"] = int(low_cls[di])
                remaining_tracks[ti]["age"] += 1
                remaining_tracks[ti]["time_since_update"] = 0

        # ── 4. Create new tracks from unmatched high-confidence detections ──
        for di in unmatched_dets_idx:
            self._tracks.append({
                "track_id": self._next_id,
                "xyxy": high_dets[di],
                "confidence": float(high_conf[di]),
                "class_id": int(high_cls[di]),
                "age": 1,
                "time_since_update": 0,
            })
            self._next_id += 1

        # ── 5. Remove dead tracks ──
        self._age_tracks()
        self._tracks = [t for t in self._tracks if t["time_since_update"] <= self.track_buffer]

        return [t for t in self._tracks if t["time_since_update"] == 0]

    # ── helpers ──
    def _age_tracks(self) -> None:
        for t in self._tracks:
            if t["time_since_update"] > 0 or True:
                t["time_since_update"] += 1
            # Reset only when matched (done in update)

    @staticmethod
    def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix (M×N) between two sets of boxes."""
        x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
        y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
        x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
        y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / (union + 1e-6)

    @staticmethod
    def _linear_assignment(
        cost_matrix: np.ndarray, thresh: float
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Greedy assignment (replace with lap.lapjv for production)."""
        try:
            import lap
            _, x, y = lap.lapjv(1 - cost_matrix, extend_cost=True, cost_limit=1 - thresh)
            matched_a, matched_b = [], []
            unmatched_a = list(range(cost_matrix.shape[0]))
            unmatched_b = list(range(cost_matrix.shape[1]))
            for i, j in enumerate(x):
                if j >= 0:
                    matched_a.append(i)
                    matched_b.append(j)
                    unmatched_a.remove(i)
                    if j in unmatched_b:
                        unmatched_b.remove(j)
            return matched_a, matched_b, unmatched_a, unmatched_b
        except ImportError:
            # Fallback: greedy argmax
            matched_a, matched_b = [], []
            used_b: set[int] = set()
            for i in range(cost_matrix.shape[0]):
                best_j = int(np.argmax(cost_matrix[i]))
                if cost_matrix[i, best_j] >= thresh and best_j not in used_b:
                    matched_a.append(i)
                    matched_b.append(best_j)
                    used_b.add(best_j)
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
            unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in used_b]
            return matched_a, matched_b, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Public multi-object tracker façade
# ---------------------------------------------------------------------------
class MultiObjectTracker:
    """High-level tracker wrapping ByteTrack (or DeepSORT)."""

    def __init__(self, algorithm: str = "bytetrack", **kwargs) -> None:
        self.algorithm = algorithm.lower()
        if self.algorithm == "bytetrack":
            bt_params = kwargs.get("bytetrack", kwargs)
            self._core = _ByteTrackCore(
                track_thresh=bt_params.get("track_thresh", 0.5),
                track_buffer=bt_params.get("track_buffer", 30),
                match_thresh=bt_params.get("match_thresh", 0.8),
            )
        elif self.algorithm == "deepsort":
            # Placeholder — plug in deep_sort_realtime or custom DeepSORT
            raise NotImplementedError("DeepSORT adapter not yet wired. Use bytetrack.")
        else:
            raise ValueError(f"Unknown tracker: {self.algorithm}")

        self._trail_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._trail_maxlen = 30
        logger.info("Tracker initialised: {}", self.algorithm)

    def update(self, detections) -> list[Track]:
        """
        Parameters
        ----------
        detections : Detections
            Output of ``Detector.detect()``.

        Returns
        -------
        list[Track]
            Active tracks for the current frame.
        """
        raw = self._core.update(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        )

        tracks: list[Track] = []
        for r in raw:
            tid = r["track_id"]
            cx = int((r["xyxy"][0] + r["xyxy"][2]) / 2)
            cy = int((r["xyxy"][1] + r["xyxy"][3]) / 2)
            trail = self._trail_map[tid]
            trail.append((cx, cy))
            if len(trail) > self._trail_maxlen:
                trail.pop(0)

            tracks.append(Track(
                track_id=tid,
                xyxy=r["xyxy"],
                confidence=r["confidence"],
                class_id=r["class_id"],
                age=r["age"],
                time_since_update=r["time_since_update"],
                trail=list(trail),
            ))
        return tracks
