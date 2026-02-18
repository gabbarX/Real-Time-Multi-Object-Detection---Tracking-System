"""
Annotated-frame renderer for live display / video recording.

Draws:
  • Bounding boxes with class label + track ID
  • Centroid trails (polyline)
  • Polygon zone overlays (semi-transparent)
  • FPS / latency HUD
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Colour palette (BGR) — visually distinct for up to 20 IDs, then cycles
_PALETTE = [
    (0, 255, 127), (255, 144, 30), (0, 215, 255), (180, 105, 255),
    (71, 99, 255), (50, 205, 50), (0, 165, 255), (205, 92, 92),
    (238, 130, 238), (0, 255, 255), (30, 105, 210), (128, 0, 0),
    (0, 128, 128), (128, 128, 0), (255, 0, 255), (0, 0, 255),
    (255, 255, 0), (0, 128, 0), (128, 0, 128), (255, 165, 0),
]


class FrameRenderer:
    """Draw detection + tracking annotations on a frame."""

    def __init__(
        self,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_trails: bool = True,
        trail_length: int = 30,
        show_zones: bool = True,
        show_fps: bool = True,
    ) -> None:
        self.show_boxes = show_boxes
        self.show_ids = show_ids
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.show_zones = show_zones
        self.show_fps = show_fps

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence,
        zones: Optional[list[tuple[str, np.ndarray]]] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Return annotated frame (draws in-place for speed)."""
        # -- Zones --
        if self.show_zones and zones:
            overlay = frame.copy()
            for name, poly in zones:
                cv2.fillPoly(overlay, [poly], (0, 0, 180))
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, name, (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # -- Tracks --
        for t in tracks:
            colour = _PALETTE[t.track_id % len(_PALETTE)]
            x1, y1, x2, y2 = map(int, t.xyxy)

            if self.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            if self.show_ids:
                label = f"ID:{t.track_id} {getattr(t, 'class_name', '')} {t.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.show_trails and hasattr(t, "trail") and len(t.trail) > 1:
                pts = np.array(t.trail[-self.trail_length:], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, colour, 2)

        # -- HUD --
        if self.show_fps:
            hud = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame
