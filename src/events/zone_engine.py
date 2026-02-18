"""
Polygon-zone intrusion & dwell-time event engine.

Events are emitted when:
  1. A tracked object's centre enters a defined polygon zone.
  2. The object remains ("dwells") in the zone longer than `dwell_time_sec`.
  3. Cooldown prevents duplicate alerts for the same track + zone pair.

All events are serialised as JSON-lines to a log file and optionally forwarded
to a webhook / MQTT endpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------
@dataclass
class ZoneEvent:
    """Immutable event record written to the alert log."""
    timestamp_utc: str
    event_type: str               # intrusion | dwell | crossing
    zone_name: str
    track_id: int
    class_id: int
    class_name: str
    dwell_time_sec: float
    bbox_xyxy: list[float]
    centroid: list[int]
    frame_id: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ---------------------------------------------------------------------------
# Zone definition
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    name: str
    polygon: np.ndarray           # (K, 2) int32
    trigger: str                  # intrusion | crossing
    dwell_time_sec: float = 2.0
    cooldown_sec: float = 10.0
    direction: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ZoneEventEngine:
    """Evaluate tracks against polygon zones and emit events."""

    def __init__(self, zone_configs: list[dict], log_path: str = "logs/events.jsonl") -> None:
        self.zones = [self._parse_zone(z) for z in zone_configs]
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Internal state: track_id → {zone_name → first_seen_time}
        self._occupancy: dict[int, dict[str, float]] = {}
        # Cooldown ledger: (track_id, zone_name) → last_alert_time
        self._cooldown: dict[tuple[int, str], float] = {}

        logger.info("ZoneEventEngine loaded {} zones.", len(self.zones))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, tracks: Sequence, frame_id: int) -> list[ZoneEvent]:
        """Check all tracks against all zones.  Returns new events."""
        now = time.time()
        events: list[ZoneEvent] = []

        active_ids: set[int] = set()
        for track in tracks:
            active_ids.add(track.track_id)
            cx = int((track.xyxy[0] + track.xyxy[2]) / 2)
            cy = int((track.xyxy[1] + track.xyxy[3]) / 2)

            for zone in self.zones:
                inside = cv2.pointPolygonTest(zone.polygon, (cx, cy), False) >= 0

                if inside:
                    occ = self._occupancy.setdefault(track.track_id, {})
                    if zone.name not in occ:
                        occ[zone.name] = now

                    dwell = now - occ[zone.name]

                    if dwell >= zone.dwell_time_sec:
                        key = (track.track_id, zone.name)
                        last_alert = self._cooldown.get(key, 0.0)
                        if now - last_alert >= zone.cooldown_sec:
                            evt = ZoneEvent(
                                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                event_type=zone.trigger,
                                zone_name=zone.name,
                                track_id=track.track_id,
                                class_id=int(track.class_id),
                                class_name=getattr(track, "class_name", ""),
                                dwell_time_sec=round(dwell, 2),
                                bbox_xyxy=[float(v) for v in track.xyxy],
                                centroid=[cx, cy],
                                frame_id=frame_id,
                            )
                            events.append(evt)
                            self._cooldown[key] = now
                            self._write(evt)
                else:
                    # Left the zone — reset dwell timer
                    if track.track_id in self._occupancy:
                        self._occupancy[track.track_id].pop(zone.name, None)

        # Purge stale entries for tracks no longer active
        stale_ids = set(self._occupancy.keys()) - active_ids
        for sid in stale_ids:
            del self._occupancy[sid]

        return events

    def get_zone_polygons(self) -> list[tuple[str, np.ndarray]]:
        """For visualization overlay."""
        return [(z.name, z.polygon) for z in self.zones]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_zone(cfg: dict) -> Zone:
        pts = np.array(cfg["polygon"], dtype=np.int32)
        return Zone(
            name=cfg["name"],
            polygon=pts,
            trigger=cfg.get("trigger", "intrusion"),
            dwell_time_sec=cfg.get("dwell_time_sec", 2.0),
            cooldown_sec=cfg.get("cooldown_sec", 10.0),
            direction=cfg.get("direction"),
        )

    def _write(self, evt: ZoneEvent) -> None:
        with open(self.log_path, "a") as f:
            f.write(evt.to_json() + "\n")
        logger.info("EVENT | {} | zone={} track={} dwell={:.1f}s",
                     evt.event_type, evt.zone_name, evt.track_id, evt.dwell_time_sec)
