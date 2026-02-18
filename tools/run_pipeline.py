#!/usr/bin/env python3
"""
CLI entry-point — Real-Time Multi-Object Detection & Tracking pipeline.

Usage
─────
    python tools/run_pipeline.py --config config/default.yaml
    python tools/run_pipeline.py --config config/default.yaml --source rtsp://...
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2
import yaml
from loguru import logger

# ── project root on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.rtsp_reader import RTSPReader
from src.detection.detector import Detector
from src.tracking.tracker import MultiObjectTracker
from src.events.zone_engine import ZoneEventEngine
from src.profiling.latency_profiler import LatencyProfiler
from src.visualization.renderer import FrameRenderer


def _setup_logging(cfg: dict) -> None:
    log_dir = Path(cfg.get("system", {}).get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=cfg["system"].get("log_level", "INFO"))
    logger.add(log_dir / "pipeline.log", rotation="50 MB", retention="7 days")


@click.command()
@click.option("--config", "-c", default="config/default.yaml", help="Path to YAML config.")
@click.option("--source", "-s", default=None, help="Override ingestion source (RTSP URL or video file).")
@click.option("--display/--no-display", default=True, help="Show live window.")
def main(config: str, source: str | None, display: bool) -> None:
    """Run the real-time detection + tracking pipeline."""
    cfg = yaml.safe_load(Path(config).read_text())
    _setup_logging(cfg)
    logger.info("Loaded config from {}", config)

    if source:
        cfg["ingestion"]["source"] = source

    # ── Initialise components ──
    reader = RTSPReader(
        source=cfg["ingestion"]["source"],
        backend=cfg["ingestion"].get("backend", "opencv"),
        buffer_size=cfg["ingestion"].get("buffer_size", 1),
        target_fps=cfg["ingestion"].get("target_fps", 30),
        reconnect_delay=cfg["ingestion"].get("reconnect_delay_sec", 3),
        max_reconnects=cfg["ingestion"].get("max_reconnect_attempts", 10),
        resolution=(
            cfg["ingestion"]["resolution"]["width"],
            cfg["ingestion"]["resolution"]["height"],
        ) if "resolution" in cfg["ingestion"] else None,
    )

    det_cfg = cfg["detection"]
    detector = Detector(
        model_path=det_cfg["model_path"],
        fallback_model=det_cfg.get("fallback_model"),
        input_size=tuple(det_cfg.get("input_size", [640, 640])),
        confidence=det_cfg.get("confidence_threshold", 0.35),
        iou=det_cfg.get("nms_iou_threshold", 0.45),
        classes=det_cfg.get("classes"),
        half=det_cfg.get("half", True),
        device=cfg["system"].get("device", "cuda:0"),
        max_det=det_cfg.get("max_detections", 100),
        agnostic_nms=det_cfg.get("agnostic_nms", False),
    )

    trk_cfg = cfg["tracking"]
    tracker = MultiObjectTracker(
        algorithm=trk_cfg.get("algorithm", "bytetrack"),
        **trk_cfg.get(trk_cfg.get("algorithm", "bytetrack"), {}),
    )

    evt_cfg = cfg.get("events", {})
    event_engine = None
    if evt_cfg.get("enabled", False):
        event_engine = ZoneEventEngine(
            zone_configs=evt_cfg.get("zones", []),
            log_path=evt_cfg.get("alert", {}).get("log_path", "logs/events.jsonl"),
        )

    prof_cfg = cfg.get("profiling", {})
    profiler = LatencyProfiler(
        gpu_sync=prof_cfg.get("gpu_sync", True),
        warmup_frames=prof_cfg.get("warmup_frames", 50),
        log_interval=prof_cfg.get("log_interval_frames", 100),
    )

    vis_cfg = cfg.get("visualization", {})
    renderer = FrameRenderer(
        show_boxes=vis_cfg.get("show_boxes", True),
        show_ids=vis_cfg.get("show_ids", True),
        show_trails=vis_cfg.get("show_trails", True),
        trail_length=vis_cfg.get("trail_length", 30),
        show_zones=vis_cfg.get("show_zones", True),
        show_fps=vis_cfg.get("show_fps", True),
    )

    video_writer = None
    if vis_cfg.get("save_video", False):
        fourcc = cv2.VideoWriter_fourcc(*vis_cfg.get("codec", "mp4v"))
        save_path = vis_cfg.get("save_path", "output/recording.mp4")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(save_path, fourcc, 25, (1920, 1080))

    # ── Main loop ──
    logger.info("Starting pipeline …")
    with reader:
        try:
            while True:
                profiler.tick("decode")
                ok, frame, fid = reader.read()
                profiler.tock("decode")

                if not ok or frame is None:
                    continue

                # Detection
                profiler.tick("inference")
                detections = detector.detect(frame)
                profiler.tock("inference")

                # Tracking
                profiler.tick("tracking")
                tracks = tracker.update(detections)
                profiler.tock("tracking")

                # Events
                profiler.tick("events")
                events = []
                if event_engine:
                    events = event_engine.process(tracks, fid)
                profiler.tock("events")

                # Visualization
                profiler.tick("visualization")
                zones = event_engine.get_zone_polygons() if event_engine else None
                annotated = renderer.render(
                    frame, tracks, zones=zones,
                    fps=profiler.current_fps,
                    latency_ms=sum(profiler._frame_times.values()) if profiler._frame_times else 0,
                )
                profiler.tock("visualization")

                profiler.end_frame()

                if video_writer:
                    video_writer.write(annotated)

                if display:
                    cv2.imshow(vis_cfg.get("window_name", "RTMODT"), annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Pipeline shutdown complete.")
    print("\n=== Profiling Summary ===")
    for k, v in profiler.summary().items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
