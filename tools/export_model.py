#!/usr/bin/env python3
"""
Export a trained YOLOv8 model to ONNX / TensorRT engine.

Usage
─────
    python tools/export_model.py --weights runs/train/rtmodt_v1/weights/best.pt --format engine
    python tools/export_model.py --weights best.pt --format onnx --half
"""

from __future__ import annotations

import click
from ultralytics import YOLO


@click.command()
@click.option("--weights", "-w", required=True, help="Path to .pt checkpoint.")
@click.option("--format", "-f", "fmt", default="engine", type=click.Choice(["onnx", "engine"]), help="Export format.")
@click.option("--imgsz", default=640, type=int, help="Input image size.")
@click.option("--half/--no-half", default=True, help="FP16 quantization.")
@click.option("--int8/--no-int8", default=False, help="INT8 quantization (requires calibration data).")
@click.option("--batch", default=1, type=int, help="Batch size baked into the engine.")
@click.option("--device", default="0", help="CUDA device.")
def export_model(weights: str, fmt: str, imgsz: int, half: bool, int8: bool, batch: int, device: str) -> None:
    """Export YOLOv8 model."""
    model = YOLO(weights)
    model.export(
        format=fmt,
        imgsz=imgsz,
        half=half,
        int8=int8,
        batch=batch,
        device=device,
    )
    click.echo("✅  Export complete.")


if __name__ == "__main__":
    export_model()
