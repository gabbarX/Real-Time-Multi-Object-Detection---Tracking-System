#!/usr/bin/env python3
"""
Training helper — wraps Ultralytics YOLOv8 CLI for convenience.

Usage
─────
    python tools/train.py                         # uses config/training.yaml
    python tools/train.py --epochs 200 --batch 8  # override
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.option("--cfg", default="config/training.yaml", help="Training YAML config.")
@click.option("--epochs", default=None, type=int, help="Override epoch count.")
@click.option("--batch", default=None, type=int, help="Override batch size.")
@click.option("--imgsz", default=None, type=int, help="Override image size.")
@click.option("--device", default="0", help="CUDA device.")
def train(cfg: str, epochs: int | None, batch: int | None, imgsz: int | None, device: str) -> None:
    """Launch YOLOv8 training."""
    cmd = [sys.executable, "-m", "ultralytics", "detect", "train", f"cfg={cfg}", f"device={device}"]
    if epochs:
        cmd.append(f"epochs={epochs}")
    if batch:
        cmd.append(f"batch={batch}")
    if imgsz:
        cmd.append(f"imgsz={imgsz}")

    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train()
