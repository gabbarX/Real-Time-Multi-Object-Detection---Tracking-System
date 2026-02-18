#!/usr/bin/env python3
"""
RTMODT — One-Click Launcher
============================
Run this single file to set up and start the entire project:

    python start.py

It will:
  1. Install all Python dependencies (if missing)
  2. Download sample images for the web demo
  3. Start the FastAPI web server on http://localhost:8000
"""

import subprocess
import sys
import os
import socket
from pathlib import Path

# ── Configuration ──
PORT = int(os.environ.get("PORT", 8000))
HOST = "0.0.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent


def free_port(port: int) -> None:
    """Kill any process currently occupying the given port (Windows)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                print(f"  ⚠️  Port {port} is in use by PID {pid} — killing…")
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True)
                print(f"  ✅ Freed port {port}.")
                return
    except Exception:
        pass  # non-Windows or netstat unavailable — skip


def run(cmd: list[str], desc: str, check: bool = True) -> int:
    """Run a command with a status message."""
    print(f"\n{'='*60}")
    print(f"  ⏳ {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if check and result.returncode != 0:
        print(f"\n  ❌ Failed: {desc}")
        sys.exit(result.returncode)
    return result.returncode


def check_module(module: str) -> bool:
    """Check if a Python module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██████  ████████ ███    ███  ██████  ██████  ████████║
    ║     ██   ██    ██    ████  ████ ██    ██ ██   ██    ██   ║
    ║     ██████     ██    ██ ████ ██ ██    ██ ██   ██    ██   ║
    ║     ██   ██    ██    ██  ██  ██ ██    ██ ██   ██    ██   ║
    ║     ██   ██    ██    ██      ██  ██████  ██████     ██   ║
    ║                                                          ║
    ║   Real-Time Multi-Object Detection & Tracking System     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # ── Step 1: Install dependencies ──
    missing = []
    for mod in ["fastapi", "uvicorn", "cv2", "numpy", "ultralytics", "click", "loguru"]:
        if not check_module(mod):
            missing.append(mod)

    if missing:
        print(f"  📦 Missing modules detected: {', '.join(missing)}")
        req_file = PROJECT_ROOT / "requirements.txt"
        run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            "Installing Python dependencies…",
        )
    else:
        print("  ✅ All core dependencies already installed.")

    # ── Step 2: Download sample images ──
    samples_dir = PROJECT_ROOT / "web" / "static" / "samples"
    sample_count = len(list(samples_dir.glob("*.jpg"))) if samples_dir.exists() else 0

    if sample_count < 3:
        run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "download_samples.py")],
            "Downloading sample images for the web demo…",
        )
    else:
        print(f"  ✅ {sample_count} sample images already present.")

    # ── Step 3: Free port & start the web server ──
    free_port(PORT)
    print(f"\n{'='*60}")
    print(f"  🚀 Starting RTMODT Web App")
    print(f"     http://localhost:{PORT}")
    print(f"{'='*60}\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        import uvicorn
        # Add project root to path so web.server can import properly
        sys.path.insert(0, str(PROJECT_ROOT))
        os.chdir(str(PROJECT_ROOT))
        uvicorn.run(
            "web.server:app",
            host=HOST,
            port=PORT,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\n  👋 RTMODT stopped. Goodbye!")


if __name__ == "__main__":
    main()
