"""Smoke tests for the RTMODT web API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Insert project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.server import app

client = TestClient(app)


class TestHealthAndStatic:
    def test_index_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "RTMODT" in r.text

    def test_samples_endpoint(self):
        r = client.get("/api/samples")
        assert r.status_code == 200
        data = r.json()
        assert "samples" in data
        assert isinstance(data["samples"], list)


class TestDetectionAPI:
    def test_detect_sample_if_exists(self):
        samples = client.get("/api/samples").json()["samples"]
        if samples:
            r = client.get(f"/api/detect/sample/{samples[0]['filename']}")
            assert r.status_code == 200
            data = r.json()
            assert "detections" in data
            assert "inference_ms" in data
            assert "num_objects" in data
            assert isinstance(data["detections"], list)

    def test_detect_sample_not_found(self):
        r = client.get("/api/detect/sample/nonexistent.jpg")
        assert r.status_code == 404

    def test_detect_image_upload(self):
        # Create a tiny valid JPEG in memory
        import numpy as np
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        r = client.post("/api/detect/image", files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert "detections" in data

    def test_detect_frame_base64(self):
        import base64
        import numpy as np
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        r = client.post("/api/detect/frame", json={"image": f"data:image/jpeg;base64,{b64}"})
        assert r.status_code == 200
        data = r.json()
        assert "detections" in data
