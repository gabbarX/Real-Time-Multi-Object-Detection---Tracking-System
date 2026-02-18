# Deploying RTMODT to Hugging Face Spaces

This guide walks you through deploying the RTMODT web app to the internet for free using **Hugging Face Spaces**.

---

## Prerequisites

- A free [Hugging Face](https://huggingface.co) account
- Git installed on your machine
- [Git LFS](https://git-lfs.com/) installed (for model weights)

---

## Step-by-Step Deployment

### 1. Create a Hugging Face Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and fill in:

| Field | Value |
|-------|-------|
| **Space name** | `rtmodt` |
| **SDK** | `Docker` |
| **Hardware** | `CPU Basic` (free) or `T4 GPU` (for faster inference) |
| **Visibility** | Public |

Click **Create Space**.

### 2. Push Your Code

```bash
# Navigate to your project
cd "d:\Project\Real-Time Multi-Object Detection & Tracking System"

# Initialize Git LFS (for large files)
git lfs install

# Add the HF Spaces remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/rtmodt

# Commit all files
git add -A
git commit -m "Deploy RTMODT to HF Spaces"

# Push to Hugging Face
git push space main
```

> **Replace `YOUR_USERNAME`** with your actual Hugging Face username.

### 3. Wait for Build

- HF Spaces will automatically build your Docker image
- The build takes **3-5 minutes** on first deploy
- Watch the build logs at `https://huggingface.co/spaces/YOUR_USERNAME/rtmodt`

### 4. Access Your App

Once built, your app is live at:

```
https://YOUR_USERNAME-rtmodt.hf.space
```

---

## What Happens During Build

The `Dockerfile` automatically:

1. Installs Python 3.11 + system dependencies
2. Installs all pip packages from `requirements.txt`
3. Downloads sample images via `tools/download_samples.py`
4. Pre-downloads YOLOv8s model weights (~22 MB)
5. Creates a non-root user (HF requirement)
6. Launches `start.py` on port 7860

---

## Updating Your Deployment

After making changes locally:

```bash
git add -A
git commit -m "Update: description of changes"
git push space main
```

HF Spaces rebuilds automatically on every push.

---

## Hardware Options

| Hardware | Cost | Inference Speed | Best For |
|----------|------|----------------|----------|
| **CPU Basic** | Free | ~200-500ms/image | Demos, testing |
| **CPU Upgrade** | $0.03/hr | ~150-300ms/image | Better demos |
| **T4 GPU** | $0.06/hr | ~12-20ms/image | Real-time, production |
| **A10G GPU** | $0.30/hr | ~5-10ms/image | High performance |

> Start with **CPU Basic** (free). Upgrade to T4 if you need real-time webcam detection.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails on pip install | Check `requirements.txt` for OS-specific packages |
| App shows "Building" forever | Check build logs for errors in the Space settings |
| Model download fails | The Dockerfile pre-downloads. Check network access |
| Port error | Ensure `PORT=7860` is set (already configured in Dockerfile) |

---

## Alternative: Docker on Any Server

If you prefer your own server instead of HF Spaces:

```bash
# Build the image
docker build -t rtmodt .

# Run it (port 8000 for local, 7860 for HF)
docker run -p 8000:7860 -e PORT=7860 rtmodt

# With GPU support
docker run --gpus all -p 8000:7860 -e PORT=7860 rtmodt
```
