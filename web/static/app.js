/* ===================================================================
   RTMODT — Frontend Application Logic
   Handles: mode switching, sample detection, image upload, webcam,
            canvas rendering with bounding boxes + labels + trails.
   =================================================================== */

// ── Colour palette for object classes ──
const COLORS = [
  '#6366f1', '#22d3ee', '#f472b6', '#34d399', '#fbbf24',
  '#f87171', '#a78bfa', '#fb923c', '#38bdf8', '#4ade80',
  '#e879f9', '#facc15', '#2dd4bf', '#818cf8', '#f97316',
];

function classColor(classId) { return COLORS[classId % COLORS.length]; }

// ── State ──
let currentMode = 'samples';
let webcamStream = null;
let webcamTimer = null;
let loadedImage = null;

// =========================================================================
// Mode Switching
// =========================================================================
function switchMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  document.getElementById('panelSamples').classList.toggle('hidden', mode !== 'samples');
  document.getElementById('panelUpload').classList.toggle('hidden', mode !== 'upload');
  document.getElementById('panelWebcam').classList.toggle('hidden', mode !== 'webcam');

  // Stop webcam if leaving
  if (mode !== 'webcam' && webcamStream) stopWebcam();
}

// =========================================================================
// Sample Images
// =========================================================================
async function loadSamples() {
  const grid = document.getElementById('sampleGrid');
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    if (!data.samples.length) {
      grid.innerHTML = `
        <div class="sample-grid__loading">
          No sample images found.<br>
          Run <code>python tools/download_samples.py</code> to download demo images.
        </div>`;
      return;
    }
    grid.innerHTML = data.samples.map(s => `
      <div class="sample-card" onclick="detectSample('${s.filename}', '${s.url}')">
        <img src="${s.url}" alt="${s.name}" loading="lazy" />
        <div class="sample-card__label">${s.name}</div>
      </div>`).join('');
  } catch (e) {
    grid.innerHTML = `<div class="sample-grid__loading">Could not load samples. Is the server running?</div>`;
  }
}

async function detectSample(filename, imgUrl) {
  showLoader();
  try {
    const res = await fetch(`/api/detect/sample/${filename}`);
    const data = await res.json();
    await drawResults(imgUrl, data);
    updateStats(data);
    toast(`Detected ${data.num_objects} objects in ${data.inference_ms}ms`, 'success');
  } catch (e) {
    toast('Detection failed: ' + e.message, 'error');
  } finally {
    hideLoader();
  }
}

// =========================================================================
// Image / Video Upload
// =========================================================================
function setupUpload() {
  const zone = document.getElementById('dropZone');
  const input = document.getElementById('fileInput');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });
}

async function handleFile(file) {
  if (file.type.startsWith('video/')) {
    handleVideo(file);
    return;
  }
  showLoader();
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/detect/image', { method: 'POST', body: formData });
    const data = await res.json();
    const url = URL.createObjectURL(file);
    await drawResults(url, data);
    updateStats(data);
    toast(`Detected ${data.num_objects} objects in ${data.inference_ms}ms`, 'success');
  } catch (e) {
    toast('Detection failed: ' + e.message, 'error');
  } finally {
    hideLoader();
  }
}

function handleVideo(file) {
  toast('Video mode: capturing first frame for detection…', 'success');
  const video = document.createElement('video');
  video.muted = true;
  video.src = URL.createObjectURL(file);
  video.onloadeddata = () => {
    video.currentTime = 0.5;
  };
  video.onseeked = async () => {
    const c = document.createElement('canvas');
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    c.toBlob(async blob => {
      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      showLoader();
      try {
        const res = await fetch('/api/detect/image', { method: 'POST', body: form });
        const data = await res.json();
        const url = c.toDataURL('image/jpeg');
        await drawResults(url, data);
        updateStats(data);
        toast(`Detected ${data.num_objects} objects in ${data.inference_ms}ms`, 'success');
      } catch (e) {
        toast('Detection failed: ' + e.message, 'error');
      } finally {
        hideLoader();
      }
    }, 'image/jpeg', 0.9);
  };
}

// =========================================================================
// Webcam
// =========================================================================
async function toggleWebcam() {
  if (webcamStream) {
    stopWebcam();
  } else {
    startWebcam();
  }
}

async function startWebcam() {
  const btn = document.getElementById('btnStartCam');
  const video = document.getElementById('webcamVideo');
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'environment' } });
    video.srcObject = webcamStream;
    video.hidden = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="1"/></svg> Stop Camera`;
    btn.className = 'btn btn--danger';

    // Start periodic detection
    const interval = parseInt(document.getElementById('webcamInterval').value, 10);
    webcamTimer = setInterval(() => captureAndDetect(video), interval);
    toast('Webcam started — detecting objects…', 'success');
  } catch (e) {
    toast('Could not access webcam: ' + e.message, 'error');
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  if (webcamTimer) { clearInterval(webcamTimer); webcamTimer = null; }
  const btn = document.getElementById('btnStartCam');
  const video = document.getElementById('webcamVideo');
  video.hidden = true;
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg> Start Camera`;
  btn.className = 'btn btn--primary';
}

async function captureAndDetect(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640;
  c.height = video.videoHeight || 480;
  c.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = c.toDataURL('image/jpeg', 0.8);

  try {
    const res = await fetch('/api/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    await drawResults(dataUrl, data);
    updateStats(data);
  } catch (e) {
    // Silently skip on network issues during live feed
  }
}

// =========================================================================
// Canvas Rendering — Draw bounding boxes + labels + trails
// =========================================================================
async function drawResults(imgSrc, data) {
  const section = document.getElementById('resultSection');
  const canvas = document.getElementById('resultCanvas');
  const ctx = canvas.getContext('2d');

  const img = await loadImage(imgSrc);
  canvas.width = img.width;
  canvas.height = img.height;

  // Draw image
  ctx.drawImage(img, 0, 0);

  // Draw detections
  const items = data.tracks && data.tracks.length ? data.tracks : data.detections;
  items.forEach((det, i) => {
    const [x1, y1, x2, y2] = det.bbox;
    const color = classColor(det.class_id);
    const w = x2 - x1, h = y2 - y1;

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, w, h);

    // Semi-transparent fill
    ctx.fillStyle = color + '18';
    ctx.fillRect(x1, y1, w, h);

    // Label background
    const label = det.track_id != null
      ? `ID:${det.track_id} ${det.class_name || det.class_id} ${(det.confidence * 100).toFixed(0)}%`
      : `${det.class_name || det.class_id} ${(det.confidence * 100).toFixed(0)}%`;

    ctx.font = 'bold 14px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 22, tw + 12, 22);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 6, y1 - 6);

    // Draw trail if available
    if (det.trail && det.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(det.trail[0][0], det.trail[0][1]);
      for (let j = 1; j < det.trail.length; j++) {
        ctx.lineTo(det.trail[j][0], det.trail[j][1]);
      }
      ctx.strokeStyle = color + 'aa';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });

  // Show result section
  section.style.display = 'block';
  document.getElementById('resultBadge').textContent = `${items.length} object${items.length !== 1 ? 's' : ''} · ${data.inference_ms}ms`;

  // Build detections list
  const list = document.getElementById('detectionsList');
  list.innerHTML = items.map((det, i) => `
    <div class="det-item" style="animation-delay:${i * 40}ms">
      <div class="det-color" style="background:${classColor(det.class_id)}"></div>
      <span class="det-class">${det.class_name || 'class_' + det.class_id}${det.track_id != null ? ' #' + det.track_id : ''}</span>
      <span class="det-conf">${(det.confidence * 100).toFixed(1)}%</span>
      <span class="det-bbox">[${det.bbox.map(v => Math.round(v)).join(', ')}]</span>
    </div>`).join('');
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// =========================================================================
// UI Helpers
// =========================================================================
function updateStats(data) {
  document.getElementById('statObjects').textContent = data.num_objects;
  document.getElementById('statLatency').textContent = data.inference_ms + 'ms';
  const fps = data.inference_ms > 0 ? Math.round(1000 / data.inference_ms) : '—';
  document.getElementById('statFPS').textContent = fps;
}

function showLoader() { document.getElementById('loader').style.display = 'flex'; }
function hideLoader() { document.getElementById('loader').style.display = 'none'; }

function toast(msg, type = 'success') {
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  setupUpload();
});
