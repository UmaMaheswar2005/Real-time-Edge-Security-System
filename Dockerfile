# ════════════════════════════════════════════════════════════════════════════════
# Stage 1 — YOLO model exporter
# Has torch + ultralytics only to export yolov8n.pt → yolov8n.onnx.
# This entire stage is DISCARDED after build; torch never enters the runtime image.
# ════════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS model-builder

WORKDIR /export

RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        ultralytics

# Download yolov8n.pt and export to ONNX (opset 12, simplified graph)
RUN python - <<'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)
print("ONNX export complete →", __import__("os").listdir("."))
EOF


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2 — Production runtime
# No torch, no TensorFlow, no ultralytics.
# Estimated peak RAM ≈ 470 MB — fits Render / Koyeb 512 MB free tier.
# ════════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/root

WORKDIR /app

# Minimal C libs needed by OpenCV headless + InsightFace ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no torch, no TF)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-bake InsightFace buffalo_sc models (~67 MB) into the image layer ──────
# This means cold-starts on free-tier hosts never stall waiting for a download.
RUN python - <<'EOF'
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(320, 320))
import os, glob
model_dir = os.path.expanduser("~/.insightface/models/buffalo_sc")
files = glob.glob(model_dir + "/*")
print(f"InsightFace buffalo_sc ready: {[os.path.basename(f) for f in files]}")
EOF

# ── Copy the ONNX model from the builder stage (~6 MB) ───────────────────────
COPY --from=model-builder /export/yolov8n.onnx /app/backend/yolov8n.onnx

# Copy application source (dataset/, frontend/ are included via COPY)
COPY . .

EXPOSE 8000

# Single worker — keeps RAM inside free-tier limit
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
