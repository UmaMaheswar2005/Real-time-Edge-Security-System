# ── Stage 1: Export YOLOv8n to ONNX (torch discarded after this stage) ────────
FROM python:3.12-slim AS model-builder

WORKDIR /export

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir ultralytics

COPY export_onnx.py .

RUN python export_onnx.py

# ── Stage 2: Lean runtime (no torch, no TF, ~470 MB RAM) ─────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/root

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY prebake_insightface.py .

RUN python prebake_insightface.py

COPY --from=model-builder /export/yolov8n.onnx /app/yolov8n.onnx

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]