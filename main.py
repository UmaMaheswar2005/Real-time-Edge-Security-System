"""
Edge Security API v2
────────────────────────────────────────────────────────────────────────────────
Stack swap summary vs v1:
  DeepFace + TensorFlow + tf-keras  →  insightface (buffalo_sc, ONNX)
  YOLOWorld + torch + torchvision   →  YOLOv8n ONNX + onnxruntime (pure)
  CLIP                              →  Gemini Vision (already cloud, free)

RAM budget (Render / Koyeb 512 MB free tier):
  base python + fastapi  ~  80 MB
  onnxruntime            ~  60 MB
  insightface buffalo_sc ~ 130 MB
  yolov8n.onnx inference ~ 120 MB  (peak, sequential)
  numpy / cv2 / PIL      ~  80 MB
  ─────────────────────────────────
  peak                   ~ 470 MB  ✓ fits in 512 MB
────────────────────────────────────────────────────────────────────────────────
"""

import gc
import glob
import io
import os
import time
import urllib.request

import cv2
import numpy as np
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
import cloudinary
import cloudinary.uploader
import onnxruntime as ort

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.onnx")

# Seconds between routine Cloudinary snapshots
SNAPSHOT_INTERVAL = 10
# Minimum seconds between alert uploads (per-event type)
ALERT_COOLDOWN    = 15
# Hold a recognised identity for this many seconds after last match (anti-flicker)
IDENTITY_GRACE    = 3.0
# Cosine similarity floor for InsightFace buffalo_sc MobileFaceNet embeddings.
# Range: [-1, 1].  Same person typically > 0.30.  Tune up to reduce false positives.
FACE_THRESHOLD    = 0.40

# COCO 80-class list (index matches YOLOv8n output)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]

# Objects that immediately escalate threat level to HIGH
HIGH_THREAT_LABELS = {"knife", "scissors", "baseball bat"}

# Gemini fallback chain  ← v1 had completely wrong model names ("gemini-3.7-flash" etc)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

SECURITY_PROMPT = (
    "You are a professional AI security analyst reviewing a live CCTV frame. "
    "Respond in ≤ 80 words covering: "
    "① how many people are visible and their behaviour, "
    "② any suspicious objects or activities, "
    "③ overall threat rating — LOW, MEDIUM, or HIGH — with one sentence of justification."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mutable globals (models + per-request state)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

yolo_session: ort.InferenceSession = None
face_app = None
# {folder_name: [normed_embedding, ...]}  — built once at startup from dataset/
admin_db: dict[str, list[np.ndarray]] = {}

_last_snapshot   = 0.0
_last_alert      = 0.0
_last_identity   = "Unknown"
_last_match_time = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO — pure ONNX inference (zero PyTorch dependency)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _letterbox(frame: np.ndarray, size: int = 640):
    """
    Scale frame to size×size with letterbox padding (grey fill, top-left origin).
    Returns (padded_frame, scale_factor).
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """Greedy non-maximum suppression. Returns list of kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_thr]
    return keep


def yolo_detect(frame: np.ndarray, conf_thr: float = 0.50) -> list[dict]:
    """
    Run YOLOv8n ONNX and return detections as:
      [{"label": str, "box": [x,y,w,h], "conf": float, "threat": bool}, ...]
    Returns [] if model not loaded.
    """
    if yolo_session is None:
        return []

    orig_h, orig_w = frame.shape[:2]
    lb, scale = _letterbox(frame)

    # Build NCHW float32 blob normalised to [0, 1]
    blob = lb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]        # → [1, 3, 640, 640]

    input_name = yolo_session.get_inputs()[0].name
    raw = yolo_session.run(None, {input_name: blob})[0]  # → [1, 84, 8400]
    preds = raw[0].T                                      # → [8400, 84]

    cls_scores = preds[:, 4:]
    confidence = cls_scores.max(axis=1)
    class_ids  = cls_scores.argmax(axis=1)

    mask = confidence > conf_thr
    if not mask.any():
        return []

    p, sc, cids = preds[mask], confidence[mask], class_ids[mask]

    # Convert centre-form (cx,cy,w,h) → corner-form in original-image pixels
    cx = p[:, 0] / scale;  cy = p[:, 1] / scale
    bw = p[:, 2] / scale;  bh = p[:, 3] / scale
    x1 = np.clip(cx - bw / 2, 0, orig_w).astype(int)
    y1 = np.clip(cy - bh / 2, 0, orig_h).astype(int)
    x2 = np.clip(cx + bw / 2, 0, orig_w).astype(int)
    y2 = np.clip(cy + bh / 2, 0, orig_h).astype(int)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep  = _nms(boxes, sc)

    return [
        {
            "label":  COCO_CLASSES[cids[i]],
            "box":    [int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])],
            "conf":   float(sc[i]),
            "threat": COCO_CLASSES[cids[i]] in HIGH_THREAT_LABELS,
        }
        for i in keep
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Face recognition — InsightFace buffalo_sc (ONNX, zero TensorFlow)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def build_face_db() -> None:
    """
    Walk every sub-folder of DATASET_DIR.
    Folder name  =  identity name (e.g. "Mahi_admin").
    Compute L2-normalised MobileFaceNet embeddings for every .jpg/.png found.
    Upgrades v1: supports multiple identities, skips undetectable images gracefully.
    """
    global admin_db
    if not os.path.exists(DATASET_DIR) or face_app is None:
        print("[FaceDB] Dataset dir missing or FaceApp not ready — skipping.")
        return

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        embeddings = []
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.png"))

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                embeddings.append(_normalise(faces[0].embedding))

        if embeddings:
            admin_db[folder_name] = embeddings
            print(f"[FaceDB] '{folder_name}': {len(embeddings)}/{len(images)} embeddings built.")
        else:
            print(f"[FaceDB] '{folder_name}': no faces detected — skipping identity.")


def identify_faces(frame: np.ndarray) -> list[tuple[list[int], str]]:
    """
    Detect all faces in frame and match each against admin_db.
    Returns [(bbox_xyxy, identity_name_or_Unknown), ...]
    Upgrade vs v1: handles multiple people in frame with per-face identity.
    """
    if face_app is None or not admin_db:
        return []

    faces = face_app.get(frame)
    results = []

    for face in faces:
        emb       = _normalise(face.embedding)
        best_name = "Unknown"
        best_sim  = -2.0  # cosine range is [-1, 1]

        for name, refs in admin_db.items():
            for ref_emb in refs:
                sim = float(np.dot(emb, ref_emb))  # dot of normed = cosine similarity
                if sim > best_sim:
                    best_sim = sim
                    if sim >= FACE_THRESHOLD:
                        best_name = name

        bbox = face.bbox.astype(int).tolist()   # [x1, y1, x2, y2]
        results.append((bbox, best_name))

    return results


def resolve_person_identity(person_box: list[int],
                            face_results: list[tuple[list[int], str]]) -> str:
    """
    Map a YOLO person box [x, y, w, h] to the identity of the face whose
    centre-point lies inside it. Returns 'Unknown' if no face matches.
    This enables correct per-person labelling when multiple people are in frame.
    """
    px, py, pw, ph = person_box
    px2, py2 = px + pw, py + ph

    for (fx1, fy1, fx2, fy2), identity in face_results:
        fcx = (fx1 + fx2) / 2
        fcy = (fy1 + fy2) / 2
        if px <= fcx <= px2 and py <= fcy <= py2:
            return identity
    return "Unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App lifespan — load models once, then free build-time RAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_session, face_app

    # ── YOLO ─────────────────────────────────────────────────────────────────
    # yolov8n.onnx is baked into the Docker image by the builder stage.
    # On bare-metal / local dev, generate it once with:
    #   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
    # then move yolov8n.onnx into backend/.
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_session = ort.InferenceSession(
            YOLO_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        print("[YOLO] yolov8n.onnx loaded — object detection ready.")
    else:
        print("[YOLO] yolov8n.onnx not found — object detection DISABLED.")
        print("[YOLO] Generate it: python -c \"from ultralytics import YOLO; "
              "YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True)\"")

    # ── InsightFace ───────────────────────────────────────────────────────────
    # Models (~67 MB) are pre-baked into the Docker image during build.
    # On first local run they auto-download to ~/.insightface/models/buffalo_sc/.
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(320, 320))
        print("[FaceApp] InsightFace buffalo_sc loaded — face recognition ready.")
        build_face_db()
    except Exception as exc:
        print(f"[FaceApp] Load failed: {exc} — face recognition DISABLED.")

    gc.collect()
    print("[App] Startup complete. Ready to serve requests.")

    yield  # ← application runs here

    print("[App] Shutting down cleanly.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

_api_key      = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=_api_key) if _api_key else None

app = FastAPI(title="Edge Security API v2", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Lock down to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health / status ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":           "Edge Security API v2 Active",
        "yolo_loaded":      yolo_session is not None,
        "face_app_loaded":  face_app is not None,
        "known_identities": list(admin_db.keys()),
        "gemini_ready":     gemini_client is not None,
    }


@app.get("/health")
def health():
    """
    Lightweight keep-alive endpoint.
    Vercel frontend pings this every 10 minutes to prevent Render/Koyeb
    free-tier spin-down.
    """
    return {"status": "ok"}


# ── Main analysis endpoint ─────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    global _last_snapshot, _last_alert, _last_identity, _last_match_time

    # ── Decode frame ─────────────────────────────────────────────────────────
    raw   = await file.read()
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image payload — could not decode frame."}

    now = time.time()

    # ── Step 1: YOLO detection ───────────────────────────────────────────────
    detections       = yolo_detect(frame, conf_thr=0.50)
    person_dets      = [d for d in detections if d["label"] == "person"]
    threat_dets      = [d for d in detections if d["threat"]]

    # ── Step 2: Face recognition ─────────────────────────────────────────────
    face_results  = []
    identity_name = "Unknown"
    any_known     = False

    if person_dets:
        face_results = identify_faces(frame)

        # Check for any recognised identity this frame
        for _, name in face_results:
            if name != "Unknown":
                _last_identity   = name
                _last_match_time = now
                any_known        = True

        # Grace-period: carry last known identity to prevent flicker
        if not any_known and (now - _last_match_time) < IDENTITY_GRACE:
            if _last_identity != "Unknown":
                # Inject virtual entry so person boxes get labelled correctly
                face_results = [([0, 0, 1, 1], _last_identity)] + face_results
                any_known    = True

        if any_known:
            identity_name = _last_identity
        else:
            _last_identity = "Unknown"

    # ── Step 3: Per-person labelling ─────────────────────────────────────────
    objects = []
    for d in detections:
        label = d["label"]
        if label == "person":
            pid   = resolve_person_identity(d["box"], face_results)
            label = f"Admin: {pid}" if pid != "Unknown" else "Unknown Human"
        objects.append({"label": label, "box": d["box"], "threat": d["threat"]})

    # ── Step 4: Threat scoring ────────────────────────────────────────────────
    unknown_intruder = any(o["label"] == "Unknown Human" for o in objects)
    if threat_dets:
        threat_level = "HIGH"
    elif unknown_intruder:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    # ── Step 5: Cloudinary DVR ────────────────────────────────────────────────
    # 5-A  Routine snapshot every SNAPSHOT_INTERVAL seconds
    if now - _last_snapshot >= SNAPSHOT_INTERVAL:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            try:
                cloudinary.uploader.upload(
                    buf.tobytes(),
                    folder="surveillance_logs",
                    public_id=f"routine_{int(now)}",
                )
            except Exception as exc:
                print(f"[CDN] Routine upload error: {exc}")
        _last_snapshot = now

    # 5-B  Alert snapshot for unknown intruder or weapon (with cooldown)
    if (unknown_intruder or threat_dets) and (now - _last_alert >= ALERT_COOLDOWN):
        af    = frame.copy()
        color = (0, 0, 255) if threat_dets else (0, 140, 255)  # red vs orange
        cv2.rectangle(af, (0, 0), (af.shape[1], af.shape[0]), color, 10)
        msg = "ALERT: WEAPON DETECTED" if threat_dets else "ALERT: UNKNOWN INTRUDER"
        cv2.putText(af, msg, (20, 56), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

        ok, buf = cv2.imencode(".jpg", af, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            try:
                folder = "weapon_alerts" if threat_dets else "surveillance_alerts"
                cloudinary.uploader.upload(
                    buf.tobytes(),
                    folder=folder,
                    public_id=f"ALERT_{int(now)}",
                )
            except Exception as exc:
                print(f"[CDN] Alert upload error: {exc}")
        _last_alert = now

    return {
        "identity":      identity_name,
        "objects":       objects,
        "threat_level":  threat_level,
        "threat_objects": [d["label"] for d in threat_dets],
    }


# ── Gemini analysis endpoint ───────────────────────────────────────────────────

@app.post("/api/gemini")
async def ask_gemini(file: UploadFile = File(...)):
    if not gemini_client:
        return {"response": "Gemini API key not configured.", "model_used": None}

    raw     = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            resp = gemini_client.models.generate_content(
                model=model_name,
                contents=[SECURITY_PROMPT, pil_img],
            )
            if resp and resp.text:
                return {"response": resp.text.strip(), "model_used": model_name}
        except Exception as exc:
            print(f"[Gemini] '{model_name}' failed: {exc}")
            last_err = exc

    return {
        "response":   f"All Gemini models unavailable. Last error: {last_err}",
        "model_used": None,
    }
