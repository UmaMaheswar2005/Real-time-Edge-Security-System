import os
import io
import time
import cv2
import yaml
import urllib.request
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from ultralytics import YOLOWorld
from deepface import DeepFace
from google import genai
import cloudinary
import cloudinary.uploader

# 1. Load Local Environment Variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

app = FastAPI(title="Edge Security API")

# 2. Allow CORS for Localhost & Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Paths & Load Local Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ADMIN_FOLDER = "Mahi_admin"

# --- Cloud DVR & State Smoothing Timers ---
last_periodic_snapshot = 0
last_human_alert = 0
SNAPSHOT_INTERVAL = 10  # Seconds between routine syncs
ALERT_COOLDOWN = 15     # Seconds between unknown human alert photos

# Identity Smoothing Memory (Prevents flickering loops)
last_confirmed_identity = "Unknown"
last_match_time = 0
IDENTITY_GRACE_PERIOD = 3.0  # Hold admin status for 3 seconds to ensure zero flapping

# Dynamically download the official Objects365 public class list
url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/Objects365.yaml"
data = yaml.safe_load(urllib.request.urlopen(url))
public_classes = list(data['names'].values())

# Load YOLO Nano with a strict confidence floor to eliminate wall hallucinations
yolo_model = YOLOWorld('yolov8s-world.pt')
yolo_model.set_classes(public_classes)

# Initialize Gemini Client
API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=API_KEY) if API_KEY else None

# --- MODERN GEMINI MODEL LIST ---
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]


@app.get("/")
def health_check():
    return {
        "status": "Security API Active",
        "dataset_exists": os.path.exists(DATASET_DIR),
        "gemini_ready": gemini_client is not None
    }


@app.post("/api/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    global last_periodic_snapshot, last_human_alert, last_confirmed_identity, last_match_time
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image payload"}

    identity = "Unknown"
    objects = []
    current_time = time.time()
    unknown_human_detected = False
    person_detected_in_frame = False

    # --- Step 1: YOLO Object & Human Detection (Strict 0.50 Confidence) ---
    try:
        results = yolo_model(frame, verbose=False, conf=0.50)
        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])].title()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if label.lower() == "person":
                    person_detected_in_frame = True
                    display_label = "Unknown Human"
                    unknown_human_detected = True
                else:
                    display_label = label

                objects.append({
                    "label": display_label,
                    "box": [x1, y1, x2 - x1, y2 - y1]
                })
    except Exception as e:
        print(f"YOLO Scanning Error: {e}")

    # --- Step 2: DeepFace Identity Detection & Memory Smoothing ---
    matched_admin = False
    if person_detected_in_frame:
        try:
            if os.path.exists(DATASET_DIR):
                dfs = DeepFace.find(
                    img_path=frame, 
                    db_path=DATASET_DIR, 
                    model_name="VGG-Face", 
                    enforce_detection=False, 
                    silent=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    distance = dfs[0].iloc[0]['distance']
                    match_path = dfs[0].iloc[0]['identity']
                    matched_folder = os.path.basename(os.path.dirname(match_path))
                    
                    # Relaxed distance slightly to 0.38 for reliable capture, backed by grace memory
                    if matched_folder == ADMIN_FOLDER and distance < 0.38:
                        matched_admin = True
                        last_confirmed_identity = ADMIN_FOLDER
                        last_match_time = current_time
        except Exception as e:
            print(f"DeepFace Scanning Error: {e}")

    # Apply Grace Period Memory to eliminate flickering loops
    if matched_admin:
        identity = ADMIN_FOLDER
        unknown_human_detected = False
    elif (current_time - last_match_time) < IDENTITY_GRACE_PERIOD and last_confirmed_identity == ADMIN_FOLDER:
        identity = ADMIN_FOLDER
        unknown_human_detected = False
    else:
        identity = "Unknown"

    # Synchronize Object Labels with Final Identity Decision
    if identity == ADMIN_FOLDER:
        for obj in objects:
            if "Human" in obj["label"] or "Person" in obj["label"]:
                obj["label"] = f"Admin: {ADMIN_FOLDER}"
    else:
        for obj in objects:
            if "Human" in obj["label"] or "Person" in obj["label"]:
                obj["label"] = "Unknown Human"

    # --- Step 3: Cloudinary Surveillance DVR Uploads ---
    
    # 3A. Routine Snapshot every 10 seconds
    if current_time - last_periodic_snapshot >= SNAPSHOT_INTERVAL:
        timestamp = int(current_time)
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
            try:
                cloudinary.uploader.upload(
                    buffer.tobytes(), 
                    folder="surveillance_logs", 
                    public_id=f"routine_sync_{timestamp}"
                )
            except Exception as e:
                print(f"Cloudinary Routine Upload Failed: {e}")
        last_periodic_snapshot = current_time

    # 3B. Immediate Alert Snapshot when an Unknown Human appears
    if unknown_human_detected and (current_time - last_human_alert >= ALERT_COOLDOWN):
        timestamp = int(current_time)
        alert_frame = frame.copy()
        cv2.rectangle(alert_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        cv2.putText(alert_frame, "ALERT: UNKNOWN HUMAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        success, buffer = cv2.imencode('.jpg', alert_frame)
        if success:
            try:
                cloudinary.uploader.upload(
                    buffer.tobytes(), 
                    folder="surveillance_alerts", 
                    public_id=f"ALERT_HUMAN_{timestamp}"
                )
            except Exception as e:
                print(f"Cloudinary Alert Upload Failed: {e}")
        last_human_alert = current_time

    return {
        "identity": identity,
        "objects": objects
    }


@app.post("/api/gemini")
async def ask_gemini(file: UploadFile = File(...)):
    if not gemini_client:
        return {"response": "Gemini API Key is not configured."}
    
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    last_error = None

    for model_name in GEMINI_MODELS:
        try:
            print(f"[Gemini] Attempting analysis using: {model_name}")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=["Describe this security frame concisely. Mention key objects or suspicious activity.", pil_img]
            )
            
            if response and response.text:
                return {
                    "response": response.text,
                    "model_used": model_name
                }
        except Exception as e:
            print(f"[Gemini Warning] Model '{model_name}' failed: {e}. Trying fallback...")
            last_error = e
            continue

    return {"response": f"All Gemini models exhausted. Error: {str(last_error)}"}