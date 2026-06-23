"""
app.py — AgeVision Backend v2.3 (Multi-Person Tracking)
Changes from v2.1:
- .predict() → .track() with ByteTrack for camera frames: each face gets a
  persistent track ID so the same person keeps their ID even as they move.
- Cooldown keyed by track_id (not slot index) — Person #1 and Person #2
  each have their own independent 5-second cooldown across frames.
- track_id returned to frontend so boxes are labelled "Person #1", "#2", etc.
- Upload/still images still use .predict() (no tracking needed).

Run: python app.py
Requires:
    pip install flask flask-cors torch torchvision pillow ultralytics huggingface_hub opencv-python-headless
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import io, os, csv, uuid, datetime, time

# ── Config ───────────────────────────────────────────────────────────────
AGE_MODEL_PATH      = 'best_model.pth'
CLASS_NAMES         = ['(0-20)', '(20-40)', '(40-100)']
IMG_SIZE            = 224
DEVICE              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FACE_CONF_THRESHOLD = 0.80   # Wajah harus terdeteksi YOLO dengan >= 80%
AGE_CONF_THRESHOLD  = 0.50   # Keyakinan umur cukup >= 50%
LOG_COOLDOWN_SEC    = 5.0

LOG_DIR   = 'logs'
FACES_DIR = os.path.join(LOG_DIR, 'faces')
LOG_CSV   = os.path.join(LOG_DIR, 'age_log.csv')

os.makedirs(FACES_DIR, exist_ok=True)

# ── App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Cooldown tracker (keyed by track_id for camera, slot for upload) ─────
_last_log_time: dict = {}

def can_log(key) -> bool:
    now = time.monotonic()
    if now - _last_log_time.get(key, 0) >= LOG_COOLDOWN_SEC:
        _last_log_time[key] = now
        return True
    return False

def cooldown_remaining(key) -> float:
    now = time.monotonic()
    return max(0.0, round(LOG_COOLDOWN_SEC - (now - _last_log_time.get(key, 0)), 1))

# ── CSV log ──────────────────────────────────────────────────────────────
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['id', 'timestamp', 'track_id', 'age_class',
                                 'age_confidence', 'face_confidence',
                                 'face_image_path', 'source'])

def append_log(row):
    with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def read_logs(limit=100):
    if not os.path.exists(LOG_CSV):
        return []
    with open(LOG_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[::-1][:limit]

def count_logs():
    if not os.path.exists(LOG_CSV):
        return 0
    with open(LOG_CSV, 'r', encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)

# ── Age Model ────────────────────────────────────────────────────────────
def load_age_model():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASS_NAMES))
    )
    checkpoint = torch.load(AGE_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    print(f'✅ Age classifier loaded | Device: {DEVICE}')
    return model

age_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

age_model = None
if os.path.exists(AGE_MODEL_PATH):
    age_model = load_age_model()
else:
    print(f'⚠️  Age model not found: {AGE_MODEL_PATH}')

# ── YOLOv8 Face Model ────────────────────────────────────────────────────
face_model = None
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    print('⬇️  Fetching YOLOv8 face weights (first run only)...')
    face_weights_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
    )
    face_model = YOLO(face_weights_path)
    print(f'✅ YOLOv8 face detector loaded | Device: {DEVICE}')
except Exception as e:
    print(f'⚠️  Could not load face model: {e}')

# ── Helpers ──────────────────────────────────────────────────────────────
def detect_faces_predict(pil_img):
    """For still images (upload) — no tracking needed."""
    if face_model is None:
        return []
    results = face_model.predict(pil_img, verbose=False,
                                  device=0 if DEVICE.type == 'cuda' else 'cpu')
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'track_id': i + 1,   # slot-based for stills
                'is_tracked': False,
            })
    return detections


def detect_faces_track(pil_img):
    """
    For camera frames — ByteTrack gives each face a persistent track_id
    so Person #1 stays #1 even as they move around between frames.
    """
    if face_model is None:
        return []
    try:
        results = face_model.track(
            pil_img,
            persist=True,             # keep tracker state between calls
            tracker="bytetrack.yaml", # ByteTrack (built-in to ultralytics)
            verbose=False,
            device=0 if DEVICE.type == 'cuda' else 'cpu',
        )
    except Exception:
        # Fallback to predict if tracking fails (e.g. first frame edge case)
        return detect_faces_predict(pil_img)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # box.id is None if tracker hasn't assigned an ID yet (first frame)
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'track_id': track_id,
                'is_tracked': track_id != -1,
            })
    return detections


def classify_age(face_crop_pil):
    tensor = age_preprocess(face_crop_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = age_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = int(torch.tensor(probs).argmax())
    return CLASS_NAMES[idx], probs[idx], {c: round(p, 4) for c, p in zip(CLASS_NAMES, probs)}


def save_face_crop(face_crop_pil):
    filename = f'{uuid.uuid4().hex[:12]}.jpg'
    path = os.path.join(FACES_DIR, filename)
    face_crop_pil.save(path, quality=85)
    return path, filename


def process_detections(detections, img, source):
    """Shared logic: crop → classify → cooldown check → log."""
    results = []
    w, h = img.size

    for det in detections:
        face_conf = det['confidence']
        x1, y1, x2, y2 = det['box']
        track_id = det['track_id']
        is_tracked = det['is_tracked']

        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        face_crop = img.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                               min(w, x2 + pad_x), min(h, y2 + pad_y)))

        age_class, age_conf, age_probs = classify_age(face_crop)
        passed = (face_conf >= FACE_CONF_THRESHOLD) and (age_conf >= AGE_CONF_THRESHOLD)

        # Use track_id as cooldown key so same person = same cooldown bucket
        cd_key = f'track_{track_id}'
        entry = {
            'track_id': track_id,
            'is_tracked': is_tracked,
            'box': [x1, y1, x2, y2],
            'face_confidence': round(face_conf, 4),
            'age_class': age_class,
            'age_confidence': round(age_conf, 4),
            'age_probabilities': age_probs,
            'passed_threshold': passed,
            'cooldown_remaining': cooldown_remaining(cd_key),
        }

        if passed and can_log(cd_key):
            face_path, face_filename = save_face_crop(face_crop)
            log_id = uuid.uuid4().hex[:10]
            timestamp = datetime.datetime.now().isoformat(timespec='seconds')
            append_log([log_id, timestamp, track_id, age_class,
                        round(age_conf, 4), round(face_conf, 4), face_path, source])
            entry['logged'] = True
            entry['log_id'] = log_id
            entry['face_image_url'] = f'/logs/faces/{face_filename}'
            entry['timestamp'] = timestamp
        else:
            entry['logged'] = False

        results.append(entry)
    return results

# ── Routes ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'age_model_loaded': age_model is not None,
        'face_model_loaded': face_model is not None,
        'device': str(DEVICE),
        'face_threshold': FACE_CONF_THRESHOLD,
        'age_threshold': AGE_CONF_THRESHOLD,
        'log_cooldown_sec': LOG_COOLDOWN_SEC,
        'tracking': 'ByteTrack',
    })


@app.route('/predict', methods=['POST'])
def predict():
    if age_model is None:
        return jsonify({'error': f'Age model not loaded ({AGE_MODEL_PATH})'}), 503
    if face_model is None:
        return jsonify({'error': 'Face detection model not loaded'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    source = request.form.get('source', 'upload')
    use_tracking = source.startswith('camera')  # track for camera, predict for upload

    try:
        img = Image.open(io.BytesIO(request.files['file'].read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot read image: {e}'}), 400

    detections = detect_faces_track(img) if use_tracking else detect_faces_predict(img)
    if not detections:
        return jsonify({'faces': [], 'message': 'No face detected.'})

    results = process_detections(detections, img, source)
    return jsonify({'faces': results})


@app.route('/logs', methods=['GET'])
def get_logs():
    limit = int(request.args.get('limit', 100))
    return jsonify({'logs': read_logs(limit), 'total': count_logs()})


@app.route('/logs/count', methods=['GET'])
def get_log_count():
    return jsonify({'count': count_logs()})


@app.route('/logs/faces/<filename>', methods=['GET'])
def get_face_image(filename):
    return send_from_directory(FACES_DIR, filename)


@app.route('/logs/clear', methods=['POST'])
def clear_logs():
    _last_log_time.clear()
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['id', 'timestamp', 'track_id', 'age_class',
                                 'age_confidence', 'face_confidence',
                                 'face_image_path', 'source'])
    return jsonify({'status': 'cleared'})


@app.route('/cooldown/reset', methods=['POST'])
def reset_cooldown():
    """Reset tracker state + cooldowns when camera stops."""
    _last_log_time.clear()
    # Reset ByteTrack internal state so IDs start fresh next session
    if face_model is not None:
        try:
            face_model.predictor = None  # forces tracker reinit on next .track() call
        except Exception:
            pass
    return jsonify({'status': 'reset'})


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('🚀 AgeVision v2.3 — Face >=80%, Age >=50% starting...')
    print(f'   Age model       : {AGE_MODEL_PATH}')
    print(f'   Device          : {DEVICE}')
    print(f'   Face threshold  : {FACE_CONF_THRESHOLD*100:.0f}%')
    print(f'   Age threshold   : {AGE_CONF_THRESHOLD*100:.0f}%')
    print(f'   Log cooldown    : {LOG_COOLDOWN_SEC}s per track ID')
    print(f'   Tracker         : ByteTrack (persistent IDs)')
    print(f'   Log file        : {LOG_CSV}')
    app.run(host='0.0.0.0', port=5000, debug=False)
