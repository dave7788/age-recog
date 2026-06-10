"""
app.py — AgeVision Flask Backend
Run: python app.py
Requires: pip install flask flask-cors torch torchvision pillow
Place this file alongside: age_classifier_efficientnet.pth
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import io, os

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = 'age_classifier_efficientnet.pth'  # sesuaikan jika beda nama
CLASS_NAMES = ['(0-20)', '(20-40)', '(40-100)']
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── App ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the HTML frontend

# ── Load Model ──────────────────────────────────────────────────────────────
def load_model():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASS_NAMES))
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Support both plain state_dict and full checkpoint dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE).eval()
    print(f'✅ Model loaded from {MODEL_PATH} | Device: {DEVICE}')
    return model

# ── Transform ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Load at startup
if os.path.exists(MODEL_PATH):
    model = load_model()
else:
    model = None
    print(f'⚠️  Model file not found: {MODEL_PATH}')
    print('   Predictions will fail until the model file is present.')

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(DEVICE),
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': f'Model not loaded. Make sure {MODEL_PATH} exists.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a multipart/form-data with key "file".'}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot read image: {e}'}), 400

    # Preprocess & infer
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    pred_idx    = int(torch.tensor(probs).argmax())
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx]
    probabilities = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs)}

    return jsonify({
        'class':         pred_class,
        'confidence':    round(confidence, 4),
        'probabilities': probabilities,
    })


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('🚀 AgeVision backend starting...')
    print(f'   Model path : {MODEL_PATH}')
    print(f'   Device     : {DEVICE}')
    print('   Endpoints  : GET /health  |  POST /predict')
    app.run(host='0.0.0.0', port=5000, debug=False)
