import os
import json
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from flask import Flask, render_template, request, redirect, flash, jsonify, url_for
from werkzeug.utils import secure_filename

# =========================
# Paths & Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shoplift")
os.environ["PYTHONUNBUFFERED"] = "1"

T = 8                       # MUST match training
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
ALLOWED = {"mp4", "avi", "mov", "mkv"}

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "shoplift_torch_state.pth"
CLASS_PATH = MODELS_DIR / "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASE_DIR      :", BASE_DIR, flush=True)
print("TEMPLATES_DIR :", TEMPLATES_DIR, "index.html exists?", (TEMPLATES_DIR / "index.html").exists(), flush=True)
print("STATIC_DIR    :", STATIC_DIR,    "style.css exists?", (STATIC_DIR / "style.css").exists(), flush=True)
print("MODEL_PATH    :", MODEL_PATH,    "exists?", MODEL_PATH.exists(), flush=True)
print("CLASS_PATH    :", CLASS_PATH,    "exists?", CLASS_PATH.exists(), flush=True)

MAX_UPLOAD_MB = 256
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

def style_version() -> int:
    """Cache-busting version for style.css."""
    try:
        return int((STATIC_DIR / "style.css").stat().st_mtime)
    except FileNotFoundError:
        return 0

# =========================
# Model (matches training)
# =========================
class TemporalClassifier(nn.Module):
    def __init__(self, num_classes=2, hidden=128, bidirectional=True):
        super().__init__()
        m = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.backbone = m.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 576

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_in = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(fc_in, num_classes)

    def encode_frames(self, x):  # (B,T,3,H,W) -> (B,T,576)
        B, Tt, C, H, W = x.shape
        x = x.view(B * Tt, C, H, W)
        f = self.backbone(x)
        f = self.gap(f).squeeze(-1).squeeze(-1)  # (B*Tt, 576)
        f = f.view(B, Tt, -1)
        return f

    def forward(self, x):
        f = self.encode_frames(x)
        out, _ = self.gru(f)
        f = out.mean(dim=1)  # mean over time
        f = self.dropout(f)
        return self.fc(f)

# =========================
# Preprocessing
# =========================
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def clip_to_tensor(clip_hwc):
    """(T,H,W,3)[0..255] -> (T,3,H,W) float32 normalized."""
    frames = []
    for f in clip_hwc:
        x = to_tensor(f.astype(np.uint8))
        x = normalize(x)
        frames.append(x)
    return torch.stack(frames, dim=0)

def sample_uniform_frames(video_path, t=T, img_size=IMG_SIZE):
    """Uniform sampling (robust)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return None
    idxs = np.linspace(0, n - 1, num=t, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok or f is None: continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(f.astype("float32"))
    cap.release()
    if not frames: return None
    clip = np.stack(frames, axis=0)
    if clip.shape[0] < t:
        pad = np.repeat(clip[-1:], t - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    return clip

# Use uniform for the app (simple & robust)
extract_clip_anyway = sample_uniform_frames

# =========================
# Load model & classes
# =========================
def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

app_model = TemporalClassifier(num_classes=2, hidden=128, bidirectional=True).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# Accept plain state_dict, checkpoint dicts, or saved module
if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()) \
   and not any(k in ckpt for k in ("state_dict", "model_state_dict", "model")):
    state = _strip_module_prefix(ckpt)
    load_res = app_model.load_state_dict(state, strict=True)
elif isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model_state_dict", "model")):
    loaded = False
    for key in ("state_dict", "model_state_dict", "model"):
        if key in ckpt:
            maybe = ckpt[key]
            if hasattr(maybe, "state_dict"):
                maybe = maybe.state_dict()
            if isinstance(maybe, dict):
                state = _strip_module_prefix(maybe)
                load_res = app_model.load_state_dict(state, strict=True)
                loaded = True
                break
    if not loaded:
        raise RuntimeError("Unrecognized checkpoint structure.")
elif hasattr(ckpt, "state_dict"):
    state = _strip_module_prefix(ckpt.state_dict())
    load_res = app_model.load_state_dict(state, strict=True)
else:
    raise RuntimeError(f"Unrecognized checkpoint type: {type(ckpt)}")

print("State dict loaded.",
      "missing:", getattr(load_res, "missing_keys", []),
      "unexpected:", getattr(load_res, "unexpected_keys", []),
      flush=True)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    APP_CLASSES = json.load(f)
print("APP_CLASSES:", APP_CLASSES, flush=True)

app_model.eval()
with torch.no_grad():
    _ = app_model(torch.zeros(1, T, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
print("Model sanity forward passed.", flush=True)

# =========================
# Flask app
# =========================
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.secret_key = "not-secret"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

def allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED

def friendly_prediction(raw_label: str):
    """Map raw class → (display_text, css_class)"""
    if raw_label == "shop_lifters":
        return ("Lifters", "pred-bad")
    else:
        return ("Shoppers", "pred-good")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            flash("Please choose a video file.")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Allowed: mp4, avi, mov, mkv")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(str(save_path))

        clip = extract_clip_anyway(save_path, t=T, img_size=IMG_SIZE)
        if clip is None:
            flash("Could not extract frames. Try re-encoding to H.264 / yuv420p.")
            return redirect(url_for("index"))

        # Debug: input stats
        print("DEBUG --- frames summary ---", flush=True)
        print("  filename:", filename, flush=True)
        print("  clip shape:", clip.shape, "dtype:", clip.dtype, flush=True)
        print("  clip min/max/std:",
              float(clip.min()), float(clip.max()), float(clip.std()), flush=True)

        xb = clip_to_tensor(clip).unsqueeze(0).to(DEVICE)
        print("  xb shape:", tuple(xb.shape),
              "xb min/max/std:",
              float(xb.min().item()),
              float(xb.max().item()),
              float(xb.std().item()),
              flush=True)

        with torch.no_grad():
            logits = app_model(xb)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

        pred_id = int(np.argmax(probs))
        raw_label = APP_CLASSES[pred_id]
        display_label, display_class = friendly_prediction(raw_label)
        conf = float(probs[pred_id])
        pairs = list(zip(probs, APP_CLASSES))

        logger.info(f"Pred for {filename} -> {display_label} ({conf:.4f}); probs={probs}")

        return render_template(
            "index.html",
            prediction=display_label,
            prediction_class=display_class,
            confidence=conf,
            filename=filename,
            probs=probs,
            classes=APP_CLASSES,
            pairs=pairs,
            style_v=style_version(),
            active_page="home",
        )

    # GET
    return render_template(
        "index.html",
        prediction=None,
        prediction_class=None,
        confidence=None,
        filename=None,
        probs=None,
        classes=APP_CLASSES,
        pairs=[],
        style_v=style_version(),
        active_page="home",
    )

@app.route("/about")
def about():
    return render_template("about.html", style_v=style_version(), active_page="about")

# Debug helpers (optional)
@app.route("/debug_predict_path")
def debug_predict_path():
    path = request.args.get("path", None)
    if not path:
        return "Usage: /debug_predict_path?path=C:\\\\full\\\\path\\\\video.mp4"
    video_path = Path(path)
    if not video_path.exists():
        return f"File not found: {video_path}"

    clip = sample_uniform_frames(video_path, t=T, img_size=IMG_SIZE)
    if clip is None:
        return "Failed to decode frames."

    xb = clip_to_tensor(clip).unsqueeze(0).to(DEVICE)

    stats = {
        "ok": True,
        "shape": list(clip.shape),
        "min": float(clip.min()),
        "max": float(clip.max()),
        "std": float(clip.std()),
        "xb_shape": list(xb.shape),
        "xb_min": float(xb.min().item()),
        "xb_max": float(xb.max().item()),
        "xb_std": float(xb.std().item()),
    }

    with torch.no_grad():
        logits = app_model(xb)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    pred_id = int(np.argmax(probs))
    stats["probs"] = {APP_CLASSES[i]: float(p) for i, p in enumerate(probs)}
    stats["prediction"] = friendly_prediction(APP_CLASSES[pred_id])[0]
    return jsonify(stats)

@app.route("/csscheck")
def csscheck():
    from flask import send_from_directory
    return send_from_directory(app.static_folder, "style.css")

@app.route("/hello")
def hello():
    return "<h1>Flask is serving HTML</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
