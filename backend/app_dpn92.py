import sys
import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64
from datetime import datetime
import uuid
import shutil
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torch.hub import download_url_to_file
from dotenv import load_dotenv
load_dotenv()

try:
    from zoo.models import Dpn92_Unet_Loc, Dpn92_Unet_Double
    print("Successfully imported DPN92 models from zoo.models")
except ImportError as e:
    print(f"Error importing models: {e}")
    raise

try:
    from utils import preprocess_inputs
    print("Using preprocess_inputs from utils")
except ImportError:
    def preprocess_inputs(x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std  = np.array([0.229, 0.224, 0.225]) * 255
        x = (x - mean) / std
        return x
    print("Using inline preprocess_inputs function")


app = Flask(__name__)
CORS(app)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FOLDER  = os.path.join(BASE_DIR, 'DPN92_weights')
UPLOAD_FOLDER   = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER  = os.path.join(BASE_DIR, 'results')

# ── Ensemble weight paths ──────────────────────────────────────────────────────
NUM_FOLDS = 3

LOC_MODEL_PATHS = [
    os.path.join(WEIGHTS_FOLDER, 'LOC', f'dpn92_loc_{i}_0_best.pth')
    for i in range(NUM_FOLDS)
]

CLS_MODEL_PATHS = [
    os.path.join(WEIGHTS_FOLDER, 'CLS', 'BEST', f'dpn92_cls_cce_{i}_1_best')
    for i in range(NUM_FOLDS)
]
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Lists that hold the 3 loaded model instances
loc_models = []
cls_models = []

DAMAGE_COLORS = {
    0: [255, 255, 255],   # No damage  – white        (BGR same as RGB)
    1: [255, 191, 105],   # Minor      – faint blue   (#69bfff in BGR)
    2: [  0, 140, 255],   # Major      – orange       (#ff8c00 in BGR)
    3: [255,   0,   0],   # Destroyed  – dark blue    (#00008b in BGR)
}

DAMAGE_LABELS = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed"
}


# # ── Model loading ──────────────────────────────────────────────────────────────

def _load_checkpoint(model, path, strict=True):
    """Load a checkpoint file; handles both plain state-dict and {'state_dict': …} formats."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model


# def load_models():
#     """Load all 3 LOC and 3 CLS DPN92 models into memory."""
#     global loc_models, cls_models
#     loc_models = []
#     cls_models = []

#     try:
#         for i in range(NUM_FOLDS):
#             # ── Localization ──
#             loc_path = LOC_MODEL_PATHS[i]
#             print(f"Loading LOC fold {i}: {loc_path}")
#             m_loc = Dpn92_Unet_Loc(pretrained=None).to(DEVICE)
#             _load_checkpoint(m_loc, loc_path, strict=True)
#             m_loc.eval()
#             loc_models.append(m_loc)
#             print(f"  ✓ LOC fold {i} loaded")

#             # ── Classification ──
#             cls_path = CLS_MODEL_PATHS[i]
#             print(f"Loading CLS fold {i}: {cls_path}")
#             m_cls = Dpn92_Unet_Double(pretrained=None).to(DEVICE)
#             _load_checkpoint(m_cls, cls_path, strict=False)
#             m_cls.eval()
#             cls_models.append(m_cls)
#             print(f"  ✓ CLS fold {i} loaded")

#         print(f"\nAll {NUM_FOLDS} LOC + {NUM_FOLDS} CLS models loaded on {DEVICE}")
#         return True

#     except Exception as e:
#         print(f"Error loading models: {e}")
#         import traceback; traceback.print_exc()
#         return False

# ─────────────────────────────────────────────────────────────────────────────
# PATCH – replace load_models() and run_ensemble() in app.py
# Strategy:
#   load model i → GPU
#   run inference → store raw numpy result in a list
#   delete model from GPU + empty_cache()
#   repeat for all folds
#   finally average all cached results
# ─────────────────────────────────────────────────────────────────────────────
 
def load_models():
    """
    Verify all weight files exist. Models are NOT kept in memory —
    they are loaded on-demand during inference and immediately discarded.
    """
    global loc_models, cls_models
    loc_models = []   # kept empty; used only as a loaded-flag check
    cls_models = []
 
    missing = []
    for path in LOC_MODEL_PATHS + CLS_MODEL_PATHS:
        if not os.path.exists(path):
            missing.append(path)
 
    if missing:
        print("Missing weight files:")
        for p in missing:
            print(f"  ✗ {p}")
        return False
 
    print("All weight files found:")
    for p in LOC_MODEL_PATHS:
        print(f"  ✓ LOC {p}")
    for p in CLS_MODEL_PATHS:
        print(f"  ✓ CLS {p}")
 
    # Populate the lists with None as a marker so health-check works
    loc_models = [None] * NUM_FOLDS
    cls_models = [None] * NUM_FOLDS
    return True


def _infer_single_fold(fold_idx, pre_tensor_cpu, comb_tensor_cpu):
    """
    Load LOC + CLS for one fold onto GPU, run inference, cache numpy results,
    then immediately delete the models and free GPU memory.
 
    Returns
    -------
    loc_out : np.ndarray (H, W)
    cls_out : np.ndarray (5, H, W)
    """
    # ── LOC ──────────────────────────────────────────────────────────────────
    print(f"  Fold {fold_idx}: loading LOC model...")
    m_loc = Dpn92_Unet_Loc(pretrained=None).to(DEVICE)
    _load_checkpoint(m_loc, LOC_MODEL_PATHS[fold_idx], strict=True)
    m_loc.eval()
 
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=(DEVICE.type == 'cuda')):
            loc_raw = m_loc(pre_tensor_cpu.to(DEVICE))
        loc_out = torch.sigmoid(loc_raw).float().cpu().numpy()[0, 0]  # (H, W)
 
    # Discard LOC model immediately
    del m_loc, loc_raw
    torch.cuda.empty_cache()
    print(f"  Fold {fold_idx}: LOC done, model freed ✓")
 
    # ── CLS ──────────────────────────────────────────────────────────────────
    print(f"  Fold {fold_idx}: loading CLS model...")
    m_cls = Dpn92_Unet_Double().to(DEVICE)
    _load_checkpoint(m_cls, CLS_MODEL_PATHS[fold_idx], strict=False)
    m_cls.eval()
 
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=(DEVICE.type == 'cuda')):
            cls_raw = m_cls(comb_tensor_cpu.to(DEVICE))
        cls_out = torch.sigmoid(cls_raw).float().cpu().numpy()[0]     # (5, H, W)
 
    # Discard CLS model immediately
    del m_cls, cls_raw
    torch.cuda.empty_cache()
    print(f"  Fold {fold_idx}: CLS done, model freed ✓")
 
    return loc_out, cls_out




# ── Pre-processing ─────────────────────────────────────────────────────────────

def pad_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if h != new_h or w != new_w:
        padded = np.zeros((new_h, new_w, 3), dtype=img.dtype)
        padded[:h, :w] = img
        return padded
    return img


def resize_keep_aspect(img, base=1024):
    h, w = img.shape[:2]
    scale = base / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def preprocess_image_for_model(cv_img):
    img = resize_keep_aspect(cv_img, base=1024)
    img = pad_to_multiple(img, multiple=32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std  = np.array([0.229, 0.224, 0.225]) * 255
    img  = (img - mean) / std
    return img


# # ── Ensemble inference ─────────────────────────────────────────────────────────

# def run_ensemble(pre_proc, post_proc):
#     """
#     Run all 3 LOC and 3 CLS models and return averaged predictions.

#     Returns
#     -------
#     loc_pred : np.ndarray, shape (H, W)   – averaged sigmoid building mask
#     cls_pred : np.ndarray, shape (5, H, W) – averaged sigmoid class probabilities
#     """
#     pre_tensor  = torch.from_numpy(pre_proc.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
#     combined    = np.concatenate([pre_proc, post_proc], axis=2)
#     comb_tensor = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

#     loc_preds = []
#     cls_preds = []

#     with torch.no_grad():
#         for i in range(NUM_FOLDS):
#             # Localization
#             loc_out = loc_models[i](pre_tensor)
#             loc_out = torch.sigmoid(loc_out).cpu().numpy()[0, 0]   # (H, W)
#             loc_preds.append(loc_out)

#             # Classification
#             cls_out = cls_models[i](comb_tensor)
#             cls_out = torch.sigmoid(cls_out).cpu().numpy()[0]       # (5, H, W)
#             cls_preds.append(cls_out)

#     # Average across folds
#     loc_pred = np.mean(loc_preds, axis=0)   # (H, W)
#     cls_pred = np.mean(cls_preds, axis=0)   # (5, H, W)

#     return loc_pred, cls_pred


def run_ensemble(pre_proc, post_proc):
    """
    For each fold:
      1. Load model onto GPU
      2. Run inference
      3. Cache numpy result
      4. Delete model + free GPU memory
    Then average all cached results.
 
    Returns
    -------
    loc_pred : np.ndarray (H, W)    – averaged sigmoid building mask
    cls_pred : np.ndarray (5, H, W) – averaged sigmoid class probabilities
    """
    # Build CPU tensors once — reused across all folds
    pre_tensor_cpu  = torch.from_numpy(pre_proc.transpose(2, 0, 1)).unsqueeze(0).float()
    combined        = np.concatenate([pre_proc, post_proc], axis=2)
    comb_tensor_cpu = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).float()
 
    loc_cache = []   # will hold NUM_FOLDS arrays of shape (H, W)
    cls_cache = []   # will hold NUM_FOLDS arrays of shape (5, H, W)
 
    for i in range(NUM_FOLDS):
        print(f"\n── Fold {i+1}/{NUM_FOLDS} ──────────────────────────────")
        loc_out, cls_out = _infer_single_fold(i, pre_tensor_cpu, comb_tensor_cpu)
        loc_cache.append(loc_out)
        cls_cache.append(cls_out)
 
    print(f"\n── Ensemble: averaging {NUM_FOLDS} cached results ──────────")
    loc_pred = np.mean(loc_cache, axis=0)   # (H, W)
    cls_pred = np.mean(cls_cache, axis=0)   # (5, H, W)
    print("Ensemble averaging done ✓")
 
    return loc_pred, cls_pred


# ── Post-processing helpers ────────────────────────────────────────────────────

def create_damage_heatmap(loc_pred, cls_pred, threshold=0.3):
    """
    Majority-vote damage class per pixel (argmax over averaged probabilities)
    restricted to pixels where the building mask exceeds *threshold*.
    """
    # cls_pred channels: [background, no-damage, minor, major, destroyed]
    # Drop background channel (index 0) and take argmax over the 4 damage classes
    damage_map   = cls_pred[1:].argmax(axis=0)   # values 0-3
    h, w         = damage_map.shape
    rgb_map      = np.zeros((h, w, 3), dtype=np.uint8)
    building_mask = loc_pred > threshold

    for class_id, color in DAMAGE_COLORS.items():
        mask = (damage_map == class_id) & building_mask
        rgb_map[mask] = color

    return rgb_map


# def calculate_damage_statistics(loc_pred, cls_pred, threshold=0.3):
#     damage_map          = cls_pred[1:].argmax(axis=0)
#     building_mask       = loc_pred > threshold
#     total_building_pixels = int(np.sum(building_mask))

#     if total_building_pixels == 0:
#         return {
#             'total_building_pixels': 0,
#             'buildings_detected':    False,
#             'damage_breakdown':   {label: 0   for label in DAMAGE_LABELS.values()},
#             'damage_percentages': {label: 0.0 for label in DAMAGE_LABELS.values()}
#         }

#     damage_counts, damage_percentages = {}, {}
#     for class_id, label in DAMAGE_LABELS.items():
#         mask  = (damage_map == class_id) & building_mask
#         count = int(np.sum(mask))
#         damage_counts[label]      = count
#         damage_percentages[label] = round(count / total_building_pixels * 100, 2)

#     return {
#         'total_building_pixels': total_building_pixels,
#         'buildings_detected':    True,
#         'damage_breakdown':      damage_counts,
#         'damage_percentages':    damage_percentages
#     }

# def calculate_damage_statistics(loc_pred, cls_pred, threshold=0.3):
#     damage_map            = cls_pred[1:].argmax(axis=0)
#     building_mask         = loc_pred > threshold
#     total_building_pixels = int(np.sum(building_mask))

#     if total_building_pixels == 0:
#         return {
#             'total_building_pixels': 0,
#             'buildings_detected':    False,
#             'damage_breakdown':   {label: 0   for label in DAMAGE_LABELS.values()},
#             'damage_percentages': {label: 0.0 for label in DAMAGE_LABELS.values()}
#         }

#     damage_counts       = {}
#     damage_percentages  = {}

#     for class_id, label in DAMAGE_LABELS.items():
#         # Only count pixels that are BOTH a building AND this damage class
#         mask  = (damage_map == class_id) & building_mask
#         count = int(np.sum(mask))
#         damage_counts[label]      = count
#         damage_percentages[label] = round(count / total_building_pixels * 100, 2)

#     return {
#         'total_building_pixels': total_building_pixels,
#         'buildings_detected':    True,
#         'damage_breakdown':      damage_counts,
#         'damage_percentages':    damage_percentages
#     }


def calculate_damage_statistics(loc_pred, cls_pred, threshold=0.3):
    damage_map            = cls_pred[1:].argmax(axis=0)   # (H, W) values 0-3
    building_mask         = loc_pred > threshold
    total_building_pixels = int(np.sum(building_mask))
    h, w                  = damage_map.shape
    total_image_pixels    = h * w

    if total_building_pixels == 0:
        return {
            'total_building_pixels': 0,
            'buildings_detected':    False,
            'damage_breakdown':      {label: 0   for label in DAMAGE_LABELS.values()},
            'damage_percentages':    {label: 0.0 for label in DAMAGE_LABELS.values()},
            # new fields — zeroed out
            'building_counts':       {label: 0   for label in DAMAGE_LABELS.values()},
            'total_buildings':       0,
            'weighted_damage_score': 0.0,
            'damage_density':        0.0,
            'hotspot':               None,
        }

    # ── 1. Pixel-level breakdown (existing) ───────────────────────────────────
    damage_counts      = {}
    damage_percentages = {}
    for class_id, label in DAMAGE_LABELS.items():
        mask  = (damage_map == class_id) & building_mask
        count = int(np.sum(mask))
        damage_counts[label]      = count
        damage_percentages[label] = round(count / total_building_pixels * 100, 2)

    # ── 2. Building count per damage class (connected components) ─────────────
    building_mask_u8 = building_mask.astype(np.uint8)
    num_labels, cc_label_map, stats, centroids = cv2.connectedComponentsWithStats(
        building_mask_u8, connectivity=8
    )
    # component 0 is background, skip it
    # filter out tiny noise components (< 50 pixels)
    MIN_BUILDING_PIXELS = 50
    valid_components = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BUILDING_PIXELS
    ]
    total_buildings = len(valid_components)

    building_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    for comp_id in valid_components:
        comp_mask      = cc_label_map == comp_id
        # majority damage class within this component
        class_pixels   = damage_map[comp_mask & building_mask]
        if len(class_pixels) == 0:
            continue
        majority_class = int(np.bincount(class_pixels).argmax())
        label          = DAMAGE_LABELS[majority_class]
        building_counts[label] += 1

    building_count_percentages = {}
    for label, count in building_counts.items():
        building_count_percentages[label] = (
            round(count / total_buildings * 100, 2) if total_buildings > 0 else 0.0
        )

    # ── 3. Weighted damage score 0–100 ────────────────────────────────────────
    # weights: no-damage=0, minor=1, major=2, destroyed=3
    CLASS_WEIGHTS = {0: 0, 1: 1, 2: 2, 3: 3}
    weighted_sum  = sum(
        CLASS_WEIGHTS[class_id] * damage_counts[label]
        for class_id, label in DAMAGE_LABELS.items()
    )
    max_possible         = 3 * total_building_pixels
    weighted_damage_score = round((weighted_sum / max_possible) * 100, 2) if max_possible > 0 else 0.0

    # ── 4. Damage density ─────────────────────────────────────────────────────
    # What fraction of the total image area is damaged building pixels
    damaged_pixels   = sum(
        damage_counts[label]
        for class_id, label in DAMAGE_LABELS.items()
        if class_id > 0   # exclude no-damage class
    )
    damage_density = round((damaged_pixels / total_image_pixels) * 100, 4)

    # ── 5. Hotspot centroid ───────────────────────────────────────────────────
    # Centroid of all destroyed (class 3) pixels; fall back to major (class 2)
    hotspot = None
    for fallback_class in [3, 2, 1]:
        hotspot_mask = (damage_map == fallback_class) & building_mask
        if np.sum(hotspot_mask) > 0:
            ys, xs   = np.where(hotspot_mask)
            hotspot  = {
                'x':           int(np.mean(xs)),
                'y':           int(np.mean(ys)),
                'image_width':  w,
                'image_height': h,
                # normalised 0–1 coordinates (useful for frontend overlay)
                'x_norm':      round(float(np.mean(xs)) / w, 4),
                'y_norm':      round(float(np.mean(ys)) / h, 4),
                'based_on':    DAMAGE_LABELS[fallback_class],
            }
            break

    return {
        # ── existing ──
        'total_building_pixels': total_building_pixels,
        'buildings_detected':    True,
        'damage_breakdown':      damage_counts,
        'damage_percentages':    damage_percentages,
        # ── new ──
        'total_buildings':             total_buildings,
        'building_counts':             building_counts,
        'building_count_percentages':  building_count_percentages,
        'weighted_damage_score':       weighted_damage_score,
        'damage_density':              damage_density,
        'hotspot':                     hotspot,
    }

def numpy_to_base64(img_array):
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', img_array)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


def create_overlay(original_img, heatmap, alpha=0.5):
    if original_img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    mask    = ~((heatmap[:,:,0]==255) & (heatmap[:,:,1]==255) & (heatmap[:,:,2]==255))
    overlay = original_img.copy()
    overlay[mask] = cv2.addWeighted(original_img[mask], 1-alpha, heatmap[mask], alpha, 0)
    return overlay


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':          'healthy',
        'models_loaded':   len(loc_models) == NUM_FOLDS and len(cls_models) == NUM_FOLDS,
        'device':          str(DEVICE),
        'num_folds':       NUM_FOLDS,
        'loc_model':       'Dpn92_Unet_Loc',
        'cls_model':       'Dpn92_Unet_Double',
        'ensemble_strategy': {
            'localization':     'average sigmoid outputs across 3 folds',
            'classification':   'average sigmoid outputs across 3 folds → argmax (majority vote per pixel)',
            'cls_weights_used': 'BEST'
        }
    })


@app.route('/api/detect-damage', methods=['POST'])
def detect_damage():
    """Main endpoint – runs 3-fold DPN92 ensemble and returns damage predictions."""
    try:
        if len(loc_models) < NUM_FOLDS or len(cls_models) < NUM_FOLDS:
            return jsonify({'error': 'Models not fully loaded'}), 500

        if 'pre_image' not in request.files or 'post_image' not in request.files:
            return jsonify({'error': 'Both pre_image and post_image are required'}), 400

        pre_image_file  = request.files['pre_image']
        post_image_file = request.files['post_image']
        threshold       = float(request.form.get('threshold', 0.3))

        # ── Decode images ──
        print("Loading images...")
        pre_img  = cv2.imdecode(np.frombuffer(pre_image_file.read(),  np.uint8), cv2.IMREAD_COLOR)
        post_img = cv2.imdecode(np.frombuffer(post_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if pre_img is None or post_img is None:
            return jsonify({'error': 'Error decoding images'}), 400
        print("✓ Images loaded")

        # ── Pre-process ──
        print("Preprocessing images...")
        pre_proc  = preprocess_image_for_model(pre_img)
        post_proc = preprocess_image_for_model(post_img)

        # Ensure spatial dims match
        if pre_proc.shape[:2] != post_proc.shape[:2]:
            h, w      = pre_proc.shape[:2]
            post_proc = cv2.resize(post_proc, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Ensemble inference ──
        print(f"Running {NUM_FOLDS}-fold DPN92 ensemble...")
        loc_pred, cls_pred = run_ensemble(pre_proc, post_proc)
        print("✓ Ensemble inference complete")

        # ── Visualisations ──
        print("Creating visualisations...")
        heatmap = create_damage_heatmap(loc_pred, cls_pred, threshold)
        overlay = create_overlay(post_img, heatmap, alpha=0.6)
        stats   = calculate_damage_statistics(loc_pred, cls_pred, threshold)

        # ── Persist results ──
        analysis_id = str(uuid.uuid4())
        timestamp   = datetime.now().isoformat()
        analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)

        cv2.imwrite(os.path.join(analysis_dir, 'pre_disaster.jpg'),  pre_img)
        cv2.imwrite(os.path.join(analysis_dir, 'post_disaster.jpg'), post_img)
        cv2.imwrite(os.path.join(analysis_dir, 'heatmap.png'),       heatmap)
        cv2.imwrite(os.path.join(analysis_dir, 'overlay.png'),       overlay)
        loc_vis = (loc_pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(analysis_dir, 'localization.png'),  loc_vis)
        thumbnail = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(analysis_dir, 'thumbnail.jpg'),     thumbnail)

        # Also keep the standalone heatmap file used by /api/download
        cv2.imwrite(os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png'), heatmap)

        result_data = {
            'id':        analysis_id,
            'timestamp': timestamp,
            'threshold': threshold,
            'statistics': stats,
            'ensemble': {
                'num_folds':   NUM_FOLDS,
                'loc_model':   'Dpn92_Unet_Loc',
                'cls_model':   'Dpn92_Unet_Double',
                'cls_weights': 'BEST'
            },
            'files': {
                'pre_disaster':  f'{analysis_id}/pre_disaster.jpg',
                'post_disaster': f'{analysis_id}/post_disaster.jpg',
                'heatmap':       f'{analysis_id}/heatmap.png',
                'overlay':       f'{analysis_id}/overlay.png',
                'localization':  f'{analysis_id}/localization.png',
                'thumbnail':     f'{analysis_id}/thumbnail.jpg'
            }
        }

        with open(os.path.join(RESULTS_FOLDER, f'{analysis_id}.json'), 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"✓ Analysis saved: {analysis_dir}")

        return jsonify({
            'success':     True,
            'analysis_id': analysis_id,
            'timestamp':   timestamp,
            'statistics':  stats,
            'ensemble_info': result_data['ensemble'],
            'visualizations': {
                'heatmap':       numpy_to_base64(heatmap),
                'overlay':       numpy_to_base64(overlay),
                'localization':  numpy_to_base64(loc_vis)
            },
            'summary': {
                'buildings_detected': stats['buildings_detected'],
                'most_severe_damage': (
                    max(stats['damage_percentages'].items(), key=lambda x: x[1])[0]
                    if stats['buildings_detected'] else None
                )
            }
        })

    except Exception as e:
        print(f"Error in detect_damage: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



# @app.route('/api/analyze-region', methods=['POST'])
# def analyze_region():
#     """
#     Re-compute damage statistics for a sub-region of an existing analysis.
#     Expects JSON body:
#       {
#         "analysis_id": "<uuid>",
#         "bbox": { "x_norm": 0.1, "y_norm": 0.2, "w_norm": 0.4, "h_norm": 0.3 }
#       }
#     All bbox values are normalised 0–1 relative to the heatmap image dimensions.
#     """
#     try:
#         data = request.json
#         if 'analysis_id' not in data:
#             return jsonify({'error': 'analysis_id is required'}), 400

#         analysis_id = data['analysis_id']
#         bbox        = data.get('bbox')          # optional — if absent, return full stats

#         result_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
#         if not os.path.exists(result_file):
#             return jsonify({'error': 'Analysis not found'}), 404

#         with open(result_file) as f:
#             result = json.load(f)

#         # If no bbox supplied just return the stored statistics
#         if not bbox:
#             return jsonify({'success': True, 'result': result,
#                             'region_statistics': result.get('statistics')})

#         # ── Load the saved heatmap and localization masks ──────────────────────
#         analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
#         heatmap_path = os.path.join(analysis_dir, 'heatmap.png')
#         loc_path     = os.path.join(analysis_dir, 'localization.png')

#         if not os.path.exists(heatmap_path) or not os.path.exists(loc_path):
#             return jsonify({'error': 'Analysis image files not found'}), 404

#         heatmap_img = cv2.imread(heatmap_path)   # BGR (H, W, 3)
#         loc_img     = cv2.imread(loc_path, cv2.IMREAD_GRAYSCALE)  # 0-255

#         h, w = loc_img.shape

#         # Convert normalised bbox → pixel coords, clamped to image bounds
#         x1 = max(0, int(bbox['x_norm'] * w))
#         y1 = max(0, int(bbox['y_norm'] * h))
#         x2 = min(w, int((bbox['x_norm'] + bbox['w_norm']) * w))
#         y2 = min(h, int((bbox['y_norm'] + bbox['h_norm']) * h))

#         if x2 <= x1 or y2 <= y1:
#             return jsonify({'error': 'Bounding box is too small or out of bounds'}), 400

#         # ── Crop both maps to the selected region ─────────────────────────────
#         loc_crop     = loc_img[y1:y2, x1:x2].astype(np.float32) / 255.0
#         heatmap_crop = heatmap_img[y1:y2, x1:x2]   # BGR

#         # ── Reconstruct a cls_pred-like array from the colour-coded heatmap ───
#         # Heatmap colours (BGR):
#         #   No damage    [255,255,255] → class 0
#         #   Minor damage [217,144, 74] → class 1   (#69bfff in RGB = [74,144,217] in BGR... wait)
#         # Actually we stored the heatmap with cv2 (BGR) DAMAGE_COLORS:
#         #   0: [255,255,255]  no damage – white
#         #   1: [255,191,105]  minor     – faint blue  (#69bfff → R=105,G=191,B=255 → BGR=[255,191,105])
#         #   2: [  0,140,255]  major     – orange      (#ff8c00 → R=255,G=140,B=0   → BGR=[0,140,255])
#         #   3: [139,  0,  0]  destroyed – dark blue   (#00008b → R=0,G=0,B=139     → BGR=[139,0,0])
#         COLOR_TO_CLASS = {
#             (255, 255, 255): 0,
#             (255, 191, 105): 1,
#             (  0, 140, 255): 2,
#             (139,   0,   0): 3,
#         }

#         region_h, region_w = heatmap_crop.shape[:2]

#         # Build a damage_map for the cropped region by matching colours
#         damage_map_crop = np.zeros((region_h, region_w), dtype=np.int32)
#         for (b, g, r), cls_id in COLOR_TO_CLASS.items():
#             mask = (
#                 (heatmap_crop[:,:,0] == b) &
#                 (heatmap_crop[:,:,1] == g) &
#                 (heatmap_crop[:,:,2] == r)
#             )
#             damage_map_crop[mask] = cls_id

#         # ── Build building mask from localization (threshold already baked in) ─
#         # loc_crop is 0-1; values > 0.3 → building
#         building_mask_crop = loc_crop > 0.3
#         total_building_pixels = int(np.sum(building_mask_crop))

#         if total_building_pixels == 0:
#             region_statistics = {
#                 'total_building_pixels': 0,
#                 'buildings_detected':    False,
#                 'damage_breakdown':      {label: 0   for label in DAMAGE_LABELS.values()},
#                 'damage_percentages':    {label: 0.0 for label in DAMAGE_LABELS.values()},
#                 'building_counts':       {label: 0   for label in DAMAGE_LABELS.values()},
#                 'building_count_percentages': {label: 0.0 for label in DAMAGE_LABELS.values()},
#                 'total_buildings':       0,
#                 'weighted_damage_score': 0.0,
#                 'damage_density':        0.0,
#                 'hotspot':               None,
#                 'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
#             }
#             return jsonify({'success': True, 'region_statistics': region_statistics})

#         # ── Pixel counts ───────────────────────────────────────────────────────
#         damage_counts      = {}
#         damage_percentages = {}
#         for class_id, label in DAMAGE_LABELS.items():
#             mask  = (damage_map_crop == class_id) & building_mask_crop
#             count = int(np.sum(mask))
#             damage_counts[label]      = count
#             damage_percentages[label] = round(count / total_building_pixels * 100, 2)

#         # ── Building count via connected components ────────────────────────────
#         building_mask_u8 = building_mask_crop.astype(np.uint8)
#         num_labels, cc_label_map, stats_cc, centroids = cv2.connectedComponentsWithStats(
#             building_mask_u8, connectivity=8
#         )
#         MIN_BUILDING_PIXELS = 50
#         valid_components = [
#             i for i in range(1, num_labels)
#             if stats_cc[i, cv2.CC_STAT_AREA] >= MIN_BUILDING_PIXELS
#         ]
#         total_buildings = len(valid_components)
#         building_counts_map = {label: 0 for label in DAMAGE_LABELS.values()}
#         for comp_id in valid_components:
#             comp_mask    = cc_label_map == comp_id
#             class_pixels = damage_map_crop[comp_mask & building_mask_crop]
#             if len(class_pixels) == 0:
#                 continue
#             majority_class = int(np.bincount(class_pixels).argmax())
#             building_counts_map[DAMAGE_LABELS[majority_class]] += 1

#         building_count_percentages = {
#             label: round(count / total_buildings * 100, 2) if total_buildings > 0 else 0.0
#             for label, count in building_counts_map.items()
#         }

#         # ── Weighted score ─────────────────────────────────────────────────────
#         CLASS_WEIGHTS = {0: 0, 1: 1, 2: 2, 3: 3}
#         weighted_sum  = sum(CLASS_WEIGHTS[cid] * damage_counts[lbl] for cid, lbl in DAMAGE_LABELS.items())
#         max_possible  = 3 * total_building_pixels
#         weighted_score = round((weighted_sum / max_possible) * 100, 2) if max_possible > 0 else 0.0

#         # ── Damage density ─────────────────────────────────────────────────────
#         region_pixels = region_h * region_w
#         damaged_pixels = sum(
#             damage_counts[lbl] for cid, lbl in DAMAGE_LABELS.items() if cid > 0
#         )
#         damage_density = round((damaged_pixels / region_pixels) * 100, 4) if region_pixels > 0 else 0.0

#         # ── Hotspot ────────────────────────────────────────────────────────────
#         hotspot = None
#         for fallback_class in [3, 2, 1]:
#             hotspot_mask = (damage_map_crop == fallback_class) & building_mask_crop
#             if np.sum(hotspot_mask) > 0:
#                 ys, xs  = np.where(hotspot_mask)
#                 # Coordinates relative to original full image
#                 abs_x   = int(np.mean(xs)) + x1
#                 abs_y   = int(np.mean(ys)) + y1
#                 full_h, full_w = h, w
#                 hotspot = {
#                     'x': abs_x, 'y': abs_y,
#                     'image_width': full_w, 'image_height': full_h,
#                     'x_norm': round(abs_x / full_w, 4),
#                     'y_norm': round(abs_y / full_h, 4),
#                     'based_on': DAMAGE_LABELS[fallback_class],
#                 }
#                 break

#         region_statistics = {
#             'total_building_pixels':      total_building_pixels,
#             'buildings_detected':         True,
#             'damage_breakdown':           damage_counts,
#             'damage_percentages':         damage_percentages,
#             'total_buildings':            total_buildings,
#             'building_counts':            building_counts_map,
#             'building_count_percentages': building_count_percentages,
#             'weighted_damage_score':      weighted_score,
#             'damage_density':             damage_density,
#             'hotspot':                    hotspot,
#             'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
#                        'width': region_w, 'height': region_h},
#         }

#         return jsonify({'success': True, 'region_statistics': region_statistics})

#     except Exception as e:
#         print(f"Error in analyze_region: {e}")
#         import traceback; traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-region', methods=['POST', 'OPTIONS'])
def analyze_region():
    """
    Re-compute damage statistics for a sub-region of an existing analysis.
    Body JSON:
      { "analysis_id": "<uuid>",
        "bbox": { "x_norm": 0.1, "y_norm": 0.2, "w_norm": 0.4, "h_norm": 0.3 } }
    """
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        if 'analysis_id' not in data:
            return jsonify({'error': 'analysis_id is required'}), 400

        analysis_id = data['analysis_id']
        bbox        = data.get('bbox')

        result_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        if not os.path.exists(result_file):
            return jsonify({'error': 'Analysis not found'}), 404

        with open(result_file) as f:
            result = json.load(f)

        if not bbox:
            return jsonify({'success': True, 'result': result,
                            'region_statistics': result.get('statistics')})

        analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        heatmap_path = os.path.join(analysis_dir, 'heatmap.png')
        loc_path     = os.path.join(analysis_dir, 'localization.png')

        if not os.path.exists(heatmap_path) or not os.path.exists(loc_path):
            return jsonify({'error': 'Analysis image files not found on disk'}), 404

        # ── Load saved PNG images (lossless, so colours are exact) ────────────
        heatmap_bgr = cv2.imread(heatmap_path)                   # (H, W, 3) BGR uint8
        loc_gray    = cv2.imread(loc_path, cv2.IMREAD_GRAYSCALE) # (H, W)    uint8

        if heatmap_bgr is None or loc_gray is None:
            return jsonify({'error': 'Could not read analysis image files'}), 500

        full_h, full_w = loc_gray.shape

        # ── Crop ──────────────────────────────────────────────────────────────
        x1 = max(0, int(bbox['x_norm'] * full_w))
        y1 = max(0, int(bbox['y_norm'] * full_h))
        x2 = min(full_w, int((bbox['x_norm'] + bbox['w_norm']) * full_w))
        y2 = min(full_h, int((bbox['y_norm'] + bbox['h_norm']) * full_h))

        if x2 <= x1 or y2 <= y1:
            return jsonify({'error': 'Bounding box too small or out of bounds'}), 400

        heatmap_crop = heatmap_bgr[y1:y2, x1:x2]
        loc_crop     = loc_gray[y1:y2, x1:x2]
        region_h, region_w = heatmap_crop.shape[:2]
        print(f"Region: ({x1},{y1})→({x2},{y2})  {region_w}×{region_h}px")

        # ── Building mask ─────────────────────────────────────────────────────
        # localization.png = (loc_pred * 255).astype(uint8), threshold was 0.3
        # → 0.3 * 255 = 76.5, so threshold at 76
        building_mask         = loc_crop >= 76
        total_building_pixels = int(np.sum(building_mask))
        print(f"Building pixels in region: {total_building_pixels}")

        if total_building_pixels == 0:
            empty = {label: 0 for label in DAMAGE_LABELS.values()}
            return jsonify({'success': True, 'region_statistics': {
                'total_building_pixels': 0, 'buildings_detected': False,
                'damage_breakdown': empty,
                'damage_percentages': {l: 0.0 for l in empty},
                'building_counts': empty,
                'building_count_percentages': {l: 0.0 for l in empty},
                'total_buildings': 0, 'weighted_damage_score': 0.0,
                'damage_density': 0.0, 'hotspot': None,
                'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            }})

        # ── Decode damage class from heatmap colour ───────────────────────────
        # DAMAGE_COLORS saved by cv2 (BGR):
        #   class 0  No Damage    [255, 255, 255]  white
        #   class 1  Minor        [255, 191, 105]  faint blue  (#69bfff → R=105,G=191,B=255 → BGR=[255,191,105])
        #   class 2  Major        [  0, 140, 255]  orange      (#ff8c00 → R=255,G=140,B=0   → BGR=[0,140,255])
        #   class 3  Destroyed    [139,   0,   0]  dark blue   (#00008b → R=0,G=0,B=139     → BGR=[139,0,0])
        #   background            [  0,   0,   0]  black
        #
        # Use nearest-colour matching (L∞) to be robust against any 1-off
        # rounding from cv2 resize operations on the overlay image.

        TARGET_BGR = np.array([
            [255, 255, 255],   # 0 – no damage
            [255, 191, 105],   # 1 – minor
            [  0, 140, 255],   # 2 – major
            [139,   0,   0],   # 3 – destroyed
        ], dtype=np.int32)

        flat       = heatmap_crop.reshape(-1, 3).astype(np.int32)        # (N, 3)
        dists      = np.max(np.abs(flat[:, None, :] - TARGET_BGR[None, :, :]), axis=2)  # (N, 4)
        nearest    = dists.argmin(axis=1).reshape(region_h, region_w)   # (H, W) 0-3

        # Mark background (near-black pixels) as -1 so they are excluded
        is_background          = np.all(heatmap_crop < 20, axis=2)
        damage_map_crop        = nearest.astype(np.int32)
        damage_map_crop[is_background] = -1

        # Debug: print unique colours found in crop
        unique_colors = np.unique(heatmap_crop.reshape(-1, 3), axis=0)
        print(f"Unique BGR colours in crop (first 10): {unique_colors[:10].tolist()}")
        print(f"Damage map unique values: {np.unique(damage_map_crop).tolist()}")

        # ── Pixel-level stats ─────────────────────────────────────────────────
        damage_counts      = {}
        damage_percentages = {}
        for class_id, label in DAMAGE_LABELS.items():
            mask  = (damage_map_crop == class_id) & building_mask
            count = int(np.sum(mask))
            damage_counts[label]      = count
            damage_percentages[label] = round(count / total_building_pixels * 100, 2)
        print(f"Damage counts: {damage_counts}")

        # ── Building-level stats (connected components) ───────────────────────
        num_labels, cc_map, stats_cc, _ = cv2.connectedComponentsWithStats(
            building_mask.astype(np.uint8), connectivity=8
        )
        valid_comps     = [i for i in range(1, num_labels)
                           if stats_cc[i, cv2.CC_STAT_AREA] >= 50]
        total_buildings = len(valid_comps)

        building_counts_map = {label: 0 for label in DAMAGE_LABELS.values()}
        for comp_id in valid_comps:
            comp_mask    = cc_map == comp_id
            class_pixels = damage_map_crop[comp_mask & building_mask]
            class_pixels = class_pixels[class_pixels >= 0]
            if len(class_pixels) == 0:
                continue
            majority = int(np.bincount(class_pixels).argmax())
            if majority in DAMAGE_LABELS:
                building_counts_map[DAMAGE_LABELS[majority]] += 1

        bld_count_pct = {
            lbl: round(cnt / total_buildings * 100, 2) if total_buildings > 0 else 0.0
            for lbl, cnt in building_counts_map.items()
        }

        # ── Weighted score ────────────────────────────────────────────────────
        W       = {0: 0, 1: 1, 2: 2, 3: 3}
        w_sum   = sum(W[cid] * damage_counts[lbl] for cid, lbl in DAMAGE_LABELS.items())
        w_score = round((w_sum / (3 * total_building_pixels)) * 100, 2)

        # ── Damage density ────────────────────────────────────────────────────
        damaged_px = sum(damage_counts[lbl] for cid, lbl in DAMAGE_LABELS.items() if cid > 0)
        density    = round((damaged_px / (region_h * region_w)) * 100, 4)

        # ── Hotspot ───────────────────────────────────────────────────────────
        hotspot = None
        for fc in [3, 2, 1]:
            mask = (damage_map_crop == fc) & building_mask
            if np.sum(mask) > 0:
                ys, xs = np.where(mask)
                ax = int(np.mean(xs)) + x1
                ay = int(np.mean(ys)) + y1
                hotspot = {
                    'x': ax, 'y': ay,
                    'image_width': full_w, 'image_height': full_h,
                    'x_norm': round(ax / full_w, 4),
                    'y_norm': round(ay / full_h, 4),
                    'based_on': DAMAGE_LABELS[fc],
                }
                break

        return jsonify({'success': True, 'region_statistics': {
            'total_building_pixels':      total_building_pixels,
            'buildings_detected':         True,
            'damage_breakdown':           damage_counts,
            'damage_percentages':         damage_percentages,
            'total_buildings':            total_buildings,
            'building_counts':            building_counts_map,
            'building_count_percentages': bld_count_pct,
            'weighted_damage_score':      w_score,
            'damage_density':             density,
            'hotspot':                    hotspot,
            'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                       'width': region_w, 'height': region_h},
        }})

    except Exception as e:
        print(f"Error in analyze_region: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<analysis_id>/<filename>')
def serve_result_file(analysis_id, filename):
    file_path = os.path.join(RESULTS_FOLDER, analysis_id, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path)


@app.route('/api/history')
def get_history():
    try:
        results = []
        for filename in os.listdir(RESULTS_FOLDER):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(RESULTS_FOLDER, filename)
            try:
                with open(filepath) as f:
                    result = json.load(f)
                if 'id' not in result or 'timestamp' not in result:
                    continue
                result['thumbnail'] = f"/api/results/{result['id']}/overlay.png"
                if 'files' in result:
                    for key in result['files']:
                        result['files'][key] = f"/api/results/{result['id']}/{os.path.basename(result['files'][key])}"
                results.append(result)
            except (json.JSONDecodeError, Exception):
                continue
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'success': True, 'count': len(results), 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    try:
        result_file  = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        heatmap_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        if not os.path.exists(result_file):
            return jsonify({'error': 'Analysis not found'}), 404
        with open(result_file) as f:
            result = json.load(f)
        if os.path.exists(heatmap_file):
            result['heatmap'] = numpy_to_base64(cv2.imread(heatmap_file))
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    try:
        analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        json_file    = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        heatmap_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        deleted = False
        if os.path.exists(analysis_dir):
            shutil.rmtree(analysis_dir); deleted = True
        if os.path.exists(json_file):
            os.remove(json_file); deleted = True
        if os.path.exists(heatmap_file):
            os.remove(heatmap_file); deleted = True
        if not deleted:
            return jsonify({'error': 'Analysis not found'}), 404
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    except Exception as e:
        return jsonify({'error': 'Failed to delete analysis'}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'ensemble': {
            'num_folds':          NUM_FOLDS,
            'loc_model_class':    'Dpn92_Unet_Loc',
            'cls_model_class':    'Dpn92_Unet_Double',
            'cls_weights_used':   'BEST',
            'loc_strategy':       'average sigmoid across folds',
            'cls_strategy':       'average sigmoid across folds then argmax (pixel-wise majority vote)',
        },
        'loc_model_paths': LOC_MODEL_PATHS,
        'cls_model_paths': CLS_MODEL_PATHS,
        'models_loaded':   len(loc_models) == NUM_FOLDS and len(cls_models) == NUM_FOLDS,
        'device':          str(DEVICE),
        'damage_classes':  DAMAGE_LABELS,
        'color_map':       DAMAGE_COLORS
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'max_image_size':    2048,
        'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        'default_threshold': 0.3,
        'device':            str(DEVICE),
        'damage_classes':    DAMAGE_LABELS
    })


@app.route('/api/download/<analysis_id>', methods=['GET'])
def download_heatmap(analysis_id):
    try:
        heatmap_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        if not os.path.exists(heatmap_file):
            return jsonify({'error': 'Heatmap not found'}), 404
        return send_file(heatmap_file, mimetype='image/png',
                         as_attachment=True,
                         download_name=f'damage_heatmap_{analysis_id}.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Infrastructure Damage Detection API  –  DPN92 Ensemble")
    print(f"  Folds : {NUM_FOLDS}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    if not load_models():
        print("WARNING: Failed to load one or more models. API will not function correctly.")

    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)