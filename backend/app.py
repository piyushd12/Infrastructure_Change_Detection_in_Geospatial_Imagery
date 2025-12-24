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
import os
from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()

try:
    from zoo.models import SeResNext50_Unet_Loc, SeResNext50_Unet_Double
    print("Successfully imported models from zoo.models")
except ImportError as e:
    print(f"Error importing models: {e}")
    raise

try:
    from utils import preprocess_inputs
    print("Using preprocess_inputs from utils")
except ImportError:
    print("utils not found, using local utils.py")
    try:
        from utils import preprocess_inputs
        print("Using preprocess_inputs from local utils")
    except ImportError:
        print("Could not import preprocess_inputs")
        print("Creating preprocess_inputs function inline...")
        
        def preprocess_inputs(x):
            """Preprocess input images for xView2 models"""
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406]) * 255
            std = np.array([0.229, 0.224, 0.225]) * 255
            x = (x - mean) / std
            return x
        print("Using inline preprocess_inputs function")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FOLDER = os.path.join(BASE_DIR, 'weights')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
LOC_MODEL_PATH = os.path.join(WEIGHTS_FOLDER, 'res50_loc_1_tuned_best')
CLS_MODEL_PATH = os.path.join(WEIGHTS_FOLDER, 'res50_cls_cce_1_0_last')

# UPLOAD_FOLDER = 'uploads'
# RESULTS_FOLDER = 'results'
# LOC_MODEL_PATH = 'backend/weights/res50_loc_1_tuned_best'
# CLS_MODEL_PATH = 'backend/weights/res50_cls_cce_1_0_last'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

loc_model = None
cls_model = None

DAMAGE_COLORS = {
    0: [255, 255, 255],  # No damage - White
    1: [255, 255, 0],    # Minor - Yellow
    2: [255, 165, 0],    # Major - Orange
    3: [255, 0, 0],      # Destroyed - Red
}

DAMAGE_LABELS = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed"
}


def download_weight(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)}...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        download_url_to_file(url, dest)
        print("Download complete")
    else:
        print(f"Weight found: {dest}")

download_weight(f"{os.getenv('HF_MODEL_REPO')}/res50_loc_1_tuned_best", LOC_MODEL_PATH)
download_weight(f"{os.getenv('HF_MODEL_REPO')}/res50_cls_cce_1_0_last", CLS_MODEL_PATH)

# def load_models():
#     global loc_model, cls_model
#     try:
#         print("Loading localization model...")
#         loc_model = SeResNext50_Unet_Loc().to(DEVICE)
#         loc_checkpoint = torch.load(LOC_MODEL_PATH, map_location=DEVICE, weights_only=False)
#         loc_model.load_state_dict(loc_checkpoint['state_dict'])
#         loc_model.eval()
#         print("Localization model loaded")
        
#         print("Loading classification model...")
#         cls_model = SeResNext50_Unet_Double().to(DEVICE)
#         cls_checkpoint = torch.load(CLS_MODEL_PATH, map_location=DEVICE, weights_only=False)
#         cls_model.load_state_dict(cls_checkpoint['state_dict'], strict=False)
#         cls_model.eval()
#         print("Classification model loaded")
        
#         print(f"Models loaded successfully on {DEVICE}")
#         return True
#     except Exception as e:
#         print(f"Error loading models: {str(e)}")
#         return False

def load_models():
    global loc_model, cls_model
    try:
        # repo_id = "piyush-s-deshmukh/change_detection_loc_and_cls"
        
        loc_url = f"{os.getenv('HF_MODEL_REPO')}/res50_loc_1_tuned_best"
        cls_url = f"{os.getenv('HF_MODEL_REPO')}/res50_cls_cce_1_0_last"
        
        os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
        
        loc_weight_path = os.path.join(WEIGHTS_FOLDER, "res50_loc_1_tuned_best")
        cls_weight_path = os.path.join(WEIGHTS_FOLDER, "res50_cls_cce_1_0_last")
        
        if not os.path.exists(loc_weight_path):
            print("Downloading localization model...")
            download_url_to_file(loc_url, loc_weight_path)
            print("✓ Localization model downloaded")
        else:
            print("✓ Localization model already exists")
        
        if not os.path.exists(cls_weight_path):
            print("Downloading classification model...")
            download_url_to_file(cls_url, cls_weight_path)
            print("✓ Classification model downloaded")
        else:
            print("✓ Classification model already exists")
        
        print("Loading models into memory...")
        loc_model = SeResNext50_Unet_Loc().to(DEVICE)
        loc_checkpoint = torch.load(loc_weight_path, map_location=DEVICE, weights_only=False)
        loc_model.load_state_dict(loc_checkpoint['state_dict'])
        loc_model.eval()
        
        cls_model = SeResNext50_Unet_Double().to(DEVICE)
        cls_checkpoint = torch.load(cls_weight_path, map_location=DEVICE, weights_only=False)
        cls_model.load_state_dict(cls_checkpoint['state_dict'], strict=False)
        cls_model.eval()
        
        print(f"Models loaded on {DEVICE}")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



# def load_models():
#     global loc_model, cls_model
#     try:
#         repo_id = os.getenv('HF_MODEL_REPO')
        
#         loc_filename = "res50_loc_1_tuned_best"
#         cls_filename = "res50_cls_cce_1_0_last"
        
#         os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
        
#         loc_weight_path = os.path.join(WEIGHTS_FOLDER, loc_filename)
#         cls_weight_path = os.path.join(WEIGHTS_FOLDER, cls_filename)
        
#         if not os.path.exists(loc_weight_path):
#             print(f"Downloading localization model from Hugging Face...")
#             loc_weight_path = hf_hub_download(
#                 repo_id=repo_id,
#                 filename=loc_filename,
#                 local_dir=WEIGHTS_FOLDER,
#                 local_dir_use_symlinks=False
#             )
#             print(f"✓ Localization model downloaded to {loc_weight_path}")
#         else:
#             print(f"✓ Localization model already exists: {loc_weight_path}")
        
#         if not os.path.exists(cls_weight_path):
#             print(f"Downloading classification model from Hugging Face...")
#             cls_weight_path = hf_hub_download(
#                 repo_id=repo_id,
#                 filename=cls_filename,
#                 local_dir=WEIGHTS_FOLDER,
#                 local_dir_use_symlinks=False
#             )
#             print(f"✓ Classification model downloaded to {cls_weight_path}")
#         else:
#             print(f"✓ Classification model already exists: {cls_weight_path}")
        
#         print("Loading localization model into memory...")
#         loc_model = SeResNext50_Unet_Loc().to(DEVICE)
#         loc_checkpoint = torch.load(loc_weight_path, map_location=DEVICE, weights_only=False)
#         loc_model.load_state_dict(loc_checkpoint['state_dict'])
#         loc_model.eval()
#         print("✓ Localization model loaded")
        
#         print("Loading classification model into memory...")
#         cls_model = SeResNext50_Unet_Double().to(DEVICE)
#         cls_checkpoint = torch.load(cls_weight_path, map_location=DEVICE, weights_only=False)
#         cls_model.load_state_dict(cls_checkpoint['state_dict'], strict=False)
#         cls_model.eval()
#         print("✓ Classification model loaded")
        
#         print(f"Models successfully loaded on {DEVICE}")
#         return True
        
#     except Exception as e:
#         print(f"Error loading models: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

def create_damage_heatmap(loc_pred, cls_pred, threshold=0.3):
    """
    Create colored damage heatmap
    
    Args:
        loc_pred: Building localization predictions
        cls_pred: Damage classification predictions
        threshold: Threshold for building detection
    
    Returns:
        RGB heatmap image
    """
    damage_map = cls_pred[1:].argmax(axis=0)
    h, w = damage_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    building_mask = loc_pred > threshold
    
    for class_id, color in DAMAGE_COLORS.items():
        mask = (damage_map == class_id) & building_mask
        rgb_map[mask] = color
    
    return rgb_map


def pad_to_multiple(img, multiple=32):
    """Pad image to make height and width divisible by multiple (default 32)"""
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    
    if h != new_h or w != new_w:
        padded = np.zeros((new_h, new_w, 3), dtype=img.dtype)
        padded[:h, :w] = img
        return padded
    return img

def resize_keep_aspect(img, base=1024):
    """Resize image keeping aspect ratio, longest side = base"""
    h, w = img.shape[:2]
    scale = base / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def preprocess_image_for_model(cv_img):
    """
    Full preprocessing pipeline compatible with xView2 models
    """
    img = resize_keep_aspect(cv_img, base=1024)
    
    img = pad_to_multiple(img, multiple=32)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img = (img - mean) / std
    
    return img


def calculate_damage_statistics(loc_pred, cls_pred, threshold=0.3):
    """Calculate damage statistics from predictions"""
    damage_map = cls_pred[1:].argmax(axis=0)
    building_mask = loc_pred > threshold
    
    total_building_pixels = np.sum(building_mask)
    
    if total_building_pixels == 0:
        return {
            'total_building_pixels': 0,
            'buildings_detected': False,
            'damage_breakdown': {label: 0 for label in DAMAGE_LABELS.values()},
            'damage_percentages': {label: 0.0 for label in DAMAGE_LABELS.values()}
        }
    
    damage_counts = {}
    damage_percentages = {}
    
    for class_id, label in DAMAGE_LABELS.items():
        mask = (damage_map == class_id) & building_mask
        count = np.sum(mask)
        percentage = (count / total_building_pixels) * 100
        
        damage_counts[label] = int(count)
        damage_percentages[label] = round(percentage, 2)
    
    return {
        'total_building_pixels': int(total_building_pixels),
        'buildings_detected': True,
        'damage_breakdown': damage_counts,
        'damage_percentages': damage_percentages
    }

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    _, buffer = cv2.imencode('.png', img_array)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"

def create_overlay(original_img, heatmap, alpha=0.5):
    """Create overlay of heatmap on original image"""
    if original_img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    mask = ~((heatmap[:,:,0] == 255) & (heatmap[:,:,1] == 255) & (heatmap[:,:,2] == 255))
    
    overlay = original_img.copy()
    overlay[mask] = cv2.addWeighted(original_img[mask], 1-alpha, heatmap[mask], alpha, 0)
    
    return overlay

@app.route('/api/history/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete a specific analysis and all its files"""
    try:
        analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        json_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        
        deleted = False
        
        if os.path.exists(analysis_dir):
            shutil.rmtree(analysis_dir)
            deleted = True
        
        if os.path.exists(json_file):
            os.remove(json_file)
            deleted = True
        
        if not deleted:
            return jsonify({'error': 'Analysis not found'}), 404
        
        print(f"Deleted analysis: {analysis_id}")
        return jsonify({
            'success': True,
            'message': 'Analysis deleted successfully'
        })
        
    except Exception as e:
        print(f"Error deleting analysis {analysis_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete analysis'}), 500



@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': loc_model is not None and cls_model is not None,
        'device': str(DEVICE),
        'localization_model': 'SeResNext50_Unet_Loc',
        'classification_model': 'SeResNext50_Unet_Double'
    })

@app.route('/api/detect-damage', methods=['POST'])
def detect_damage():
    """Main endpoint for damage detection"""
    try:
        if loc_model is None or cls_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        if 'pre_image' not in request.files or 'post_image' not in request.files:
            return jsonify({'error': 'Both pre_image and post_image are required'}), 400
        
        pre_image_file = request.files['pre_image']
        post_image_file = request.files['post_image']
        
        threshold = float(request.form.get('threshold', 0.3))
        
        print("Loading images...")
        pre_img_bytes = np.frombuffer(pre_image_file.read(), np.uint8)
        post_img_bytes = np.frombuffer(post_image_file.read(), np.uint8)
        
        pre_img = cv2.imdecode(pre_img_bytes, cv2.IMREAD_COLOR)
        post_img = cv2.imdecode(post_img_bytes, cv2.IMREAD_COLOR)
        
        if pre_img is None or post_img is None:
            return jsonify({'error': 'Error decoding images'}), 400
        
        print("✓ Images loaded")
        
        # # Preprocess images
        # print("Preprocessing images...")
        # pre_proc = preprocess_inputs(pre_img)
        # post_proc = preprocess_inputs(post_img)
        # Preprocess images with fixed pipeline
        print("Preprocessing images...")
        pre_proc = preprocess_image_for_model(pre_img)
        post_proc = preprocess_image_for_model(post_img)
        
        print(f"Pre image shape after processing: {pre_proc.shape}")
        print(f"Post image shape after processing: {post_proc.shape}")
        
        if pre_proc.shape[:2] != post_proc.shape[:2]:
            target_h, target_w = pre_proc.shape[:2]
            post_proc = cv2.resize(post_proc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            print(f"Resized post image to match pre: {post_proc.shape}")
        
        print("Running localization...")
        pre_tensor = torch.from_numpy(pre_proc.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            loc_pred = loc_model(pre_tensor)
            loc_pred = torch.sigmoid(loc_pred).cpu().numpy()[0, 0]
        
        print("Localization complete")
        
        print("Running classification...")
        combined = np.concatenate([pre_proc, post_proc], axis=2)
        combined_tensor = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            cls_pred = cls_model(combined_tensor)
            cls_pred = torch.sigmoid(cls_pred).cpu().numpy()[0]
        
        print("Classification complete")
        
        print("Creating visualizations...")
        heatmap = create_damage_heatmap(loc_pred, cls_pred, threshold)
        
        overlay = create_overlay(post_img, heatmap, alpha=0.6)
        
        stats = calculate_damage_statistics(loc_pred, cls_pred, threshold)
        
        # # Generate unique ID for this analysis
        # analysis_id = str(uuid.uuid4())
        # timestamp = datetime.now().isoformat()
        
        # # Save results
        # result_data = {
        #     'id': analysis_id,
        #     'timestamp': timestamp,
        #     'threshold': threshold,
        #     'statistics': stats
        # }

        # analysis_id = str(uuid.uuid4())
        # timestamp = datetime.now().isoformat()
        
        # result_data = {
        #     'id': analysis_id,
        #     'timestamp': timestamp,
        #     'threshold': threshold,
        #     'statistics': stats,
        #     'files': {
        #         'pre_disaster': f'{analysis_id}/pre_disaster.jpg',
        #         'post_disaster': f'{analysis_id}/post_disaster.jpg',
        #         'heatmap': f'{analysis_id}/heatmap.png',
        #         'overlay': f'{analysis_id}/overlay.png',
        #         'localization': f'{analysis_id}/localization.png',
        #         'thumbnail': f'{analysis_id}/thumbnail.jpg'
        #     }
        # }

        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        result_data = {
            'id': analysis_id,
            'timestamp': timestamp,
            'threshold': threshold,
            'statistics': stats,
            'files': {
                'pre_disaster': f'{analysis_id}/pre_disaster.jpg',
                'post_disaster': f'{analysis_id}/post_disaster.jpg',
                'heatmap': f'{analysis_id}/heatmap.png',
                'overlay': f'{analysis_id}/overlay.png',
                'localization': f'{analysis_id}/localization.png',
                'thumbnail': f'{analysis_id}/thumbnail.jpg'
            }
        }
        
        json_path = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # json_path = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        # with open(json_path, 'w') as f:
        #     json.dump(result_data, f, indent=2)
        
        heatmap_path = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        cv2.imwrite(heatmap_path, heatmap)
        
        result_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
        
        print("✓ Analysis complete")

        analysis_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(analysis_dir, 'pre_disaster.jpg'), pre_img)
        cv2.imwrite(os.path.join(analysis_dir, 'post_disaster.jpg'), post_img)
        
        cv2.imwrite(os.path.join(analysis_dir, 'heatmap.png'), heatmap)
        cv2.imwrite(os.path.join(analysis_dir, 'overlay.png'), overlay)
        
        loc_vis = (loc_pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(analysis_dir, 'localization.png'), loc_vis)
        
        thumbnail = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(analysis_dir, 'thumbnail.jpg'), thumbnail)
        
        print(f"✓ Full analysis saved to: {analysis_dir}")

        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'timestamp': timestamp,
            'statistics': stats,
            'visualizations': {
                'heatmap': numpy_to_base64(heatmap),
                'overlay': numpy_to_base64(overlay),
                'localization': numpy_to_base64((loc_pred * 255).astype(np.uint8))
            },
            'summary': {
                'buildings_detected': stats['buildings_detected'],
                'most_severe_damage': max(stats['damage_percentages'].items(), 
                                         key=lambda x: x[1])[0] if stats['buildings_detected'] else None
            }
        })
        
    except Exception as e:
        print(f"Error in detect_damage: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/analyze-region', methods=['POST'])
def analyze_region():
    """Analyze specific region of interest"""
    try:
        data = request.json
        
        if 'analysis_id' not in data:
            return jsonify({'error': 'analysis_id is required'}), 400
        
        result_file = os.path.join(RESULTS_FOLDER, f"{data['analysis_id']}.json")
        
        if not os.path.exists(result_file):
            return jsonify({'error': 'Analysis not found'}), 404
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<analysis_id>/<filename>')
def serve_result_file(analysis_id, filename):
    """Serve any file from a specific analysis"""
    file_path = os.path.join(RESULTS_FOLDER, analysis_id, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)

@app.route('/api/history')
def get_history():
    """Enhanced history with thumbnails and file info"""
    try:
        results = []
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.endswith('.json'):
                filepath = os.path.join(RESULTS_FOLDER, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                
                result['thumbnail'] = f"/api/results/{result['id']}/overlay.png"
                
                if 'files' in result:
                    for key in result['files']:
                        result['files'][key] = f"/api/results/{result['id']}/{os.path.basename(result['files'][key])}"
                
                results.append(result)
        
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get specific analysis by ID"""
    try:
        result_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}.json')
        heatmap_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        
        if not os.path.exists(result_file):
            return jsonify({'error': 'Analysis not found'}), 404
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        if os.path.exists(heatmap_file):
            heatmap = cv2.imread(heatmap_file)
            result['heatmap'] = numpy_to_base64(heatmap)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'localization_model': {
            'name': 'SeResNext50_Unet_Loc',
            'loaded': loc_model is not None,
            'checkpoint': LOC_MODEL_PATH
        },
        'classification_model': {
            'name': 'SeResNext50_Unet_Double',
            'loaded': cls_model is not None,
            'checkpoint': CLS_MODEL_PATH
        },
        'device': str(DEVICE),
        'damage_classes': DAMAGE_LABELS,
        'color_map': DAMAGE_COLORS
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get configuration settings"""
    return jsonify({
        'max_image_size': 2048,
        'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        'default_threshold': 0.3,
        'device': str(DEVICE),
        'damage_classes': DAMAGE_LABELS
    })

@app.route('/api/download/<analysis_id>', methods=['GET'])
def download_heatmap(analysis_id):
    """Download heatmap image"""
    try:
        heatmap_file = os.path.join(RESULTS_FOLDER, f'{analysis_id}_heatmap.png')
        
        if not os.path.exists(heatmap_file):
            return jsonify({'error': 'Heatmap not found'}), 404
        
        return send_file(heatmap_file, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'damage_heatmap_{analysis_id}.png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Infrastructure Damage Detection API")
    print("=" * 50)
    
    if not load_models():
        print("Warning: Failed to load models. API will not function correctly.")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=True)