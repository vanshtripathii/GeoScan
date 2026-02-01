from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load trained model (if available). Use a safe fallback so the server can run
model = None
model_path = os.path.join(os.path.dirname(__file__), 'model', 'weights', 'best_model.h5')
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print(f"Model file not found at {model_path}; starting server without model.")
except Exception as e:
    print('Failed to load model:', e)
    model = None

def preprocess_images(image1, image2):
    """Preprocess two images for change detection"""
    # Resize
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))
    
    # Combine images
    combined = np.concatenate([image1, image2], axis=-1)
    combined = combined.astype(np.float32) / 255.0
    
    return np.expand_dims(combined, axis=0)

@app.route('/detect-changes', methods=['POST'])
def detect_changes():
    try:
        # Get images from request
        data = request.json
        image1_data = data['image1']  # Base64 encoded
        image2_data = data['image2']  # Base64 encoded
        
        # Decode base64 images
        image1 = decode_base64_image(image1_data)
        image2 = decode_base64_image(image2_data)
        
        # Preprocess
        processed_input = preprocess_images(image1, image2)
        
        # Predict (use loaded model if available). If no model, use a simple image-diff fallback
        if model is not None:
            prediction = model.predict(processed_input)
        else:
            # Simple fallback: compute absolute difference between the two images
            # Convert to grayscale, resize to 256x256, compute absdiff, threshold and normalize
            try:
                img1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if image1.ndim == 3 else image1
                img2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if image2.ndim == 3 else image2
            except Exception:
                img1_gray = cv2.resize(image1, (256, 256))
                img2_gray = cv2.resize(image2, (256, 256))

            img1_rs = cv2.resize(img1_gray, (256, 256))
            img2_rs = cv2.resize(img2_gray, (256, 256))
            diff = cv2.absdiff(img1_rs, img2_rs)

            # Normalize and threshold the difference to create a mask
            diff_norm = diff.astype(np.float32) / 255.0
            # Use a fixed threshold (tunable) to detect changes
            _, diff_thresh = cv2.threshold((diff_norm * 255).astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
            pred = (diff_thresh.astype(np.float32) / 255.0)

            # prediction shape expected by downstream code: (1, H, W, 1)
            prediction = np.expand_dims(pred, axis=(0, -1))
        
        # Post-process prediction
        change_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
        
        # Convert mask to base64 for response
        mask_base64 = encode_image_to_base64(change_mask)
        
        return jsonify({
            'success': True,
            'change_mask': mask_base64,
            'change_percentage': float(np.mean(prediction > 0.5))
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/region-analysis', methods=['GET'])
def region_analysis():
    """Run analysis for a named region by loading dataset images server-side."""
    try:
        region = request.args.get('region', 'delhi').lower()

        # Resolve datasets directory (repo root `/datasets`)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        datasets_dir = os.path.join(repo_root, 'datasets')

        # Known mapping for sample regions (override if you have specific filenames)
        region_map = {
            'delhi': {
                'before': os.path.join(datasets_dir, 'images', 'A', '2014.png'),
                'after': os.path.join(datasets_dir, 'images', 'B', '2024.png')
            }
        }

        if region in region_map:
            before_path = region_map[region]['before']
            after_path = region_map[region]['after']
        else:
            # Fallback: pick the first file from A and B
            a_dir = os.path.join(datasets_dir, 'images', 'A')
            b_dir = os.path.join(datasets_dir, 'images', 'B')
            a_files = sorted([os.path.join(a_dir, f) for f in os.listdir(a_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            b_files = sorted([os.path.join(b_dir, f) for f in os.listdir(b_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            if not a_files or not b_files:
                return jsonify({'success': False, 'error': 'No dataset files found for region analysis.'})
            before_path = a_files[0]
            after_path = b_files[0]

        # Load images
        from PIL import Image as PILImage

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            return jsonify({'success': False, 'error': 'Region dataset files not found on server.'})

        img1 = PILImage.open(before_path).convert('RGB')
        img2 = PILImage.open(after_path).convert('RGB')
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        # Reuse prediction logic from detect_changes
        processed_input = preprocess_images(img1_np, img2_np)

        if model is not None:
            prediction = model.predict(processed_input)
        else:
            # Fallback image-diff as in detect_changes
            try:
                img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY) if img1_np.ndim == 3 else img1_np
                img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY) if img2_np.ndim == 3 else img2_np
            except Exception:
                img1_gray = cv2.resize(img1_np, (256, 256))
                img2_gray = cv2.resize(img2_np, (256, 256))

            img1_rs = cv2.resize(img1_gray, (256, 256))
            img2_rs = cv2.resize(img2_gray, (256, 256))
            diff = cv2.absdiff(img1_rs, img2_rs)
            diff_norm = diff.astype(np.float32) / 255.0
            _, diff_thresh = cv2.threshold((diff_norm * 255).astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
            pred = (diff_thresh.astype(np.float32) / 255.0)
            prediction = np.expand_dims(pred, axis=(0, -1))

        change_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
        mask_base64 = encode_image_to_base64(change_mask)

        return jsonify({
            'success': True,
            'change_mask': mask_base64,
            'change_percentage': float(np.mean(prediction > 0.5)),
            'before_path': os.path.relpath(before_path, repo_root).replace('\\', '/'),
            'after_path': os.path.relpath(after_path, repo_root).replace('\\', '/')
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def decode_base64_image(base64_string):
    """Decode base64 string to numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def encode_image_to_base64(image_array):
    """Encode numpy array to base64 string"""
    # Normalize array and handle common shapes (batch dim, single-channel)
    arr = np.array(image_array)

    # Remove batch dimension if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # If single channel with shape (H, W, 1), squeeze to (H, W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # Convert boolean to uint8 (0/255)
    if arr.dtype == np.bool_:
        arr = (arr.astype(np.uint8)) * 255

    # If float in [0,1], scale to 0-255
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # Ensure uint8 for PIL
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # For 2D arrays use 'L', for 3D with 3 channels PIL will auto-detect RGB
    image = Image.fromarray(arr)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':

import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=True)
