import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import io
import logging
from pathlib import Path
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define paths
base_path = Path(__file__).parent.parent
model_path = base_path / "models/best_segformer"
UPLOAD_FOLDER = base_path / "web_app/static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Load model and processor
processor = SegformerImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
processor.size = {"height": 512, "width": 512}  # Set reasonable target size
model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model = model.to(device)
model.eval()

# Class mappings (matching process_labels.py)
SIMPLIFIED_CLASSES = {
    0: 'background',  # unlabeled, conflicting, ar-marker, etc.
    1: 'pavement',    # paved-area, gravel, dirt, etc.
    2: 'grass',       # usable grass areas
    3: 'vegetation',  # trees, bushes, other vegetation
    4: 'roof',        # roof
    5: 'water',       # water, pool
    6: 'car',         # car (optional)
}

# Colors for visualization (RGB)
SIMPLIFIED_COLORS = {
    0: (0, 0, 0),        # background: black
    1: (128, 64, 128),   # pavement: purple
    2: (0, 255, 0),      # grass: bright green
    3: (0, 102, 0),      # vegetation: dark green
    4: (70, 70, 70),     # roof: dark gray
    5: (28, 42, 168),    # water: blue
    6: (9, 143, 150),    # car: cyan
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_meters_per_pixel(latitude, zoom_level):
    """Calculate meters per pixel based on latitude and zoom level"""
    C = 40075016.686  # Earth's circumference in meters
    return (C * np.cos(np.radians(latitude))) / (2 ** (zoom_level + 8))

def resize_image_fixed(image, target_size=512):
    """Resize the image directly to target size (512x512)."""
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def process_image(image):
    logger.info(f"Original image shape: {image.shape}")

    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_height, original_width = image.shape[:2]

    # Resize to 512x512
    image = resize_image_fixed(image, target_size=512)
    logger.info(f"Resized image shape: {image.shape}")

    inputs = processor(images=image, return_tensors="pt")
    logger.info(f"Model input shape: {inputs['pixel_values'].shape}")

    return inputs, (original_height, original_width)


def create_colored_mask(pred_mask, original_shape):
    logger.info(f"Prediction mask shape before resizing: {pred_mask.shape}")

    # Convert class indices to color
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in SIMPLIFIED_COLORS.items():
        colored_mask[pred_mask == class_idx] = color

    # Resize mask to match original input shape
    colored_mask = cv2.resize(colored_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    logger.info(f"Final colored mask shape: {colored_mask.shape}")

    return colored_mask


def calculate_statistics(pred_mask, original_shape):
    logger.info(f"Prediction mask shape before resizing: {pred_mask.shape}")

    # Resize prediction to original shape
    pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    logger.info(f"Resized prediction shape: {pred_mask.shape}")

    total_pixels = pred_mask.size
    stats = {}

    for class_idx, class_name in SIMPLIFIED_CLASSES.items():
        class_pixels = np.sum(pred_mask == class_idx)
        percentage = (class_pixels / total_pixels) * 100
        color = SIMPLIFIED_COLORS[class_idx]
        color_hex = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # BGR → HEX

        stats[class_name] = {
            'percentage': round(percentage, 2),
            'pixels': int(class_pixels),
            'color': color_hex
        }

    return stats


def process_base64_image(image_data):
    """Process base64 image data"""
    # Remove the data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image format")
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        lat = float(data['lat'])
        lng = float(data['lng'])
        zoom = int(data['zoom'])
        image_data = data['image']
        
        logger.info(f"Received coordinates: lat={lat}, lng={lng}, zoom={zoom}")
        
        # Calculate meters per pixel
        meters_per_pixel = calculate_meters_per_pixel(lat, zoom)
        logger.info(f"Calculated meters per pixel: {meters_per_pixel}")
        
        # Process base64 image
        image = process_base64_image(image_data)
        logger.info(f"Base64 image decoded shape: {image.shape}")
        
        # Process image
        inputs, original_shape = process_image(image)
        logger.info(f"Model inputs shape: {inputs['pixel_values'].shape}")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            logger.info(f"Model output logits shape: {logits.shape}")
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            logger.info(f"Prediction mask shape after argmax: {pred_mask.shape}")
        
        # Calculate statistics
        stats = calculate_statistics(pred_mask, original_shape)
        
        # Calculate areas in square meters
        total_area_m2 = (original_shape[1] * meters_per_pixel) * (original_shape[0] * meters_per_pixel)
        logger.info(f"Total area in square meters: {total_area_m2}")
        
        for class_name in stats:
            stats[class_name]['area_m2'] = round(total_area_m2 * (stats[class_name]['percentage'] / 100), 2)
        
        # Create colored mask
        colored_mask = create_colored_mask(pred_mask, original_shape)
        
        # Convert mask to base64
        _, buffer = cv2.imencode('.png', colored_mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'meters_per_pixel': meters_per_pixel,
            'segmentation_result': stats,
            'segmented_image': f'data:image/png;base64,{mask_base64}',
            'image_dimensions': {
                'width': original_shape[1],
                'height': original_shape[0]
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save original image
        filename = secure_filename(file.filename)
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(original_path)
        
        # Read and process image
        image = cv2.imread(original_path)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process image
        inputs, original_shape = process_image(image)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Calculate statistics
        stats = calculate_statistics(pred_mask, original_shape, padding_info)
        
        # Create colored mask
        colored_mask = create_colored_mask(pred_mask, original_shape, padding_info)
        
        # Save mask
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
        cv2.imwrite(mask_path, colored_mask)
        
        # Format statistics for display
        formatted_stats = []
        for class_name, data in stats.items():
            if data['percentage'] > 0:  # Only show classes that are present
                formatted_stats.append({
                    'class': class_name,
                    'percentage': f"{data['percentage']:.1f}%",
                    'pixels': f"{data['pixels']:,}",
                    'color': data['color']
                })
        
        # Sort by percentage (highest first)
        formatted_stats.sort(key=lambda x: float(x['percentage'].rstrip('%')), reverse=True)
        
        response = {
            'original': f'/static/uploads/{filename}',
            'mask': f'/static/uploads/{mask_filename}',
            'statistics': formatted_stats,
            'image_dimensions': {
                'width': original_shape[1],
                'height': original_shape[0]
            }
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 