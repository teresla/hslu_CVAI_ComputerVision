import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, SegformerForSemanticSegmentation
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

try:
    # Load both models and processors
    huggingface_processor = AutoImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    huggingface_processor.size = {"height": 512, "width": 512}
    huggingface_model = SegformerForSemanticSegmentation.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    huggingface_model = huggingface_model.to(device)
    huggingface_model.eval()

    custom_processor = AutoImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    custom_processor.size = {"height": 512, "width": 512}
    custom_model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    custom_model = custom_model.to(device)
    custom_model.eval()
    
    logger.info("Successfully loaded both models")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Original class mappings (for Hugging Face model)
ORIGINAL_CLASSES = {
    0: 'unlabeled',
    1: 'paved-area',
    2: 'dirt',
    3: 'grass',
    4: 'gravel',
    5: 'water',
    6: 'rocks',
    7: 'pool',
    8: 'vegetation',
    9: 'roof',
    10: 'wall',
    11: 'window',
    12: 'door',
    13: 'fence',
    14: 'fence-pole',
    15: 'person',
    16: 'dog',
    17: 'car',
    18: 'bicycle',
    19: 'tree',
    20: 'bald-tree',
    21: 'ar-marker',
    22: 'obstacle',
    23: 'conflicting'
}

# Simplified class mappings
SIMPLIFIED_CLASSES = {
    0: 'background',  # unlabeled, conflicting, ar-marker, wall, window, door, fence, fence-pole, person, dog, bicycle, obstacle
    1: 'pavement',    # paved-area, gravel, dirt, rocks
    2: 'grass',       # grass
    3: 'vegetation',  # tree, bald-tree, bushes, other vegetation
    4: 'roof',        # roof
    5: 'water',       # water, pool
    6: 'car',         # car
}

# Color mappings for visualization
ORIGINAL_COLORS = {
    0: [128, 128, 128],  # unlabeled - gray
    1: [255, 0, 0],      # paved-area - red
    2: [139, 69, 19],    # dirt - brown
    3: [0, 255, 0],      # grass - green
    4: [160, 82, 45],    # gravel - sandy brown
    5: [0, 0, 255],      # water - blue
    6: [105, 105, 105],  # rocks - dark gray
    7: [0, 191, 255],    # pool - light blue
    8: [34, 139, 34],    # vegetation - forest green
    9: [128, 0, 0],      # roof - dark red
    10: [192, 192, 192], # wall - silver
    11: [135, 206, 235], # window - sky blue
    12: [165, 42, 42],   # door - brown
    13: [128, 128, 0],   # fence - olive
    14: [128, 128, 0],   # fence-pole - olive
    15: [255, 192, 203], # person - pink
    16: [255, 165, 0],   # dog - orange
    17: [0, 0, 128],     # car - navy
    18: [128, 0, 128],   # bicycle - purple
    19: [0, 100, 0],     # tree - dark green
    20: [85, 107, 47],   # bald-tree - dark olive green
    21: [255, 255, 0],   # ar-marker - yellow
    22: [255, 0, 255],   # obstacle - magenta
    23: [128, 128, 128]  # conflicting - gray
}

SIMPLIFIED_COLORS = {
    0: [128, 128, 128],  # background - gray
    1: [255, 0, 0],      # pavement - red
    2: [0, 255, 0],      # grass - green
    3: [34, 139, 34],    # vegetation - forest green
    4: [128, 0, 0],      # roof - dark red
    5: [0, 0, 255],      # water - blue
    6: [0, 0, 128],      # car - navy
}

# Mapping from original to simplified classes
ORIGINAL_TO_SIMPLIFIED = {
    0: 0,   # unlabeled -> background
    1: 1,   # paved-area -> pavement
    2: 1,   # dirt -> pavement
    3: 2,   # grass -> grass
    4: 1,   # gravel -> pavement
    5: 5,   # water -> water
    6: 1,   # rocks -> pavement
    7: 5,   # pool -> water
    8: 3,   # vegetation -> vegetation
    9: 4,   # roof -> roof
    10: 0,  # wall -> background
    11: 0,  # window -> background
    12: 0,  # door -> background
    13: 0,  # fence -> background
    14: 0,  # fence-pole -> background
    15: 0,  # person -> background
    16: 0,  # dog -> background
    17: 6,  # car -> car
    18: 0,  # bicycle -> background
    19: 3,  # tree -> vegetation
    20: 3,  # bald-tree -> vegetation
    21: 0,  # ar-marker -> background
    22: 0,  # obstacle -> background
    23: 0   # conflicting -> background
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_meters_per_pixel(latitude, zoom_level):
    """Calculate meters per pixel based on latitude and zoom level"""
    C = 40075016.686  # Earth's circumference in meters
    return (C * np.cos(np.radians(latitude))) / (2 ** (zoom_level + 8))

def process_image(image, processor):
    logger.info(f"Original image shape: {image.shape}")

    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_height, original_width = image.shape[:2]

    # Process image for model
    inputs = processor(images=image, return_tensors="pt")
    logger.info(f"Model input shape: {inputs['pixel_values'].shape}")

    return inputs, (original_height, original_width)


def create_colored_mask(pred_mask, original_shape, use_simplified=True):
    logger.info(f"Prediction mask shape before resizing: {pred_mask.shape}")

    # Convert class indices to color
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    if use_simplified:
        # Convert original classes to simplified classes
        simplified_mask = np.zeros_like(pred_mask)
        for orig_class, simplified_class in ORIGINAL_TO_SIMPLIFIED.items():
            simplified_mask[pred_mask == orig_class] = simplified_class
        
        # Apply simplified colors
        for class_idx, color in SIMPLIFIED_COLORS.items():
            colored_mask[simplified_mask == class_idx] = color
    else:
        # Use original classes and colors
        for class_idx, color in ORIGINAL_COLORS.items():
            colored_mask[pred_mask == class_idx] = color

    # Resize mask to match original input shape
    colored_mask = cv2.resize(colored_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    logger.info(f"Final colored mask shape: {colored_mask.shape}")

    return colored_mask


def calculate_statistics(pred_mask, original_shape, use_simplified=True):
    logger.info(f"Prediction mask shape before resizing: {pred_mask.shape}")

    # Resize prediction to original shape
    pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    logger.info(f"Resized prediction shape: {pred_mask.shape}")

    total_pixels = pred_mask.size
    stats = {}

    if use_simplified:
        # Convert to simplified classes
        simplified_mask = np.zeros_like(pred_mask)
        for orig_class, simplified_class in ORIGINAL_TO_SIMPLIFIED.items():
            simplified_mask[pred_mask == orig_class] = simplified_class
        
        # Calculate statistics for simplified classes
        for class_idx, class_name in SIMPLIFIED_CLASSES.items():
            class_pixels = np.sum(simplified_mask == class_idx)
            percentage = (class_pixels / total_pixels) * 100
            color = SIMPLIFIED_COLORS[class_idx]
            color_hex = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # BGR → HEX

            stats[class_name] = {
                'percentage': round(percentage, 2),
                'pixels': int(class_pixels),
                'color': color_hex
            }
    else:
        # Calculate statistics for original classes
        for class_idx, class_name in ORIGINAL_CLASSES.items():
            class_pixels = np.sum(pred_mask == class_idx)
            percentage = (class_pixels / total_pixels) * 100
            color = ORIGINAL_COLORS[class_idx]
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
        use_simplified = data.get('use_simplified', True)  # Default to simplified classes
        use_custom_model = data.get('use_custom_model', True)  # Default to custom model
        
        logger.info(f"Received coordinates: lat={lat}, lng={lng}, zoom={zoom}")
        logger.info(f"Using {'simplified' if use_simplified else 'original'} classes")
        logger.info(f"Using {'custom' if use_custom_model else 'Hugging Face'} model")
        
        # Select model and processor
        model = custom_model if use_custom_model else huggingface_model
        processor = custom_processor if use_custom_model else huggingface_processor
        
        # Calculate meters per pixel
        meters_per_pixel = calculate_meters_per_pixel(lat, zoom)
        logger.info(f"Calculated meters per pixel: {meters_per_pixel}")
        
        # Process base64 image
        image = process_base64_image(image_data)
        logger.info(f"Base64 image decoded shape: {image.shape}")
        
        # Process image
        inputs, original_shape = process_image(image, processor)
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
        stats = calculate_statistics(pred_mask, original_shape, use_simplified)
        
        # Calculate areas in square meters
        total_area_m2 = (original_shape[1] * meters_per_pixel) * (original_shape[0] * meters_per_pixel)
        logger.info(f"Total area in square meters: {total_area_m2}")
        
        for class_name in stats:
            stats[class_name]['area_m2'] = round(total_area_m2 * (stats[class_name]['percentage'] / 100), 2)
        
        # Create colored mask
        colored_mask = create_colored_mask(pred_mask, original_shape, use_simplified)
        
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
        inputs, original_shape = process_image(image, custom_processor)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = custom_model(**inputs)
            logits = outputs.logits
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Calculate statistics
        stats = calculate_statistics(pred_mask, original_shape)
        
        # Create colored mask
        colored_mask = create_colored_mask(pred_mask, original_shape)
        
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