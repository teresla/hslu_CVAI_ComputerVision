from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import torch
import base64
import cv2

app = Flask(__name__)

# Load model
processor = AutoImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
model = SegformerForSemanticSegmentation.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")

def calculate_meters_per_pixel(latitude, zoom_level):
    """Calculate meters per pixel based on latitude and zoom level"""
    C = 40075016.686  # Earth's circumference in meters
    return (C * np.cos(np.radians(latitude))) / (2 ** (zoom_level + 8))

def process_image(image_data):
    """Process the base64 image data and return the segmented image"""
    # Remove the data URL prefix
    image_data = image_data.split(',')[1]
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Process image for model
    inputs = processor(images=image, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Process the output
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    # Get the predicted segmentation mask
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # Convert to numpy array
    pred_seg = pred_seg.numpy()
    
    # Create a colored mask (you can adjust the colors based on your model's classes)
    mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    mask[pred_seg == 1] = [255, 0, 0]  # Paved areas in red
    mask[pred_seg == 0] = [0, 255, 0]  # Non-paved areas in green
    
    # Calculate percentage of paved area
    total_pixels = pred_seg.size
    paved_pixels = np.sum(pred_seg == 1)
    paved_percentage = (paved_pixels / total_pixels) * 100
    
    # Convert mask to base64
    _, buffer = cv2.imencode('.png', mask)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return mask_base64, paved_percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    lat = float(data['lat'])
    lng = float(data['lng'])
    zoom = 18  # Fixed zoom level
    image_data = data['image']
    
    # Calculate meters per pixel
    meters_per_pixel = calculate_meters_per_pixel(lat, zoom)
    
    # Process image and get segmentation results
    segmented_image, paved_percentage = process_image(image_data)
    
    # Calculate area in square meters
    # Assuming the image is 125x125 pixels
    total_area_m2 = (125 * meters_per_pixel) * (125 * meters_per_pixel)
    paved_area_m2 = total_area_m2 * (paved_percentage / 100)
    
    return jsonify({
        'meters_per_pixel': meters_per_pixel,
        'segmentation_result': {
            'paved': round(paved_percentage, 2),
            'area_m2': round(paved_area_m2, 2)
        },
        'segmented_image': f'data:image/png;base64,{segmented_image}'
    })

if __name__ == '__main__':
    app.run(debug=True) 