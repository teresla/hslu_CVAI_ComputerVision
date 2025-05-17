import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse
from pathlib import Path

def load_model():
    """Load and return the model and processor"""
    print("Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    
    # Print model information
    print("\nModel configuration:")
    print(model.config)
    print(f"Number of classes: {model.config.num_labels}")
    
    return processor, model

def process_image(image_path, processor, model):
    """Process a single image and return original, mask, and paved percentage"""
    print(f"\nProcessing {image_path}...")
    
    try:
        # Load image and ensure it's in RGB format
        image = Image.open(image_path)
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        # Process image for model
        inputs = processor(images=image, return_tensors="pt")
        print(f"Input tensor shape: {inputs['pixel_values'].shape}")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"Logits shape: {logits.shape}")
        
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
        print(f"Unique classes in prediction: {np.unique(pred_seg)}")
        
        # Create a colored mask with more detailed classes
        mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        
        # Color mapping for different classes
        color_map = {
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
        
        # Apply colors to mask
        for class_id, color in color_map.items():
            mask[pred_seg == class_id] = color
        
        # Calculate percentage of paved area
        total_pixels = pred_seg.size
        paved_pixels = np.sum(pred_seg == 1)
        paved_percentage = (paved_pixels / total_pixels) * 100
        
        # Calculate percentages for other major classes
        class_percentages = {}
        for class_id, class_name in model.config.id2label.items():
            class_pixels = np.sum(pred_seg == int(class_id))
            class_percentage = (class_pixels / total_pixels) * 100
            if class_percentage > 0.1:  # Only show classes with more than 0.1% coverage
                class_percentages[class_name] = class_percentage
        
        return image, mask, paved_percentage, class_percentages
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        raise

def display_results(original, mask, paved_percentage, class_percentages, model_config, show_plot=True):
    """Display the original image with segmentation overlay and legend"""
    # Create figure with three subplots: original, overlay, and legend
    fig = plt.figure(figsize=(20, 8))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Overlay image
    ax2 = plt.subplot(1, 3, 2)
    # Convert original to numpy array for overlay
    original_np = np.array(original)
    # Create overlay by blending original and mask
    overlay = cv2.addWeighted(original_np, 0.7, mask, 0.3, 0)
    ax2.imshow(overlay)
    ax2.set_title('Segmentation Overlay')
    ax2.axis('off')
    
    # Legend
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    # Sort classes by percentage (descending)
    sorted_classes = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Color mapping for different classes
    color_map = {
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
    
    # Create legend with color swatches
    legend_text = "Class Coverage:\n\n"
    y_pos = 0.9  # Starting y position for legend items
    
    for class_name, percentage in sorted_classes:
        # Find the class ID for this class name
        class_id = None
        for id, name in model_config.id2label.items():
            if name == class_name:
                class_id = int(id)
                break
        
        if class_id is not None:
            # Get the color for this class
            color = np.array(color_map[class_id]) / 255.0  # Normalize to [0,1] for matplotlib
            
            # Add color swatch
            ax3.add_patch(plt.Rectangle((0.1, y_pos - 0.02), 0.05, 0.05, 
                                      facecolor=color, edgecolor='black'))
            
            # Add text
            ax3.text(0.2, y_pos, f"{class_name}: {percentage:.1f}%", 
                    fontsize=10, verticalalignment='center')
            
            y_pos -= 0.1  # Move down for next item
    
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig

def analyze_model_output(image_path, processor, model):
    """Analyze the model's output format and class predictions"""
    print(f"\nAnalyzing model output for {image_path}...")
    
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # Get raw model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Print output shape and class probabilities
    print(f"Output shape: {logits.shape}")
    print(f"Number of classes: {logits.shape[1]}")
    
    # Get class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Print class probabilities for a sample pixel
    print("\nClass probabilities for center pixel:")
    center_probs = probabilities[0, :, logits.shape[2]//2, logits.shape[3]//2]
    for i, prob in enumerate(center_probs):
        print(f"Class {i}: {prob.item():.4f}")
    
    return logits, probabilities

def save_processed_images(image_path, output_dir, processor, model):
    """Process and save the original and segmented images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process image
    original, mask, paved_percentage, class_percentages = process_image(image_path, processor, model)
    
    # Save original and mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    original.save(os.path.join(output_dir, f"{base_name}_original.png"))
    
    # Save overlay
    original_np = np.array(original)
    overlay = cv2.addWeighted(original_np, 0.7, mask, 0.3, 0)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Save mask
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    
    # Save class percentages to a text file
    with open(os.path.join(output_dir, f"{base_name}_percentages.txt"), 'w') as f:
        f.write("Class Coverage Percentages:\n")
        for class_name, percentage in class_percentages.items():
            f.write(f"{class_name}: {percentage:.1f}%\n")
    
    print(f"Saved processed images for {base_name}")
    print("Class Coverage Percentages:")
    for class_name, percentage in class_percentages.items():
        print(f"{class_name}: {percentage:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Experiment with aerial image segmentation model')
    parser.add_argument('--input-dir', type=str, default='map_samples',
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='processed_samples',
                        help='Directory to save processed images')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze model output format')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display plots')
    args = parser.parse_args()

    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, args.input_dir)
    output_dir = os.path.join(current_dir, args.output_dir)

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files in input directory: {os.listdir(input_dir)}")

    # Load model
    processor, model = load_model()

    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            print(f"\nFound image: {image_path}")
            
            try:
                # Process image
                original, mask, paved_percentage, class_percentages = process_image(image_path, processor, model)
                
                # Display results
                if not args.no_display:
                    display_results(original, mask, paved_percentage, class_percentages, model.config)
                
                # Save processed images
                save_processed_images(image_path, output_dir, processor, model)
                
                # Analyze model output if requested
                if args.analyze:
                    analyze_model_output(image_path, processor, model)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == '__main__':
    main() 