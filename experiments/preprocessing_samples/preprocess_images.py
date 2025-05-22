import cv2
import numpy as np
from pathlib import Path
import sys

def load_image(image_path):
    """Load an image and return it as a numpy array"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Apply Gaussian blur to an image"""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_median_blur(image, kernel_size=15):
    """Apply median blur to an image"""
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_blur(image, d=15, sigma_color=75, sigma_space=75):
    """Apply bilateral blur to an image (preserves edges better)"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def create_comparison_image(images, titles):
    """Create a side-by-side comparison of images"""
    # Get the height of the first image
    height = images[0].shape[0]
    # Calculate total width
    total_width = sum(img.shape[1] for img in images)
    
    # Create a blank image
    comparison = np.zeros((height, total_width, 3), dtype=np.uint8)
    
    # Place each image side by side
    x_offset = 0
    for img in images:
        comparison[0:height, x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1]
    
    return comparison

def save_image(image, output_path):
    """Save an image to the specified path"""
    cv2.imwrite(str(output_path), image)

def main():
    try:
        print("Starting image processing...")
        
        # Define paths
        base_path = Path(__file__).parent
        drone_path = base_path / "drone.png"
        google_path = base_path / "google.png"
        gt_path = Path("/Users/teres/Documents/GitHub/hslu_CVAI_ComputerVision/data/SemanticDrone_groundtruth/semantic/label_images/002.png")
        
        # Create output directory
        output_dir = base_path / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Load images
        print("Loading images...")
        drone_img = load_image(drone_path)
        google_img = load_image(google_path)
        gt_img = load_image(gt_path)
        
        print(f"Original drone image shape: {drone_img.shape}")
        print(f"Original Google image shape: {google_img.shape}")
        print(f"Ground truth image shape: {gt_img.shape}")
        
        # Resize drone image to match Google image's width while maintaining aspect ratio
        target_width = google_img.shape[1]
        aspect_ratio = drone_img.shape[0] / drone_img.shape[1]
        target_height = int(target_width * aspect_ratio)
        drone_resized = cv2.resize(drone_img, (target_width, target_height))
        print(f"Resized drone image shape: {drone_resized.shape}")
        
        # Resize ground truth to match drone image
        gt_resized = cv2.resize(gt_img, (drone_resized.shape[1], drone_resized.shape[0]))
        
        # Apply bilateral blur to both images
        print("Applying bilateral blur...")
        drone_bilateral = apply_bilateral_blur(drone_resized, d=15, sigma_color=75, sigma_space=75)
        google_bilateral = apply_bilateral_blur(google_img, d=15, sigma_color=75, sigma_space=75)
        
        # Create comparison image
        print("Creating comparison image...")
        comparison = np.hstack([
            drone_resized,     # Original drone
            drone_bilateral,   # Bilateral blur drone
            google_img,        # Original Google
            google_bilateral   # Bilateral blur Google
        ])
        
        # Save comparison image
        save_image(comparison, output_dir / "comparison_bilateral_both.png")
        print("Processing complete. Check the 'processed' directory for results.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main() 