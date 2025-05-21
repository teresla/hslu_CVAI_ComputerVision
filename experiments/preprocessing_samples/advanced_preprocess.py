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

def resize_to_match(source, target):
    """Resize source image to match target image dimensions"""
    return cv2.resize(source, (target.shape[1], target.shape[0]), 
                     interpolation=cv2.INTER_AREA)

def lab_histogram_transfer(source, target):
    """Transfer color histogram from target to source image in LAB color space"""
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    # Split channels
    source_l, source_a, source_b = cv2.split(source_lab)
    target_l, target_a, target_b = cv2.split(target_lab)
    
    # Calculate histograms
    hist_source_l = cv2.calcHist([source_l], [0], None, [256], [0, 256])
    hist_target_l = cv2.calcHist([target_l], [0], None, [256], [0, 256])
    
    # Calculate cumulative histograms
    cdf_source_l = hist_source_l.cumsum()
    cdf_target_l = hist_target_l.cumsum()
    
    # Normalize cumulative histograms
    cdf_source_l = cdf_source_l * 255 / cdf_source_l[-1]
    cdf_target_l = cdf_target_l * 255 / cdf_target_l[-1]
    
    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 0
        while j < 256 and cdf_source_l[i] > cdf_target_l[j]:
            j += 1
        lookup_table[i] = j
    
    # Apply lookup table
    source_l = cv2.LUT(source_l, lookup_table)
    
    # Merge channels
    result_lab = cv2.merge([source_l, source_a, source_b])
    
    # Convert back to BGR
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def detect_shadow(image):
    """Detect shadows in the image using LAB color space"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate shadow mask using L channel
    shadow_mask = (l < 128).astype(np.float32)
    
    # Apply Gaussian blur to smooth the mask
    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    
    return shadow_mask

def gaussian_soften(image, sigma=1.2):
    """Apply Gaussian blur to soften the image"""
    return cv2.GaussianBlur(image, (0, 0), sigma)

def main():
    try:
        print("Starting advanced image processing...")
        
        # Define paths
        base_path = Path(__file__).parent
        drone_path = base_path / "drone.png"
        google_path = base_path / "google.png"
        
        # Create output directory
        output_dir = base_path / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Load images
        print("Loading images...")
        drone_img = load_image(drone_path)
        google_img = load_image(google_path)
        
        print(f"Original drone image shape: {drone_img.shape}")
        print(f"Original Google image shape: {google_img.shape}")
        
        # Resize drone image to match Google image dimensions
        print("Resizing drone image...")
        drone_resized = resize_to_match(drone_img, google_img)
        print(f"Resized drone image shape: {drone_resized.shape}")
        
        # Process drone image
        print("Processing drone image...")
        
        # 1. Color transfer
        print("Applying color transfer...")
        drone_color_transferred = lab_histogram_transfer(drone_resized, google_img)
        
        # 2. Shadow detection and softening
        print("Detecting and softening shadows...")
        shadow_mask = detect_shadow(drone_color_transferred)
        shadow_mask_3ch = np.repeat(shadow_mask[:, :, np.newaxis], 3, axis=2)
        drone_shadow_softened = drone_color_transferred / (1 - 0.5 * shadow_mask_3ch)
        drone_shadow_softened = np.clip(drone_shadow_softened, 0, 255).astype(np.uint8)
        
        # 3. Blur
        print("Applying blur...")
        drone_blurred = gaussian_soften(drone_shadow_softened, sigma=1.2)
        
        # Save results
        print("Saving results...")
        cv2.imwrite(str(output_dir / "drone_resized.png"), drone_resized)
        cv2.imwrite(str(output_dir / "drone_color_transferred.png"), drone_color_transferred)
        cv2.imwrite(str(output_dir / "drone_shadow_softened.png"), drone_shadow_softened)
        cv2.imwrite(str(output_dir / "drone_blurred.png"), drone_blurred)
        
        # Create comparison image
        print("Creating comparison image...")
        # Resize all images to the same height for comparison
        height = google_img.shape[0]
        width = google_img.shape[1]
        
        drone_resized = cv2.resize(drone_resized, (width, height))
        drone_color_transferred = cv2.resize(drone_color_transferred, (width, height))
        drone_shadow_softened = cv2.resize(drone_shadow_softened, (width, height))
        drone_blurred = cv2.resize(drone_blurred, (width, height))
        
        comparison = np.hstack([
            drone_resized,
            google_img,
            drone_color_transferred,
            drone_shadow_softened,
            drone_blurred
        ])
        
        cv2.imwrite(str(output_dir / "comparison_advanced.png"), comparison)
        
        print("Processing complete. Check the 'processed' directory for results.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main() 