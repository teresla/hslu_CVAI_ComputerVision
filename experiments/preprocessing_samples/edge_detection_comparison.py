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

def apply_sobel_edges(image):
    """Apply Sobel edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Sobel in both x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Combine the results
    sobel = np.sqrt(sobelx**2 + sobely**2)
    # Normalize to 0-255
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def apply_canny_edges(image, threshold1=100, threshold2=200):
    """Apply Canny edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_laplacian_edges(image):
    """Apply Laplacian edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Convert to absolute values
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

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
        print("Starting edge detection comparison...")
        
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
        
        # Resize drone image to match Google image's width while maintaining aspect ratio
        target_width = google_img.shape[1]
        aspect_ratio = drone_img.shape[0] / drone_img.shape[1]
        target_height = int(target_width * aspect_ratio)
        drone_resized = cv2.resize(drone_img, (target_width, target_height))
        print(f"Resized drone image shape: {drone_resized.shape}")
        
        # Apply different edge detection methods to drone image
        print("Applying edge detection to drone image...")
        drone_sobel = apply_sobel_edges(drone_resized)
        drone_canny = apply_canny_edges(drone_resized)
        drone_laplacian = apply_laplacian_edges(drone_resized)
        
        # Apply different edge detection methods to Google image
        print("Applying edge detection to Google image...")
        google_sobel = apply_sobel_edges(google_img)
        google_canny = apply_canny_edges(google_img)
        google_laplacian = apply_laplacian_edges(google_img)
        
        # Create comparison images
        print("Creating comparison images...")
        drone_comparison = np.hstack([
            drone_resized,     # Original
            drone_sobel,       # Sobel
            drone_canny,       # Canny
            drone_laplacian    # Laplacian
        ])
        
        google_comparison = np.hstack([
            google_img,        # Original
            google_sobel,      # Sobel
            google_canny,      # Canny
            google_laplacian   # Laplacian
        ])
        
        # Stack the comparisons vertically
        final_comparison = np.vstack([drone_comparison, google_comparison])
        
        # Save comparison image
        save_image(final_comparison, output_dir / "edge_detection_comparison.png")
        print("Processing complete. Check the 'processed' directory for results.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main() 