import cv2
import numpy as np
from pathlib import Path

def main():
    # Define paths
    base_path = Path(__file__).parent
    drone_path = base_path / "drone.png"
    google_path = base_path / "google.png"
    
    # Create output directory
    output_dir = base_path / "processed"
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    drone_img = cv2.imread(str(drone_path))
    google_img = cv2.imread(str(google_path))
    
    # Resize drone to match Google's width while maintaining aspect ratio
    target_width = google_img.shape[1]
    aspect_ratio = drone_img.shape[0] / drone_img.shape[1]
    target_height = int(target_width * aspect_ratio)
    drone_resized = cv2.resize(drone_img, (target_width, target_height))
    
    # Apply median blur to both images
    drone_blurred = cv2.medianBlur(drone_resized, 5)  # Kernel size 5 for median blur
    google_blurred = cv2.medianBlur(google_img, 5)
    
    # Create side-by-side comparison
    comparison = np.hstack([drone_blurred, google_blurred])
    
    # Save result
    cv2.imwrite(str(output_dir / "median_drone-google.png"), comparison)
    print("Saved median blur comparison to processed/median_drone-google.png")

if __name__ == "__main__":
    main() 