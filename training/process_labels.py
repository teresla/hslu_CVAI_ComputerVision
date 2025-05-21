import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Simplified class mapping with separate grass class
SIMPLIFIED_CLASSES = {
    0: 'background',  # unlabeled, conflicting, ar-marker, etc.
    1: 'pavement',    # paved-area, gravel, dirt, etc.
    2: 'grass',       # usable grass areas
    3: 'vegetation',  # trees, bushes, other vegetation
    4: 'roof',        # roof
    5: 'water',       # water, pool
    6: 'car',         # car (optional)
}

# New colors for simplified classes (RGB)
SIMPLIFIED_COLORS = {
    0: (0, 0, 0),        # background: black
    1: (128, 64, 128),   # pavement: purple
    2: (0, 255, 0),      # grass: bright green
    3: (0, 102, 0),      # vegetation: dark green
    4: (70, 70, 70),     # roof: dark gray
    5: (28, 42, 168),    # water: blue
    6: (9, 143, 150),    # car: cyan
}

# Original class to simplified class mapping
CLASS_MAPPING = {
    # Background (0)
    'unlabeled': 0,
    'conflicting': 0,
    'ar-marker': 0,
    'obstacle': 0,
    'wall': 0,
    'window': 0,
    'door': 0,
    'fence': 0,
    'fence-pole': 0,
    'person': 0,
    'dog': 0,
    'bicycle': 0,
    
    # Pavement (1)
    'paved-area': 1,
    'gravel': 1,
    'dirt': 1,
    'rocks': 1,
    
    # Grass (2)
    'grass': 2,
    
    # Vegetation (3)
    'vegetation': 3,
    'tree': 3,
    'bald-tree': 3,
    
    # Roof (4)
    'roof': 4,
    
    # Water (5)
    'water': 5,
    'pool': 5,
    
    # Car (6)
    'car': 6,
}

def load_class_dict(csv_path):
    """Load the original class dictionary"""
    return pd.read_csv(csv_path, skipinitialspace=True)

def create_color_mapping(class_dict):
    """Create a mapping from RGB colors to class indices"""
    color_to_class = {}
    for _, row in class_dict.iterrows():
        color = (row['r'], row['g'], row['b'])
        color_to_class[color] = row['name']
    return color_to_class

def remap_label(label_img, color_to_class):
    """Remap the label image to simplified classes using vectorized operations"""
    # Create a mapping from original colors to simplified class indices
    color_to_simplified = {}
    for color, class_name in color_to_class.items():
        if class_name in CLASS_MAPPING:
            color_to_simplified[color] = CLASS_MAPPING[class_name]
        else:
            color_to_simplified[color] = 0
    
    # Initialize output array
    remapped = np.zeros(label_img.shape[:2], dtype=np.uint8)
    
    # Process each unique color in the image
    unique_colors = np.unique(label_img.reshape(-1, 3), axis=0)
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in color_to_simplified:
            # Create a mask for this color
            mask = np.all(label_img == color, axis=-1)
            # Assign the simplified class index
            remapped[mask] = color_to_simplified[color_tuple]
    
    return remapped

def colorize_label(label_img):
    """Convert class indices to RGB colors"""
    height, width = label_img.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_idx, color in SIMPLIFIED_COLORS.items():
        mask = label_img == class_idx
        colored[mask] = color
    
    return colored

def preprocess_image(image, is_label=False, target_size=512):
    """
    Crop → Resize → Blur (only if not label)
    This avoids applying expensive operations on huge images.
    """
    h, w = image.shape[:2]

    # Center crop to square
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    image = image[top:top + min_dim, left:left + min_dim]

    # Resize early
    image = cv2.resize(image, (target_size, target_size),
                       interpolation=cv2.INTER_LINEAR if not is_label else cv2.INTER_NEAREST)

    # Blur only the resized version (much faster)
    if not is_label:
        image = cv2.GaussianBlur(image, (7, 7), 0)

    return image


def process_label(label_path, output_dir, color_to_class):
    """Process a single label image"""
    # Load and remap label
    label_img = cv2.imread(str(label_path))
    if label_img is None:
        print(f"Failed to load {label_path}")
        return
    
    # Convert BGR to RGB
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    
    # Remap to simplified classes
    remapped = remap_label(label_img, color_to_class)
    
    # Colorize the remapped labels
    colored = colorize_label(remapped)
    
    # Apply preprocessing to the label (resize and padding, but no blur)
    preprocessed_label = preprocess_image(colored, is_label=True)
    
    # Save the preprocessed label
    output_path = output_dir / f"simplified_{label_path.name}"
    cv2.imwrite(str(output_path), cv2.cvtColor(preprocessed_label, cv2.COLOR_RGB2BGR))

def process_image(image_path, output_dir):
    """Process a single image"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    # Apply preprocessing (blur and resize)
    preprocessed = preprocess_image(image, is_label=False)
    
    # Save preprocessed image
    output_path = output_dir / f"preprocessed_{image_path.name}"
    cv2.imwrite(str(output_path), preprocessed)

def main():
    try:
        # Define paths
        base_path = Path(__file__).parent.parent
        class_dict_path = base_path / "data/SemanticDrone_groundtruth/semantic/class_dict.csv"
        label_dir = base_path / "data/SemanticDrone_groundtruth/semantic/label_images"
        image_dir = base_path / "data/SemanticDrone_images"
        output_dir = base_path / "data/processed/SemanticDrone_groundtruth"
        output_images_dir = base_path / "data/processed/SemanticDrone_images"
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load class dictionary
        print("Loading class dictionary...")
        class_dict = load_class_dict(class_dict_path)
        color_to_class = create_color_mapping(class_dict)
        
        # Get list of files
        label_files = list(label_dir.glob("*.png"))
        image_files = list(image_dir.glob("*.jpg"))
        
        # Process labels in parallel
        print("Processing label images...")
        with ThreadPoolExecutor() as executor:
            # Create a list of arguments for each task
            label_tasks = [(label_path, output_dir, color_to_class) for label_path in label_files]
            # Process labels in parallel with progress bar
            list(tqdm(
                executor.map(lambda x: process_label(*x), label_tasks),
                total=len(label_tasks),
                desc="Processing labels"
            ))
        
        # Process images in parallel
        print("Processing images...")
        with ThreadPoolExecutor() as executor:
            # Create a list of arguments for each task
            image_tasks = [(image_path, output_images_dir) for image_path in image_files]
            # Process images in parallel with progress bar
            list(tqdm(
                executor.map(lambda x: process_image(*x), image_tasks),
                total=len(image_tasks),
                desc="Processing images"
            ))
        
        # Save the simplified class dictionary
        simplified_dict = pd.DataFrame({
            'name': list(SIMPLIFIED_CLASSES.values()),
            'r': [SIMPLIFIED_COLORS[i][0] for i in range(len(SIMPLIFIED_CLASSES))],
            'g': [SIMPLIFIED_COLORS[i][1] for i in range(len(SIMPLIFIED_CLASSES))],
            'b': [SIMPLIFIED_COLORS[i][2] for i in range(len(SIMPLIFIED_CLASSES))]
        })
        simplified_dict.to_csv(output_dir / "simplified_class_dict.csv", index=False)
        

    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main() 