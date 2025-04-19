# config.py
import os

# Data directories
DATA_DIR = "data"
DEEPGLOBE_IMG_DIR = os.path.join(DATA_DIR, "DeepGlobe_images")
DEEPGLOBE_MASK_DIR = os.path.join(DATA_DIR, "DeepGlobe_groundtruth")
SUBURB_IMG_DIR = os.path.join(DATA_DIR, "USSuburb_images")
SUBURB_MASK_DIR = os.path.join(DATA_DIR, "USSuburb_groundtruth")
GEN_IMG_DIR = os.path.join(DATA_DIR, "USSuburbGen_images")
GEN_MASK_DIR = os.path.join(DATA_DIR, "USSuburbGen_groundtruth")

IMG_SIZE = 256
BATCH_SIZE = 8
TEST_SPLIT = 0.2
SEED = 42
NUM_SAMPLES_TO_LOG = 5

DEVICE = "mps" if hasattr(__import__('torch').backends, "mps") and __import__('torch').backends.mps.is_available() else "cpu"
