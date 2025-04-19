import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import logging
import random
from sklearn.model_selection import train_test_split

def get_logger(name="seg_logger", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def compute_iou(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()

def compute_dice(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return ((2. * intersection + eps) / (pred.sum() + target.sum() + eps)).item()

class RoadDataset(Dataset):
    """Single‑class road dataset for both training and validation."""

    def __init__(self, image_paths, mask_paths, transform, return_path=False):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 128).astype(np.float32)
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0)
        if self.return_path:
            return image_t, mask_t, self.image_paths[idx]
        return image_t, mask_t


def get_image_mask_pairs(image_dir, mask_dir, extension="*.png"):
    image_paths = sorted(glob(os.path.join(image_dir, extension)))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in image_paths]
    return image_paths, mask_paths

def sample_dataset(images, masks, sample_size, seed=42):
    """Randomly samples N pairs from image + mask lists."""
    random.seed(seed)
    sample = random.sample(list(zip(images, masks)), sample_size)
    return zip(*sample)

def split_dataset(images, masks, test_size=0.2, seed=42):
    """Train/test split."""
    return train_test_split(images, masks, test_size=test_size, random_state=seed)

def prepare_train_test_sets(
    deepglobe_paths,
    suburb_paths,
    generated_paths,
    deepglobe_sample_size=6000,
    suburb_sample_size=100,
    test_split=0.2,
    seed=42):
    # Unpack all paths
    dg_imgs, dg_masks = deepglobe_paths
    sb_imgs, sb_masks = suburb_paths
    gen_imgs, gen_masks = generated_paths

    # Sample DeepGlobe and Suburb
    dg_imgs_s, dg_masks_s = sample_dataset(dg_imgs, dg_masks, deepglobe_sample_size, seed)
    sb_imgs_s, sb_masks_s = sample_dataset(sb_imgs, sb_masks, suburb_sample_size, seed)

    # Split both (test set excludes generated)
    dg_train_i, dg_test_i, dg_train_m, dg_test_m = split_dataset(dg_imgs_s, dg_masks_s, test_split, seed)
    sb_train_i, sb_test_i, sb_train_m, sb_test_m = split_dataset(sb_imgs_s, sb_masks_s, test_split, seed)

    # Generated goes only to train
    train_images = list(dg_train_i) + list(sb_train_i) + list(gen_imgs)
    train_masks = list(dg_train_m) + list(sb_train_m) + list(gen_masks)

    test_images = list(dg_test_i) + list(sb_test_i)
    test_masks = list(dg_test_m) + list(sb_test_m)

    return train_images, train_masks, test_images, test_masks
