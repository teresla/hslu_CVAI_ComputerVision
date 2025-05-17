"""
Evaluation script for an **untuned DeepLab v3+ (ResNet‑50) checkpoint** under *any* torchvision version.
It now handles cases where the weights’ `meta` **do not provide mean / std** (as in the
COCO‑with‑VOC labels file you pasted) by falling back to standard ImageNet statistics.

It still
* auto‑detects a road‑like class (if present) or falls back to “non‑background = road”,
* supports CUDA / CPU / **MPS** via your existing `DEVICE` variable, and
* leaves the rest of your pipeline unchanged.
"""

import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

from baseline_config import *
from baseline_helpers import *



# ---------------------------------------------------------------------------
# 0. SEEDS & LOGGER
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logger = get_logger()

# ---------------------------------------------------------------------------
# 1. DATA SPLIT (unchanged)
# ---------------------------------------------------------------------------

deepglobe  = get_image_mask_pairs(DEEPGLOBE_IMG_DIR, DEEPGLOBE_MASK_DIR)
suburb     = get_image_mask_pairs(SUBURB_IMG_DIR,     SUBURB_MASK_DIR)
generated  = get_image_mask_pairs(GEN_IMG_DIR,        GEN_MASK_DIR)

train_images, train_masks, test_images, test_masks = prepare_train_test_sets(
    deepglobe, suburb, generated,
    deepglobe_sample_size=6000,
    suburb_sample_size=100,
    test_split=TEST_SPLIT,
    seed=SEED,
)
logger.info(
    f"Train {len(train_images)} | Test {len(test_images)} | Device={DEVICE}"
)

# ---------------------------------------------------------------------------
# 2. WEIGHTS & TRANSFORMS
# ---------------------------------------------------------------------------

weights = DeepLabV3_ResNet50_Weights.DEFAULT   # in your install: COCO + VOC 20 classes

# some older snapshots don’t ship mean/std — fall back to ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_MEAN = weights.meta.get("mean", IMAGENET_MEAN)
IMG_STD  = weights.meta.get("std",  IMAGENET_STD)

logger.info(
    f"Using weights: {weights.name} | mean/std = {IMG_MEAN}/{IMG_STD} | classes = {len(weights.meta['categories'])}"
)

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ToTensorV2(),
])

test_ds = RoadTestDataset(test_images, test_masks, transform=transform)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------------

model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(DEVICE)
model.eval()

# locate a "road"‑like index — COCO/VOC weights have none, so expect fallback
road_keywords = ["road", "street", "highway", "route", "pavement"]
categories = [c.lower() for c in weights.meta.get("categories", [])]
road_idx = next((i for i, c in enumerate(categories) if any(k in c for k in road_keywords)), None)

if road_idx is not None:
    logger.info(f"Found road‑like class at index {road_idx}: '{weights.meta['categories'][road_idx]}'")
else:
    logger.warning("No explicit road class in the checkpoint — using non‑background pixels as proxy.")

THRESH = 0.30

# ---------------------------------------------------------------------------
# 4. WANDB
# ---------------------------------------------------------------------------

wandb.init(project="road-segmentation", name="eval_deeplabv3_resnet50_untuned")
wandb.config.update({
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "model": weights.name,
    "checkpoint_classes": len(categories),
    "device": DEVICE,
})

# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------

all_ious, all_dices = [], []
sample_count = 0

with torch.no_grad():
    for images, masks, paths in tqdm(test_loader, desc="Evaluating"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        logits = model(images)["out"]  # [B, C, H, W]
        probs  = torch.softmax(logits, dim=1)

        if road_idx is not None:
            road_prob = probs[:, road_idx : road_idx + 1]
        else:
            road_prob = 1.0 - probs[:, 0:1]  # assume channel‑0 is background

        preds_bin = (road_prob > THRESH).float()

        for i in range(images.size(0)):
            iou  = compute_iou(preds_bin[i], masks[i])
            dice = compute_dice(preds_bin[i], masks[i])
            all_ious.append(iou)
            all_dices.append(dice)

            if sample_count < NUM_SAMPLES_TO_LOG:
                img_np  = images[i].cpu().permute(1, 2, 0).numpy()
                pred_np = preds_bin[i][0].cpu().numpy()
                mask_np = masks[i][0].cpu().numpy()

                wandb.log({
                    f"sample_{sample_count}": [
                        wandb.Image(img_np,  caption="Image"),
                        wandb.Image(pred_np, caption="Predicted Mask"),
                        wandb.Image(mask_np, caption="Ground Truth"),
                    ]
                })
                sample_count += 1

mean_iou  = float(np.mean(all_ious))
mean_dice = float(np.mean(all_dices))

wandb.log({"mean_iou": mean_iou, "mean_dice": mean_dice})
logger.info(f"✅ Evaluation complete — mIoU: {mean_iou:.4f} | Dice: {mean_dice:.4f}")
