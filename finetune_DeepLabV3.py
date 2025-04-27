"""
train_finetune_deeplabv3.py — single‑class (road / background) fine‑tuning script
==================================================================
This replaces the evaluation‑only notebook with a full training pipeline
that can run on **MPS, CUDA or CPU**.  Drop it into your repo, adjust the
`hyper‑params` block if needed, and launch:

```bash
python train_finetune_deeplabv3.py
```

Key features
------------
* **Binary head** (1 logit) replaces the original 21‑class classifier.
* **Mixed Dice + BCE loss** — good for class‑imbalance (<5 % road pixels).
* **Layer‑wise LR** — head learns 10× faster than the frozen backbone for the
  first `N_FREEZE_EPOCHS`, then the whole network fine‑tunes.
* **WandB logging** of losses & metrics; best checkpoint saved as
  `best_deeplabv3_road.pth`.
* **Evaluation loop** re‑uses the helper metrics you already have.
"""

# ---------------------------------------------------------------------------
# 0. Imports & Repro ----------------------------------------------------------------

import os, random, time, math, json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from sklearn.model_selection import KFold
from torch.utils.data import Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

from baseline_config import *   # IMG_SIZE, BATCH_SIZE, SEED, DEVICE …
from baseline_helpers import *  # get_image_mask_pairs, RoadTrainDataset, RoadTestDataset, compute_iou, compute_dice

# ---------------------------------------------------------------------------
# 1. Hyper‑Parameters --------------------------------------------------------
# ---------------------------------------------------------------------------

EPOCHS             = 12
N_FREEZE_EPOCHS    = 7       # freeze backbone for first N epochs
LR_BACKBONE        = 1e-4
LR_HEAD            = 1e-3
WEIGHT_DECAY       = 1e-4
THRESH             = 0.5      # binarise sigmoid output
CHECKPOINT_PATH    = "models/best_deeplabv3_road.pth"
K_FOLDS            = 5
FIXED_SAMPLE_COUNT = 8

# ---------------------------------------------------------------------------
# 2. Repro & Logger ----------------------------------------------------------
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = get_logger()
logger.info(f"Running on {DEVICE}")

# ---------------------------------------------------------------------------
# 3. Data Split -------------------------------------------------------------
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

logger.info(f"Train {len(train_images)} | Test {len(test_images)}")

full_ds = RoadDataset(train_images, train_masks, transform=None)

# choose N fixed indices once so they stay the same for every fold & epoch
FIXED_IDX = random.sample(range(len(full_ds)), FIXED_SAMPLE_COUNT)

# ---------------------------------------------------------------------------
# 4. Transforms & Datasets ---------------------------------------------------
# ---------------------------------------------------------------------------

IMG_MEAN, IMG_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]    # ImageNet

train_transform = A.Compose([
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ToTensorV2(),
])

# K‑fold wrapper ------------------------------------------------------------
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kfold.split(full_ds)):
    train_subset = Subset(full_ds, train_idx)
    val_subset  = Subset(full_ds, val_idx)

    # apply transforms on‑the‑fly via wrapper datasets -------------
    train_ds = RoadDatasetWrapper(train_subset, transform=train_transform)
    val_ds  = RoadDatasetWrapper(val_subset,  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  drop_last=True)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                              shuffle=False)
train_set_this_fold = set(train_idx)
# ---------------------------------------------------------------------------
# 5. Model -------------------------------------------------------------------
# ---------------------------------------------------------------------------

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)

# replace classifier & aux layer with binary heads
model.classifier[-1]      = nn.Conv2d(256, 1, kernel_size=1)
if model.aux_classifier is not None:
    model.aux_classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

model = model.to(DEVICE)

# ---------------------------------------------------------------------------
# 6. Loss, Optimiser, Scheduler ---------------------------------------------
# ---------------------------------------------------------------------------

def dice_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

bce = nn.BCEWithLogitsLoss()

def mixed_loss(pred, target):
    return 0.2 * bce(pred, target) + 0.8 * dice_loss(pred, target)

# param groups
backbone_params = []
head_params     = []
for name, param in model.named_parameters():
    if "backbone" in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)



# freeze backbone for first N epochs
for m in model.backbone.parameters():
    m.requires_grad = False

# ---------------------------------------------------------------------------
# 7. WandB -------------------------------------------------------------------
# ---------------------------------------------------------------------------
wandb.init(project="road-segmentation", name="finetune_deeplabv3_resnet50_binary")
wandb.config.update({
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr_backbone": LR_BACKBONE,
    "lr_head": LR_HEAD,
    "img_size": IMG_SIZE,
    "freeze_epochs": N_FREEZE_EPOCHS,
    "device": DEVICE,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "test_split": TEST_SPLIT,
})

# ---------------------------------------------------------------------------
# 8. Training Loop -----------------------------------------------------------
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_fixed_samples(model, dataset, train_set, epoch):
    model.eval()
    for k, global_idx in enumerate(FIXED_IDX):
        img_t_, mask_t_ = dataset[global_idx]           # tensors, CHW & 1HW

        img_np  = img_t_.permute(1, 2, 0).cpu().numpy()   # HWC
        mask_np = mask_t_.squeeze(0).cpu().numpy()        # HW
        aug     = val_transform(image=img_np, mask=mask_np)

        img_t   = aug["image"].unsqueeze(0).to(DEVICE)    # 1×3×H×W
        mask_r  = aug["mask"]                             # H×W

        pred = (torch.sigmoid(model(img_t)["out"]) > THRESH).float()[0]

        phase = "train" if global_idx in train_set else "val"
        wandb.log({f"{phase}_fixed_{k}": [
            wandb.Image(img_t[0].cpu().permute(1,2,0).numpy(),
                        caption=f"{phase} img"),
            wandb.Image(pred[0].cpu().numpy(),  caption="pred"),
            wandb.Image(mask_r.cpu().numpy(),    caption="gt"),
        ]}, step=epoch)



best_val_loss = math.inf            # unified, typo fixed
best_val_dice   = -float("inf")    # higher = better
es_patience     = 3                # stop after 3 non‑improving epochs
es_wait         = 0                # epochs since last improvement

for epoch in range(1, EPOCHS + 1):
    # ---------- TRAIN ------------------------------------------------------
    model.train()
    tr_loss, tr_iou, tr_dice = 0, [], []

    for images, masks, _ in tqdm(train_loader,
                                 desc=f"Train {epoch}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)["out"]
        loss   = mixed_loss(logits, masks)   # or loss_fn(...)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * images.size(0)

        # metrics -----------------------------------------------------------
        preds_bin = (torch.sigmoid(logits) > THRESH).float()
        for i in range(preds_bin.size(0)):
            tr_iou.append(compute_iou (preds_bin[i], masks[i]))
            tr_dice.append(compute_dice(preds_bin[i], masks[i]))

    tr_loss  /= len(train_loader.dataset)
    miou_tr   = np.mean(tr_iou)
    dice_tr   = np.mean(tr_dice)

    # ---------- VALIDATION --------------------------------------------------
    model.eval()
    val_loss, v_iou, v_dice = 0, [], []

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Val"):
            images, masks = images.to(DEVICE), masks.float().to(DEVICE)
            logits   = model(images)["out"]
            val_loss += mixed_loss(logits, masks).item() * images.size(0)

            preds_bin = (torch.sigmoid(logits) > THRESH).float()
            for i in range(preds_bin.size(0)):
                v_iou.append(compute_iou(preds_bin[i], masks[i]))
                v_dice.append(compute_dice(preds_bin[i], masks[i]))


    val_loss /= len(val_loader.dataset)
    miou_val  = np.mean(v_iou)
    dice_val  = np.mean(v_dice)

    # ---------- HOUSEKEEPING -----------------------------------------------
    if epoch == N_FREEZE_EPOCHS:
        for p in model.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen.")

    scheduler.step(val_loss)
    current_lr = scheduler.get_last_lr()[0]

    wandb.log({
        "epoch":      epoch,
        "lr":         current_lr,
        "train_loss": tr_loss,
        "train_miou": miou_tr,
        "train_dice": dice_tr,
        "val_loss":   val_loss,
        "val_miou":   miou_val,
        "val_dice":   dice_val,
    })

    logger.info(
        f"Epoch {epoch}: "
        f"train {tr_loss:.4f} (mIoU {miou_tr:.3f}, Dice {dice_tr:.3f}) | "
        f"val {val_loss:.4f} (mIoU {miou_val:.3f}, Dice {dice_val:.3f})"
    )
    log_fixed_samples(model, full_ds, train_set_this_fold, epoch)

    # ---------- CHECKPOINT --------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        logger.info(f"✅ Saved new best model @ {CHECKPOINT_PATH}")

    # ---------- EARLY‑STOPPING on validation Dice -----------------------------
    if dice_val > best_val_dice + 1e-6:          # tiny epsilon to avoid ties
        best_val_dice = dice_val
        es_wait = 0
    else:
        es_wait += 1
        logger.info(f"No val‑Dice improvement for {es_wait}/{es_patience} epoch(s)")

    if es_wait >= es_patience:
        logger.info(f"⏹ Early stopping triggered at epoch {epoch} "
                    f"(best val‑Dice = {best_val_dice:.4f})")
        break


logger.info("Training complete. Best val loss = %.4f", best_val_loss)
