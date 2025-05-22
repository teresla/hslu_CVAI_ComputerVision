import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your model and processor loading functions
from web_app.app import custom_model, custom_processor, process_image

def dice_coefficient(pred, target, eps=1e-6):
    intersection = np.sum((pred > 0) & (target > 0))
    return (2. * intersection + eps) / (np.sum(pred > 0) + np.sum(target > 0) + eps)

def iou_score(pred, target, eps=1e-6):
    intersection = np.sum((pred > 0) & (target > 0))
    union = np.sum((pred > 0) | (target > 0))
    return (intersection + eps) / (union + eps)

def precision_score(pred, target, eps=1e-6):
    tp = np.sum((pred > 0) & (target > 0))
    fp = np.sum((pred > 0) & (target == 0))
    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, target, eps=1e-6):
    tp = np.sum((pred > 0) & (target > 0))
    fn = np.sum((pred == 0) & (target > 0))
    return (tp + eps) / (tp + fn + eps)

def f1_score(pred, target, eps=1e-6):
    prec = precision_score(pred, target, eps)
    rec = recall_score(pred, target, eps)
    return 2 * (prec * rec) / (prec + rec + eps)

def main():
    image_dir = 'data/USSuburb_images'
    mask_dir = 'data/USSuburb_groundtruth'
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = custom_model.to(device)
    processor = custom_processor
    model.eval()

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    print(f"Evaluating on {len(image_files[:10])} images...")
    for idx, fname in enumerate(image_files[:10]):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            print(f"No ground truth for {fname}, skipping.")
            continue

        # Load image and mask
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 128).astype(np.uint8)  # binarize ground truth

        # Preprocess and predict
        inputs, original_shape = process_image(image_rgb, processor)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # Assume class 1 is 'pavement' or 'street' in your model
        pred_bin = (pred_mask == 1).astype(np.uint8)

        # Resize prediction to match ground truth
        pred_bin_resized = cv2.resize(pred_bin, (mask_bin.shape[1], mask_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Metrics
        dice = dice_coefficient(pred_bin_resized, mask_bin)
        iou = iou_score(pred_bin_resized, mask_bin)
        prec = precision_score(pred_bin_resized, mask_bin)
        rec = recall_score(pred_bin_resized, mask_bin)
        f1 = f1_score(pred_bin_resized, mask_bin)

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

        print(f"{fname}: Dice={dice:.4f}, IoU={iou:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    print("\nMean metrics over all evaluated images:")
    print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
    print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall:    {np.mean(recall_scores):.4f}")
    print(f"Mean F1:        {np.mean(f1_scores):.4f}")

if __name__ == '__main__':
    main() 