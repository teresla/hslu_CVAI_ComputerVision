import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable albumentations update check

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score
import random
import wandb
import cv2

NO_ALBUMENTATIONS_UPDATE = True
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class mappings (matching process_labels.py)
SIMPLIFIED_CLASSES = {
    0: 'background',  # unlabeled, conflicting, ar-marker, etc.
    1: 'pavement',    # paved-area, gravel, dirt, etc.
    2: 'grass',       # usable grass areas
    3: 'vegetation',  # trees, bushes, other vegetation
    4: 'roof',        # roof
    5: 'water',       # water, pool
    6: 'car',         # car (optional)
}

# Colors for visualization (RGB)
SIMPLIFIED_COLORS = {
    0: (0, 0, 0),        # background: black
    1: (128, 64, 128),   # pavement: purple
    2: (0, 255, 0),      # grass: bright green
    3: (0, 102, 0),      # vegetation: dark green
    4: (70, 70, 70),     # roof: dark gray
    5: (28, 42, 168),    # water: blue
    6: (9, 143, 150),    # car: cyan
}

# Dataset class
class DroneDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.processor = processor
        self.transform = transform
        
        # Get all preprocessed files
        self.image_files = sorted([f for f in self.image_dir.glob("preprocessed_*.jpg")])
        self.label_files = sorted([f for f in self.label_dir.glob("simplified_*.png")])
        
        assert len(self.image_files) == len(self.label_files), "Number of images and labels must match"
        logger.info(f"Found {len(self.image_files)} image-label pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load preprocessed image
        image = cv2.imread(str(self.image_files[idx]))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load preprocessed label
        label = cv2.imread(str(self.label_files[idx]))
        if label is None:
            raise ValueError(f"Failed to load label: {self.label_files[idx]}")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        # Convert label to class indices
        label_indices = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        for class_idx, color in SIMPLIFIED_COLORS.items():
            mask = np.all(label == color, axis=-1)
            label_indices[mask] = class_idx
        
        # Apply additional transformations if specified
        if self.transform:
            transformed = self.transform(image=image, mask=label_indices)
            image = transformed['image']
            label_indices = transformed['mask']
        
        # Process image for model
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Convert label_indices to tensor if it's not already
        if isinstance(label_indices, np.ndarray):
            inputs['labels'] = torch.from_numpy(label_indices).long()
        else:
            inputs['labels'] = label_indices.long()
        
        return inputs

# Training function
def train_epoch(model, dataloader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move batch to device
        pixel_values = batch['pixel_values'].squeeze(1).to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Log batch loss to wandb
        wandb.log({
            "batch_loss": loss.item() * accumulation_steps,
            "batch": i
        })
    
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pixel_values = batch['pixel_values'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            # Move tensors to CPU and convert to numpy
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Upsample predictions to match label size
            if preds.shape != labels.shape:
                preds = torch.nn.functional.interpolate(
                    torch.from_numpy(preds).unsqueeze(0).float(),
                    size=labels.shape[1:],
                    mode='nearest'
                ).squeeze(0).numpy().astype(np.uint8)
            
            # Flatten the predictions and labels
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)
            
            # Remove padding (where label is 0)
            mask = labels != 0
            preds = preds[mask]
            labels = labels[mask]
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            total_loss += loss.item()
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_model_parameters(model):
    """Print detailed information about model parameters and their freezing status"""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    logger.info("\nModel Parameter Status:")
    logger.info("-" * 80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            status = "Trainable"
        else:
            frozen_params += param_count
            status = "Frozen"
        
        logger.info(f"{name:<60} | {status:<10} | Params: {param_count:,}")
    
    logger.info("-" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    logger.info("-" * 80)

def unfreeze_layers(model, epoch):
    """Gradually unfreeze layers based on epoch number"""
    if epoch == 0:
        # Initially freeze all layers except the final classifier
        for name, param in model.named_parameters():
            if "decode_head.classifier" not in name:
                param.requires_grad = False
    elif epoch == 1:
        # Unfreeze the last transformer block and decode head classifier
        for name, param in model.named_parameters():
            if ("encoder.block.3" in name or 
                "decode_head.classifier" in name or
                "decode_head.batch_norm" in name):
                param.requires_grad = True
    elif epoch == 2:
        # Unfreeze the last two transformer blocks and more decode head layers
        for name, param in model.named_parameters():
            if ("encoder.block.2" in name or 
                "encoder.block.3" in name or 
                "decode_head.classifier" in name or
                "decode_head.batch_norm" in name or
                "decode_head.linear_fuse" in name or
                "decode_head.linear_c.3" in name):
                param.requires_grad = True
    elif epoch == 3:
        # Unfreeze all transformer blocks and most decode head layers
        for name, param in model.named_parameters():
            if ("encoder.block" in name or 
                "decode_head.classifier" in name or
                "decode_head.batch_norm" in name or
                "decode_head.linear_fuse" in name or
                "decode_head.linear_c" in name):
                param.requires_grad = True
    elif epoch == 4:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

def main():
    # Initialize wandb with more detailed config
    wandb.init(
        project="drone-segmentation",
        name="segformer_finetuning",  # Add a specific run name
        config={
            "learning_rate": 1e-5,
            "batch_size": 1,
            "accumulation_steps": 4,
            "architecture": "Segformer",
            "dataset": "SemanticDrone",
            "epochs": 5,  # Increased epochs for gradual unfreezing
            "early_stopping_patience": 5,
            "image_size": "3000x4500",
            "optimizer": "Adam",
            "scheduler": "LinearLR",
            "num_classes": len(SIMPLIFIED_CLASSES),
            "classes": SIMPLIFIED_CLASSES,
            "device": str(device) if 'device' in locals() else "cpu"
        }
    )
    
    # Set random seed
    set_seed(42)
    
    # Define paths
    base_path = Path(__file__).parent.parent
    image_dir = base_path / "data/processed/SemanticDrone_images"
    label_dir = base_path / "data/processed/SemanticDrone_groundtruth"
    models_dir = base_path / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Define device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    processor = SegformerImageProcessor.from_pretrained("Thalirajesh/Aerial-Drone-Image-Segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "Thalirajesh/Aerial-Drone-Image-Segmentation",
        num_labels=len(SIMPLIFIED_CLASSES),
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Log model architecture and parameters
    wandb.watch(model, log="all", log_freq=10)
    
    # Define transformations (only augmentations, no resizing needed)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
        ToTensorV2(),
    ])
    
    # Create dataset
    dataset = DroneDataset(image_dir, label_dir, processor, transform)
    
    # Use a smaller subset for testing
    subset_size = 401 # Entire dataset  
    if len(dataset) > subset_size:
        logger.info(f"Using subset of {subset_size} images for testing")
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Training loop
    best_val_accuracy = 0
    early_stopping_patience = 5
    patience_counter = 0
    
    for epoch in range(50):  # Increased to 5 epochs for gradual unfreezing
        logger.info(f"\nEpoch {epoch + 1}/5")
        
        # Unfreeze layers based on epoch
        unfreeze_layers(model, epoch)
        logger.info(f"\nLayer freezing status for epoch {epoch + 1}:")
        print_model_parameters(model)
        
        # Create new optimizer for each epoch (since parameters might change)
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-08
        )
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Training Loss: {train_loss:.4f}")
        wandb.log({
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        })
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, device)
        logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch + 1
        })
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save model
            model_save_path = models_dir / "best_segformer"
            model.save_pretrained(model_save_path)
            logger.info(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")
            
            # Log model to wandb
            wandb.save(str(model_save_path))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break
    
    # Close wandb run
    wandb.finish()
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 