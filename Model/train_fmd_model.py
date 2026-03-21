"""
🐄 Cattle FMD Detection Model - Complete Training Script
Production-ready training automation for PyTorch model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================
# CONFIGURATION
# ============================================
class Config:
    """Training configuration"""
    # Paths
    DATA_ROOT = Path('.')
    CATTLE_HEALTHY_DIR = DATA_ROOT / 'cattle_healthy'
    CATTLE_INFECTED_DIR = DATA_ROOT / 'cattle_infected'
    NON_CATTLE_DIR = DATA_ROOT / 'not_cattle_animals'
    OUTPUT_DIR = Path('pipeline_output')
    MODELS_DIR = OUTPUT_DIR / 'models'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    LOGS_DIR = OUTPUT_DIR / 'logs'
    
    # Data
    IMAGE_SIZE = 224
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Training - Phase 1
    PHASE1_EPOCHS = 20
    PHASE1_LR = 0.001
    PHASE1_BATCH_SIZE = 32
    
    # Training - Phase 2
    PHASE2_EPOCHS = 15
    PHASE2_LR = 0.0001
    PHASE2_BATCH_SIZE = 32
    
    # General
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_BEST_ONLY = True
    PATIENCE = 5  # Early stopping patience

# ============================================
# LOGGING SETUP
# ============================================
def setup_logging():
    """Setup logging configuration"""
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = Config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()

# ============================================
# DATA PREPARATION
# ============================================
class CattleDataset(Dataset):
    """Custom dataset for cattle health detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Tuple of (id_label, diag_label) for each image
            transform: Albumentations transform
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        id_label, diag_label = self.labels[idx]
        
        return {
            'image': image,
            'id_label': torch.tensor(id_label, dtype=torch.long),
            'diag_label': torch.tensor(diag_label, dtype=torch.long),
            'path': str(image_path)
        }

def get_augmentation_transforms(stage='train'):
    """Get augmentation transforms for different stages"""
    if stage == 'train':
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def prepare_dataset():
    """Prepare dataset from directory structure"""
    logger.info("Preparing dataset...")
    
    image_paths = []
    id_labels = []      # 0 = Cattle, 1 = Not-Cattle
    diag_labels = []    # 0 = Healthy, 1 = FMD
    
    # Healthy cattle (Cattle + Healthy)
    if Config.CATTLE_HEALTHY_DIR.exists():
        for img_path in Config.CATTLE_HEALTHY_DIR.glob('*.jpg') + Config.CATTLE_HEALTHY_DIR.glob('*.png'):
            image_paths.append(img_path)
            id_labels.append(0)      # Cattle
            diag_labels.append(0)    # Healthy
    
    # Infected cattle (Cattle + FMD)
    if Config.CATTLE_INFECTED_DIR.exists():
        for img_path in Config.CATTLE_INFECTED_DIR.glob('*.jpg') + Config.CATTLE_INFECTED_DIR.glob('*.png'):
            image_paths.append(img_path)
            id_labels.append(0)      # Cattle
            diag_labels.append(1)    # FMD
    
    # Non-cattle animals
    if Config.NON_CATTLE_DIR.exists():
        for img_path in Config.NON_CATTLE_DIR.glob('*.jpg') + Config.NON_CATTLE_DIR.glob('*.png'):
            image_paths.append(img_path)
            id_labels.append(1)      # Not-Cattle
            diag_labels.append(0)    # No diagnosis for non-cattle
    
    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"  Healthy cattle: {sum((id_labels[i] == 0 and diag_labels[i] == 0) for i in range(len(image_paths)))}")
    logger.info(f"  Infected cattle: {sum((id_labels[i] == 0 and diag_labels[i] == 1) for i in range(len(image_paths)))}")
    logger.info(f"  Non-cattle: {sum(id_labels[i] == 1 for i in range(len(image_paths)))}")
    
    return image_paths, id_labels, diag_labels

def split_dataset(image_paths, id_labels, diag_labels):
    """Split dataset into train/val/test"""
    logger.info("Splitting dataset...")
    
    total = len(image_paths)
    train_size = int(total * Config.TRAIN_SPLIT)
    val_size = int(total * Config.VAL_SPLIT)
    
    # Indices
    indices = np.arange(total)
    np.random.seed(Config.RANDOM_SEED)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split
    train_paths = [image_paths[i] for i in train_indices]
    train_ids = [id_labels[i] for i in train_indices]
    train_diags = [diag_labels[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_ids = [id_labels[i] for i in val_indices]
    val_diags = [diag_labels[i] for i in val_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_ids = [id_labels[i] for i in test_indices]
    test_diags = [diag_labels[i] for i in test_indices]
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return (
        (train_paths, list(zip(train_ids, train_diags))),
        (val_paths, list(zip(val_ids, val_diags))),
        (test_paths, list(zip(test_ids, test_diags)))
    )

def create_dataloaders():
    """Create train/val/test dataloaders"""
    logger.info("Creating dataloaders...")
    
    image_paths, id_labels, diag_labels = prepare_dataset()
    train_data, val_data, test_data = split_dataset(image_paths, id_labels, diag_labels)
    
    # Create datasets
    train_transform = get_augmentation_transforms('train')
    val_transform = get_augmentation_transforms('val')
    
    train_dataset = CattleDataset(train_data[0], train_data[1], transform=train_transform)
    val_dataset = CattleDataset(val_data[0], val_data[1], transform=val_transform)
    test_dataset = CattleDataset(test_data[0], test_data[1], transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info("Dataloaders created successfully")
    
    return train_loader, val_loader, test_loader

# ============================================
# MODEL ARCHITECTURE
# ============================================
class CattleMultiTaskModel(nn.Module):
    """Dual-head model for cattle identification and FMD diagnosis"""
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Load backbone
        if pretrained:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[0].in_features
        
        # Replace classifier with identity
        self.backbone.classifier = nn.Identity()
        
        # Identification head (Cattle vs Non-Cattle)
        self.id_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Diagnosis head (Healthy vs FMD)
        self.diag_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Dual head outputs
        id_output = self.id_head(features)
        diag_output = self.diag_head(features)
        
        return id_output, diag_output
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

# ============================================
# TRAINING FUNCTIONS
# ============================================
class LossMeter:
    """Meter for tracking losses"""
    def __init__(self):
        self.id_loss = []
        self.diag_loss = []
        self.total_loss = []
    
    def update(self, losses):
        self.id_loss.append(losses['id_loss'])
        self.diag_loss.append(losses['diag_loss'])
        self.total_loss.append(losses['total_loss'])
    
    def avg(self):
        return {
            'id_loss': np.mean(self.id_loss),
            'diag_loss': np.mean(self.diag_loss),
            'total_loss': np.mean(self.total_loss)
        }
    
    def reset(self):
        self.id_loss = []
        self.diag_loss = []
        self.total_loss = []

class MetricsMeter:
    """Meter for tracking metrics"""
    def __init__(self):
        self.id_preds = []
        self.id_targets = []
        self.diag_preds = []
        self.diag_targets = []
    
    def update(self, preds, targets):
        self.id_preds.extend(preds['id'].cpu().numpy())
        self.id_targets.extend(targets['id'].cpu().numpy())
        self.diag_preds.extend(preds['diag'].cpu().numpy())
        self.diag_targets.extend(targets['diag'].cpu().numpy())
    
    def compute(self):
        id_acc = accuracy_score(self.id_targets, self.id_preds)
        diag_acc = accuracy_score(self.diag_targets, self.diag_preds)
        
        return {
            'id_accuracy': id_acc,
            'diag_accuracy': diag_acc
        }
    
    def reset(self):
        self.id_preds = []
        self.id_targets = []
        self.diag_preds = []
        self.diag_targets = []

def train_epoch(model, train_loader, criterion_id, criterion_diag, optimizer, device):
    """Train for one epoch"""
    model.train()
    loss_meter = LossMeter()
    metrics_meter = MetricsMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        id_labels = batch['id_label'].to(device)
        diag_labels = batch['diag_label'].to(device)
        
        # Forward pass
        id_logits, diag_logits = model(images)
        
        # Loss
        id_loss = criterion_id(id_logits, id_labels)
        diag_loss = criterion_diag(diag_logits, diag_labels)
        total_loss = id_loss + diag_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track loss
        loss_meter.update({
            'id_loss': id_loss.item(),
            'diag_loss': diag_loss.item(),
            'total_loss': total_loss.item()
        })
        
        # Track metrics
        id_preds = id_logits.argmax(1)
        diag_preds = diag_logits.argmax(1)
        
        metrics_meter.update(
            {'id': id_preds, 'diag': diag_preds},
            {'id': id_labels, 'diag': diag_labels}
        )
        
        pbar.set_postfix(loss_meter.avg())
    
    metrics = metrics_meter.compute()
    metrics.update(loss_meter.avg())
    
    return metrics

def validate(model, val_loader, criterion_id, criterion_diag, device):
    """Validate model"""
    model.eval()
    loss_meter = LossMeter()
    metrics_meter = MetricsMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            images = batch['image'].to(device)
            id_labels = batch['id_label'].to(device)
            diag_labels = batch['diag_label'].to(device)
            
            # Forward pass
            id_logits, diag_logits = model(images)
            
            # Loss
            id_loss = criterion_id(id_logits, id_labels)
            diag_loss = criterion_diag(diag_logits, diag_labels)
            total_loss = id_loss + diag_loss
            
            # Track loss
            loss_meter.update({
                'id_loss': id_loss.item(),
                'diag_loss': diag_loss.item(),
                'total_loss': total_loss.item()
            })
            
            # Track metrics
            id_preds = id_logits.argmax(1)
            diag_preds = diag_logits.argmax(1)
            
            metrics_meter.update(
                {'id': id_preds, 'diag': diag_preds},
                {'id': id_labels, 'diag': diag_labels}
            )
            
            pbar.set_postfix(loss_meter.avg())
    
    metrics = metrics_meter.compute()
    metrics.update(loss_meter.avg())
    
    return metrics

def test(model, test_loader, device):
    """Test model"""
    model.eval()
    metrics_meter = MetricsMeter()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for batch in pbar:
            images = batch['image'].to(device)
            id_labels = batch['id_label'].to(device)
            diag_labels = batch['diag_label'].to(device)
            
            # Forward pass
            id_logits, diag_logits = model(images)
            
            # Predictions
            id_preds = id_logits.argmax(1)
            diag_preds = diag_logits.argmax(1)
            
            metrics_meter.update(
                {'id': id_preds, 'diag': diag_preds},
                {'id': id_labels, 'diag': diag_labels}
            )
    
    metrics = metrics_meter.compute()
    
    return metrics

# ============================================
# TRAINING PIPELINE
# ============================================
def train_phase1(model, train_loader, val_loader, device):
    """Phase 1: Train heads with frozen backbone"""
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Training Heads (Frozen Backbone)")
    logger.info("="*60)
    
    model.freeze_backbone()
    
    # Optimizer - only optimize heads
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=Config.PHASE1_LR
    )
    
    # Loss functions
    criterion_id = nn.CrossEntropyLoss()
    criterion_diag = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(Config.PHASE1_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{Config.PHASE1_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion_id, criterion_diag, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion_id, criterion_diag, device)
        
        # Log
        logger.info(f"Train - ID Acc: {train_metrics['id_accuracy']:.4f}, Diag Acc: {train_metrics['diag_accuracy']:.4f}")
        logger.info(f"Val   - ID Acc: {val_metrics['id_accuracy']:.4f}, Diag Acc: {val_metrics['diag_accuracy']:.4f}")
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        val_acc = (val_metrics['id_accuracy'] + val_metrics['diag_accuracy']) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = Config.MODELS_DIR / 'model_phase1_best.pt'
            Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"\nPhase 1 completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return history

def train_phase2(model, train_loader, val_loader, device):
    """Phase 2: Fine-tune full model"""
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Fine-tuning Full Model")
    logger.info("="*60)
    
    # Load best phase1 model
    model_path = Config.MODELS_DIR / 'model_phase1_best.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded Phase 1 model from {model_path}")
    
    model.unfreeze_backbone()
    
    # Optimizer - optimize all parameters
    optimizer = optim.Adam(model.parameters(), lr=Config.PHASE2_LR)
    
    # Loss functions
    criterion_id = nn.CrossEntropyLoss()
    criterion_diag = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(Config.PHASE2_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{Config.PHASE2_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion_id, criterion_diag, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion_id, criterion_diag, device)
        
        # Log
        logger.info(f"Train - ID Acc: {train_metrics['id_accuracy']:.4f}, Diag Acc: {train_metrics['diag_accuracy']:.4f}")
        logger.info(f"Val   - ID Acc: {val_metrics['id_accuracy']:.4f}, Diag Acc: {val_metrics['diag_accuracy']:.4f}")
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        val_acc = (val_metrics['id_accuracy'] + val_metrics['diag_accuracy']) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = Config.MODELS_DIR / 'model_phase2_best.pt'
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"\nPhase 2 completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return history

def main():
    """Main training pipeline"""
    logger.info("\n" + "="*60)
    logger.info("CATTLE FMD DETECTION - MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"Random seed: {Config.RANDOM_SEED}")
    
    # Set random seeds
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Create model
    logger.info("Creating model...")
    model = CattleMultiTaskModel(pretrained=True, freeze_backbone=True).to(Config.DEVICE)
    logger.info(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: Train heads
    history_phase1 = train_phase1(model, train_loader, val_loader, Config.DEVICE)
    
    # Phase 2: Fine-tune
    history_phase2 = train_phase2(model, train_loader, val_loader, Config.DEVICE)
    
    # Load best final model
    model_path = Config.MODELS_DIR / 'model_phase2_best.pt'
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    
    # Test
    logger.info("\n" + "="*60)
    logger.info("TESTING")
    logger.info("="*60)
    
    test_metrics = test(model, test_loader, Config.DEVICE)
    logger.info(f"Test - ID Accuracy: {test_metrics['id_accuracy']:.4f}, Diag Accuracy: {test_metrics['diag_accuracy']:.4f}")
    
    # Save final model
    final_model_path = Config.MODELS_DIR / 'model_final.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"\nFinal model saved to {final_model_path}")
    
    # Save training history
    history = {
        'phase1': history_phase1,
        'phase2': history_phase2,
        'test_metrics': test_metrics
    }
    
    history_path = Config.RESULTS_DIR / 'training_history.json'
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    history_json = {
        'phase1': {
            'train': [
                {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in h.items()}
                for h in history['phase1']['train']
            ],
            'val': [
                {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in h.items()}
                for h in history['phase1']['val']
            ]
        },
        'phase2': {
            'train': [
                {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in h.items()}
                for h in history['phase2']['train']
            ],
            'val': [
                {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in h.items()}
                for h in history['phase2']['val']
            ]
        },
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)

if __name__ == "__main__":
    main()
