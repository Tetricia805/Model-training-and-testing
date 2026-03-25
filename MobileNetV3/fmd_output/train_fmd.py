import os
import cv2
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import onnx
import onnxruntime as ort

# ---------------------------------------------------------
# 1. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ---------------------------------------------------------
CONFIG = {
    "seed": 42,
    "input_size": 224,
    "batch_size": 32,
    "num_epochs_phase1": 5,   # Freeze backbone
    "num_epochs_phase2": 15,  # Unfreeze partial
    "num_epochs_phase3": 20,  # Full fine-tuning
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "data_dir": ".",          # Script is now inside AniLinkData2
    "output_dir": "fmd_output",
    "dropout_rate": 0.3,
    "blur_threshold": 100.0,
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Ensure reproducibility
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# ---------------------------------------------------------
# 2. UTILITY FUNCTIONS (Cleaning & Preparation)
# ---------------------------------------------------------

def extract_farm_id(filename):
    """
    Extracts Farm ID from filename. 
    Source logic: Handles various patterns found in the dataset.
    """
    parts = filename.split('-')
    if len(parts) > 1: return parts[0]
    if "farm_" in filename:
        idx = filename.find("farm_")
        end_idx = filename.find("_", idx + 5)
        return filename[idx:end_idx] if end_idx != -1 else filename[idx:]
    return "unknown_farm"

def generate_catalog(data_dir):
    """
    Scans data_dir, identifies categories, and returns a catalog DataFrame.
    """
    data = []
    categories = {
        "cattle_healthy": {"gatekeeper": 1, "diagnostic": 0},
        "cattle_infected": {"gatekeeper": 1, "diagnostic": 1},
        "not_cattle_animals": {"gatekeeper": 0, "diagnostic": -1},
        "not_cattle_environment": {"gatekeeper": 0, "diagnostic": -1},
        "not_cattle_human": {"gatekeeper": 0, "diagnostic": -1},
        "not_cattle_text_images": {"gatekeeper": 0, "diagnostic": -1}
    }
    
    print("--- Phase 1: Data Cataloging ---")
    for cat_dir, labels in categories.items():
        full_path = Path(data_dir) / cat_dir
        if not full_path.exists():
            print(f"Warning: Directory {cat_dir} not found.")
            continue
            
        files = list(full_path.glob("*.jpg")) + list(full_path.glob("*.jpeg")) + list(full_path.glob("*.png"))
        print(f"Scanning {cat_dir}: {len(files)} files found.")
        
        for f in tqdm(files, desc=f"Processing {cat_dir}"):
            farm_id = extract_farm_id(f.name)
            is_early = 1 if (cat_dir == "cattle_infected" and any(k in f.name.lower() for k in ["1 day", "early", "vesicle"])) else 0
            
            data.append({
                "path": str(f),
                "farm_id": farm_id,
                "gatekeeper_label": labels["gatekeeper"],
                "diagnostic_label": labels["diagnostic"],
                "is_early_stage": is_early,
                "category": cat_dir
            })
            
    return pd.DataFrame(data)

# ---------------------------------------------------------
# 3. LOSS FUNCTIONS
# ---------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ---------------------------------------------------------
# 4. DATASET & ARCHITECTURE (MobileNetV3 + CBAM)
# ---------------------------------------------------------

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.ca(x)
        x = x * avg_out
        # Spatial Attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sam = self.sa(torch.cat([avg_s, max_s], dim=1))
        return x * sam

class MultiTaskFMDModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MultiTaskFMDModel, self).__init__()
        backbone = models.mobilenet_v3_small(weights='DEFAULT')
        self.features = backbone.features
        # MobileNetV3 small last feature layer: features[12] output is 576 channels
        feature_dim = 576 
        self.cbam = CBAM(feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Heads
        self.gatekeeper_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 2)
        )
        self.diagnostic_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.gatekeeper_head(x), self.diagnostic_head(x)

# ---------------------------------------------------------
# 4. DATA LOADER & TRANSFORMS
# ---------------------------------------------------------

class FMDDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, {
            "gk": torch.tensor(row['gatekeeper_label'], dtype=torch.long),
            "dx": torch.tensor(row['diagnostic_label'], dtype=torch.long)
        }

def get_transforms(phase="train"):
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ---------------------------------------------------------
# 5. GRAD-CAM (EXPLAINABILITY)
# ---------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, target_index=1):
        self.model.eval()
        gk_logits, dx_logits = self.model(input_image)
        
        # We target the diagnostic head (FMD vs Healthy)
        self.model.zero_grad()
        dx_logits[0, target_index].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

def overlay_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, colored_heatmap, 0.4, 0)
    return overlay

# ---------------------------------------------------------
# 6. INFERENCE & PREDICTON
# ---------------------------------------------------------

class FMDPredictor:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = MultiTaskFMDModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device).eval()
        self.transform = get_transforms("val")
        
    def predict(self, image_path, threshold=0.5):
        orig_img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(orig_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gk_logits, dx_logits = self.model(input_tensor)
            gk_probs = F.softmax(gk_logits, dim=1)
            dx_probs = F.softmax(dx_logits, dim=1)
            
        gk_conf, gk_class = torch.max(gk_probs, 1)
        # Rejection Logic
        if gk_class.item() == 0:
            return {"status": "NOT_CATTLE", "confidence": gk_conf.item()}
            
        dx_conf_infected = dx_probs[0, 1].item()
        diagnosis = "FMD_POSITIVE" if dx_conf_infected > threshold else "HEALTHY"
        
        return {
            "status": "CATTLE_DETECTED",
            "diagnosis": diagnosis,
            "fmd_probability": dx_conf_infected,
            "gk_confidence": gk_conf.item()
        }

# ---------------------------------------------------------
# 7. TRAINING & EXPORT ENGINE
# ---------------------------------------------------------

class FMDPipeline:
    def __init__(self):
        self.config = CONFIG
        output_dir = str(CONFIG["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        
    def export_model(self, model):
        """Exports the model to ONNX for mobile deployment."""
        print("\n--- Phase 4: Exporting Model ---")
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(CONFIG["device"])
        output_dir = str(CONFIG["output_dir"])
        onnx_path = os.path.join(output_dir, "fmd_multitask.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, 
                         input_names=['input'], 
                         output_names=['gatekeeper', 'diagnostic'],
                         dynamic_axes={'input': {0: 'batch_size'}},
                         opset_version=12)
        print(f"Model exported to {onnx_path}")

    def validate(self, model, loader, criterion_gk, criterion_dx):
        model.eval()
        total_loss = 0
        all_gk_preds, all_gk_targets = [], []
        all_dx_preds, all_dx_targets = [], []
        
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Validating"):
                imgs = imgs.to(CONFIG["device"])
                gk_targets = lbls["gk"].to(CONFIG["device"])
                dx_targets = lbls["dx"].to(CONFIG["device"])
                
                gk_logits, dx_logits = model(imgs)
                
                loss_gk = criterion_gk(gk_logits, gk_targets)
                mask = (gk_targets == 1)
                if mask.any():
                    # Use Focal for validation
                    loss_dx = criterion_dx(dx_logits[mask], dx_targets[mask])
                    loss = 0.3 * loss_gk + 0.7 * loss_dx
                else:
                    loss = loss_gk
                    
                total_loss += loss.item() if torch.is_tensor(loss) else loss
                
                all_gk_preds.extend(torch.argmax(gk_logits, dim=1).cpu().numpy())
                all_gk_targets.extend(gk_targets.cpu().numpy())
                
                if mask.any():
                    all_dx_preds.extend(torch.argmax(dx_logits[mask], dim=1).cpu().numpy())
                    all_dx_targets.extend(dx_targets[mask].cpu().numpy())
                    
        metrics = {
            "loss": total_loss / len(loader),
            "gk_acc": (np.array(all_gk_preds) == np.array(all_gk_targets)).mean(),
            "dx_f1": f1_score(all_dx_targets, all_dx_preds, zero_division=0) if all_dx_targets else 0
        }
        return metrics

    def run(self):
        df = generate_catalog(CONFIG["data_dir"])
        if len(df) == 0:
            print("Error: No data found.")
            return

        # Split based on FarmID to prevent leakage
        farms = df['farm_id'].unique()
        tr_farms, val_farms = train_test_split(farms, test_size=0.2, random_state=42)
        
        train_df = df[df['farm_id'].isin(tr_farms)]
        val_df = df[df['farm_id'].isin(val_farms)]
        
        train_loader = DataLoader(FMDDataset(train_df, get_transforms("train")), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(FMDDataset(val_df, get_transforms("val")), batch_size=CONFIG["batch_size"], num_workers=4)
        
        model = MultiTaskFMDModel().to(CONFIG["device"])
        
        # Calculate class weights for DX head to handle imbalance
        infected_count = len(train_df[train_df['diagnostic_label'] == 1])
        healthy_count = len(train_df[train_df['diagnostic_label'] == 0])
        dx_weight = torch.tensor([1.0, healthy_count/infected_count]).to(CONFIG["device"])
        
        criterion_gk = nn.CrossEntropyLoss()
        criterion_dx = FocalLoss(alpha=0.25, gamma=2.0)
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
        
        print(f"Training on {len(train_df)} images. Validating on {len(val_df)} images.")
        print(f"DX Imbalance Ratio: {healthy_count/infected_count:.2f}")
        
        best_f1 = 0
        history = []
        
        # Combined Phase Strategy
        total_epochs = int(CONFIG["num_epochs_phase2"]) + int(CONFIG["num_epochs_phase3"])
        
        for epoch in range(total_epochs):
            if epoch == int(CONFIG["num_epochs_phase2"]):
                print("\n>>> Phase 3: Unfreezing All Layers & Reducing LR")
                for param in model.parameters(): param.requires_grad = True
                for g in optimizer.param_groups: g['lr'] = 1e-4
                
            model.train()
            train_loss = 0
            for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                imgs = imgs.to(CONFIG["device"])
                gk_targets = lbls["gk"].to(CONFIG["device"])
                dx_targets = lbls["dx"].to(CONFIG["device"])
                
                optimizer.zero_grad()
                gk_logits, dx_logits = model(imgs)
                
                loss_gk = criterion_gk(gk_logits, gk_targets)
                mask = (gk_targets == 1)
                
                if mask.any():
                    # Diagnostic loss with Focal Loss
                    loss_dx = criterion_dx(dx_logits[mask], dx_targets[mask])
                    # Weight Diagnostic task higher to ensure F1 improvement
                    loss = 0.3 * loss_gk + 0.7 * loss_dx
                else:
                    loss = loss_gk
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Real Validation
            val_metrics = self.validate(model, val_loader, criterion_gk, criterion_dx)
            print(f"--- Epoch {epoch+1} Results ---")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"GK Acc:     {val_metrics['gk_acc']:.4f}")
            print(f"DX F1:      {val_metrics['dx_f1']:.4f}")
            
            history.append(val_metrics)
            
            if val_metrics['dx_f1'] > best_f1:
                best_f1 = val_metrics['dx_f1']
                best_model_path = os.path.join(str(CONFIG["output_dir"]), "best_fmd_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(">> Saved New Best Model!")
                
        self.export_model(model)
        self.plot_results(history)

    def plot_results(self, history):
        """Generates and saves training visualization plots."""
        print("\n--- Phase 5: Generating Visualizations ---")
        plt.figure(figsize=(12, 5))
        
        epochs = list(range(1, len(history) + 1))
        accs = [h['gk_acc'] for h in history]
        f1s = [h['dx_f1'] for h in history]
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, accs, 'b-o', label='GK Accuracy')
        plt.title('Gatekeeper Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, f1s, 'r-o', label='DX F1 Score')
        plt.title('Diagnostic F1 Score')
        plt.legend()
        
        plot_path = os.path.join(str(CONFIG["output_dir"]), "training_history.png")
        plt.savefig(plot_path)
        print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    pipeline = FMDPipeline()
    pipeline.run()
