import os
import re
import cv2
import glob
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "num_workers": 4, # Fallback to 4 to be safer, was 6
    "img_size": 224,
    "epochs": 25,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 10,
    "rejection_threshold": 0.5,
    "categories": {
        "cattle_healthy": (1, 0, "healthy"),
        "cattle_infected": (1, 1, "fmd"),
        "not_cattle_animals": (0, -1, "non_cattle"),
        "not_cattle_environment": (0, -1, "non_cattle"),
        "not_cattle_human": (0, -1, "non_cattle"),
        "not_cattle_text_images": (0, -1, "non_cattle"),
    },
    "patterns": [
        r"farm[_-]?(\d{1,6})",
        r"herd[_-]?(\d{1,6})",
        r"ranch[_-]?(\d{1,6})",
        r"(\d{4,6})",
        r"([a-z]{2,10})\d{2,}",
    ],
}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# PHASE 1: DATA PREPARATION
# ==========================================
def extract_farm_id(filename, parent_dir=""):
    for pattern in CONFIG["patterns"]:
        match = re.search(pattern, filename.lower())
        if match:
            return f"farm_{match.group(1)}"
    if re.search(r'\d+', parent_dir):
        return f"farm_dir_{parent_dir}"
    return "unknown_farm"

def safe_load_image(path):
    try:
        path_str = str(path)
        if not os.path.exists(path_str): return None
        img = cv2.imread(path_str)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None

def is_blurry(img, threshold=100.0):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold
    except:
        return True

def get_phash(img):
    try:
        pil_img = Image.fromarray(img)
        return imagehash.phash(pil_img)
    except:
        return None

def is_similar_hash(hash1, hash2, max_dist=4):
    if hash1 is None or hash2 is None: return False
    return hash1 - hash2 <= max_dist

def analyze_symptoms(img):
    """
    Returns (symptoms_visible, is_early_stage) based on simple CV heuristics.
    """
    try:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 200
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByInertia = False
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        inverts = cv2.bitwise_not(gray)
        keypoints = detector.detect(inverts)
        blob_count = len(keypoints)
        
        # 2. Red ratio in lower half
        lower_half = img[h//2:, :]
        lower_hsv = hsv[h//2:, :]
        mask1 = cv2.inRange(lower_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(lower_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        red_ratio = cv2.countNonZero(red_mask) / (lower_half.shape[0] * lower_half.shape[1] + 1e-6)
        
        # 3. Edge strength
        edges = cv2.Canny(gray, 100, 200)
        edge_strength = cv2.countNonZero(edges) / (h * w + 1e-6)
        
        # 4. Saliva detection
        saliva_mask = cv2.inRange(hsv, np.array([20, 10, 200]), np.array([40, 100, 255]))
        saliva_ratio = cv2.countNonZero(saliva_mask) / (h * w + 1e-6)
        
        symptoms_visible = (blob_count > 2) or (red_ratio > 0.03) or (edge_strength > 0.5) or (saliva_ratio > 0.01)
        is_early_stage = (blob_count > 0 and blob_count <= 2) or (0.01 < red_ratio <= 0.03)
        
        return bool(symptoms_visible), bool(is_early_stage)
    except:
        return False, False

def cleanup_dataset_dirs(data_root):
    records = []
    hashes = {}
    stats = {"corrupt": 0, "blurry": 0, "duplicates": 0, "subtle_fmd": 0, "valid": 0}
    
    print("Scanning Data Folders...")
    for folder, (is_cattle, disease_status, source) in CONFIG["categories"].items():
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path):
            continue
            
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        for img_path in tqdm(files, desc=f"Processing {folder}"):
            filename = os.path.basename(img_path)
            
            img = safe_load_image(img_path)
            if img is None:
                stats["corrupt"] += 1
                continue
                
            if is_blurry(img):
                stats["blurry"] += 1
                continue
                
            ph = get_phash(img)
            is_dup = False
            for h in hashes.keys():
                if is_similar_hash(ph, h):
                    is_dup = True
                    break
            
            if is_dup:
                stats["duplicates"] += 1
                continue
                
            if ph is not None:
                hashes[ph] = img_path
            
            symptoms_visible, is_early_stage = False, False
            if is_cattle == 1 and disease_status == 1:
                symptoms_visible, is_early_stage = analyze_symptoms(img)
                if is_early_stage:
                    stats["subtle_fmd"] += 1
            
            farm_id = extract_farm_id(filename, folder)
            
            records.append({
                "image_path": img_path,
                "is_cattle": is_cattle,
                "disease_status": disease_status,
                "farm_id": farm_id,
                "source": source,
                "notes": "",
                "symptoms_visible": symptoms_visible,
                "is_early_stage": is_early_stage
            })
            stats["valid"] += 1
                
    df = pd.DataFrame(records)
    return df, stats

# ==========================================
# PHASE 2: PREPROCESSING & AUGMENTATION
# ==========================================
def get_transforms(phase, is_train=True):
    sz = CONFIG["img_size"]
    if not is_train:
        return A.Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    transforms = [A.Resize(sz, sz)]
    # Phase 1: Mild augmentation
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5)
    ])
    # Phase 2: Moderate augmentation
    if phase >= 2:
        transforms.extend([
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ])
    # Phase 3: Aggressive augmentation
    if phase >= 3:
        transforms.extend([
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.ElasticTransform(p=0.3)
        ])
        
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return A.Compose(transforms)

class FMDDataset(Dataset):
    def __init__(self, df, phase=1, is_train=True, current_epoch=1):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.is_train = is_train
        self.current_epoch = current_epoch
        self.transform = get_transforms(phase, is_train)
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def set_phase(self, phase):
        self.phase = phase
        self.transform = get_transforms(phase, self.is_train)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        img = safe_load_image(img_path)
        
        if img is None: # Handled smoothly within DataLoader
            img = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
            
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        # Curriculum learning for early-stage FMD
        sample_weight = 1.0
        if self.is_train and self.current_epoch >= 10 and row.get('is_early_stage', False):
            sample_weight = 2.0
            
        return img, row['is_cattle'], row['disease_status'], sample_weight, img_path

# ==========================================
# PHASE 3: MODEL ARCHITECTURE
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MultiTaskMobileNetV3(nn.Module):
    def __init__(self):
        super(MultiTaskMobileNetV3, self).__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        in_channels = 576 
        
        self.cbam = CBAM(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.disease_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.cbam, self.shared_bottleneck, self.binary_head, self.disease_head]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        pooled = self.pool(x).flatten(1)
        shared = self.shared_bottleneck(pooled)
        return self.binary_head(shared).squeeze(-1), self.disease_head(shared).squeeze(-1)

# ==========================================
# PHASE 4: TRAINING & LOSS STRATEGY
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if weights is not None:
            focal_loss = focal_loss * weights
        return focal_loss.mean()

def get_current_phase(epoch):
    if epoch <= 5: return 1
    elif epoch <= 12: return 2
    else: return 3

def set_parameter_requires_grad(model, phase):
    if phase == 1: # Freeze backbone
        for param in model.backbone.parameters(): param.requires_grad = False
    elif phase == 2: # Unfreeze last 4 blocks
        for param in model.backbone.parameters(): param.requires_grad = False
        for param in list(model.backbone.parameters())[-30:]: # aprox last 4 blocks in MobileNetV3 features
            param.requires_grad = True
    else: # Full fine-tune
        for param in model.backbone.parameters(): param.requires_grad = True

def get_optimizer(model, phase):
    if phase == 1:
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=CONFIG['weight_decay'])
    elif phase == 2:
        return torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.cbam.parameters(), 'lr': 1e-4},
            {'params': model.shared_bottleneck.parameters(), 'lr': 1e-4},
            {'params': model.binary_head.parameters(), 'lr': 1e-4},
            {'params': model.disease_head.parameters(), 'lr': 1e-4}
        ], weight_decay=CONFIG['weight_decay'])
    else:
        return torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=CONFIG['weight_decay'])

# ==========================================
# PHASE 5: EVALUATION & GRAD-CAM
# ==========================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = self.model.backbone[-1]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, task='disease'):
        self.model.eval()
        self.model.zero_grad()
        
        binary_out, disease_out = self.model(img_tensor)
        output = binary_out if task == 'binary' else disease_out
        
        # We assume label 1 implies prob > 0.5, thus positive logits
        loss = output
        loss.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.detach().cpu().numpy()

def save_gradcam(img_path, heatmap, save_path):
    img = cv2.imread(img_path)
    if img is None: return
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed_img)

def optimize_threshold(y_true, y_probs):
    thresholds = np.arange(0.3, 0.81, 0.02)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

# ==========================================
# PHASE 6: DEPLOYMENT OPTIMIZATION
# ==========================================
def prune_and_export_model(model, output_dir):
    model.eval()
    
    # Pruning 30% of weights
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export TorchScript
    ts_path = os.path.join(output_dir, "mobilenetv3_fmd_torchscript.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(ts_path)
    print(f"TorchScript exported to {ts_path}")
    
    # Export ONNX
    onnx_path = os.path.join(output_dir, "mobilenetv3_fmd_int8.onnx") # Naming as requested
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path,
            opset_version=13,
            input_names=["input"],
            output_names=["binary_out", "disease_out"],
            dynamic_axes={"input": {0: "batch_size"}}
        )
        print(f"ONNX exported to {onnx_path}")
    except RuntimeError as e:
        print(f"ONNX export failed: {e}")

# ==========================================
# INFERENCE ENGINE / MAIN LOOP
# ==========================================
def inference_triage(binary_prob, disease_prob, img_path, logs_csv):
    """Production inference mock with confidence triage."""
    rej_thresh = CONFIG["rejection_threshold"]
    if binary_prob < rej_thresh:
        res = "Non-Cattle Rejected"
    elif disease_prob > 0.8:
        res = "FMD High Confidence"
    elif disease_prob < 0.2:
        res = "Healthy High Confidence"
    else:
        res = "Low Confidence - Flagged for Review"
    
    # Log logic here
    with open(logs_csv, "a") as f:
        f.write(f"{img_path},{binary_prob:.4f},{disease_prob:.4f},{res}\n")
    return res

def train_pipeline(data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Prep
    annotations_path = os.path.join(output_dir, "annotations.csv")
    if os.path.exists(annotations_path):
        print("Loading existing annotations.csv...")
        df = pd.read_csv(annotations_path)
    else:
        df, stats = cleanup_dataset_dirs(data_root)
        df.to_csv(annotations_path, index=False)
        print(f"QC Summary: {stats}")
    
    if len(df) == 0:
        print("No valid images found for training! Please check Data Root.")
        return
        
    # Split Data by Farm ID
    df['combined_label'] = df['is_cattle'].astype(str) + "_" + df['disease_status'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
    train_val_idx, test_idx = next(sgkf.split(df, df['combined_label'], df['farm_id']))
    df_train_val, df_test = df.iloc[train_val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    
    train_idx, val_idx = next(sgkf.split(df_train_val, df_train_val['combined_label'], df_train_val['farm_id']))
    df_train, df_val = df_train_val.iloc[train_idx].reset_index(drop=True), df_train_val.iloc[val_idx].reset_index(drop=True)
    
    print(f"Data Split -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Compute Weights
    class_weights_bin = list(1.0 / df_train['is_cattle'].value_counts(normalize=True))
    sample_weights = [class_weights_bin[int(row['is_cattle'])] for _, row in df_train.iterrows()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Datasets
    train_dataset = FMDDataset(df_train, phase=1, is_train=True)
    val_dataset = FMDDataset(df_val, phase=1, is_train=False)
    test_dataset = FMDDataset(df_test, phase=1, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    # Model Setup
    model = MultiTaskMobileNetV3().to(device)
    criterion = FocalLoss().to(device)
    scaler = GradScaler()
    optimizer = get_optimizer(model, phase=1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Training Loop
    for epoch in range(1, CONFIG["epochs"] + 1):
        phase = get_current_phase(epoch)
        train_dataset.set_epoch(epoch)
        train_dataset.set_phase(phase)
        
        # Adjust Model Graph
        set_parameter_requires_grad(model, phase)
        optimizer = get_optimizer(model, phase)
        
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} Phase {phase}")
        for imgs, is_cattle, disease, w, _ in pbar:
            imgs, is_cattle, disease, w = imgs.to(device), is_cattle.float().to(device), disease.float().to(device), w.to(device)
            optimizer.zero_grad()
            
            with autocast():
                out_binary, out_disease = model(imgs)
                loss_binary = criterion(out_binary, is_cattle)
                # Compute disease loss only on cattle images
                cattle_mask = is_cattle == 1
                loss_disease = 0
                if cattle_mask.sum() > 0:
                    loss_disease = criterion(out_disease[cattle_mask], disease[cattle_mask], weights=w[cattle_mask])
                
                loss = loss_binary + loss_disease
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        train_loss /= len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss, all_bin_true, all_bin_probs, all_dis_true, all_dis_probs = 0.0, [], [], [], []
        with torch.no_grad(), autocast():
            for imgs, is_cattle, disease, _, _ in val_loader:
                imgs, is_cattle, disease = imgs.to(device), is_cattle.float().to(device), disease.float().to(device)
                out_binary, out_disease = model(imgs)
                loss_binary = criterion(out_binary, is_cattle)
                loss_disease = 0
                
                cattle_mask = is_cattle == 1
                if cattle_mask.sum() > 0:
                    loss_disease = criterion(out_disease[cattle_mask], disease[cattle_mask])
                loss = loss_binary + loss_disease
                val_loss += loss.item() * imgs.size(0)
                
                all_bin_true.extend(is_cattle.cpu().numpy())
                all_bin_probs.extend(torch.sigmoid(out_binary).cpu().numpy())
                
                if cattle_mask.sum() > 0:
                    all_dis_true.extend(disease[cattle_mask].cpu().numpy())
                    all_dis_probs.extend(torch.sigmoid(out_disease[cattle_mask]).cpu().numpy())
                    
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            early_stop_counter = 0
            print("=> Best Model Saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training finished! Testing Set Evaluation...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.eval()
    
    # Optimize Thresholds on Val
    opt_bin_thresh = optimize_threshold(np.array(all_bin_true), np.array(all_bin_probs))
    opt_dis_thresh = optimize_threshold(np.array(all_dis_true), np.array(all_dis_probs))
    print(f"Optimal Thresholds -> Binary: {opt_bin_thresh:.2f}, Disease: {opt_dis_thresh:.2f}")
    
    # Test Loop
    t_bin_true, t_bin_probs, t_dis_true, t_dis_probs = [], [], [], []
    gradcam_samples = []
    
    with torch.no_grad():
        for imgs, is_cattle, disease, _, paths in test_loader:
            imgs = imgs.to(device)
            out_binary, out_disease = model(imgs)
            t_bin_true.extend(is_cattle.numpy())
            t_bin_probs.extend(torch.sigmoid(out_binary).cpu().numpy())
            
            # Identify true cattle images for disease test
            cattle_mask = is_cattle == 1
            if cattle_mask.sum() > 0:
                t_dis_true.extend(disease[cattle_mask].numpy())
                t_dis_probs.extend(torch.sigmoid(out_disease[cattle_mask]).cpu().numpy())
                
                # Save just a few gradcam samples
                if len(gradcam_samples) < 5:
                    c_idx = torch.nonzero(cattle_mask).squeeze(1).tolist()
                    if isinstance(c_idx, int): c_idx = [c_idx]
                    for idx in c_idx:
                        gradcam_samples.append((imgs[idx:idx+1], paths[idx]))
                        if len(gradcam_samples) >= 5: break

    # Test Metrics computation
    t_bin_pred = (np.array(t_bin_probs) >= opt_bin_thresh).astype(int)
    t_dis_pred = (np.array(t_dis_probs) >= opt_dis_thresh).astype(int)
    
    metrics = f"""
    FINAL TEST METRICS:
    ===================
    Binary Task (Cattle vs Non-Cattle):
    - Accuracy: {accuracy_score(t_bin_true, t_bin_pred):.4f}
    - F1 Score: {f1_score(t_bin_true, t_bin_pred):.4f}
    - Precision: {precision_score(t_bin_true, t_bin_pred):.4f}
    - Recall: {recall_score(t_bin_true, t_bin_pred):.4f}
    
    Disease Task (Healthy vs FMD):
    - F1 Score: {f1_score(t_dis_true, t_dis_pred):.4f}
    - Precision: {precision_score(t_dis_true, t_dis_pred):.4f}
    - Recall (FMD Detection R): {recall_score(t_dis_true, t_dis_pred):.4f}
    """
    print(metrics)
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(metrics)
        
    # Plot Confusion Matrices
    plt.figure()
    sns.heatmap(confusion_matrix(t_bin_true, t_bin_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Binary Classification Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "cm_binary.png"))
    
    if len(t_dis_true) > 0:
        plt.figure()
        sns.heatmap(confusion_matrix(t_dis_true, t_dis_pred), annot=True, fmt='d', cmap='Reds')
        plt.title("Disease Classification Confusion Matrix")
        plt.savefig(os.path.join(output_dir, "cm_disease.png"))
        
    # Generate Grad-CAM for samples
    print("Generating Grad-CAM visualization for 5 test samples...")
    gcam = GradCAM(model)
    for i, (img_tensor, path) in enumerate(gradcam_samples):
        # We need gradients so we have to enable them temporarily here
        with torch.enable_grad():
            img_tensor.requires_grad_(True)
            heatmap = gcam.generate(img_tensor, task='disease')
            if heatmap is not None:
                save_path = os.path.join(output_dir, f"gradcam_{i}.png")
                save_gradcam(path, heatmap, save_path)
                
    print("Optimizing, pruning and exporting model...")
    export_model = MultiTaskMobileNetV3().to(device)
    export_model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    prune_and_export_model(export_model, output_dir)
    print("Pipeline Execution Complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMD Multi-Task MobileNetV3 Pipeline")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Data directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Path to save outputs")
    
    args = parser.parse_args()
    set_seed(CONFIG["seed"])
    train_pipeline(args.data_root, args.output_dir)
