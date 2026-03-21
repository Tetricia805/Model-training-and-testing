# 🐄 Model Training and Automation Guide

Complete Python script for training and retraining the Cattle FMD Detection model with PyTorch.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Script Overview](#script-overview)
3. [Configuration](#configuration)
4. [Installation](#installation)
5. [Running the Training](#running-the-training)
6. [Output Files](#output-files)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)
9. [Two-Phase Training Explained](#two-phase-training-explained)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU) or CPU (slower)
- ~2GB disk space for models

### Installation

```bash
# 1. Install required packages
pip install torch torchvision torchaudio
pip install pytorch-lightning scikit-learn pillow
pip install albumentations tqdm pandas seaborn matplotlib numpy

# Or use requirements file (if created)
pip install -r requirements_training.txt
```

### Run Training

```bash
# Simple - uses default configuration
python train_fmd_model.py

# With GPU check
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
Logging initialized. Log file: pipeline_output/logs/training_20260321_143022.log
Preparing dataset...
Found 500 images
  Healthy cattle: 150
  Infected cattle: 100
  Non-cattle: 250
...
PHASE 1: Training Heads (Frozen Backbone)
Epoch 1/20
Training: 100%|████████| 16/16 [00:45<00:00, 2.81s/it]
Validation: 100%|████████| 4/4 [00:08<00:00, 2.00s/it]
Train - ID Acc: 0.8750, Diag Acc: 0.7200
Val   - ID Acc: 0.8500, Diag Acc: 0.7100
```

---

## 📖 Script Overview

### Main Components

#### 1. **Configuration Class**
```python
class Config:
    # Paths
    DATA_ROOT = Path('.')
    CATTLE_HEALTHY_DIR = Path('cattle_healthy')
    CATTLE_INFECTED_DIR = Path('cattle_infected')
    NON_CATTLE_DIR = Path('not_cattle_animals')
    
    # Training parameters
    IMAGE_SIZE = 224
    PHASE1_EPOCHS = 20
    PHASE1_LR = 0.001
    PHASE2_EPOCHS = 15
    PHASE2_LR = 0.0001
```

#### 2. **CattleDataset Class**
Custom PyTorch Dataset that:
- Loads images from disk
- Applies augmentations (flip, rotate, color jitter)
- Returns (image, id_label, diag_label) tuples
- Handles batching automatically

#### 3. **CattleMultiTaskModel Class**
Dual-head neural network:
```
Input Image (224×224)
        ↓
MobileNetV3-Small Backbone (2.5M params)
        ↓
    Features (576dim)
    ↙           ↘
ID Head      Diag Head
(2 neurons)  (2 neurons)
Cattle/       Healthy/
Not-Cattle    FMD
```

#### 4. **Training Pipeline**

**Phase 1 (20 epochs):**
- Freeze backbone (MobileNetV3-Small)
- Train only heads
- Lower learning rate: 0.001
- Fast convergence, prevents overfitting

**Phase 2 (15 epochs):**
- Unfreeze entire model
- Fine-tune all layers
- Much lower learning rate: 0.0001
- Adapts pre-trained features to specific task

#### 5. **Metrics Tracking**
- Accuracy for identification and diagnosis
- Loss tracking (separate for each head)
- Validation during training
- Test set evaluation

---

## ⚙️ Configuration

### Data Paths

Update these to match your directory structure:

```python
CATTLE_HEALTHY_DIR = Path('cattle_healthy')      # Healthy cattle images
CATTLE_INFECTED_DIR = Path('cattle_infected')    # FMD-infected cattle
NON_CATTLE_DIR = Path('not_cattle_animals')      # Other animals
OUTPUT_DIR = Path('pipeline_output')             # Save models/results
```

### Training Hyperparameters

**Phase 1 Configuration:**
```python
PHASE1_EPOCHS = 20          # Number of epochs for head training
PHASE1_LR = 0.001           # Learning rate (0.001 = 0.1%)
PHASE1_BATCH_SIZE = 32      # Images per batch
```

**Phase 2 Configuration:**
```python
PHASE2_EPOCHS = 15          # Number of epochs for fine-tuning
PHASE2_LR = 0.0001          # Much lower for fine-tuning
PHASE2_BATCH_SIZE = 32
```

### Data Split

```python
TRAIN_SPLIT = 0.70          # 70% training
VAL_SPLIT = 0.15            # 15% validation
TEST_SPLIT = 0.15           # 15% testing
```

### Training Settings

```python
RANDOM_SEED = 42            # Reproducibility (same results each run)
NUM_WORKERS = 4             # Parallel data loading (CPU cores)
PATIENCE = 5                # Early stopping patience
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Modifying Configuration

```python
# Edit in train_fmd_model.py

class Config:
    # Change learning rate for Phase 1
    PHASE1_LR = 0.0005  # From 0.001 to 0.0005
    
    # Increase training epochs
    PHASE2_EPOCHS = 20  # From 15 to 20
    
    # Larger batch size (if GPU memory allows)
    PHASE1_BATCH_SIZE = 64  # From 32 to 64
```

---

## 💾 Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv training_env
training_env\Scripts\activate

# Linux/Mac
python3 -m venv training_env
source training_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core PyTorch (GPU - CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional packages
pip install scikit-learn pillow albumentations tqdm pandas matplotlib seaborn numpy
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}')"
```

### Full Requirements File

Create `requirements_training.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
scikit-learn>=1.3.0
pillow>=9.0.0
albumentations>=1.3.0
tqdm>=4.65.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
```

Then install:
```bash
pip install -r requirements_training.txt
```

---

## 🎯 Running the Training

### Basic Training

```bash
# Run with default configuration
python train_fmd_model.py
```

This will:
1. Load data from directories
2. Create train/val/test splits (70/15/15)
3. Run Phase 1: Head training (20 epochs)
4. Run Phase 2: Fine-tuning (15 epochs)
5. Evaluate on test set
6. Save final model

### Resuming Training

If training is interrupted:

```bash
# Modify train_fmd_model.py to load previous best model
model.load_state_dict(torch.load('pipeline_output/models/model_phase2_best.pt'))

# Then run again - it will continue from there
python train_fmd_model.py
```

### GPU Acceleration

```bash
# Use GPU (if available)
# Automatically detected - no changes needed

# Force CPU (for testing)
# Modify in script: DEVICE = torch.device('cpu')
```

### Monitoring Training

```bash
# View logs in real-time
Get-Content -Path pipeline_output/logs/training_*.log -Tail 20 -Wait  # Windows

tail -f pipeline_output/logs/training_*.log  # Linux/Mac
```

---

## 📁 Output Files

### Directory Structure After Training

```
pipeline_output/
├── models/
│   ├── model_phase1_best.pt        # Best model after Phase 1
│   ├── model_phase2_best.pt        # Best model after Phase 2
│   └── model_final.pt              # Final trained model (for deployment)
├── results/
│   ├── training_history.json       # Metrics and loss history
│   └── confusion_matrices.json     # Per-class performance
└── logs/
    └── training_20260321_143022.log # Training log
```

### Model Files Explained

| File | Purpose | When to Use |
|------|---------|------------|
| `model_phase1_best.pt` | Checkpoint after head training | Debugging/analysis |
| `model_phase2_best.pt` | Best fine-tuned model | Testing/evaluation |
| `model_final.pt` | Final model for deployment | Production use |

### Training History Format

```json
{
  "phase1": {
    "train": [
      {
        "id_accuracy": 0.875,
        "diag_accuracy": 0.72,
        "id_loss": 0.28,
        "diag_loss": 0.42,
        "total_loss": 0.70
      }
    ],
    "val": [...]
  },
  "phase2": {...},
  "test_metrics": {
    "id_accuracy": 0.96,
    "diag_accuracy": 0.86
  }
}
```

### Log File Example

```
2026-03-21 14:30:22,123 - INFO - Logging initialized. Log file: pipeline_output/logs/training_20260321_143022.log
2026-03-21 14:30:25,456 - INFO - Found 500 images
2026-03-21 14:30:25,789 - INFO -   Healthy cattle: 150
2026-03-21 14:30:25,901 - INFO -   Infected cattle: 100
2026-03-21 14:30:25,945 - INFO -   Non-cattle: 250
2026-03-21 14:31:02,123 - INFO - 
============================================================
PHASE 1: Training Heads (Frozen Backbone)
============================================================
...
```

---

## 🔧 Customization

### Modify Augmentations

Edit `get_augmentation_transforms()` function:

```python
def get_augmentation_transforms(stage='train'):
    if stage == 'train':
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),           # Horizontal flip probability
            A.VerticalFlip(p=0.2),             # Vertical flip probability
            A.Rotate(limit=20, p=0.5),         # Rotate ±20 degrees
            A.GaussNoise(p=0.2),               # Add noise
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            # Add more transformations as needed:
            # A.RandomBrightnessContrast(p=0.2),
            # A.GaussianBlur(blur_limit=3, p=0.2),
            # A.ElasticTransform(p=0.2),
            A.Normalize(...),
            ToTensorV2()
        ])
```

### Use Different Backbone

```python
# Change from MobileNetV3-Small to other models:

# Option 1: MobileNetV3-Large (slower but more accurate)
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Option 2: ResNet-50 (more accurate, slower)
from torchvision.models import resnet50, ResNet50_Weights
self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Option 3: EfficientNet (good balance)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
```

### Change Loss Function

```python
# Use weighted loss (if class imbalance exists)
weights = [1.0, 3.0]  # 3x weight for FMD class
criterion_diag = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

# Use focal loss (handles hard examples better)
# pip install focal-loss
from focal_loss.focal_loss import FocalLoss
criterion_diag = FocalLoss(gamma=2, alpha=0.25)
```

### Adjust Learning Rates

```python
# For faster convergence (riskier)
PHASE1_LR = 0.01      # 10x higher
PHASE2_LR = 0.001     # 10x higher

# For more stable training
PHASE1_LR = 0.0001    # 10x lower
PHASE2_LR = 0.00001   # 10x lower

# Rule of thumb: If loss explodes → decrease LR
#                If loss decreases slowly → increase LR
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```python
# Reduce batch size
PHASE1_BATCH_SIZE = 16  # From 32 to 16
PHASE2_BATCH_SIZE = 16

# Use CPU instead
DEVICE = torch.device('cpu')

# Clear cache
torch.cuda.empty_cache()
```

### Issue: Files Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'cattle_healthy'
```

**Solution:**
```python
# Check current working directory
import os
print(os.getcwd())  # Should be Data_AniLink directory

# Verify directories exist
import os
print(os.listdir('.'))  # Should show cattle_healthy, etc.

# Update paths if needed
CATTLE_HEALTHY_DIR = Path('/full/path/to/cattle_healthy')
```

### Issue: Model Training is Slow

**Check GPU usage:**
```python
# In training loop add:
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()} bytes")
```

**Optimize:**
```python
# Increase batch size (if memory allows)
PHASE1_BATCH_SIZE = 64

# Reduce number of workers
NUM_WORKERS = 2

# Use mixed precision (faster GPUs)
# pip install apex
from torch.cuda.amp import autocast, GradScaler
```

### Issue: Validation Accuracy Not Improving

**Possible causes and solutions:**

```python
# 1. Learning rate too high → Reduce it
PHASE1_LR = 0.0005  # From 0.001

# 2. Learning rate too low → Increase it
PHASE1_LR = 0.005   # From 0.001

# 3. Model not seeing enough data → Check dataset size
# Add data augmentation

# 4. Model might be underfitting → Make it bigger
# Use larger backbone (ResNet instead of MobileNet)

# 5. Early stopping too aggressive → Increase patience
PATIENCE = 10  # From 5
```

### Issue: NaN Loss

```
Warning: gradient contains NaN
```

**Solutions:**
```python
# Reduce learning rate
PHASE1_LR = 0.0001  # Much lower

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for corrupted images in dataset
# Add image validation in CattleDataset.__getitem__()
```

---

## 📚 Two-Phase Training Explained

### Why Two Phases?

**Phase 1: Head Training (Frozen Backbone)**
```
Goal: Quickly adapt pre-trained features to your task
Why frozen: Prevents throwing away learned features
Duration: 20 epochs (fast)
Learning rate: 0.001 (relatively high)

Process:
1. Keep MobileNetV3 weights from ImageNet (general vision)
2. Train only the head layers (cattle/FMD specific)
3. Adapts general features to cattle detection
```

**Phase 2: Fine-tuning (Unfrozen Backbone)**
```
Goal: Slightly adjust backbone for your specific images
Why unfrozen: Fine-tune for cattle images specifically
Duration: 15 epochs (slower, finer adjustments)
Learning rate: 0.0001 (much lower → smaller steps)

Process:
1. Load best model from Phase 1
2. Unfreeze all layers
3. Train entire network with tiny learning rate
4. Carefully adjusts backbone without destroying it
```

### Learning Rate Visualization

```
Learning Rate Schedule:
Phase 1                    Phase 2
0.001 ────────────────    0.0001 ───────────────
           │                        │
      Fast learning            Slow fine-tuning
      ↓ Loss quickly          ↓ Loss gradually
      (20 epochs)             (15 epochs)
```

### Example Results

```
After Phase 1 (20 epochs):
- Identification Accuracy: 89.5%
- Diagnosis Accuracy: 76.2%
- Time: ~15 minutes

After Phase 2 (15 epochs):
- Identification Accuracy: 96.3%  ← +6.8%
- Diagnosis Accuracy: 85.8%       ← +9.6%
- Time: ~20 minutes (total 35)
```

### Why Not Single Phase?

| Approach | Speed | Accuracy | Memory | Stability |
|----------|-------|----------|--------|-----------|
| No pre-training | Fast | Low | High | Unstable |
| Single phase | Slow | Medium | High | Okay |
| Two phases ✓ | Medium | High | Low | Stable |

---

## 📊 Monitoring Training Progress

### Check Logs in Real-time

```bash
# Windows PowerShell
Get-Content pipeline_output/logs/training_*.log -Tail 10 -Wait

# Linux/Mac
tail -f pipeline_output/logs/training_*.log
```

### View Training History

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('pipeline_output/results/training_history.json') as f:
    history = json.load(f)

# Plot identification accuracy
train_id_acc = [h['id_accuracy'] for h in history['phase1']['train']]
val_id_acc = [h['id_accuracy'] for h in history['phase1']['val']]

plt.plot(train_id_acc, label='Train')
plt.plot(val_id_acc, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## ✅ Checklist Before Training

- [ ] Data directories exist and contain images
- [ ] PyTorch is installed and GPU is detected (if using GPU)
- [ ] Enough disk space (at least 2GB)
- [ ] Configuration parameters are set correctly
- [ ] Installation script can be run as: `python train_fmd_model.py`
- [ ] `pipeline_output` directory is writable

---

## 📝 Next Steps

After training completes:

1. **Verify model performance:**
   ```python
   # Check test metrics
   cat pipeline_output/results/training_history.json
   ```

2. **Export model for deployment:**
   ```python
   # Convert to ONNX for mobile/web
   python export_model.py
   ```

3. **Run inference on new images:**
   ```bash
   python test_model_ui.py            # Developer version
   streamlit run test_model_farmer_ui.py  # Farmer version
   ```

4. **Deploy to production:**
   - Use `pipeline_output/models/model_final.pt`
   - Refer to MODEL_INTEGRATION_GUIDE.md for deployment options

---

## 🤝 Support & Questions

For issues or questions:
- Check the troubleshooting section above
- Review logs: `pipeline_output/logs/`
- Verify data format and paths
- Check PyTorch documentation: https://pytorch.org/

