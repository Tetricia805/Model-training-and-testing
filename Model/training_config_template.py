"""
Training Configuration Template
========================================
Copy this file and modify parameters as needed for different training scenarios.
Use as: python train_fmd_model.py --config training_config.py
"""

from pathlib import Path
import torch

# ============================================
# FILE PATHS
# ============================================

# Root data directory
DATA_ROOT = Path('.')

# Training data directories
CATTLE_HEALTHY_DIR = DATA_ROOT / 'cattle_healthy'
CATTLE_INFECTED_DIR = DATA_ROOT / 'cattle_infected'
NON_CATTLE_DIR = DATA_ROOT / 'not_cattle_animals'

# Output directories
OUTPUT_DIR = Path('pipeline_output')
MODELS_DIR = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
LOGS_DIR = OUTPUT_DIR / 'logs'

# ============================================
# DATA CONFIGURATION
# ============================================

# Image size for model input
IMAGE_SIZE = 224

# Train/Val/Test split ratios
TRAIN_SPLIT = 0.70  # 70% training
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% testing

# Random seed for reproducibility
RANDOM_SEED = 42

# Data augmentation settings
AUGMENTATION_ENABLED = True
AUGMENTATION_PROBABILITY = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.2,
    'rotation': 20,  # degrees
    'gauss_noise': 0.2,
    'color_jitter': 0.3,
}

# ============================================
# PHASE 1: HEAD TRAINING (FROZEN BACKBONE)
# ============================================

# Number of training epochs
PHASE1_EPOCHS = 20

# Learning rate
PHASE1_LR = 0.001

# Batch size
PHASE1_BATCH_SIZE = 32

# Dropout rate
PHASE1_DROPOUT = 0.5

# ============================================
# PHASE 2: FINE-TUNING (UNFROZEN BACKBONE)
# ============================================

# Number of training epochs
PHASE2_EPOCHS = 15

# Learning rate (much lower than Phase 1)
PHASE2_LR = 0.0001

# Batch size
PHASE2_BATCH_SIZE = 32

# Dropout rate
PHASE2_DROPOUT = 0.3

# ============================================
# TRAINING SETTINGS
# ============================================

# Device (GPU or CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of worker threads for data loading
NUM_WORKERS = 4

# Loss function weights (if class imbalance exists)
ID_LOSS_WEIGHT = 1.0      # Identification loss weight
DIAG_LOSS_WEIGHT = 1.0    # Diagnosis loss weight

# ============================================
# MODEL SETTINGS
# ============================================

# Use pre-trained backbone from ImageNet
PRETRAINED_BACKBONE = True

# Freeze backbone in Phase 1
FREEZE_BACKBONE_PHASE1 = True

# ============================================
# OPTIMIZATION SETTINGS
# ============================================

# Optimizer type ('adam', 'sgd')
OPTIMIZER = 'adam'

# Gradient clipping (prevent exploding gradients)
CLIP_GRADIENTS = True
GRADIENT_CLIP_VALUE = 1.0

# ============================================
# CALLBACKS & EARLY STOPPING
# ============================================

# Early stopping patience (stop if no improvement for N epochs)
PATIENCE = 5

# Save only best model
SAVE_BEST_ONLY = True

# Validation frequency (every N batches)
VALIDATION_FREQUENCY = 1  # Every epoch

# ============================================
# LOGGING & MONITORING
# ============================================

# Log level ('DEBUG', 'INFO', 'WARNING')
LOG_LEVEL = 'INFO'

# Save training history
SAVE_TRAINING_HISTORY = True

# Generate visualizations
GENERATE_PLOTS = True

# ============================================
# ADVANCED SETTINGS
# ============================================

# Use mixed precision training (faster on GPUs)
MIXED_PRECISION = False

# Use data parallel (multi-GPU training)
DATA_PARALLEL = False

# Resume training from checkpoint
RESUME_FROM_CHECKPOINT = None  # Path to checkpoint or None

# ============================================
# PRESET CONFIGURATIONS
# ============================================

# Quick training (for testing only)
PRESET_QUICK_TRAIN = False
if PRESET_QUICK_TRAIN:
    PHASE1_EPOCHS = 2
    PHASE2_EPOCHS = 1
    PHASE1_BATCH_SIZE = 8
    PHASE2_BATCH_SIZE = 8

# Fast training (small batches, few epochs)
PRESET_FAST_TRAIN = False
if PRESET_FAST_TRAIN:
    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 8
    PHASE1_BATCH_SIZE = 16
    PHASE2_BATCH_SIZE = 16
    PHASE1_LR = 0.005
    PHASE2_LR = 0.0005

# Accurate training (large batches, many epochs)
PRESET_ACCURATE_TRAIN = False
if PRESET_ACCURATE_TRAIN:
    PHASE1_EPOCHS = 30
    PHASE2_EPOCHS = 20
    PHASE1_BATCH_SIZE = 64
    PHASE2_BATCH_SIZE = 64
    PHASE1_LR = 0.0005
    PHASE2_LR = 0.00005
    PATIENCE = 10

# ============================================
# CUSTOM CONFIGURATIONS
# ============================================

# Example 1: Low GPU memory configuration
# Uncomment to use:
"""
PHASE1_BATCH_SIZE = 8
PHASE2_BATCH_SIZE = 8
NUM_WORKERS = 0
"""

# Example 2: High accuracy configuration
# Uncomment to use:
"""
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20
PHASE1_LR = 0.0005
PHASE2_LR = 0.00005
DEVICE = torch.device('cuda')
MIXED_PRECISION = True
"""

# Example 3: Stable training configuration
# Uncomment to use:
"""
PHASE1_LR = 0.0001
PHASE2_LR = 0.00001
CLIP_GRADIENTS = True
PATIENCE = 10
"""

print("✓ Training configuration loaded successfully")
print(f"  Device: {DEVICE}")
print(f"  Phase 1: {PHASE1_EPOCHS} epochs, LR={PHASE1_LR}, BS={PHASE1_BATCH_SIZE}")
print(f"  Phase 2: {PHASE2_EPOCHS} epochs, LR={PHASE2_LR}, BS={PHASE2_BATCH_SIZE}")
