# 🚀 QUICK START GUIDE - Cattle Detection Pipeline

## ⏱️ 5-Minute Setup

1. **Open Notebook**
   - File: `c:\Users\USER\Desktop\Data_AniLink\cattle_detection_pipeline.ipynb`
   - Type: Jupyter Notebook

2. **Run First 3 Cells** (2 minutes)
   - Cell 1: Library imports + setup
   - Cell 2: Configuration settings
   - Cell 3: Data exploration

3. **Watch Progress** (rest of time)
   - Cells auto-execute sequentially
   - Plots update in real-time
   - Training progress shown with progress bars

---

## 📊 What Happens in Each Phase

### PHASE 1: Data Preparation (Cells 1-10)
**Time**: ~10 minutes
**Output**: 
- Cleaned dataset (removed blurry/corrupted)
- Train/Val/Test splits (70/15/15)
- Sample images visualization
- Class distribution charts

**Key Variables Created**:
```python
df_annotations       # All images with 2 labels each
df_train, df_val, df_test  # Split datasets
```

### PHASE 2: Preprocessing (Cells 11-20)
**Time**: ~5 minutes (setup only)
**Output**:
- PyTorch DataLoaders ready
- Augmentation pipeline configured
- Batch creation verified

**Key Code**:
```python
train_loader   # DataLoader for training
val_loader     # DataLoader for validation
test_loader    # DataLoader for testing
```

### PHASE 3: Model Setup (Cells 21-25)
**Time**: ~2 minutes
**Output**:
- MobileNetV3-Small model initialized
- Dual heads created
- GPU/CPU detection

**Key Variables**:
```python
model                  # Main model
criterion_id, criterion_diag  # Loss functions
optimizer             # Adam optimizer
```

### PHASE 4: Training (Cells 26-35)
**Time**: 🔴 **1-3 HOURS** (depends on GPU)
**What Happens**:
- Phase 4A: 20 epochs training frozen backbone
- Phase 4B: 15 epochs fine-tuning with unfrozen layers
- Progress bars show real-time metrics
- Best models saved automatically

**Expected Output**:
- Training loss decreases smoothly
- Validation accuracy improves
- Best model auto-saved when ready

**What to Monitor**:
```
📈 Good: Train loss ↓, Val loss ↓ (both decreasing)
⚠️ Warning: Train loss ↓, Val loss ↑ (overfitting)
❌ Bad: Both losses flat (learning rate too low)
```

### PHASE 5: Evaluation (Cells 36-42)
**Time**: ~10 minutes
**Output**:
- Accuracy/Precision/Recall/F1 scores
- Confusion matrices (visual)
- Detailed metrics report
- 🎯 FMD Recall highlighted

**Key Outputs**:
```
Identification Accuracy: XX%
Diagnosis Accuracy: YY%
FMD Recall: ZZ% (most important!)
```

### PHASE 6: Optimization (Cells 43-50)
**Time**: ~15 minutes
**Output**:
- Model quantized to INT8 (4x smaller)
- ONNX format exported
- Deployment-ready files created
- Model size: 3-11MB

**Final Files**:
```
model_final.pt              (11MB, Float32)
model_quantized_int8.pt    (3MB, optimized)
model_final.onnx           (11MB, mobile-ready)
```

---

## 🎮 Real-Time Monitoring During Training

### Phase 4 Training View
```
Epoch 1/20
────────────────
Training: 100%|██████████| 50/50 [00:15<00:00]
Metrics:
  Train Loss: 0.4532 | Val Loss: 0.3821
  Train ID Acc: 0.8632 | Val ID Acc: 0.9104
  Learning Rate: 0.001000
```

### What to Watch For
✅ **Good Signs**:
- Validation loss decreases for 5+ epochs
- No sudden spikes
- Smooth curves without noise

⚠️ **Warning Signs**:
- Validation loss stops improving for 5 epochs → Early stopping activates
- Training loss very different from validation → Overfitting
- Loss NaN or Inf → Gradient explosion

---

## 📁 Output Files Explained

### After Phase 1
```
pipeline_output/processed_data/
├── train_annotations.csv    ← 70% training data list
├── val_annotations.csv      ← 15% validation data list
└── test_annotations.csv     ← 15% test data list
```

### After Phase 4
```
pipeline_output/models/
├── model_phase1_final.pt    ← After frozen training
└── model_phase2_best.pt     ← Best fine-tuned model
```

### After Phase 6
```
pipeline_output/models/
├── model_final.pt           ← Production model (Float32)
├── model_quantized_int8.pt  ← Optimized for mobile
└── model_final.onnx         ← Cross-platform format
```

### Results
```
pipeline_output/results/
├── sample_images.png        ← Dataset visualization
├── class_distribution.png   ← Class balance
├── training_curves.png      ← Loss/accuracy graphs
├── confusion_matrices.png   ← Detailed evaluation
└── final_report.txt         ← Complete summary
```

---

## 🎯 Key Metrics to Watch

### Identification Task (Cattle vs Not-Cattle)
- **Target Accuracy**: > 85%
- **Why**: Clear distinction, easier task
- **Trade-off**: Recall vs Precision (both important equally)

### Diagnosis Task (Healthy vs FMD)
- **Target Accuracy**: > 75%
- **Why**: FMD is rare, harder to detect
- **🎯 Critical Metric**: FMD Recall > 90%
  - Better to have false alarms than miss disease
  - Better approach: Alert on suspicion, vet confirms

---

## 🚨 Troubleshooting During Training

| Symptom | Likely Cause | Solution |
|---------|---|---|
| Out of memory (OOM) | Batch size too large | ↓ batch_size to 16 or 8 |
| Loss is NaN/Inf | Gradient explosion | ↓ learning_rate by 2x |
| Validation loss flat | Learning rate too low | ↑ learning_rate by 2x |
| Overfitting after epoch 10 | Not enough regularization | ↑ weight_decay, add dropout |
| GPU not detected | PyTorch CUDA issue | Use CPU (slower but works) |
| Takes 10+ hours on GPU | NumWorkers bottleneck | ↓ workers or use different batch_size |

---

## 📱 Quick Inference Test

After training, test on new images:

```python
# Create inference engine
inference = CattleInferenceEngine('pipeline_output/models/model_final.pt')

# Predict on image
result = inference.predict('path/to/image.jpg')

# Print results
print(f"Cattle? {result['identification']['class']}")
print(f"Disease? {result['diagnosis']['class']}")
print(f"Alert: {result['alert']}")
```

### Expected Output
```
Cattle? Cattle (92.3%)
Disease? Healthy (87.5%)
Alert: ✅ Cattle appears healthy. Monitor regularly.
```

---

## 📊 Important Cells for Customization

### To Change Learning Rates
**Cell**: Configuration (Phase 3)
```python
CONFIG['learning_rate_phase1'] = 0.001  # Original
CONFIG['learning_rate_phase1'] = 0.0005  # Try lower if unstable
```

### To Change Training Epochs
**Cell**: Configuration (Phase 3)
```python
CONFIG['epochs_phase1'] = 20
CONFIG['epochs_phase2'] = 15
# Increase if underfitting, decrease if overfitting
```

### To Change Batch Size
**Cell**: Configuration (Phase 3)
```python
CONFIG['batch_size'] = 32  # Original
CONFIG['batch_size'] = 64  # Faster training (needs more VRAM)
```

### To Skip Evaluation
**Cell**: Evaluation (Phase 5)
```python
# Comment out confusion matrix plotting to save time
# Just use the metric numbers
```

---

## 💡 Pro Tips

### Tip 1: Save Time During Development
Run only first 3 phases initially to verify data loading works. Then commit to full training.

### Tip 2: Monitor Memory
If GPU memory issues, reduce:
- batch_size (32 → 16)
- num_workers (4 → 2)
- image_size (224 → 192)

### Tip 3: Set Reminders
- Phase 4 takes 1-3 hours → Come back later to check progress
- Email yourself when training ends (optional)

### Tip 4: Save Checkpoints
Models auto-saved every epoch. If interrupted:
```python
# Load best model and continue
model.load_state_dict(torch.load('pipeline_output/models/model_phase2_best.pt'))
# Continue from this point
```

### Tip 5: Run on Colab for Free GPU
If no local GPU:
1. Upload data to Google Drive
2. Open notebook in Colab
3. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
4. Run all cells (10-20x faster than CPU!)

---

## ✅ Success Checklist

- [ ] Notebook opens without errors
- [ ] Data loads (Phase 1: ~X images found)
- [ ] Model initializes (Phase 3: MobileNetV3 loaded)
- [ ] Training starts (Phase 4: Loss decreases)
- [ ] Evaluation runs (Phase 5: Accuracy > 80%)
- [ ] Model exports (Phase 6: ONNX file created)
- [ ] Report generated (Phase 6: final_report.txt)
- [ ] FMD Recall > 90% (critical success metric!)

---

## 🔗 File Locations

| Item | Location |
|------|----------|
| Notebook | `Data_AniLink/cattle_detection_pipeline.ipynb` |
| README | `Data_AniLink/PIPELINE_README.md` |
| Configuration | Cell: Configuration (Phase 3) |
| Training Data | `Data_AniLink/cattle_healthy/`, `cattle_infected/`, etc. |
| Output Models | `Data_AniLink/pipeline_output/models/` |
| Results | `Data_AniLink/pipeline_output/results/` |
| Annotations | `Data_AniLink/pipeline_output/processed_data/` |

---

## 🎓 Learning Path

Not familiar with ML? Read in this order:
1. **Concepts**: "What are neural networks?" → MobileNet paper
2. **Transfer Learning**: "How to reuse pre-trained models?"
3. **Data Augmentation**: "How to make small datasets bigger?"
4. **Multi-task Learning**: "How to train 2 tasks together?"
5. **Quantization**: "How to make models faster on phones?"

Then run the notebook cell-by-cell and understand each part.

---

## 🆘 Emergency Fixes

### Notebook Won't Run
```
Error: ModuleNotFoundError: No module named 'torch'
↓ Solution: Run cell 1 (pip install packages)
```

### Model Training Crashes
```
Error: CUDA out of memory
↓ Solution: ↓ batch_size from 32 to 16 in CONFIG
```

### ONNX Export Fails
```
Error: Unsupported operation
↓ Solution: Check PyTorch version (must be 1.9+)
↓ Update: pip install --upgrade torch
```

### Results Look Bad
```
Accuracy < 70%, FMD Recall < 80%
↓ Solution: More training data (100+ FMD images)
↓ Review: confusion matrix for patterns
↓ Adjust: hyperparameters (LR, epochs, etc.)
```

---

## 📞 Quick Reference Links

- **MobileNet Paper**: https://arxiv.org/abs/1905.02175
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **ONNX Format**: https://onnx.ai/
- **Android ONNX Runtime**: https://github.com/microsoft/onnxruntime/blob/main/docs/Operators.md

---

**Ready to start? Open the notebook and run first cell! 🚀**

Questions? Check the full README.md or review relevant notebook sections.

Good luck with your cattle detection model! 🐄✨
