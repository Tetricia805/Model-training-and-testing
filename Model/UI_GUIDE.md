# 🐄 Cattle FMD Detection - UI Guide

## Overview

This project includes **two separate UI applications** for testing the **Foot-and-Mouth Disease (FMD) detection model**:

| Version | File | Purpose | Users | Focus |
|---------|------|---------|-------|-------|
| **🌾 Farmer-Friendly** | `test_model_farmer_ui.py` | Easy disease detection for farmers | Farmers, Agricultural workers | Action-based guidance |
| **🧪 Technical/Developer** | `test_model_ui.py` | Model validation & testing | ML Engineers, Developers, QA | Detailed metrics & probabilities |

---

## 🌾 Farmer-Friendly Version

**File:** `test_model_farmer_ui.py`

### Who Should Use This?
- Agricultural farmers
- Veterinary assistants
- Farm managers
- People with no technical background

### What It Shows
✅ **Simple, actionable results**  
✅ **Large, clear text**  
✅ **Step-by-step guidance**  
✅ **Immediate action items**  
✅ **No technical jargon**  

### Key Features

#### 1️⃣ **Simple Photo Upload**
```
Take a clear photo of your cattle:
• Good lighting (daytime)
• Can see whole cattle
• Close enough for detail
```

#### 2️⃣ **Clear Results**
- ✅ **Healthy** → "Cattle appears healthy. Check again in 2 weeks."
- ⚠️ **Possible FMD** → "Isolate cattle & call veterinarian today"
- 🚨 **FMD Detected** → "URGENT: Isolate immediately & call vet now"
- 📸 **Not Cattle** → "Please upload a cattle image"

#### 3️⃣ **Action-Based Guidance**
Each result includes specific, numbered actions:
```
✅ What To Do:
1. Isolate this cattle from others
2. Schedule veterinarian visit TODAY
3. Get professional confirmation
```

#### 4️⃣ **Expandable Details**
- Hidden by default (keeps interface clean)
- Click "Detailed Information" for technical metrics
- Good for farmers who want to keep records

#### 5️⃣ **Download Results**
- Save results as text file
- Good for record-keeping
- Easy to share with veterinarian

### Run the Farmer UI
```bash
streamlit run test_model_farmer_ui.py
```

Then open: `http://localhost:8501`

---

## 🧪 Technical/Developer Version

**File:** `test_model_ui.py`

### Who Should Use This?
- ML Engineers
- Developers
- QA/Testing teams
- Model validation specialists

### What It Shows
✅ **Detailed probabilities**  
✅ **Confidence scores**  
✅ **Preprocessing visualization**  
✅ **Multiple model formats**  
✅ **Advanced configuration**  

### Key Features

#### 1️⃣ **Sidebar Configuration**
- Model format selection (PyTorch, INT8)
- Confidence threshold slider (0.0 - 1.0)
- Display option toggles

#### 2️⃣ **Preprocessing Visualization**
View all 4 preprocessing steps:
- Original image
- Resized image (224×224)
- Normalization parameters
- Tensor shape

#### 3️⃣ **Multiple Result Tabs**
- **🎯 Alert** → Decision & severity level
- **📊 Probabilities** → Detailed percentage bars
- **📈 Details** → Metric cards & confidence levels

#### 4️⃣ **Probability Visualization**
For both tasks:
- Identification (Cattle vs Not-Cattle)
- Diagnosis (Healthy vs FMD)

#### 5️⃣ **Model Statistics**
- Test set accuracy metrics
- Recall rates
- False alarm rates

#### 6️⃣ **Expert Controls**
- Adjustable confidence threshold
- Model format selection
- Preprocessing toggle

### Run the Technical UI
```bash
streamlit run test_model_ui.py
```

Then open: `http://localhost:8501`

---

## 📋 Key Difference: Diagnosis Only for Cattle

**IMPORTANT:** Both UIs implement the same core logic:

### When Image is NOT Cattle:
```
❌ Result: "This is NOT a cattle"
❌ Diagnosis: NOT COMPUTED
❌ Action: "Upload a cattle image"
```

**Why?** FMD (Foot-and-Mouth Disease) is specific to cattle. 
There's no point diagnosing FMD on non-cattle animals.

### When Image IS Cattle:
```
✅ Result: "Cattle detected"
✅ Diagnosis: COMPUTED (Healthy or FMD)
✅ Action: Based on health status
```

---

## 🔍 Comparison Table

| Feature | Farmer UI | Developer UI |
|---------|-----------|--------------|
| **Page Layout** | Centered, simple | Wide, detailed |
| **Sidebar** | Hidden | Expandable with settings |
| **Probabilities** | Hidden by default | Always visible |
| **Preprocessing** | Not shown | 4-step visualization |
| **Model Format** | Fixed (Float32) | Selectable |
| **Threshold** | Fixed (0.7) | Adjustable slider |
| **Alert Messages** | Action-based | Technical-based |
| **Text Size** | Large (18px) | Standard (14px) |
| **Color Scheme** | Simple (3 states) | Detailed (5+ states) |
| **Export Format** | Text (.txt) | Markdown (.md) |
| **Target User** | Farmers | Engineers |

---

## 🚀 Quick Start

### For Farmers:
```bash
streamlit run test_model_farmer_ui.py
```
- Upload photo of cattle
- Read the result
- Follow the action (do NOT think about percentages)

### For Developers:
```bash
streamlit run test_model_ui.py
```
- Adjust threshold & settings
- Validate model performance
- Check probabilities
- Export detailed results

---

## 📊 Result Flow

Both UIs follow the same logic:

```
1. Image Uploaded
        ↓
2. Identification Task
   Is this a cattle?
        ↓
   YES ──────────────→ Diagnosis Task
   NO               (Only if cattle)
   ↓                    ↓
   Alert:           Healthy? or FMD?
   "Not Cattle"     ↓
   Stop             Alert: Health Status
                    (with action items)
```

---

## ⚠️ Important Notes

### Diagnosis ONLY for Cattle Images
- If you upload a dog, cat, horse, etc. → No diagnosis shown
- The app will tell you to upload a cattle image
- This is correct behavior (FMD only in cattle)

### Alert Levels (for both UIs)

**Farmer Version:**
- ✅ Healthy
- ⚠️ Possible FMD
- 🚨 FMD Detected
- 📸 Not Cattle

**Developer Version:**
- ✅ OK (Green)
- ❓ UNCLEAR (Orange)
- ⚠️ WARNING (Orange-Red)
- 🚨 CRITICAL (Red)
- ℹ️ INFO (Blue)

### Confidence Thresholds
**Farmer Version:**
- Built-in (0.7 for cattle detection)
- Cannot change

**Developer Version:**
- Adjustable slider (0.0 - 1.0)
- For testing model sensitivity

---

## 📥 For Farmers - Recording Results

Each test generates a downloadable text file:
```
CATTLE DISEASE CHECK - 2026-03-21 14:35:22
=====================================================

RESULT: CATTLE IS HEALTHY
✅ (Very confident: 85%)

CATTLE DETECTION: Cattle
Confidence: 96.2%

HEALTH STATUS: Healthy
Confidence: 85.1%

ACTION REQUIRED:
Continue normal care. Check again in 2 weeks.

=====================================================
```

**Save this file:**
- Print it for veterinarian
- Keep records by date
- Share with farm management

---

## 🔧 For Developers - Testing Workflow

1. **Load test images** (cattle, non-cattle, various conditions)
2. **Adjust threshold** (experiment with sensitivity)
3. **Review probabilities** (understand model confidence)
4. **Check preprocessing** (verify image transformation)
5. **Change model format** (test PyTorch vs ONNX)
6. **Validate accuracy** (compare to test metrics)
7. **Export results** (save for documentation)

---

## ℹ️ Project Information

**Project:** Cattle Disease Detection System  
**Disease:** Foot-and-Mouth Disease (FMD)  
**Model:** MobileNetV3-Small (Dual-task learning)  

**Test Set Performance:**
- Cattle identification: **96.35%** accuracy
- FMD diagnosis: **85.82%** accuracy
- FMD recall: **91.82%** (catches 9 out of 10 FMD cases)

**Model Architecture:**
- Task 1: Cattle vs Other Animals
- Task 2: Healthy vs FMD (only if cattle)
- Input: 224×224 RGB images
- Output: Identification + Diagnosis

---

## 🆘 Troubleshooting

### UI won't start?
```bash
pip install streamlit torch torchvision pillow opencv-python
streamlit run test_model_farmer_ui.py
```

### Model not found?
Check: `pipeline_output/models/model_final.pt` exists

### Slow inference?
- Use CPU if GPU not available (it will auto-switch)
- First run is slower (model loads into memory)

### Wrong results?
- Upload clearer photo (better lighting, closer)
- Check that it's actually a cattle
- Contact developer with result screenshots

---

## 📚 Related Documentation

- `TESTING_GUIDE.md` - Complete testing instructions
- `TESTING_UPDATE.md` - Conditional diagnosis logic
- `MODEL_INTEGRATION_GUIDE.md` - Integration examples
- `PROJECT_DOCUMENTATION.md` - Complete architecture
- `DEPLOYMENT_ARCHITECTURE.md` - Production deployment

---

**Last Updated:** March 21, 2026

---

## 🎯 Which Version Should I Use?

### Use **Farmer UI** if:
- You're testing with actual farmers
- You want simple, action-based results
- You don't care about percentages
- You need to make quick decisions

### Use **Developer UI** if:
- You're validating model performance
- You need detailed metrics
- You're adjusting thresholds
- You're debugging the model

**Both are correct. They just speak different languages.** 🌾 ↔️ 🧪
