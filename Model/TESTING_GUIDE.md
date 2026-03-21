# 🧪 Model Testing Guide

Complete guide for testing the **Cattle Foot-and-Mouth Disease (FMD) Detection Model** before deployment.

## ⚡ Quick Start: Choose Your Testing Method

### 🌾 For Farmers & Decision-Making
```bash
streamlit run test_model_farmer_ui.py
```
Simple, action-focused results. No technical jargon. What to do based on results.

---

### 🧪 For Developers & Model Testing
```bash
pip install streamlit pillow opencv-python
streamlit run test_model_ui.py
```
Detailed metrics, adjustable thresholds, probability visualization.

---

### 🐍 For Programmatic Testing
```python
from simple_test import CattleModelTester

tester = CattleModelTester('pipeline_output/models/model_final.pt')
result = tester.predict('cattle_image.jpg')
alert = tester.generate_alert(result)
```

---

## 📋 Table of Contents
1. [Two UI Versions](#two-ui-versions)
2. [Farmer-Friendly UI](#farmer-friendly-ui)
3. [Developer/Technical UI](#developertechnical-ui)
4. [Simple Python Script](#simple-python-script)
5. [Jupyter Notebook Testing](#jupyter-notebook-testing)
6. [Batch Testing](#batch-testing)
7. [Results Interpretation](#results-interpretation)
8. [Key Feature: Diagnosis Only for Cattle](#key-feature-diagnosis-only-for-cattle)

---

## Two UI Versions

### 🌾 Farmer-Friendly UI
**File:** `test_model_farmer_ui.py`

Perfect for:
- Farmers with no technical background
- Making quick decisions
- Getting action-based guidance
- Recording results

Run: `streamlit run test_model_farmer_ui.py`

### 🧪 Technical/Developer UI
**File:** `test_model_ui.py`

Perfect for:
- ML Engineers & Developers
- Model validation & testing
- Adjusting thresholds
- Viewing detailed metrics

Run: `streamlit run test_model_ui.py`

**See `UI_GUIDE.md` for detailed comparison**

---

## 🌾 Farmer-Friendly UI

**File:** `test_model_farmer_ui.py`

### Setup & Run

```bash
# 1. Install dependencies
pip install streamlit pillow opencv-python

# 2. Run the app
streamlit run test_model_farmer_ui.py

# 3. Open browser
# http://localhost:8501
```

### What You See

#### 1️⃣ Upload Photo
- Clear instructions on how to take a good photo
- Drag-and-drop upload

#### 2️⃣ Get Result
- ✅ **Healthy** - No action needed
- ⚠️ **Possible FMD** - Call veterinarian
- 🚨 **FMD Detected** - Isolate immediately
- 📸 **Not Cattle** - Upload cattle image

#### 3️⃣ Follow Action
- Specific steps (numbered)
- Clear guidance
- No percentages or confusing metrics

#### 4️⃣ Download Record
- Save result as text file
- Share with veterinarian
- Keep farm records

---

## 🧪 Developer/Technical UI

### Setup & Run

```bash
# 1. Install dependencies
pip install streamlit pillow opencv-python

# 2. Run the app
streamlit run test_model_ui.py

# 3. Open browser
# http://localhost:8501
```

### Features

#### 🔧 Sidebar Configuration
- Select model format (PyTorch, INT8)
- Adjust confidence threshold (0.0 - 1.0)
- Toggle display options

#### 📸 Image Upload
- Drag-and-drop upload
- Supported formats: JPG, JPEG, PNG, BMP
- Automatic preprocessing to 224×224

#### 🔍 Preprocessing Visualization
- Original image
- Resized image (224×224)
- Normalization parameters
- Tensor shape

#### 🤖 Model Predictions
**Two Tasks:**

**Quick example:**
```python
from simple_test import CattleModelTester
from pathlib import Path

# Initialize tester
tester = CattleModelTester('pipeline_output/models/model_final.pt')

# Test single image
result = tester.predict('path/to/image.jpg')
alert = tester.generate_alert(result, confidence_threshold=0.7)

# Print results
print(f"Cattle: {result['identification']['class_name']} "
      f"({result['identification']['confidence']:.1%})")
print(f"Disease: {result['diagnosis']['class_name']} "
      f"({result['diagnosis']['confidence']:.1%})")
print(f"Alert: {alert['message']}")

# Visualize
fig = tester.visualize_results('path/to/image.jpg', result)
import matplotlib.pyplot as plt
plt.show()
```

---

## Streamlit UI (Interactive)

### Setup

```bash
# 1. Install dependencies
pip install streamlit pillow opencv-python

# 2. Run the app
streamlit run test_model_ui.py

# 3. Open browser
# http://localhost:8501
```

### Features

#### 📸 Image Upload
- Drag-and-drop upload
- Supported formats: JPG, JPEG, PNG, BMP
- Automatic preprocessing to 224×224

#### 🔍 Preprocessing Visualization
- Original image
- Resized image (224×224)
- Normalization parameters
- Tensor shape

#### 🤖 Model Predictions
**Two Tasks:**

1. **Identification Task** (Cattle vs Non-Cattle)
   - Probability: 0.0 - 1.0
   - Threshold: Adjustable via sidebar
   
2. **Diagnosis Task** (Healthy vs FMD)
   - Probability: 0.0 - 1.0
   - Only computed if cattle detected

#### 📊 Visualization
- Probability bar charts
- Confidence metrics
- Alert level indicators
- Test statistics

#### 💾 Export
- Download test results as Markdown
- Results include:
  - Image metadata
  - All predictions with confidence
  - Alert decision
  - Timestamp

### Example Workflow

```
1. Upload image → See preprocessing
2. View probabilities → Check confidence
3. Check alert level → See recommended action
4. Download results → Save for records
```

---

## Simple Python Script

### Setup

```bash
# No extra dependencies needed beyond training requirements
python simple_test.py
```

### Basic Usage

```python
from simple_test import CattleModelTester

# 1. Initialize
tester = CattleModelTester('pipeline_output/models/model_final.pt')

# 2. Test single image
result = tester.predict('image.jpg')
print(result)

# 3. Generate alert
alert = tester.generate_alert(result, confidence_threshold=0.7)
print(alert['message'])

# 4. Visualize
fig = tester.visualize_results('image.jpg', result)
```

### API Reference

#### `CattleModelTester(model_path, device='auto')`

**Parameters:**
- `model_path` (str): Path to model file (model_final.pt)
- `device` (str): 'cuda', 'cpu', or 'auto'

**Methods:**

##### `predict(image_path) -> Dict`
Run inference on an image.

**Returns:**
```python
{
    'identification': {
        'class_id': 0,                    # 0=Cattle, 1=Not-Cattle
        'class_name': 'Cattle',
        'confidence': 0.9635,             # 0.0 - 1.0
        'probabilities': {
            'cattle': 0.9635,
            'not_cattle': 0.0365
        }
    },
    'diagnosis': {
        'class_id': 0,                    # 0=Healthy, 1=FMD
        'class_name': 'Healthy',
        'confidence': 0.8245,
        'probabilities': {
            'healthy': 0.8245,
            'fmd': 0.1755
        }
    }
}
```

##### `generate_alert(result, confidence_threshold=0.7) -> Dict`
Generate alert based on predictions.

**Returns:**
```python
{
    'level': 'OK',                        # OK, UNCLEAR, WARNING, CRITICAL, INFO
    'emoji': '✅',
    'message': '✅ Cattle appears healthy',
    'action': 'Continue monitoring',
    'severity': 0                          # 0-3 (higher = more severe)
}
```

##### `visualize_results(image_path, result, title=None) -> Figure`
Create visualization figure.

##### `test_batch(image_dir, save_results=True) -> List[Dict]`
Test multiple images in a directory.

---

## Jupyter Notebook Testing

You can also test directly in your notebook:

```python
# In your notebook

import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# Load model
model_path = 'pipeline_output/models/model_final.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [Load model code here - see simple_test.py for full definition]
model = CattleMultiTaskModel(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Preprocess image
image = Image.open('test_image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    id_logits, diag_logits = model(tensor)
    id_probs = torch.softmax(id_logits, dim=1)
    diag_probs = torch.softmax(diag_logits, dim=1)

# Display results
print(f"Cattle: {id_probs[0, 0]:.1%}")
print(f"Not-Cattle: {id_probs[0, 1]:.1%}")
print(f"Healthy: {diag_probs[0, 0]:.1%}")
print(f"FMD: {diag_probs[0, 1]:.1%}")
```

---

## Batch Testing

### Test Multiple Images

```python
from simple_test import CattleModelTester

tester = CattleModelTester('pipeline_output/models/model_final.pt')

# Test all images in a directory
results = tester.test_batch('path/to/test_images/', save_results=True)

# Results saved to: path/to/test_images/test_results.json
```

### Results are saved as JSON:

```json
[
  {
    "image": "cattle_1.jpg",
    "result": {
      "identification": {
        "class_id": 0,
        "class_name": "Cattle",
        "confidence": 0.9635,
        ...
      },
      "diagnosis": {...}
    },
    "alert": {
      "level": "OK",
      "message": "✅ Cattle appears healthy",
      ...
    }
  }
]
```

---

## Results Interpretation

### Alert Levels

#### ✅ **OK** (Green)
```
Message: Cattle appears healthy
Action: Continue monitoring (check again in 2 weeks)
Severity: 0 (Low)
```

**Meaning:** Model is confident the cattle is healthy.
**What to do:** Regular monitoring schedule.

---

#### ℹ️ **INFO** (Blue)
```
Message: This is not a cattle
Action: Try another image with cattle
Severity: 0 (Low)
```

**Meaning:** Model classified image as non-cattle animal.
**What to do:** Upload image with cattle.

---

#### ❓ **UNCLEAR** (Orange)
```
Message: Unclear if this is cattle (confidence: 45%)
Action: Ask for better angle/lighting
Severity: 1 (Low-Medium)
```

**Meaning:** Model is unsure if the animal is cattle (below threshold).
**What to do:** Take another photo with better lighting/angle.

---

#### ⚠️ **WARNING** (Orange-Red)
```
Message: ⚠️ CAUTION: Possible FMD (confidence: 72%)
Action: Contact veterinarian for confirmation
Severity: 2 (Medium)
```

**Meaning:** Model detected possible FMD but not highly confident.
**What to do:** Contact veterinarian for secondary confirmation.

---

#### 🚨 **CRITICAL** (Red)
```
Message: 🚨 HIGH RISK: FMD DETECTED (confidence: 92%)
Action: ISOLATE HERD - Contact veterinarian immediately
Severity: 3 (High)
```

**Meaning:** Model is highly confident FMD is present.
**What to do:** Immediately isolate animal and contact veterinarian.

---

### Understanding Confidence Scores

**Identification Confidence (Cattle vs Not-Cattle):**
- **>95%**: Very confident cattle
- **80-95%**: Confident cattle
- **70-80%**: Moderately confident
- **50-70%**: Low confidence (get clearer image)
- **<50%**: Likely not cattle

**Diagnosis Confidence (Healthy vs FMD):**
- **>90%**: Very high confidence
- **75-90%**: High confidence
- **60-75%**: Moderate confidence
- **50-60%**: Low confidence (needs vet confirmation)

---

## 🐄 Key Feature: Diagnosis ONLY for Cattle Images

### Important Rule
**FMD (Foot-and-Mouth Disease) detection only works for cattle images.**

This is by design - not a limitation.

### When Image is NOT Cattle

Both UIs will show:
```
❌ This is NOT a cattle
❌ Diagnosis: NOT COMPUTED
❌ Action: "Upload a cattle image"
```

**Examples of non-cattle:**
- Dogs, cats, horses
- Other farm animals
- People, objects, etc.

**What happens:**
1. Image uploaded
2. Identification task runs: "Not cattle"
3. Diagnosis task SKIPPED (no point in diagnosing FMD on a dog!)
4. Result: "Please upload cattle image"

### When Image IS Cattle

Both UIs will show:
```
✅ Cattle detected
✅ Diagnosis: COMPUTED (Healthy or FMD)
✅ Full result with specific action
```

**Examples of cattle:**
- Dairy cattle
- Beef cattle
- Various cattle breeds

**What happens:**
1. Image uploaded
2. Identification task runs: "Cattle detected"
3. Diagnosis task RUNS: "Healthy" or "FMD"
4. Result: Health status with action items

### Why This Design?

FMD is a cattle-specific disease. It doesn't exist in other animals. Running disease diagnosis on a non-cattle image would produce meaningless results.

**Example of what we DON'T do:**
```
❌ Wrong: Upload dog photo
❌ Wrong: System says "Dog is healthy" or "Dog has FMD"
❌ Wrong: Confusing for users
```

**What we DO instead:**
```
✅ Right: Upload dog photo
✅ Right: System says "This is not cattle"
✅ Right: Clear guidance: "Upload cattle image for FMD check"
```

### Testing This Feature

To verify this works:
1. **Test with cattle image** → Should show diagnosis
2. **Test with non-cattle image** → Should NOT show diagnosis
3. **Check result carefully** → Diagnosis field should be `null` for non-cattle

**Developer View:**
```python
result = model_test(non_cattle_image)
print(result['diagnosis'])  # Should be None
```

**Farmer View:**
```
📸 This is NOT a cattle!
Please take a picture of your cattle and try again.
```

Both show the same thing, just different language.

---

### Model Performance Context

**Test Set Metrics:**

| Metric | Value | Meaning |
|--------|-------|---------|
| **Identification Accuracy** | 96.35% | Model correctly identifies cattle 96% of the time |
| **Diagnosis Accuracy** | 85.82% | Model correctly diagnoses health status 86% of the time |
| **FMD Recall** | 91.82% | Model catches 9 out of 10 FMD cases |
| **False Alarm Rate** | 8.2% | ~8% of healthy cattle flagged as FMD (acceptable) |

**Interpretation:**
- ✅ Cattle identification is very reliable (96%+)
- ✅ FMD detection catches most cases (91.82%)
- ✅ False alarms are acceptable (vet can verify)

---

## Troubleshooting

### Issue: Model not found
**Solution:** Ensure `pipeline_output/models/model_final.pt` exists

### Issue: Out of memory error
**Solution:** Use CPU instead
```python
tester = CattleModelTester(model_path, device='cpu')
```

### Issue: Image not loading
**Solution:** Ensure image format is JPG, PNG, or BMP

### Issue: Very low confidence scores
**Cause:** Image quality is poor (blurry, dark, unusual angle)
**Solution:** Retake photo with better lighting and focus

### Issue: Wrong predictions
**Cause:** Edge case image the model hasn't seen
**Report:** Save the image for model retraining

---

## Batch Testing Example

```bash
# Test all images in a directory
python -c "
from simple_test import CattleModelTester
tester = CattleModelTester('pipeline_output/models/model_final.pt')
results = tester.test_batch('test_images/')
print(f'Tested {len(results)} images')
"
```

---

## Next Steps

After testing locally:

1. **Export Model**: Use testing results to validate model performance
2. **Fine-tune if needed**: If accuracy is unsatisfactory, retrain
3. **Deploy**: Use the model in production (see DEPLOYMENT_ARCHITECTURE.md)
4. **Monitor**: Continue testing with field data
5. **Improve**: Use farmer feedback for retraining

---

## Questions?

Refer to:
- **MODEL_INTEGRATION_GUIDE.md** - Integration examples
- **PROJECT_DOCUMENTATION.md** - Complete architecture
- **DEPLOYMENT_ARCHITECTURE.md** - Production setup

Good luck with your testing! 🚀
