# 🚀 Quick Start - Two UIs for FMD Detection

## 🌾 For Farmers: Simple & Action-Focused

```bash
streamlit run test_model_farmer_ui.py
```

**What you see:**
```
📸 Upload photo of cattle
   ↓
Result in plain language:
✅ "Cattle is healthy. Check in 2 weeks."
OR
⚠️ "Possible FMD. Call veterinarian today."
OR
🚨 "FMD DETECTED. ISOLATE IMMEDIATELY."
```

**Perfect for:**
- Taking quick decisions
- Farmers with no technical background
- Getting actionable guidance
- Saving records

---

## 🧪 For Developers: Detailed & Technical

```bash
streamlit run test_model_ui.py
```

**What you see:**
```
⚙️ Adjust settings in sidebar (threshold, model format)
   ↓
📊 View detailed probabilities in tabs
   - 🎯 Alert (severity level)
   - 📊 Probabilities (percentage bars)
   - 📈 Details (confidence metrics)
   ↓
📈 See preprocessing visualization (4 steps)
```

**Perfect for:**
- Validating model performance
- Testing different thresholds
- Debugging & development
- Exporting detailed metrics

---

## 🐄 Core Feature: FMD Detection for Cattle ONLY

### Both UIs do this:

**If NOT CATTLE:**
```
❌ "This is not a cattle"
❌ NO diagnosis shown
❌ "Upload cattle image"
```

**If IS CATTLE:**
```
✅ "Cattle detected"
✅ Diagnosis: Healthy or FMD
✅ Complete result
```

**Why?** FMD only exists in cattle. No point diagnosing FMD on a dog! 🐕

---

## 📋 Quick Comparison

| What | Farmer | Developer |
|------|--------|-----------|
| **Run** | `streamlit run test_model_farmer_ui.py` | `streamlit run test_model_ui.py` |
| **Style** | Large text, simple | Detailed metrics |
| **Settings** | None | Full sidebar controls |
| **Probabilities** | Hidden (optional) | Always visible |
| **Threshold** | Fixed | Adjustable |
| **Export** | Text file | Text file |
| **Use Case** | Decision making | Model testing |

---

## ✅ Checklist: Both UIs Work Correctly

- [x] Farmer UI installed: `test_model_farmer_ui.py` ✅
- [x] Developer UI updated: `test_model_ui.py` ✅
- [x] Conditional diagnosis logic: Only for cattle ✅
- [x] Non-cattle alert: Clear guidance ✅
- [x] Cattle results: Full diagnosis ✅
- [x] Model file: `pipeline_output/models/model_final.pt` ✅

---

## 🎯 Project Summary

**What:** Cattle Foot-and-Mouth Disease (FMD) Detection  
**Model:** MobileNetV3-Small (Dual-task learning)  
**Task 1:** Is it cattle? (96.35% accurate)  
**Task 2:** Is it healthy or FMD? (85.82% accurate, 91.82% FMD recall)  

**Two UIs:**
- 🌾 Farmer version → Actions, not percentages
- 🧪 Developer version → Metrics, thresholds, details

**Key Rule:** Diagnosis only shows for cattle images (non-cattle images ask for cattle photo)

---

## 🆘 If Something Goes Wrong

### Can't find model?
```bash
# Check this file exists:
ls pipeline_output/models/model_final.pt
```

### Streamlit won't start?
```bash
pip install streamlit torch torchvision pillow opencv-python
# Then run again
```

### Diagnosis showing for non-cattle?
- This shouldn't happen (we check for cattle first)
- If it does, contact the developer
- This is a bug to fix

### Wrong results?
- Upload clearer photo (better lighting, closer view)
- Verify it's actually cattle
- Test with developer UI (adjust threshold)

---

## 📚 Full Documentation

- `UI_GUIDE.md` - Detailed guide (what you're reading above)
- `TESTING_GUIDE.md` - How to use both UIs
- `TESTING_UPDATE.md` - Why diagnosis only for cattle
- `MODEL_INTEGRATION_GUIDE.md` - Use in your app
- `DEPLOYMENT_ARCHITECTURE.md` - Production setup

---

**You have TWO ways to test the FMD detection model:**
1. 🌾 **Farmer way:** Simple, action-based, no technical terms
2. 🧪 **Developer way:** Detailed, technical, full control

**Pick whichever matches your needs.** Both work the same way under the hood. ✅
