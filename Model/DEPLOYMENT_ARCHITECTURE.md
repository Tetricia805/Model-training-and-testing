# 🚀 Deployment Architecture & Application Integration Guide

## Executive Summary

This document provides a complete application architecture for deploying the cattle disease detection model into production. It covers farmer mobile apps, veterinarian dashboards, and backend systems.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Farmer Mobile App](#farmer-mobile-app)
3. [Veterinarian Dashboard](#veterinarian-dashboard)
4. [Backend Services](#backend-services)
5. [Data Pipeline & Feedback Loop](#data-pipeline--feedback-loop)
6. [Deployment Guide](#deployment-guide)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FARMER MOBILE APPS                          │
│  (iOS / Android) - Offline-first model inference on device      │
└────────────────────┬────────────────────────────────────────────┘
                     │ Upload flagged cases
                     │ (wifi when available)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  API SERVER (FastAPI)                           │
│  - Case management  - Model inference  - Analytics              │
│  - User authentication  - Role-based access                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    PostgreSQL   Redis Cache   S3 Storage
    (Case data)  (Model cache) (Images)
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           VETERINARIAN WEB DASHBOARD                            │
│  (React.js) - Review cases, approve alerts, manage follow-up    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Farmer takes photo 📸
    ↓
Phone preprocesses image (224×224)
    ↓
On-device model inference (INT8 quantized)
    ↓
    ├─→ If Cattle + Healthy → Log to local database
    │   ├─→ No alert (monitoring continues)
    │   └─→ Sync with server when online
    │
    ├─→ If Cattle + FMD → 🚨 LOCAL ALERT
    │   ├─→ Blue notification on phone
    │   ├─→ Save case locally
    │   ├─→ Upload to server immediately (if connected)
    │   └─→ Store encrypted on device (if offline)
    │
    └─→ If Not-Cattle → Info message
        ├─→ Ask user to re-image
        └─→ Log for model retraining
```

---

## Farmer Mobile App

### Requirements

**Target Device:**
- Android 8.0+ (API Level 26+)
- RAM: 2GB minimum
- Storage: 100MB free
- Network: Offline-first (optional internet)

**Technology Stack:**
```
Framework: React Native / Flutter / Native (Kotlin/Swift)
Model Runtime: ONNX Runtime Mobile
Image Processing: Camera2 API (Android) / AVFoundation (iOS)
Local Storage: SQLite
Sync: Background sync with server
```

### Feature Specification

#### 1. Camera Capture Module

```python
# User interface flow
1. Open Camera App
2. Farmer frames cattle (guidance overlay: "Show muzzle/hooves")
3. Capture photo
4. Auto-preprocessing (resize, normalize)
5. Display inference loading indicator
```

**Code Example (React Native):**
```javascript
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

export function CattleCamera({ onDetection }) {
  const runInference = async (imagePath) => {
    // Load ONNX Session
    const session = await ort.InferenceSession.create(
      'models/model_final.onnx'
    );
    
    // Preprocess
    const tensor = await preprocessImage(imagePath);
    
    // Inference
    const result = await session.run({ image: tensor });
    
    // Parse results
    const identification = result.identification.data;
    const diagnosis = result.diagnosis.data;
    
    onDetection({
      cattle: identification[0] > identification[1],
      disease: diagnosis[1] > diagnosis[0],
      confidence: Math.max(
        Math.max(...identification),
        Math.max(...diagnosis)
      )
    });
  };
  
  return (
    <Camera style={styles.camera} onPictureSaved={runInference}>
      <GuideOverlay />
    </Camera>
  );
}
```

#### 2. Result Display & Actions

**Alert Levels:**

```
┌─────────────────────────────────────────────────────┐
│                RESULT SCREENS                       │
├─────────────────────────────────────────────────────┤

CASE 1: Cattle + Healthy
┌───────────────────────────────────────────────────┐
│  ✅ HEALTHY CATTLE                                │
│                                                   │
│  Identification: 96.8%                           │
│  Status: Healthy                                 │
│                                                   │
│  [MONITOR]  [RETAKE]  [SAVE]                    │
│                                                   │
│  Tips: Check again in 2 weeks                    │
└───────────────────────────────────────────────────┘

CASE 2: Cattle + FMD
┌───────────────────────────────────────────────────┐
│  🚨 ALERT: POSSIBLE FMD DETECTED                 │
│                                                   │
│  Identification: 99.2%                           │
│  Condition: FMD Risk (87% confidence)            │
│                                                   │
│  URGENT ACTIONS:                                 │
│  [CONTACT VET]  [QUARANTINE]  [SAVE]            │
│                                                   │
│  ⚠️  Isolate from herd immediately               │
│  📞 Call veterinarian: [VET PHONE]               │
│  📍 Case saved & queued for upload               │
└───────────────────────────────────────────────────┘

CASE 3: Not-Cattle
┌───────────────────────────────────────────────────┐
│  ℹ️  NOT A CATTLE                                 │
│                                                   │
│  Detected: Goat / Sheep / Other Animal           │
│                                                   │
│  [RETAKE PHOTO]  [NEXT ANIMAL]                   │
│                                                   │
│  Note: Model works best with cattle photos       │
└───────────────────────────────────────────────────┘
```

#### 3. Local Database Schema

```sql
-- Captured cases (stores locally, syncs later)
CREATE TABLE cases (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  image_path TEXT NOT NULL,
  cattle_confidence FLOAT NOT NULL,
  disease_confidence FLOAT NOT NULL,
  disease_label INTEGER NOT NULL,  -- 0=Healthy, 1=FMD
  synced BOOLEAN DEFAULT FALSE,
  farmer_notes TEXT
);

-- Sync log (tracks what was uploaded)
CREATE TABLE sync_log (
  id INTEGER PRIMARY KEY,
  case_id INTEGER,
  sync_timestamp DATETIME,
  server_response_code INTEGER,
  FOREIGN KEY(case_id) REFERENCES cases(id)
);

-- Cached model info
CREATE TABLE model_metadata (
  version TEXT,
  accuracy FLOAT,
  last_updated DATETIME
);
```

#### 4. Background Sync

```python
# Runs when device connects to WiFi
def sync_cases_to_server():
    unsynced_cases = db.query("SELECT * FROM cases WHERE synced = 0")
    
    for case in unsynced_cases:
        try:
            # Upload case
            response = requests.post(
                'https://api.cattledetector.org/cases/upload',
                files={'image': open(case.image_path, 'rb')},
                data={
                    'disease_detected': case.disease_label,
                    'confidence': case.disease_confidence,
                    'timestamp': case.timestamp
                },
                headers={'Authorization': f'Bearer {farmer_token}'}
            )
            
            if response.status_code == 200:
                db.execute("UPDATE cases SET synced = 1 WHERE id = ?", [case.id])
                db.execute(
                    "INSERT INTO sync_log VALUES (?, ?, ?, ?)",
                    [None, case.id, datetime.now(), 200]
                )
        except requests.ConnectionError:
            # Retry later
            continue
```

#### 5. User Authentication

```javascript
// Login flow
async function farmLogin(phoneNumber, pin) {
  // Send OTP to phone
  const response = await fetch('https://api.cattledetector.org/auth/otp', {
    method: 'POST',
    body: JSON.stringify({ phone: phoneNumber })
  });
  
  // User enters OTP
  const verified = await verifyOTP(userOTP);
  
  if (verified) {
    // Get access token
    const token = await fetch('https://api.cattledetector.org/auth/verify', {
      method: 'POST',
      body: JSON.stringify({ phone: phoneNumber, otp: userOTP })
    });
    
    // Store token securely (Keychain/Keystore)
    SecureStore.setItem('auth_token', token.access_token);
    
    // Return to home screen
    return { success: true, farmer_id: token.farmer_id };
  }
}
```

---

## Veterinarian Dashboard

### Technology Stack

```
Frontend: React.js (TypeScript)
UI Components: Material-UI / Ant Design
Real-time Updates: WebSocket / Socket.io
State Management: Redux / Zustand
Charts: Chart.js / Recharts
```

### Dashboard Views

#### 1. Alert Management

```
┌────────────────────────────────────────────────────────────┐
│  🏥 VETERINARIAN PORTAL - Alert Management               │
├────────────────────────────────────────────────────────────┤

FILTERS: [High Priority] [All Cases] [Awaiting Review]     [By Region]

┌─ CASE #2847 ─────── PRIORITY: CRITICAL ─────── TODAY 10:45 AM ─┐
│                                                                  │
│ Farmer: Okech Francis (Kampala District)                       │
│ Cattle: Female Holstein (Brown/White)                          │
│ Identified: Cattle (99.2%)                                     │
│ Disease: FMD Risk (89% confidence) 🚨                          │
│                                                                │
│ [IMAGE] [IMAGE] [IMAGE] [DOWNLOAD ALL]                        │
│  (3 photos submitted)                                          │
│                                                                │
│ Model Prediction: FMD Positive                              │
│ Veterinarian Assessment:                                      │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ ✓ Approve - This is FMD (high confidence)              │  │
│ │ ✗ Reject - False positive (actually healthy)           │  │
│ │ ❓ Unclear - Need more images / information             │  │
│ │                                                        │  │
│ │ Notes: [________________________________________]       │  │
│ │                                                        │  │
│ │ [SUBMIT ASSESSMENT]                                   │  │
│ │                                                        │  │
│ │ Actions:                                              │  │
│ │ ☐ Send reply to farmer                                │  │
│ │ ☐ Schedule follow-up visit                            │  │
│ │ ☐ Report to agricultural authorities                  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                                │
│ Model Performance on This Case:                               │
│ • Identification Accuracy: 96.35%                             │
│ • Diagnosis Accuracy: 85.82%                                 │
│ • FMD Recall: 91.82% (strong on detection)                   │
│                                                                │
└──────────────────────────────────────────────────────────────┘

[Previous Case] [Next Case] [Mark as Done]
```

#### 2. Analytics Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  📊 ANALYTICS                                           │
├─────────────────────────────────────────────────────────┤

Period: [Last 30 Days] [This Quarter] [All Time]

┌─────────────────────────────────────────────────────────┐
│ CASE STATISTICS                                         │
├────────────┬──────────┬──────────┬──────────────────────┤
│ Total:     │ 456      │ ▲ 45%    │ vs. Previous Month  │
│ FMD Cases: │ 23       │ ▼ 8%     │ vs. Previous Month  │
│ Pending:   │ 12       │ ▲ 120%   │ vs. Previous Month  │
│ Approved:  │ 89       │          │ (Model predictions) │
│ Rejected:  │ 15       │          │ (False positives)   │
└────────────┴──────────┴──────────┴──────────────────────┘

CASE TIMELINE (30 days)
┌─────────────────────────────────────────────────────────┐
│ Cases Submitted:  ███████████████ 456                  │
│ FMD Detected:     ███ 23                                │
│ Confirmed FMD:    ██ 18                                 │
│                                                        │
│ Confirmation Rate: 78% (18/23)                         │
│ False Positive Rate: 22% (5/23 were healthy)           │
└─────────────────────────────────────────────────────────┘

MODEL PERFORMANCE
┌──────────────────────────────────────────────────────────┐
│ Identification Accuracy: 96.35% (Field Validation)      │
│ ─────────────────────────────────────────────────────── │
│ ████████████████ ▌ 96.35%                              │
│                                                        │
│ FMD Recall: 91.82% (Catches disease)                   │
│ ─────────────────────────────────────────────────────── │
│ ████████████████ ▌ 91.82%                              │
│                                                        │
│ False Alarm Rate: 8.2% (acceptable)                    │
│ ─────────────────────────────────────────────────────── │
│ █ 8.2%                                                 │
└──────────────────────────────────────────────────────────┘
```

#### 3. Model Performance Monitoring

```javascript
// React component example
export function ModelPerformanceMonitor() {
  const [metrics, setMetrics] = useState(null);
  
  useEffect(() => {
    // Fetch real-time metrics
    fetch('/api/model/performance?days=30')
      .then(r => r.json())
      .then(data => setMetrics(data));
  }, []);
  
  return (
    <div className="performance-dashboard">
      <MetricCard 
        title="Cattle Detection Accuracy"
        value={metrics?.identification_accuracy}
        benchmark={0.9635}
        trend="+2.1%"
      />
      <MetricCard 
        title="FMD Recall (Critical)"
        value={metrics?.fmd_recall}
        benchmark={0.9182}
        trend="+1.5%"
      />
      <ConfusionMatrixChart data={metrics?.confusion_matrix} />
      <CaseTrendChart data={metrics?.case_timeline} />
    </div>
  );
}
```

---

## Backend Services

### FastAPI Server Architecture

```python
# main.py - FastAPI Backend

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from datetime import datetime

app = FastAPI(title="Cattle Disease Detection API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.cattledetector.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (cached)
@app.on_event("startup")
async def load_model():
    global onnx_session
    onnx_session = ort.InferenceSession(
        "models/model_final.onnx",
        providers=['CPUExecutionProvider']  # Or ['CUDAExecutionProvider']
    )

def preprocess_image(image_bytes: bytes):
    """Convert uploaded image to model input tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Normalize
    img_array = np.array(img).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Reshape: (H, W, C) → (1, C, H, W)
    return np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]

@app.post("/api/cases/upload")
async def upload_case(
    image: UploadFile = File(...),
    farmer_token: str = Header(...)
):
    """Upload case and run inference"""
    
    # Authenticate farmer
    farmer = verify_auth_token(farmer_token)
    if not farmer:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Read and preprocess image
    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes)
    
    # Run inference
    outputs = onnx_session.run(
        None,  # Run all outputs
        {'image': tensor}
    )
    
    id_logits, diag_logits = outputs
    id_probs = softmax(id_logits[0])
    diag_probs = softmax(diag_logits[0])
    
    # Store in database
    case = Case(
        farmer_id=farmer.id,
        timestamp=datetime.utcnow(),
        image_url=save_image(image_bytes),
        cattle_detected=id_logits[0][0] > id_logits[0][1],
        cattle_confidence=float(id_probs[0]),
        disease_detected=diag_logits[0][1] > diag_logits[0][0],
        disease_confidence=float(diag_probs[1]),
        model_version='1.0'
    )
    
    db.add(case)
    db.commit()
    
    # Alert if FMD
    if case.disease_detected and case.disease_confidence > 0.85:
        send_alert_to_veterinarians(case)
    
    return {
        'case_id': case.id,
        'identification': {
            'cattle': bool(case.cattle_detected),
            'confidence': case.cattle_confidence
        },
        'diagnosis': {
            'fmd': bool(case.disease_detected),
            'confidence': case.disease_confidence
        }
    }

@app.get("/api/vet/pending-cases")
async def get_pending_cases(vet_token: str = Header(...)):
    """Get cases awaiting veterinarian review"""
    vet = verify_auth_token(vet_token)
    if not vet:
        raise HTTPException(status_code=401)
    
    # Get all high-confidence cases not yet reviewed
    cases = db.query(Case).filter(
        Case.disease_confidence > 0.7,
        Case.veterinarian_assessment == None
    ).order_by(Case.timestamp.desc()).all()
    
    return [case.to_dict() for case in cases]

@app.post("/api/vet/assess")
async def submit_assessment(
    case_id: int,
    assessment: str,  # 'approve', 'reject', 'unclear'
    notes: str,
    vet_token: str = Header(...)
):
    """Veterinarian submits assessment"""
    vet = verify_auth_token(vet_token)
    case = db.query(Case).filter(Case.id == case_id).first()
    
    if not case:
        raise HTTPException(status_code=404)
    
    # Save assessment
    case.veterinarian_assessment = assessment
    case.veterinarian_notes = notes
    case.assessed_by_vet_id = vet.id
    case.assessment_timestamp = datetime.utcnow()
    
    db.commit()
    
    # Record for model improvement
    if assessment == 'approve' and case.disease_detected:
        # True positive - FMD correctly identified
        log_for_retraining('TP', case)
    elif assessment == 'reject' and case.disease_detected:
        # False positive - healthy but flagged
        log_for_retraining('FP', case)
    
    return {'status': 'success', 'assessment_recorded': True}

@app.get("/api/model/performance")
async def get_model_performance(days: int = 30):
    """Get model performance metrics"""
    # Calculate metrics from database
    cases = db.query(Case).filter(
        Case.assessment_timestamp >= datetime.utcnow() - timedelta(days=days)
    ).all()
    
    tp = sum(1 for c in cases if c.veterinarian_assessment == 'approve')
    fp = sum(1 for c in cases if c.veterinarian_assessment == 'reject')
    
    return {
        'accuracy': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'cases_processed': len(cases),
        'fmd_recall': tp / sum(1 for c in cases if c.disease_detected) 
                       if any(c.disease_detected for c in cases) else 0
    }

def softmax(x):
    """Convert logits to probabilities"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

### Database Schema

```sql
-- Users
CREATE TABLE farmers (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE veterinarians (
    id SERIAL PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(100),
    license_number VARCHAR(50),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cases
CREATE TABLE cases (
    id SERIAL PRIMARY KEY,
    farmer_id INTEGER NOT NULL REFERENCES farmers(id),
    timestamp TIMESTAMP NOT NULL,
    image_url VARCHAR(500),
    cattle_detected BOOLEAN,
    cattle_confidence FLOAT,
    disease_detected BOOLEAN,
    disease_confidence FLOAT,
    model_version VARCHAR(20),
    
    veterinarian_assessment VARCHAR(20),  -- 'approve', 'reject', 'unclear'
    veterinarian_id INTEGER REFERENCES veterinarians(id),
    veterinarian_notes TEXT,
    assessment_timestamp TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model retraining logs
CREATE TABLE retraining_data (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES cases(id),
    prediction_type VARCHAR(20),  -- 'TP', 'FP', 'FN', 'TN'
    veterinarian_feedback VARCHAR(20),  -- 'correct', 'incorrect'
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Data Pipeline & Feedback Loop

### Continuous Improvement Workflow

```
1. FARMER USES APP
   ├─→ Takes photo
   ├─→ Model predicts locally
   ├─→ Case synced to server

2. VETERINARIAN REVIEWS
   ├─→ Sees alerts in dashboard
   ├─→ Reviews multiple photos
   ├─→ Submits assessment (approve/reject/unclear)

3. MODEL LEARNS
   ├─→ Assessment recorded in database
   ├─→ True positives & false positives logged
   ├─→ Hard examples identified

4. MONTHLY RETRAINING
   ├─→ Extract corrected cases
   ├─→ Create new training set: original + corrections
   ├─→ Re-train with same 2-phase strategy
   ├─→ Deploy improved model v1.1

5. CONTINUOUS MONITORING
   ├─→ Track metrics: accuracy, recall, false negatives
   ├─→ Alert if performance degrades
   ├─→ Analyze misclassifications
```

### Retraining Pipeline Example

```python
# monthly_retrain.py - Scheduled monthly job

from datetime import datetime, timedelta
from sqlalchemy import func

def monthly_retraining():
    """Retrain model with vet feedback"""
    
    # 1. Gather corrected cases from past month
    start_date = datetime.utcnow() - timedelta(days=30)
    
    corrected_cases = db.query(Case).filter(
        Case.assessment_timestamp >= start_date,
        Case.veterinarian_assessment != None
    ).all()
    
    print(f"Found {len(corrected_cases)} cases for retraining")
    
    # 2. Create new training set
    new_train_data = []
    for case in corrected_cases:
        # Load original image
        image = download_image(case.image_url)
        
        # Get corrected label (from vet)
        if case.veterinarian_assessment == 'approve':
            # If vet approved: use model prediction as label
            label = 1 if case.disease_detected else 0
        else:
            # If vet rejected: flip the label
            label = 0 if case.disease_detected else 1
        
        new_train_data.append((image, label))
    
    # 3. Merge with original training data
    original_train_set = load_original_training_data('processed_data/train_annotations.csv')
    combined_train_set = original_train_set + new_train_data
    
    # 4. Retrain model (same 2-phase strategy)
    model_v1_1 = train_model_two_phase(
        combined_train_set,
        epochs_phase1=20,
        epochs_phase2=15
    )
    
    # 5. Evaluate on held-out test set
    metrics = evaluate_model(model_v1_1, load_test_set())
    
    print(f"New Model Performance:")
    print(f"  ID Accuracy: {metrics['id_accuracy']:.1%}")
    print(f"  Diagnosis Accuracy: {metrics['diag_accuracy']:.1%}")
    print(f"  FMD Recall: {metrics['fmd_recall']:.1%}")
    
    # 6. If improved, deploy
    if metrics['fmd_recall'] > 0.918:  # Must beat v1.0
        save_model(model_v1_1, 'models/model_v1.1.onnx')
        update_deployment('models/model_v1.1.onnx')
        notify_farmers("Model updated: improved FMD detection!")
    else:
        print("Model improvement insufficient, keeping v1.0")
```

---

## Deployment Guide

### Deployment Stages

#### Stage 1: Validation Deployment (Weeks 1-2)
```
✓ Deploy to 5-10 veterinarians
✓ Run in validation mode (model predicts, but doesn't alert)
✓ Collect veterinarian feedback
✓ Measure accuracy on ground truth
✓ Decide: proceed or iterate?
```

#### Stage 2: Pilot Deployment (Weeks 3-8)
```
✓ Deploy to 50-100 farmers in 1 region
✓ Monitor model predictions vs vet feedback
✓ Gather performance data
✓ Fix any bugs or UX issues
✓ Measure real-world accuracy
```

#### Stage 3: Regional Rollout (Months 2-3)
```
✓ Expand to other regions (100-1000 farmers)
✓ Establish feedback loop with local veterinarians
✓ Monitor system performance
✓ Begin monthly retraining with new data
```

#### Stage 4: National Deployment (Month 4+)
```
✓ Full rollout across Uganda
✓ Integrate with agricultural authorities
✓ Continuous model improvement
✓ Support local language versions
```

### Server Infrastructure

**Recommended AWS Setup:**

```yaml
Application Server:
  - EC2: t3.large (2vCPU, 8GB RAM)
  - Auto-scaling group (2-10 instances)
  - Load balancer (ALB)

Database:
  - RDS PostgreSQL: db.t3.medium (2vCPU, 4GB RAM)
  - Automated backups (daily)
  - Multi-AZ for HA

Cache:
  - ElastiCache Redis: cache.t3.small
  - Model cache (20GB)
  - Session cache

Storage:
  - S3: Case images (SSL encrypted)
  - CloudFront: CDN for image delivery
  - Lifecycle: Delete after 90 days

Monitoring:
  - CloudWatch: Logs, metrics, alarms
  - New Relic / DataDog: APM
  - PagerDuty: On-call alerting
```

**Cost Estimate (Monthly):**
```
EC2:              $100-300
RDS:              $200-400
ElastiCache:      $50-100
S3:               $50-200  (depends on image volume)
Bandwidth:        $100-300
Monitoring:       $50-100
─────────────
TOTAL:           $550-1,400/month
```

### Monitoring & Alerting

```python
# monitoring_setup.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
inference_counter = Counter(
    'model_inferences_total',
    'Total model inferences',
    ['status']  # 'success', 'error'
)

inference_latency = Histogram(
    'model_inference_seconds',
    'Model inference latency'
)

active_users = Gauge(
    'active_farmers',
    'Number of active farmers (last 7 days)'
)

# Instrument API
@app.post("/api/cases/upload")
async def upload_case(...):
    
    start_time = time.time()
    
    try:
        # ... inference code ...
        
        inference_counter.labels(status='success').inc()
        
    except Exception as e:
        inference_counter.labels(status='error').inc()
        raise
        
    finally:
        inference_latency.observe(time.time() - start_time)

# Alerts
ALERT_RULES = {
    'HighErrorRate': 'error_rate > 0.05',
    'HighLatency': 'p95_latency > 500ms',
    'ModelDrift': 'accuracy_drop > 0.05',
    'DatabaseDown': 'db_connections == 0'
}
```

---

## 🎯 Deployment Checklist

- [ ] **Infrastructure**
  - [ ] Web server configured and tested
  - [ ] Database created and backed up
  - [ ] Redis cache configured
  - [ ] S3 bucket created with encryption
  - [ ] CDN set up for image serving

- [ ] **Backend**
  - [ ] API endpoints tested locally
  - [ ] Database migrations applied
  - [ ] Authentication system working
  - [ ] Error handling implemented
  - [ ] Rate limiting configured

- [ ] **Mobile App**
  - [ ] ONNX model bundled and tested
  - [ ] Local database working
  - [ ] Camera integration functional
  - [ ] Offline mode tested
  - [ ] Background sync working

- [ ] **Frontend Dashboard**
  - [ ] Case upload interface working
  - [ ] Real-time alerts functional
  - [ ] Assessment submission working
  - [ ] Analytics dashboard loading

- [ ] **Security**
  - [ ] HTTPS/TLS enabled everywhere
  - [ ] API keys rotated
  - [ ] Database encrypted
  - [ ] Images encrypted at rest
  - [ ] User authentication tested

- [ ] **Monitoring**
  - [ ] Logging configured
  - [ ] Alerts set up
  - [ ] Dashboards created
  - [ ] Performance baselines recorded

- [ ] **Documentation**
  - [ ] API documentation complete
  - [ ] User guides written
  - [ ] Training materials prepared
  - [ ] Support procedures documented

---

**Ready to deploy! Good luck with your cattle disease detection system.** 🚀

For questions or issues during deployment, refer to the [MODEL_INTEGRATION_GUIDE.md](MODEL_INTEGRATION_GUIDE.md) and [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
