"""
🐄 CATTLE HEALTH CHECK - Simple Farmer App
Easy-to-use interface for checking cattle disease (FMD)
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🐄 Cattle Health Check",
    page_icon="🐄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    /* Make text larger and easier to read */
    body {
        font-size: 16px;
    }
    .big-text {
        font-size: 24px;
        font-weight: bold;
    }
    .alert-success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        color: #155724;
        font-size: 18px;
        font-weight: bold;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        padding: 20px;
        border-radius: 10px;
        color: #721c24;
        font-size: 18px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 2px solid #ftic7d;
        padding: 20px;
        border-radius: 10px;
        color: #856404;
        font-size: 18px;
        font-weight: bold;
    }
    .action-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL LOADING (CACHED)
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = Path("pipeline_output/models/model_final.pt")
        
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        class CattleMultiTaskModel(torch.nn.Module):
            def __init__(self, pretrained=False):
                super().__init__()
                if pretrained:
                    self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = mobilenet_v3_small(weights=None)
                
                self.feature_dim = self.backbone.classifier[0].in_features
                self.backbone.classifier = torch.nn.Identity()
                
                self.id_head = torch.nn.Sequential(
                    torch.nn.Linear(self.feature_dim, 256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 2)
                )
                
                self.diag_head = torch.nn.Sequential(
                    torch.nn.Linear(self.feature_dim, 256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 2)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                id_output = self.id_head(features)
                diag_output = self.diag_head(features)
                return id_output, diag_output
        
        model = CattleMultiTaskModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    except Exception as e:
        return None

@st.cache_resource
def get_device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# PREPROCESSING
# ============================================
def preprocess_image(image_pil):
    """Preprocess image for model input"""
    img = image_pil.convert('RGB')
    img_resized = img.resize((224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, img_resized

# ============================================
# INFERENCE
# ============================================
def run_inference(model, image_tensor, device):
    """Run model inference"""
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        id_logits, diag_logits = model(image_tensor)
        id_probs = torch.softmax(id_logits, dim=1)
        id_pred = id_logits.argmax(1).item()
        
        result = {
            'identification': {
                'class_id': id_pred,
                'class_name': 'Cattle' if id_pred == 0 else 'Not-Cattle',
                'confidence': id_probs[0, id_pred].item(),
                'probabilities': {
                    'cattle': id_probs[0, 0].item(),
                    'not_cattle': id_probs[0, 1].item()
                }
            },
            'diagnosis': None
        }
        
        # Only run diagnosis if cattle is detected
        if id_pred == 0:
            diag_probs = torch.softmax(diag_logits, dim=1)
            diag_pred = diag_logits.argmax(1).item()
            
            result['diagnosis'] = {
                'class_id': diag_pred,
                'class_name': 'Healthy' if diag_pred == 0 else 'FMD (Disease)',
                'confidence': diag_probs[0, diag_pred].item(),
                'probabilities': {
                    'healthy': diag_probs[0, 0].item(),
                    'fmd': diag_probs[0, 1].item()
                }
            }
    
    return result

# ============================================
# GENERATE SIMPLE ALERT FOR FARMERS
# ============================================
def generate_farmer_alert(result):
    """Generate simple, actionable alert for farmers"""
    
    id_conf = result['identification']['confidence']
    
    # NOT CATTLE
    if result['identification']['class_id'] == 1:
        return {
            'status': 'NOT_CATTLE',
            'emoji': '📸',
            'title': 'This is NOT a cattle!',
            'message': 'The picture does not show a cattle.',
            'action': 'Please take a picture of your cattle and try again.',
            'color': 'warning',
            'confidence_text': f"({id_conf*100:.0f}% sure)"
        }
    
    # LOW CONFIDENCE CATTLE
    elif id_conf < 0.7:
        return {
            'status': 'UNCLEAR_CATTLE',
            'emoji': '🤔',
            'title': 'Cannot see the cattle clearly',
            'message': 'The picture is too unclear to check if it is cattle.',
            'action': 'Take another picture with:\n• Better lighting\n• Clear view of the animal\n• Close enough to see detail',
            'color': 'warning',
            'confidence_text': f"({id_conf*100:.0f}% sure)"
        }
    
    # CATTLE DETECTED - NOW CHECK HEALTH
    if result['diagnosis'] is not None:
        diag_conf = result['diagnosis']['confidence']
        
        # HEALTHY CATTLE
        if result['diagnosis']['class_id'] == 0:
            return {
                'status': 'HEALTHY',
                'emoji': '✅',
                'title': 'CATTLE IS HEALTHY',
                'message': 'No signs of FMD disease detected.',
                'action': 'Continue normal care. Check again in 2 weeks.',
                'color': 'success',
                'confidence_text': f"(Very confident: {diag_conf*100:.0f}%)"
            }
        
        # FMD SUSPECTED
        else:
            if diag_conf > 0.85:
                return {
                    'status': 'CRITICAL_FMD',
                    'emoji': '🚨',
                    'title': 'URGENT: FMD DISEASE DETECTED!',
                    'message': 'The cattle shows strong signs of FMD (Foot-and-Mouth Disease).',
                    'action': '⚠️ IMMEDIATE ACTION NEEDED:\n1. ISOLATE this cattle from the herd\n2. CALL the veterinarian TODAY\n3. Do NOT move the animal',
                    'color': 'danger',
                    'confidence_text': f"(Very confident: {diag_conf*100:.0f}%)"
                }
            else:
                return {
                    'status': 'WARNING_FMD',
                    'emoji': '⚠️',
                    'title': 'POSSIBLE FMD DETECTED',
                    'message': 'The cattle might have FMD disease.',
                    'action': '🔔 RECOMMENDED ACTION:\n1. Isolate this cattle from others\n2. Schedule veterinarian visit TODAY\n3. Get professional confirmation',
                    'color': 'warning',
                    'confidence_text': f"({diag_conf*100:.0f}% confidence)"
                }
    
    return {
        'status': 'UNKNOWN',
        'emoji': '❓',
        'title': 'Unable to determine',
        'message': 'Something went wrong.',
        'action': 'Please try again with a clear photo.',
        'color': 'warning',
        'confidence_text': ''
    }

# ============================================
# MAIN UI
# ============================================
def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("# 🐄")
    with col2:
        st.markdown("# CATTLE HEALTH CHECK")
        st.markdown("_Simple disease detection tool for farmers_")
    
    st.markdown("---")
    
    # Load model
    model = load_model()
    device = get_device()
    
    if model is None:
        st.error("❌ Error: Cannot load the model. Please check the system.")
        return
    
    # Upload section
    st.markdown("## 📸 Take or Upload a Photo")
    st.markdown("Take a clear photo of your cattle. Make sure:")
    st.markdown("""
    - ✓ Good lighting (daytime or bright area)
    - ✓ Can see the whole cattle clearly
    - ✓ Close enough to see details
    - ✓ Photo is square or landscape (not vertical)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )
    
    # Process image
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Show image
        st.markdown("### Your Photo:")
        st.image(image, use_column_width=True)
        
        # Process
        st.markdown("---")
        st.markdown("### 🔍 Analyzing...")
        
        # Preprocess
        tensor, preprocessed_img = preprocess_image(image)
        
        # Run inference
        with st.spinner("Checking the cattle..."):
            result = run_inference(model, tensor, device)
        
        # Generate alert
        alert = generate_farmer_alert(result)
        
        # Display alert
        st.markdown("---")
        st.markdown("## 📋 RESULT")
        
        if alert['color'] == 'success':
            st.markdown(f"""
            <div class="alert-success">
                <div style="font-size: 32px; margin-bottom: 10px;">{alert['emoji']}</div>
                <div style="font-size: 24px; margin-bottom: 5px;">{alert['title']}</div>
                <div style="font-size: 16px; margin-bottom: 15px;">{alert['message']}</div>
                <div style="font-size: 18px; font-weight: bold; color: green;">{alert['confidence_text']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="action-box">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">✅ What To Do:</div>
                <div style="font-size: 16px; line-height: 1.8;">{alert['action'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        elif alert['color'] == 'danger':
            st.markdown(f"""
            <div class="alert-danger">
                <div style="font-size: 32px; margin-bottom: 10px;">{alert['emoji']}</div>
                <div style="font-size: 24px; margin-bottom: 5px;">{alert['title']}</div>
                <div style="font-size: 16px; margin-bottom: 15px;">{alert['message']}</div>
                <div style="font-size: 18px; font-weight: bold; color: darkred;">{alert['confidence_text']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="action-box" style="background-color: #ffe7e7; border-left-color: #f00;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">🚨 URGENT ACTION REQUIRED:</div>
                <div style="font-size: 16px; line-height: 1.8; color: darkred;">{alert['action'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:  # warning
            st.markdown(f"""
            <div class="alert-warning">
                <div style="font-size: 32px; margin-bottom: 10px;">{alert['emoji']}</div>
                <div style="font-size: 24px; margin-bottom: 5px;">{alert['title']}</div>
                <div style="font-size: 16px; margin-bottom: 15px;">{alert['message']}</div>
                <div style="font-size: 18px; font-weight: bold; color: #856404;">{alert['confidence_text']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="action-box">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">⚡ Recommended Action:</div>
                <div style="font-size: 16px; line-height: 1.8;">{alert['action'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional info (expandable for interested users)
        with st.expander("📊 Detailed Information (for record keeping)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Is This a Cattle?**")
                st.markdown(f"- Detection: {result['identification']['class_name']}")
                st.markdown(f"- Confidence: {result['identification']['confidence']*100:.1f}%")
            
            with col2:
                if result['diagnosis'] is not None:
                    st.markdown("**Health Status**")
                    st.markdown(f"- Status: {result['diagnosis']['class_name']}")
                    st.markdown(f"- Confidence: {result['diagnosis']['confidence']*100:.1f}%")
                else:
                    st.markdown("**Health Status**")
                    st.markdown("- Not Available (Not a cattle)")
        
        # Save info
        st.markdown("---")
        st.markdown("### 💾 Save This Result")
        
        result_text = f"""
CATTLE DISEASE CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================================

RESULT: {alert['title']}
{alert['emoji']} {alert['confidence_text']}

CATTLE DETECTION: {result['identification']['class_name']}
Confidence: {result['identification']['confidence']*100:.1f}%

{f'HEALTH STATUS: {result["diagnosis"]["class_name"]}{chr(10)}Confidence: {result["diagnosis"]["confidence"]*100:.1f}%' if result['diagnosis'] else 'HEALTH STATUS: Not applicable (not a cattle)'}

ACTION REQUIRED:
{alert['action']}

=====================================================
        """
        
        st.download_button(
            label="📥 Download Result",
            data=result_text,
            file_name=f"cattle_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    else:
        # No image uploaded yet
        st.info("👆 Please upload a photo of your cattle to get started.")
        
        st.markdown("---")
        st.markdown("### 📚 How to Use This Tool")
        st.markdown("""
        1. **Take a Photo** - Take a clear photo of your cattle
        2. **Upload** - Click above and select your photo
        3. **Wait** - The app will analyze the photo
        4. **Get Result** - See if your cattle is healthy or needs attention
        5. **Take Action** - Follow the recommended actions
        """)

if __name__ == "__main__":
    main()
