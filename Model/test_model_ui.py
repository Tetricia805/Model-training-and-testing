"""
Cattle Disease Detection Model - Testing UI
A Streamlit app to test the trained model on new images
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Cattle Disease Detection - Model Testing",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
st.sidebar.title("🔧 Model Testing Configuration")

# Model selection
model_format = st.sidebar.radio(
    "Select Model Format:",
    ["PyTorch (Float32)", "Quantized (INT8)"],
    index=0
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Lower = more sensitive, Higher = more confident"
)

# Display options
show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=True)
show_probabilities = st.sidebar.checkbox("Show Probability Scores", value=True)
show_statistics = st.sidebar.checkbox("Show Statistics", value=True)

# ============================================
# MODEL LOADING (CACHED)
# ============================================
@st.cache_resource
def load_model(format_type="pytorch"):
    """Load the trained model"""
    try:
        if format_type == "pytorch":
            # Load PyTorch model
            model_path = Path("pipeline_output/models/model_final.pt")
            
            # Define model architecture (must match training)
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            
            class CattleMultiTaskModel(torch.nn.Module):
                def __init__(self, pretrained=False, freeze_backbone=False):
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
        
        else:  # Quantized
            st.warning("Quantized model requires special handling. Using Float32 for testing.")
            return load_model("pytorch")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def get_device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def preprocess_image(image_path_or_pil, show_steps=False):
    """Preprocess image for model input"""
    
    # Convert PIL to numpy if needed
    if isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil
    else:
        img = Image.open(image_path_or_pil)
    
    img = img.convert('RGB')
    
    if show_steps:
        st.write("#### Preprocessing Steps:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(img, caption="1. Original Image", use_column_width=True)
        
        # Resize
        img_resized = img.resize((224, 224))
        with col2:
            st.image(img_resized, caption="2. Resized (224×224)", use_column_width=True)
        
        # Convert to array for visualization
        img_array = np.array(img_resized)
    else:
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
    
    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(img_resized).unsqueeze(0)
    
    if show_steps:
        with col3:
            st.write("3. Normalized")
            st.info("Mean: [0.485, 0.456, 0.406]\nStd: [0.229, 0.224, 0.225]")
        
        with col4:
            st.write("4. Tensor Shape")
            st.info(f"Shape: {tensor.shape}\nDtype: {tensor.dtype}")
    
    return tensor, img_resized

# ============================================
# INFERENCE FUNCTION
# ============================================
def run_inference(model, image_tensor, device):
    """Run model inference - only run diagnosis if cattle is detected"""
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        id_logits, diag_logits = model(image_tensor)
        
        # Convert to probabilities
        id_probs = torch.softmax(id_logits, dim=1)
        
        # Get identification prediction
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
            'diagnosis': None  # Will only be filled if cattle is detected
        }
        
        # ONLY run diagnosis if cattle is detected (class_id == 0)
        if id_pred == 0:  # Is cattle
            diag_probs = torch.softmax(diag_logits, dim=1)
            diag_pred = diag_logits.argmax(1).item()
            
            result['diagnosis'] = {
                'class_id': diag_pred,
                'class_name': 'Healthy' if diag_pred == 0 else 'FMD',
                'confidence': diag_probs[0, diag_pred].item(),
                'probabilities': {
                    'healthy': diag_probs[0, 0].item(),
                    'fmd': diag_probs[0, 1].item()
                }
            }
    
    return result

# ============================================
# DECISION LOGIC
# ============================================
def generate_alert(result, threshold):
    """Generate alert based on predictions"""
    
    id_conf = result['identification']['confidence']
    
    # FIRST CHECK: Is it cattle?
    if result['identification']['class_id'] == 1:  # Not cattle
        return {
            'level': 'NOT_CATTLE',
            'message': '❌ This is NOT a cattle - Please upload a cattle image',
            'color': '#FF6600',
            'emoji': '❌',
            'action': 'Upload an image containing CATTLE to diagnose FMD'
        }
    
    # SECOND CHECK: Is it confident cattle?
    elif id_conf < threshold:  # Unclear cattle
        return {
            'level': 'UNCLEAR',
            'message': f'❓ Unclear if this is cattle (confidence: {id_conf:.1%})',
            'color': '#FFA500',
            'emoji': '❓',
            'action': 'Ask for better angle/lighting'
        }
    
    # THIRD CHECK: Cattle confirmed, now check diagnosis
    # Diagnosis is only available if cattle was detected
    elif result['diagnosis'] is not None:
        diag_conf = result['diagnosis']['confidence']
        
        if result['diagnosis']['class_id'] == 1:  # FMD
            if diag_conf > 0.85:
                return {
                    'level': 'CRITICAL',
                    'message': '🚨 HIGH RISK: FMD DETECTED',
                    'color': '#FF0000',
                    'emoji': '🚨',
                    'action': 'ISOLATE HERD - Contact veterinarian immediately'
                }
            else:
                return {
                    'level': 'WARNING',
                    'message': '⚠️ CAUTION: Possible FMD',
                    'color': '#FF6600',
                    'emoji': '⚠️',
                    'action': 'Contact veterinarian for confirmation'
                }
        
        else:  # Healthy
            return {
                'level': 'OK',
                'message': '✅ Cattle appears healthy',
                'color': '#00CC00',
                'emoji': '✅',
                'action': 'Continue monitoring (check again in 2 weeks)'
            }
    
    # Fallback (should not reach here)
    return {
        'level': 'UNKNOWN',
        'message': 'Unable to determine status',
        'color': '#999999',
        'emoji': '❓',
        'action': 'Try again with a clearer image'
    }

# ============================================
# MAIN UI
# ============================================
def main():
    st.title("🐄 Cattle Disease Detection - Model Testing")
    st.markdown("---")
    
    # Load model
    model = load_model("pytorch")
    device = get_device()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # System info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Device", "GPU" if device.type == 'cuda' else "CPU")
    with col2:
        st.metric("Model Format", model_format)
    with col3:
        st.metric("Threshold", f"{confidence_threshold:.0%}")
    
    st.markdown("---")
    
    # Upload section
    st.header("📸 Upload Image for Testing")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a cattle image or test image"
    )
    
    # Sample images section
    sample_mode = st.checkbox("Use Sample Image Instead", value=False)
    
    if sample_mode:
        sample_dir = Path("pipeline_output/results")
        # For demo, we'll create placeholder text
        st.info("Sample images would be loaded from pipeline_output/results/")
    
    if uploaded_file is not None or sample_mode:
        
        # Process image
        st.header("🔍 Model Analysis")
        
        # Preprocess
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_name = uploaded_file.name
        else:
            st.write("No sample images available for demo")
            return
        
        # Create two columns for image and results
        col_img, col_results = st.columns([1, 1.2])
        
        with col_img:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
        
        with col_results:
            st.subheader("Image Properties")
            img_array = np.array(image)
            st.write(f"**Dimensions:** {img_array.shape}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size (MB):** {uploaded_file.size / (1024*1024):.2f}")
        
        # Show preprocessing steps
        if show_preprocessing:
            st.markdown("---")
            tensor, preprocessed_img = preprocess_image(image, show_steps=True)
        else:
            tensor, preprocessed_img = preprocess_image(image, show_steps=False)
        
        # Run inference
        st.markdown("---")
        st.header("🤖 Model Predictions")
        
        with st.spinner("Running inference..."):
            result = run_inference(model, tensor, device)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Alert", "📊 Probabilities", "📈 Details"])
        
        with tab1:
            alert = generate_alert(result, confidence_threshold)
            
            # Display alert with color
            st.markdown(f"""
            <div style="background-color: {alert['color']}; padding: 20px; border-radius: 10px; color: white;">
                <h2 style="margin: 0;">{alert['emoji']} {alert['message']}</h2>
                <p style="margin: 10px 0 0 0;"><b>Action:</b> {alert['action']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            if show_probabilities:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Identification Task")
                    st.write(f"**Cattle:** {result['identification']['probabilities']['cattle']:.1%}")
                    st.write(f"**Not-Cattle:** {result['identification']['probabilities']['not_cattle']:.1%}")
                    
                    # Probability bar
                    fig, ax = plt.subplots(figsize=(8, 3))
                    categories = ['Cattle', 'Not-Cattle']
                    values = [
                        result['identification']['probabilities']['cattle'],
                        result['identification']['probabilities']['not_cattle']
                    ]
                    colors = ['#00CC00' if values[0] > values[1] else '#CCCCCC',
                              '#FF0000' if values[1] > values[0] else '#CCCCCC']
                    
                    ax.barh(categories, values, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    for i, v in enumerate(values):
                        ax.text(v + 0.02, i, f'{v:.1%}', va='center')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    # Only show diagnosis if cattle was detected
                    if result['diagnosis'] is not None:
                        st.subheader("Diagnosis Task (FMD Detection)")
                        st.write(f"**Healthy:** {result['diagnosis']['probabilities']['healthy']:.1%}")
                        st.write(f"**FMD:** {result['diagnosis']['probabilities']['fmd']:.1%}")
                        
                        # Probability bar
                        fig, ax = plt.subplots(figsize=(8, 3))
                        categories = ['Healthy', 'FMD']
                        values = [
                            result['diagnosis']['probabilities']['healthy'],
                            result['diagnosis']['probabilities']['fmd']
                        ]
                        colors = ['#00CC00' if values[0] > values[1] else '#CCCCCC',
                                  '#FF0000' if values[1] > values[0] else '#CCCCCC']
                        
                        ax.barh(categories, values, color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        for i, v in enumerate(values):
                            ax.text(v + 0.02, i, f'{v:.1%}', va='center')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Please upload a CATTLE image to see FMD diagnosis results.")
        
        with tab3:
            st.write("#### Identification Task")
            id_col1, id_col2, id_col3 = st.columns(3)
            with id_col1:
                st.metric("Prediction", result['identification']['class_name'])
            with id_col2:
                st.metric("Confidence", f"{result['identification']['confidence']:.1%}")
            with id_col3:
                status = "✅ PASS" if result['identification']['confidence'] > confidence_threshold else "⚠️ LOW"
                st.metric("Threshold Status", status)
            
            # Only show diagnosis details if cattle was detected
            if result['diagnosis'] is not None:
                st.write("#### Diagnosis Task (FMD Detection)")
                diag_col1, diag_col2, diag_col3 = st.columns(3)
                with diag_col1:
                    st.metric("Prediction", result['diagnosis']['class_name'])
                with diag_col2:
                    st.metric("Confidence", f"{result['diagnosis']['confidence']:.1%}")
                with diag_col3:
                    if result['diagnosis']['class_name'] == 'FMD':
                        status = "🚨 ALERT" if result['diagnosis']['confidence'] > 0.85 else "⚠️ WARNING"
                        st.metric("Risk Level", status)
                    else:
                        st.metric("Risk Level", "✅ OK")
            else:
                st.warning("⚠️ Diagnosis not available. Please upload a CATTLE image to diagnose FMD.")
        
        # Statistics
        if show_statistics:
            st.markdown("---")
            st.header("📉 Model Performance Context")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ID Accuracy (Test)", "96.35%", help="Cattle identification accuracy on test set")
            with col2:
                st.metric("Diagnosis Accuracy", "85.82%", help="FMD diagnosis accuracy on test set")
            with col3:
                st.metric("FMD Recall", "91.82%", help="Catches 9 out of 10 FMD cases")
            with col4:
                st.metric("False Alarm Rate", "8.2%", help="False positives on test set")
        
        # Export results
        st.markdown("---")
        st.header("💾 Test Results")
        
        results_text = f"""
# Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Image Info
- **File Name:** {image_name}
- **Dimensions:** {img_array.shape}

## Identification (Cattle vs Non-Cattle)
- **Prediction:** {result['identification']['class_name']}
- **Confidence:** {result['identification']['confidence']:.1%}
- **Cattle Probability:** {result['identification']['probabilities']['cattle']:.1%}
- **Not-Cattle Probability:** {result['identification']['probabilities']['not_cattle']:.1%}

## Diagnosis (Healthy vs FMD)
{f"- **Prediction:** {result['diagnosis']['class_name']}" if result['diagnosis'] is not None else "- **Diagnosis:** Not available (not a cattle image)"}
{f"- **Confidence:** {result['diagnosis']['confidence']:.1%}" if result['diagnosis'] is not None else ""}
- **Healthy Probability:** {result['diagnosis']['probabilities']['healthy']:.1%}
- **FMD Probability:** {result['diagnosis']['probabilities']['fmd']:.1%}

## Alert Decision
{alert['message']}
- **Action:** {alert['action']}
- **Confidence Threshold:** {confidence_threshold:.0%}
"""
        
        st.download_button(
            label="📥 Download Test Results",
            data=results_text,
            file_name=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
