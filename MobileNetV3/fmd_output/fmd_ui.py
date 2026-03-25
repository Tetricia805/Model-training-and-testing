import gradio as gr
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from fmd_tools import MultiTaskFMDModel, GradCAM, get_transforms

# Initialize Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fmd_output/best_fmd_model.pth"

def load_fmd_model():
    model = MultiTaskFMDModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_fmd_model()
transform = get_transforms("val")

def predict_and_visualize(input_img):
    if input_img is None: return None, "No Image Uploaded"
    
    # PIL image for transforms
    pil_img = Image.fromarray(input_img).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # Explainability (Grad-CAM)
    cam = GradCAM(model, model.features[12])
    heatmap = cam.generate_heatmap(input_tensor)
    
    # Inference
    with torch.no_grad():
        gk_logits, dx_logits = model(input_tensor)
        gk_probs = F.softmax(gk_logits, dim=1)
        dx_probs = F.softmax(dx_logits, dim=1)
        
    gk_conf, gk_class = torch.max(gk_probs, 1)
    
    # Reject non-cattle with high confidence logic
    if gk_class.item() == 0 or gk_conf.item() < 0.85:
        status_text = f"❌ STATUS: REJECTED\nConfidence: {gk_conf.item():.2%}\nResult: Not Cattle or Uncertain Image."
        return input_img, status_text
        
    dx_conf_inf = dx_probs[0][1].item()
    diagnosis = "🏥 FMD POSITIVE (Infected)" if dx_conf_inf > 0.5 else "✅ HEALTHY"
    
    # Combine heatmap with original image
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) # Back to RGB for Gradio
    
    summary = (
        f"🏆 STATUS: CATTLE DETECTED\n"
        f"------------------------------\n"
        f"Diagnosis: {diagnosis}\n"
        f"FMD Probability: {dx_conf_inf:.2%}\n"
        f"Gatekeeper Conf: {gk_conf.item():.2%}\n"
        f"------------------------------\n"
        f"Explainability: Heatmap shows model focus area (Red = High focus)"
    )
    
    return overlay, summary

# Create UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐮 BAKO AI: Multi-Task FMD Detector")
    gr.Markdown("Upload a cattle image to detect Foot-and-Mouth Disease (FMD) with Grad-CAM explainability.")
    
    with gr.Row():
        input_image = gr.Image()
        output_image = gr.Image(label="Explainability Heatmap (CAM)")
        
    output_text = gr.Textbox(label="Diagnostic Summary", lines=8)
    submit_btn = gr.Button("🔍 Run Diagnosis", variant="primary")
    
    submit_btn.click(fn=predict_and_visualize, inputs=input_image, outputs=[output_image, output_text])
    
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)
