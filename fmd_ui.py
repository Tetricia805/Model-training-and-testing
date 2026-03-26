import gradio as gr
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the model architecture from the training script
from fmd_pipeline import MultiTaskMobileNetV3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskMobileNetV3().to(device)

# Load the trained model weights
model.load_state_dict(torch.load('d:/AniLinkData2/fmd_output/best_model.pth', map_location=device, weights_only=True))
model.eval()

# Validation transforms
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def predict(img):
    if img is None:
        return "No image provided."
    
    # img is a numpy RGB array from Gradio
    augmented = transform(image=img)
    tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_binary, out_disease = model(tensor)
        prob_binary = torch.sigmoid(out_binary).item()
        prob_disease = torch.sigmoid(out_disease).item()

    # Optimal thresholds found during training
    opt_bin_thresh = 0.54
    opt_dis_thresh = 0.66

    if prob_binary < opt_bin_thresh:
        return f"Prediction: **Not Cattle**\nConfidence: {(1 - prob_binary)*100:.1f}%"
    else:
        if prob_disease >= opt_dis_thresh:
            return f"Prediction: **FMD Infected Cattle**\nConfidence: {prob_disease*100:.1f}%"
        else:
            return f"Prediction: **Healthy Cattle**\nConfidence: {(1 - prob_disease)*100:.1f}%"

with gr.Blocks(title="FMD Cattle Detection") as demo:
    gr.Markdown("# Foot and Mouth Disease (FMD) Cattle Detection")
    gr.Markdown("Upload an image of a cow to detect if it has Foot and Mouth Disease using the locally trained Multi-Task MobileNetV3 model.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Image")
            predict_btn = gr.Button("Analyze Image")
        with gr.Column():
            output_text = gr.Markdown(label="Results", value="Prediction will appear here.")
            
    predict_btn.click(fn=predict, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
