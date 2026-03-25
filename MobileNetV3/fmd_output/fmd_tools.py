import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageFile
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn.utils.prune as prune

# Import architecture pieces from training script
try:
    from train_fmd import MultiTaskFMDModel, get_transforms
except ImportError:
    # Redefine if import fails for some reason
    class CBAM(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super(CBAM, self).__init__()
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
                nn.Sigmoid()
            )
            self.sa = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3, bias=False),
                nn.Sigmoid()
            )
        def forward(self, x):
            avg_out = self.ca(x); x = x * avg_out
            avg_s = torch.mean(x, dim=1, keepdim=True)
            max_s, _ = torch.max(x, dim=1, keepdim=True)
            sam = self.sa(torch.cat([avg_s, max_s], dim=1))
            return x * sam

    class MultiTaskFMDModel(nn.Module):
        def __init__(self, dropout_rate=0.3):
            super(MultiTaskFMDModel, self).__init__()
            backbone = models.mobilenet_v3_small(weights='DEFAULT')
            self.features = backbone.features
            feature_dim = 576 
            self.cbam = CBAM(feature_dim)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.gatekeeper_head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 2))
            self.diagnostic_head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 2))
        def forward(self, x):
            x = self.features(x); x = self.cbam(x); x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.gatekeeper_head(x), self.diagnostic_head(x)

    def get_transforms(phase="val"):
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# --- 1. EXPLAINABILITY (Grad-CAM) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def generate_heatmap(self, input_image, target_index=1):
        self.model.zero_grad()
        gk_logits, dx_logits = self.model(input_image)
        dx_logits[0, target_index].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap = cv2.resize(heatmap, (224, 224))
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

# --- 2. OPTIMIZATION (Pruning & Quantization) ---
def deploy_optimize(model_path, output_path):
    print("--- Loading Model for Optimization ---")
    model = MultiTaskFMDModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    print("Applying 30% L1-Unstructured Pruning (Conv layers)...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')
            
    print("Applying Dynamic INT8 Quantization...")
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    print(f"Saving Optimized model to {output_path}...")
    torch.save(quantized_model.state_dict(), output_path)
    
    # Export to ONNX
    print("Exporting optimized model to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path.replace(".pth", ".onnx"), 
                     input_names=['input'], output_names=['gatekeeper', 'diagnostic'], opset_version=12)

# --- 3. INFERENCE RUNNER ---
def run_prediction(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskFMDModel().to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    transform = get_transforms("val")
    orig_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    
    # CAM for Explainability
    cam = GradCAM(model, model.features[12]) # Last feature layer
    heatmap = cam.generate_heatmap(input_tensor)
    
    # Inference
    with torch.no_grad():
        gk_logits, dx_logits = model(input_tensor)
        gk_probs = F.softmax(gk_logits, dim=1)
        dx_probs = F.softmax(dx_logits, dim=1)
    
    gk_conf, gk_class = torch.max(gk_probs, 1)
    
    results = {"image": image_path}
    if gk_class.item() == 0 or gk_conf.item() < 0.85:
        results["status"] = "REJECTED (Not Cattle or low confidence)"
        results["confidence"] = gk_conf.item()
    else:
        dx_conf_inf = dx_probs[0][1].item()
        results["status"] = "CATTLE DETECTED"
        results["diagnosis"] = "FMD POSITIVE" if dx_conf_inf > 0.5 else "HEALTHY"
        results["fmd_prob"] = dx_conf_inf
        
        # Save Overlay
        img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_c, 0.4, 0)
        cv2.imwrite("prediction_heatmap.jpg", overlay)
        results["heatmap_saved"] = "prediction_heatmap.jpg"
        
    return results

if __name__ == "__main__":
    import sys
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 fmd_tools.py [predict|optimize] [path]")
    elif sys.argv[1] == "predict":
        # CHANGE THIS LINE:
        res = run_prediction(sys.argv[2], "best_fmd_model.pth")
        print(res)
    elif sys.argv[1] == "optimize":
        # AND CHANGE THIS LINE:
        deploy_optimize("best_fmd_model.pth", "optimized_model.pth")
