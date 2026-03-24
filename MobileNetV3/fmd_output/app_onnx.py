import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "fmd_multitask.onnx"

# Example label mappings based on your original logic
# Gatekeeper: 0 = not cattle, 1 = cattle
GATEKEEPER_LABELS = ["Not Cattle", "Cattle"]

# Diagnosis head: 0 = healthy, 1 = infected
DIAGNOSIS_LABELS = ["Healthy", "FMD Positive"]

# Confidence threshold from your original code
GATEKEEPER_THRESHOLD = 0.85


# =========================
# LOAD ONNX MODEL
# =========================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()

INPUT_NAME = input_meta.name
INPUT_SHAPE = input_meta.shape  # usually [1, 3, 224, 224]

# Try to get height and width dynamically
if len(INPUT_SHAPE) == 4 and isinstance(INPUT_SHAPE[2], int) and isinstance(INPUT_SHAPE[3], int):
    INPUT_HEIGHT = INPUT_SHAPE[2]
    INPUT_WIDTH = INPUT_SHAPE[3]
else:
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224


# =========================
# HELPERS
# =========================
def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def preprocess_image(input_img):
    """
    Matches the likely training format:
    - RGB
    - resize to model input size
    - normalize to [0,1]
    - HWC -> CHW
    - add batch dimension
    """
    pil_img = Image.fromarray(input_img).convert("RGB")
    pil_img = pil_img.resize((INPUT_WIDTH, INPUT_HEIGHT))

    img = np.array(pil_img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # add batch dim

    return pil_img, img


def predict_and_visualize(input_img):
    if input_img is None:
        return None, "No Image Uploaded"

    # Preprocess
    pil_img, input_tensor = preprocess_image(input_img)

    # ONNX inference
    outputs = session.run(None, {INPUT_NAME: input_tensor})

    # Expecting 2 outputs:
    # outputs[0] = gatekeeper logits
    # outputs[1] = diagnosis logits
    if len(outputs) < 2:
        return np.array(pil_img), (
            "Model did not return the expected 2 outputs.\n\n"
            f"Number of outputs returned: {len(outputs)}\n"
            "Please inspect the ONNX model output structure."
        )

    gk_logits = np.array(outputs[0])
    dx_logits = np.array(outputs[1])

    # Convert logits to probabilities
    gk_probs = softmax(gk_logits)
    dx_probs = softmax(dx_logits)

    # Predictions
    gk_class = int(np.argmax(gk_probs, axis=1)[0])
    gk_conf = float(np.max(gk_probs, axis=1)[0])

    # Reject non-cattle or uncertain images
    if gk_class == 0 or gk_conf < GATEKEEPER_THRESHOLD:
        status_text = (
            f"❌ STATUS: REJECTED\n"
            f"Confidence: {gk_conf:.2%}\n"
            f"Predicted Gatekeeper Class: {GATEKEEPER_LABELS[gk_class]}\n"
            f"Result: Not Cattle or Uncertain Image."
        )
        return np.array(pil_img), status_text

    # Diagnosis
    dx_class = int(np.argmax(dx_probs, axis=1)[0])
    dx_conf_inf = float(dx_probs[0][1]) if dx_probs.shape[1] > 1 else float(dx_probs[0][0])

    diagnosis = "🏥 FMD POSITIVE (Infected)" if dx_class == 1 else "✅ HEALTHY"

    summary = (
        f"🏆 STATUS: CATTLE DETECTED\n"
        f"------------------------------\n"
        f"Diagnosis: {diagnosis}\n"
        f"Diagnosis Class: {DIAGNOSIS_LABELS[dx_class]}\n"
        f"FMD Probability: {dx_conf_inf:.2%}\n"
        f"Gatekeeper Class: {GATEKEEPER_LABELS[gk_class]}\n"
        f"Gatekeeper Conf: {gk_conf:.2%}\n"
        f"------------------------------\n"
        f"Note: ONNX version runs prediction only.\n"
        f"Grad-CAM explainability was removed because it depends on the PyTorch model graph."
    )

    return np.array(pil_img), summary


# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐮 BAKO AI: Multi-Task FMD Detector")
    gr.Markdown(
        "Upload a cattle image to detect Foot-and-Mouth Disease (FMD) using the ONNX multitask model."
    )

    with gr.Row():
        input_image = gr.Image(label="Upload Image")
        output_image = gr.Image(label="Processed Output")

    output_text = gr.Textbox(label="Diagnostic Summary", lines=10)
    submit_btn = gr.Button("🔍 Run Diagnosis", variant="primary")

    submit_btn.click(
        fn=predict_and_visualize,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)