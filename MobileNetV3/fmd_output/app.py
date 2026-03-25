import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort

# =========================
# CONFIG
# =========================
MODEL_PATH = "fmd_multitask.onnx"

# Adjust these if your label order is different
GATEKEEPER_LABELS = ["Not Cattle", "Cattle"]
DIAGNOSIS_LABELS = ["Healthy", "FMD Positive"]

# Same logic from your original code
GATEKEEPER_THRESHOLD = 0.85


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


session = load_model()
inputs = session.get_inputs()
outputs = session.get_outputs()

INPUT_NAME = inputs[0].name
INPUT_SHAPE = inputs[0].shape

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


def preprocess_image(image, input_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(input_size)

    img = np.array(image).astype(np.float32) / 255.0

    # If your training used ImageNet normalization, uncomment this:
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dimension
    return img


def run_inference(image):
    input_tensor = preprocess_image(image, input_size=(INPUT_WIDTH, INPUT_HEIGHT))
    results = session.run(None, {INPUT_NAME: input_tensor})
    return results


# =========================
# UI
# =========================
st.set_page_config(page_title="FMD Cattle Detector", layout="centered")

st.title("🐄 Foot-and-Mouth Disease Detection in Cattle")
st.write("Upload an image to test the multitask ONNX model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Diagnosis"):
        try:
            results = run_inference(image)

            if len(results) < 2:
                st.error(
                    f"The model returned {len(results)} output(s), but 2 outputs were expected "
                    f"(gatekeeper + diagnosis head)."
                )
                st.stop()

            # -------------------------
            # OUTPUT 1: Gatekeeper
            # -------------------------
            gk_logits = np.array(results[0])
            gk_probs = softmax(gk_logits)

            gk_class = int(np.argmax(gk_probs, axis=1)[0])
            gk_conf = float(np.max(gk_probs, axis=1)[0])

            st.subheader("Gatekeeper Result")
            st.write(f"Predicted class: **{GATEKEEPER_LABELS[gk_class]}**")
            st.write(f"Confidence: **{gk_conf:.2%}**")

            # STRICT STOP CONDITION
            if gk_class == 0 or gk_conf < GATEKEEPER_THRESHOLD:
                st.error("❌ Rejected: Not cattle or uncertain image.")
                st.info("Diagnosis stopped at gatekeeper stage. No FMD diagnosis was performed.")
                st.stop()

            # -------------------------
            # OUTPUT 2: Diagnosis
            # -------------------------
            dx_logits = np.array(results[1])
            dx_probs = softmax(dx_logits)

            dx_class = int(np.argmax(dx_probs, axis=1)[0])
            dx_conf = float(np.max(dx_probs, axis=1)[0])

            st.subheader("Diagnosis Result")

            if dx_class == 1:
                st.error(f"🏥 Diagnosis: {DIAGNOSIS_LABELS[dx_class]}")
            else:
                st.success(f"✅ Diagnosis: {DIAGNOSIS_LABELS[dx_class]}")

            st.write(f"Diagnosis confidence: **{dx_conf:.2%}**")

            # Optional detailed probabilities
            st.subheader("Detailed Scores")

            st.write("**Gatekeeper probabilities**")
            for i, label in enumerate(GATEKEEPER_LABELS):
                st.write(f"{label}: {gk_probs[0][i]:.2%}")

            st.write("**Diagnosis probabilities**")
            for i, label in enumerate(DIAGNOSIS_LABELS):
                st.write(f"{label}: {dx_probs[0][i]:.2%}")

        except Exception as e:
            st.error(f"An error occurred during inference: {e}")