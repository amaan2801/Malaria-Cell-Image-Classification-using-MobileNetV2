import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Page Config
st.set_page_config(page_title="Malaria Detection", layout="centered")

# UI
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    h1 {
        color: #00ffcc;
        text-align: center;
    }
    .stFileUploader {
        border: 1px dashed #00ffcc;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Malaria Cell Detection")
st.markdown("<h4 style='text-align:center;'>AI-powered malaria diagnosis with visual explanation</h4>", unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/malaria_detector_final.h5")

model = load_model()
st.success("Model loaded successfully")

# ADVANCED GRAD-CAM
def get_gradcam(model, img_array):
    # Find LAST conv layer dynamically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv layer found")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Guided gradients (sharper maps)
    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads

    weights = tf.reduce_mean(guided_grads, axis=(1, 2))

    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * conv_outputs[0, :, :, i]

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    return cam

# Upload Image
uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    st.markdown("## Prediction Result")

    if prediction > 0.5:
        st.error(f"Parasitized (Malaria Infected)\n\nConfidence: {confidence:.2f}")
    else:
        st.success(f"Uninfected (Healthy)\n\nConfidence: {confidence:.2f}")

    # Confidence Graph
    st.markdown("## Prediction Confidence")

    labels = ["Uninfected", "Parasitized"]
    values = [1 - prediction, prediction]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim([0, 1])
    ax.set_title("Model Confidence")

    st.pyplot(fig)

    # SHARP GRAD-CAM
    st.markdown("## Model Attention (Grad-CAM)")

    try:
        heatmap = get_gradcam(model, img_array)

        # Resize
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

        # Smooth for better quality
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # Convert to color
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Sharper overlay
        overlay = cv2.addWeighted(np.array(img), 0.65, heatmap, 0.45, 0)

        st.image(overlay, caption="Grad-CAM Heatmap (Enhanced)")

    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")