import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import gdown

st.set_page_config(page_title="Malaria Detection", layout="centered")

st.title("Malaria Cell Detection")
st.write("Upload a blood smear image to detect malaria infection.")

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=13h6j0Y_i6m7Is70ScjAgBFo-6qFE-ExV"
    output = "model.h5"
    gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

model = load_model()
st.success("Model loaded successfully")

def get_gradcam(model, img_array):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(guided_grads, axis=(1, 2))

    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * conv_outputs[0, :, :, i]

    cam = np.maximum(cam, 0)

    if np.max(cam) == 0:
        return None

    cam = cam / np.max(cam)
    return cam

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error(f"Parasitized (Malaria Infected)\nConfidence: {confidence:.2f}")
    else:
        st.success(f"Uninfected (Healthy)\nConfidence: {confidence:.2f}")

    st.subheader("Confidence")

    labels = ["Uninfected", "Parasitized"]
    values = [1 - prediction, prediction]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    st.subheader("Grad-CAM")

    heatmap = get_gradcam(model, img_array)

    if heatmap is None:
        st.warning("Grad-CAM could not highlight regions.")
    else:
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(img), 0.65, heatmap, 0.45, 0)
        st.image(overlay, caption="Heatmap")
