import os
import gdown
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import efficientnet

CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']
IMG_SIZE = (224, 224)

# -------------------
# Load model from Google Drive
# -------------------
def load_keras_model_from_drive(file_id: str, local_path="koa_model.h5"):
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False, fuzzy=True)

    # Patch DepthwiseConv2D for older TF compatibility
    orig_init = DepthwiseConv2D.__init__
    def patched_init(self, *args, **kwargs):
        if "groups" in kwargs:
            kwargs.pop("groups")
        return orig_init(self, *args, **kwargs)
    DepthwiseConv2D.__init__ = patched_init

    model = load_model(local_path, compile=False)
    return model

# -------------------
# Preprocess image array
# -------------------
def preprocess_array_image(img_array, target_size=IMG_SIZE):
    if img_array.dtype != np.uint8:
        img = (img_array * 255).astype(np.uint8)
    else:
        img = img_array.copy()

    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    den = cv2.bilateralFilter(eq_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    resized = cv2.resize(den, target_size, interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = efficientnet.preprocess_input(rgb)
    return rgb

# -------------------
# Grad-CAM
# -------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    import tensorflow as tf

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            last_conv_layer_name = 'top_conv'

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_batch = np.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)

    for i in range(int(pooled_grads.shape[-1])):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0 or tf.math.is_nan(max_val):
        return np.zeros((int(heatmap.shape[0]), int(heatmap.shape[1])))
    heatmap /= max_val
    return heatmap.numpy()

# -------------------
# Overlay heatmap on image
# -------------------
def overlay_heatmap_on_image(orig_rgb, heatmap, alpha=0.4):
    if orig_rgb.dtype != np.uint8:
        img = np.clip(orig_rgb * 255, 0, 255).astype(np.uint8)
    else:
        img = orig_rgb.copy()

    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -------------------
# Convert numpy image to base64
# -------------------
def pil_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# -------------------
# Prediction function
# -------------------
def predict(model, img_array):
    pre = img_array
    if len(pre.shape) == 3:
        pre = np.expand_dims(pre, axis=0)
    probs = model.predict(pre)[0]
    pred_class = int(np.argmax(probs))
    result = {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "top_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

    # Grad-CAM overlay
    heatmap = make_gradcam_heatmap(np.squeeze(pre), model)
    overlay = overlay_heatmap_on_image(np.squeeze(pre), heatmap, alpha=0.4)
    result["gradcam_base64"] = pil_to_base64(overlay)
    return result
