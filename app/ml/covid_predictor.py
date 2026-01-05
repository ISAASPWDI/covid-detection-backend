import os
import pickle
import numpy as np
import tensorflow as tf
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from cv2 import resize, imread
import matplotlib.cm as cm
from tensorflow import keras

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")


# ---------- CARGA DE MODELOS (UNA SOLA VEZ) ----------

DNN_MODEL = DenseNet169(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg",
    weights="imagenet"
)

with open(MODEL_PATH, "rb") as f:
    XGB_MODEL = pickle.load(f)


# ---------- PREDICCIÓN ----------
def predict_image(image_path, heatmap_output_path):
    # Leer imagen con OpenCV (BGR)
    img = imread(image_path)
    img = resize(img, (224, 224))

    # Batch dimension
    img_array = np.array([img])  # EXACTAMENTE como el primero

    # Feature extraction
    features = DNN_MODEL.predict(img_array)

    # XGBoost prediction
    prediction = XGB_MODEL.predict(features)[0]

    # Opcional: probabilidad
    if hasattr(XGB_MODEL, "predict_proba"):
        confidence = float(np.max(XGB_MODEL.predict_proba(features)) * 100)
    else:
        confidence = None

    label = "COVID" if prediction == 1 else "NORMAL"

    # Grad-CAM (misma idea que antes)
    heatmap = generate_gradcam(image_path)
    save_gradcam(image_path, heatmap, heatmap_output_path)

    return label, confidence


# ---------- GRAD-CAM ----------

def generate_gradcam(img_path):
    img_size = (224, 224)
    last_conv_layer_name = "conv5_block32_concat"

    # EXACTAMENTE igual al primer código
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = DenseNet169(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)

        # 🔥 ESTA ES LA DIFERENCIA CLAVE
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(output_path)
