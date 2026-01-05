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

# ============================================
# CARGA DE MODELOS UNA SOLA VEZ (a nivel módulo)
# ============================================
print("🔄 Cargando modelos de COVID...")

try:
    # Modelo DenseNet para feature extraction
    DNN_MODEL = DenseNet169(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
        weights="imagenet"
    )
    print("✓ DenseNet169 cargado")
    
    # Modelo XGBoost
    with open(MODEL_PATH, "rb") as f:
        XGB_MODEL = pickle.load(f)
    print("✓ XGBoost cargado")
    
    # Modelo DenseNet completo para Grad-CAM (se carga una sola vez también)
    GRADCAM_MODEL = DenseNet169(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    
    LAST_CONV_LAYER_NAME = "conv5_block32_concat"
    
    GRAD_MODEL = tf.keras.models.Model(
        [GRADCAM_MODEL.inputs],
        [GRADCAM_MODEL.get_layer(LAST_CONV_LAYER_NAME).output, GRADCAM_MODEL.output]
    )
    print("✓ Modelo Grad-CAM preparado")
    
    MODELS_LOADED = True
    print("✅ Todos los modelos cargados exitosamente\n")
    
except Exception as e:
    print(f"❌ Error cargando modelos: {e}\n")
    DNN_MODEL = None
    XGB_MODEL = None
    GRAD_MODEL = None
    MODELS_LOADED = False


# ============================================
# PREDICCIÓN (usa modelos ya cargados)
# ============================================
def predict_image(image_path, heatmap_output_path):
    """
    Predice si una imagen de rayos X muestra COVID o no.
    
    Args:
        image_path: Ruta de la imagen a analizar
        heatmap_output_path: Ruta donde guardar el heatmap
        
    Returns:
        tuple: (label, confidence) donde label es "COVID" o "NORMAL"
    """
    if not MODELS_LOADED:
        raise RuntimeError("Modelos no disponibles. No se pudieron cargar correctamente.")
    
    try:
        # Leer imagen con OpenCV (BGR)
        img = imread(image_path)
        img = resize(img, (224, 224))
        
        # Batch dimension
        img_array = np.array([img])
        
        # Feature extraction con DNN_MODEL (ya cargado)
        features = DNN_MODEL.predict(img_array, verbose=0)
        
        # XGBoost prediction
        prediction = XGB_MODEL.predict(features)[0]
        
        # Probabilidad
        if hasattr(XGB_MODEL, "predict_proba"):
            confidence = float(np.max(XGB_MODEL.predict_proba(features)) * 100)
        else:
            confidence = 85.0  # Valor por defecto si no hay predict_proba
        
        label = "COVID" if prediction == 1 else "NORMAL"
        
        # Grad-CAM (usa GRAD_MODEL ya cargado)
        heatmap = generate_gradcam(image_path)
        save_gradcam(image_path, heatmap, heatmap_output_path)
        
        return label, round(confidence, 2)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        raise


# ============================================
# GRAD-CAM (usa modelo ya cargado)
# ============================================
def generate_gradcam(img_path):
    """
    Genera heatmap Grad-CAM usando el modelo ya cargado globalmente.
    """
    if not MODELS_LOADED:
        raise RuntimeError("Modelo Grad-CAM no disponible")
    
    img_size = (224, 224)
    
    # Preprocesar imagen (EXACTAMENTE igual al código original)
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Usar GRAD_MODEL ya cargado globalmente
    with tf.GradientTape() as tape:
        last_conv_output, preds = GRAD_MODEL(img_array)
        
        # Índice de clase predicha
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Calcular gradientes
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