from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
import os

# ── Universal Keras Layer Compatibility Patch ────────────────────────────────
ALLOWED_BASE_KWARGS = {
    'input_shape', 'batch_input_shape', 'batch_size', 'weights', 'dynamic',
    'name', 'trainable', 'dtype', 'autocast', 'activity_regularizer'
}

original_tf_layer_init = tf.keras.layers.Layer.__init__
def patched_tf_layer_init(self, *args, **kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_BASE_KWARGS}
    original_tf_layer_init(self, *args, **filtered_kwargs)
tf.keras.layers.Layer.__init__ = patched_tf_layer_init

try:
    import keras
    original_keras_layer_init = keras.layers.Layer.__init__
    def patched_keras_layer_init(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_BASE_KWARGS}
        original_keras_layer_init(self, *args, **filtered_kwargs)
    keras.layers.Layer.__init__ = patched_keras_layer_init
except (ImportError, AttributeError):
    pass

app = Flask(__name__)

# ── Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dr_model_finetuned.keras")
model = tf.keras.models.load_model(MODEL_PATH)

classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Retina Image Validation Heuristics ──────────────────────────────────────
def is_retinal_image(img):
    img_np = np.array(img)
    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
        return False
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Flat/blank image check
    if np.std(gray) < 5:
        return False
        
    # 2. Too dark/bright check
    mean_val = np.mean(gray)
    if mean_val < 5 or mean_val > 250:
        return False
        
    # 3. Color channel check (Retinas usually have more red than blue in the central region)
    h, w = img_np.shape[:2]
    center_r = np.mean(img_np[h//4:3*h//4, w//4:3*w//4, 0])
    center_b = np.mean(img_np[h//4:3*h//4, w//4:3*w//4, 2])
    
    # If the center is overwhelmingly blue, it's not a retina.
    # We add a small buffer (+10) to account for different lighting.
    if center_b > center_r + 20:
        return False

    return True


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    return heatmap


def overlay_heatmap(original_img, heatmap):
    original = np.array(original_img)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return superimposed


# ── Find the last conv layer name automatically ────────────────────────────
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D,
                              tf.keras.layers.Activation)):
            return layer.name
    return None

LAST_CONV_LAYER = get_last_conv_layer(model)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image = None
    gradcam = None

    if request.method == 'POST':
        img = None

        # 📸 Camera input
        if 'image_data' in request.form and request.form['image_data']:
            try:
                image_data = request.form['image_data'].split(',')[1]
                img_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception:
                prediction = "Wrong image provided"
                return render_template('index.html', prediction=prediction)

        # 📁 File upload
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                try:
                    img = Image.open(BytesIO(file.read())).convert("RGB")
                except Exception:
                    prediction = "Wrong image provided"
                    return render_template('index.html', prediction=prediction)
            else:
                prediction = "Wrong image provided"
                return render_template('index.html', prediction=prediction)

        if img is not None:
            # Check if it is a retinal image
            if not is_retinal_image(img):
                prediction = "Wrong image provided"
                # Still show preview of the invalid upload
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                image = base64.b64encode(buffered.getvalue()).decode()
            else:
                # preview
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                image = base64.b64encode(buffered.getvalue()).decode()

                # preprocess
                img_resized = img.resize((160, 160))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                # predict
                pred = model.predict(img_array)
                class_index = np.argmax(pred)
                confidence = float(np.max(pred)) * 100

                # Grad-CAM
                if LAST_CONV_LAYER:
                    try:
                        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
                        gradcam_img = overlay_heatmap(img, heatmap)
                        gradcam_pil = Image.fromarray(gradcam_img)
                        buffered = BytesIO()
                        gradcam_pil.save(buffered, format="PNG")
                        gradcam = base64.b64encode(buffered.getvalue()).decode()
                    except Exception:
                        gradcam = None  # Grad-CAM is optional, don't crash

                prediction = classes[class_index]

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        image=image,
        gradcam=gradcam
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)