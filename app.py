from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
import os

# ── Monkeypatch BatchNormalization ──────────────────────────────────────────
# The model was saved with a TF/Keras version that stored 'renorm' params in
# BatchNormalization. Keras 3 removed these options, causing load_model to fail.
# Monkeypatch the constructor of BatchNormalization in both tf.keras and keras.

original_tf_bn_init = tf.keras.layers.BatchNormalization.__init__
def patched_tf_bn_init(self, *args, **kwargs):
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    original_tf_bn_init(self, *args, **kwargs)
tf.keras.layers.BatchNormalization.__init__ = patched_tf_bn_init

try:
    import keras
    original_keras_bn_init = keras.layers.BatchNormalization.__init__
    def patched_keras_bn_init(self, *args, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        original_keras_bn_init(self, *args, **kwargs)
    keras.layers.BatchNormalization.__init__ = patched_keras_bn_init
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
                prediction = "Invalid camera image. Please capture again."
                return render_template('index.html', prediction=prediction)

        # 📁 File upload
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                try:
                    img = Image.open(BytesIO(file.read())).convert("RGB")
                except Exception:
                    prediction = "Invalid image file."
                    return render_template('index.html', prediction=prediction)
            else:
                prediction = "Invalid file type! Use PNG/JPG."
                return render_template('index.html', prediction=prediction)

        if img is not None:
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

            if confidence < 50:
                prediction = "Not a valid retinal image"
            else:
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