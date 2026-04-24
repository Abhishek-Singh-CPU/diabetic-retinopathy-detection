from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("dr_model_finetuned.keras")

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


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image = None
    gradcam = None

    if request.method == 'POST':
        img = None

        # 📸 Camera input (SAFE)
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
            img_array = np.expand_dims(img_array, axis=0)

            # predict
            pred = model.predict(img_array)
            class_index = np.argmax(pred)
            confidence = float(np.max(pred)) * 100

            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model, "out_relu")
            gradcam_img = overlay_heatmap(img, heatmap)

            gradcam_pil = Image.fromarray(gradcam_img)
            buffered = BytesIO()
            gradcam_pil.save(buffered, format="PNG")
            gradcam = base64.b64encode(buffered.getvalue()).decode()

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
    app.run(debug=True)