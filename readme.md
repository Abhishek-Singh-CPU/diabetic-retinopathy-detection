# Diabetic Retinopathy Detection

An AI-powered web application for detecting diabetic retinopathy from retinal images using deep learning.

---

## 🔍 Overview
This project uses a Convolutional Neural Network (MobileNetV2 with transfer learning) to classify retinal images into different stages of diabetic retinopathy.

The system also includes Grad-CAM visualization to highlight the regions of the image that influenced the prediction, making the model more interpretable.

---

## 🚀 Features
- Image upload for prediction
- Real-time camera input
- Prediction with confidence score
- Grad-CAM visualization (Explainable AI)

---

## 🧠 Tech Stack
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow

---

## 📊 Model Details
- Architecture: MobileNetV2 (Transfer Learning)
- Input size: 160x160
- Classes:
  - No DR
  - Mild
  - Moderate
  - Severe
  - Proliferative DR

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
python app.py