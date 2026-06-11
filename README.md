---
title: Diabetic Retinopathy Detection
emoji: 👁️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# Diabetic Retinopathy Detection

A deep learning web app that classifies diabetic retinopathy severity from retinal fundus images using a fine-tuned MobileNetV2 model with Grad-CAM visualization.

## Classes
- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

How to Use
1. Upload a retinal fundus image (PNG/JPG)
2. Or use your webcam to capture one
3. Click **Predict** to see the classification and Grad-CAM heatmap
