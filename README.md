# Image Classification Web App (ViT)

This project is a simple image classification web application built using a pretrained Vision Transformer model from Hugging Face.

It allows users to upload images and get top-5 predictions with confidence scores.

---

## 🚀 Model

- Model: google/vit-base-patch16-224  
- Framework: PyTorch + Hugging Face Transformers  
- Task: Image Classification (ImageNet classes)

---

## 🧠 Features

- Upload image via web interface
- Top-5 predictions with confidence scores
- Batch image processing (folder-based inference)
- Simple Streamlit UI
- CPU/GPU support

---

## 🛠️ Installation

```bash
git clone https://github.com/yusufgney/vit-base-patch16-224-image-classification-webapp.git
cd vit-base-patch16-224-image-classification-webapp

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate (Windows)

pip install -r requirements.txt
