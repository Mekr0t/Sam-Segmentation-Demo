# 🎯 Segment Anything Model (SAM) Demo

An interactive web application for object segmentation using [Meta AI's Segment Anything Model (SAM)](https://segment-anything.com/), built with [Gradio](https://gradio.app/).

This demo allows you to:
- Upload a custom image or choose an example
- Click on any object in the image
- Instantly segment the clicked object and visualize it with a green mask

---

## 🧠 Supported Models

You can choose from multiple model variants:

| Model   | Description                            |
|---------|----------------------------------------|
| `vit_b` | Fastest, lower accuracy                |
| `vit_l` | Balanced speed and accuracy (default) |
| `vit_h` | Most accurate, slowest                 |

Model checkpoints are downloaded automatically from:  
👉 [`UwUrquell/sam-vit-checkpoints`](https://huggingface.co/UwUrquell/sam-vit-checkpoints)

---

## 🚀 How It Works

1. The `vit_l` model loads by default on app launch
2. Upload your own image or pick an example
3. Click on an object in the image
4. A green mask will be generated around the selected object

---

## 📦 Run Locally (Optional)

To run the app on your machine:

```bash
git clone https://huggingface.co/spaces/UwUrquell/sam-segmentation-demo
cd sam-segmentation-demo

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install -r requirements.txt

python app.py
