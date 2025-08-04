import os
import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamPredictor
import requests
from io import BytesIO
from pathlib import Path


# Global variabiles
sam_models = {} # Cache for loaded models
current_predictor = None # Current active predictor

# Function to load SAM model based on user selection
def load_sam_model(model_type):
    global sam_models, current_predictor
    
    if model_type in sam_models:
        current_predictor = sam_models[model_type]
        return f"Model {model_type} is loaded and active."
    
    # Mapping model types to their checkpoint files
    model_files = {
        "vit_b": "sam_vit_b.pth",
        "vit_l": "sam_vit_l.pth", 
        "vit_h": "sam_vit_h.pth"
    }
    
    try:
        # Download checkpoint from Hugging Face Hub
        if model_type not in model_files:
            return f"Model {model_type} is not supported."
        checkpoint_path = hf_hub_download(
            repo_id="UwUrquell/sam-vit-checkpoints",
            filename=model_files[model_type]
        )
        
        # Initialize SAM model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        # Set the model to the global cache
        sam_models[model_type] = predictor
        current_predictor = predictor
        
        return f"Model {model_type} was loaded successfully!"
        
    except Exception as e:
        return f"Error while loading model {model_type}: {str(e)}"


# Function to handle segmentation on click
def segment_on_click(image, model_type, evt: gr.SelectData):
    global current_predictor
    
    if current_predictor is None or image is None:
        return None
    
    try:
        x_coord, y_coord = evt.index
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Set the image for the predictor
        current_predictor.set_image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        
        # Mapping click coordinates to the image
        point_coords = np.array([[x_coord, y_coord]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        
        masks, scores, _ = current_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Choose the best mask based on scores
        best_mask = masks[np.argmax(scores)]
        
        # Convert mask to uint8 for visualization
        mask = best_mask.astype(np.uint8) * 255
        
        # Create an overlay with the mask
        colored_mask = np.zeros_like(cv2_img)
        colored_mask[:, :, 1] = mask  # Green channel for the mask
        
        overlay = cv2.addWeighted(cv2_img, 0.7, colored_mask, 0.3, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(overlay_rgb)
    except Exception as e:
        print(f"Error while segmenting: {e}")
        return None

# Function to create sample images if they cannot be downloaded
def create_sample_images():
    # Create a few simple sample images with colored circles
    sample_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, color in enumerate(colors):
        # Create a blank image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img.fill(50)  # Fill with a dark gray background
        
        # Draw colored circles
        cv2.circle(img, (150, 150), 80, color, -1)
        cv2.circle(img, (100, 100), 30, (255, 255, 255), -1)
        
        sample_images.append(Image.fromarray(img))
    
    return sample_images

# Function to get example images from Google Drive
def get_example_images():
    # List of example images with their Google Drive IDs
    file_ids = {
        "img1.jpg": "1SMJcyvUb33g1twOqk-b_i_s4jBgKufUw",  # Google Drive IDs
        "img2.jpg": "1pGWwdYOfJKiV2TVtS1YvuaEa-qiBEjQ7",
        "img3.jpg": "1HluOBYTRH-Uurw4oeILgbeEQMZq3So7L",
        "img4.jpg": "1LWVz2TxkzGJiJtQgqbS12ko0kLUkD3wd"
    }
    

    example_images = []

    # Check if the images can be downloaded, otherwise create sample images
    for fname, file_id in file_ids.items():
        try:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).resize((300, 300))
            example_images.append(img)
        except Exception as e:
            print(f"Cannot load {fname}: {e}")

    return example_images

# Function to handle selection of example images
def select_example_image(evt: gr.SelectData):
    example_images = get_example_images()
    if evt.index < len(example_images):
        return example_images[evt.index], None  # Return selected image and None for output
    return None, None

# Function to toggle help section visibility
def toggle_help():
    return gr.update(visible=True)

# Function to close help section
def close_help():
    return gr.update(visible=False)

# Function to initialize the default model (vit_l)
def initialize_default_model():
    return load_sam_model("vit_l")

# Custom CSS for styling the help button and content
custom_css = """
.help-button {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    z-index: 1000 !important;
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    font-size: 20px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

.help-button:hover {
    background: #1d4ed8 !important;
    transform: scale(1.1) !important;
}

.help-content {
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 25px !important;
    margin: 20px 0 !important;
    color: #1a202c !important;
    line-height: 1.6 !important;
}

.help-content h2 {
    color: #2d3748 !important;
    margin-bottom: 15px !important;
    font-size: 1.5em !important;
}

.help-content h3 {
    color: #4a5568 !important;
    margin: 20px 0 10px 0 !important;
    font-size: 1.2em !important;
}

.help-content ul, .help-content ol {
    margin: 10px 0 !important;
    padding-left: 25px !important;
}

.help-content li {
    margin: 8px 0 !important;
    color: #2d3748 !important;
}

.help-content strong {
    color: #1a202c !important;
    font-weight: 600 !important;
}

#seg_output img {
    width: 100% !important;
    height: auto !important;
    max-width: 600px !important;
}
"""

# Gradio app setup
with gr.Blocks(title="SAM Demo", css=custom_css) as demo:
    # Header
    with gr.Row():
        gr.Markdown("# 🎯 Segment Anything Model (SAM) Demo")
    
    # Help Button (floating)
    help_button = gr.Button("❓", elem_classes=["help-button"], size="sm")
    
    # Help Section (Initially hidden)
    with gr.Column(visible=False) as help_section:
        gr.HTML("""
        <div class="help-content">
            <h2>📖 Guide to Using SAM Demo</h2>
            
            <h3>🔧 Steps:</h3>
            <ol>
                <li><strong>The model loads automatically</strong> (vit_l is the default)</li>
                <li><strong>Upload an image</strong> or select one from the examples</li>
                <li><strong>Click on an object</strong> in the image to segment it</li>
                <li><strong>Result:</strong> The segmented object will be displayed with a green mask</li>
            </ol>
            
            <h3>ℹ️ Model Information:</h3>
            <ul>
                <li><strong>vit_b</strong>: Fastest, suitable for basic use</li>
                <li><strong>vit_l</strong> (default): Balanced speed/accuracy ratio, optimal for most cases</li>
                <li><strong>vit_h</strong>: Most accurate, slower, best results</li>
            </ul>
            
            <h3>💡 Tips:</h3>
            <ul>
                <li>Click precisely on the object you want to segment</li>
                <li>For better results, use sharp, high-quality images</li>
                <li>The model works best on clearly defined objects</li>
            </ul>
        </div>
        """)
        
        close_help_btn = gr.Button("📖 Hide guide", variant="secondary")
    
    gr.Markdown("Select a model and upload an image, then click on an object to segment it!")
    
    # Model selection
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["vit_b", "vit_l", "vit_h"],
            value="vit_l",  # Default model
            label="🎛️ Select SAM Model",
            info="vit_b = fastest, vit_l = balanced (default), vit_h = most accurate"
        )
        
        model_status = gr.Textbox(
            label="📊 Model Status",
            value="Loading default model vit_l...",
            interactive=False,
            scale=2
        )
    
    # Images section
    with gr.Row(equal_height=True):
        with gr.Column():
            uploaded_image_input = gr.Image(
                label="📁 Upload an image",
                type="pil",
                sources=["upload"]
            )
            
            # Example images
            gr.Markdown("### 🖼️ Or select an example image:")
            example_gallery = gr.Gallery(
                value=get_example_images(),
                label="Example Images",
                show_label=False,
                elem_id="example_gallery",
                columns=2,
                rows=2,
                height="300px",
                allow_preview=False
            )
        
        with gr.Column():
            segmented_output = gr.Image(
                label="🎯 Segmented Output",
                type="pil",
                elem_id="seg_output"
            )
    
    # Event handlers
    # Initialize default model on load
    demo.load(
        fn=initialize_default_model,
        outputs=[model_status]
    )
    
    # Model selection change handler
    model_dropdown.change(
        fn=load_sam_model,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    # Segmentation on click handler
    uploaded_image_input.select(
        fn=segment_on_click,
        inputs=[uploaded_image_input, model_dropdown],
        outputs=[segmented_output]
    )
    
    # Example image selection handler
    example_gallery.select(
        fn=select_example_image,
        outputs=[uploaded_image_input, segmented_output]
    )
    
    # Help button and close button handlers
    help_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=[help_section]
    )
    
    # Close help section button handler
    close_help_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[help_section]
    )

if __name__ == "__main__":
    demo.launch()