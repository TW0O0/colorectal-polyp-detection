import streamlit as st
import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image
import time
import matplotlib.pyplot as plt
import io

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.utils.visualization import overlay_mask
from src.data.data_utils import denormalize

# Set page config
st.set_page_config(
    page_title="Colorectal Polyp Detection",
    page_icon="🔬",
    layout="wide"
)

# Constants
MODEL_PATH = "models/best_model.pt"
IMAGE_SIZE = (256, 256)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Cache the model loading to avoid reloading with every rerun
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device
    else:
        return None, device

def preprocess_image(image, target_size):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
    img_tensor = torch.nn.functional.normalize(img_tensor, MEAN, STD)
    
    return img_tensor.unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    
    return prediction.cpu().numpy()

def main():
    # Sidebar
    st.sidebar.title("Colorectal Polyp Detection")
    st.sidebar.markdown("---")
    
    # Model loading
    model, device = load_model(MODEL_PATH)
    
    if model is None:
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        st.info("Please train the model first or update the model path.")
        return
    
    # Main content
    st.title("Colorectal Polyp Detection")
    st.markdown("""
    This application uses deep learning to detect colorectal polyps in images.
    Upload an image or use the camera to capture one.
    """)
    
    # Image upload
    upload_option = st.radio("Select input method:", ["Upload Image", "Use Camera"])
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            process_image(image, model, device)
    else:
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image = Image.open(camera_input)
            process_image(image, model, device)

def process_image(image, model, device):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Create a placeholder for the prediction
    with col2:
        st.subheader("Prediction")
        prediction_placeholder = st.empty()
    
    # Process the image
    with st.spinner('Processing image...'):
        start_time = time.time()
        
        # Preprocess
        image_tensor = preprocess_image(image, IMAGE_SIZE)
        
        # Predict
        mask = predict(model, image_tensor, device)[0, 0]
        
        # Create visualization
        original_np = np.array(image.resize(IMAGE_SIZE))
        overlay_image = overlay_mask(original_np, mask, alpha=0.5)
        
        end_time = time.time()
        
        # Display results
        prediction_placeholder.image(overlay_image, use_column_width=True)
        
        # Display metrics
        st.write(f"Processing time: {(end_time - start_time):.3f} seconds")
        
        # Calculate polyp area statistics
        polyp_area = np.sum(mask)
        total_area = mask.shape[0] * mask.shape[1]
        area_percentage = (polyp_area / total_area) * 100
        
        st.markdown(f"""
        **Polyp Detection Results:**
        - Polyp detected: {'Yes' if polyp_area > 0 else 'No'}
        - Polyp area: {polyp_area} pixels ({area_percentage:.2f}% of image)
        """)
        
        # Add download button for the prediction image
        buffered = io.BytesIO()
        Image.fromarray((overlay_image * 255).astype(np.uint8)).save(buffered, format="PNG")
        st.download_button(
            label="Download Result",
            data=buffered.getvalue(),
            file_name="polyp_detection_result.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
