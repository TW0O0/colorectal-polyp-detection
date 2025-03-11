import streamlit as st
import os

def render_sidebar():
    """Render the sidebar for the Streamlit app."""
    
    st.sidebar.title("Controls & Settings")
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("Model")
    model_files = get_available_models()
    selected_model = st.sidebar.selectbox(
        "Select model",
        model_files,
        index=0 if model_files else None
    )
    
    # Prediction threshold
    threshold = st.sidebar.slider(
        "Prediction threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Threshold for binary segmentation"
    )
    
    # Visualization options
    st.sidebar.subheader("Visualization")
    
    overlay_alpha = st.sidebar.slider(
        "Overlay opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Opacity of the mask overlay"
    )
    
    show_contours = st.sidebar.checkbox(
        "Show contours",
        value=True,
        help="Display contours around polyps"
    )
    
    contour_color = st.sidebar.color_picker(
        "Contour color", 
        "#FF0000",
        help="Color for polyp contours"
    )
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This app demonstrates colorectal polyp detection using deep learning.
        
        The model is trained on the Kvasir-SEG dataset.
        """
    )
    
    return {
        "model": selected_model,
        "threshold": threshold,
        "overlay_alpha": overlay_alpha,
        "show_contours": show_contours,
        "contour_color": contour_color
    }

def get_available_models():
    """Get a list of available model files."""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        return []
    
    model_files = [
        f for f in os.listdir(models_dir) 
        if f.endswith((".pt", ".pth")) and os.path.isfile(os.path.join(models_dir, f))
    ]
    
    return model_files
