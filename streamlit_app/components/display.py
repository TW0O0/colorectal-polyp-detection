import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

def display_results(original_image, prediction_mask, settings):
    """
    Display prediction results.
    
    Args:
        original_image (numpy.ndarray): Original input image
        prediction_mask (numpy.ndarray): Predicted binary mask
        settings (dict): Display settings from sidebar
    """
    col1, col2 = st.columns(2)
    
    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)
    
    # Display prediction
    with col2:
        st.subheader("Polyp Detection")
        
        # Create overlay
        overlay_image = create_overlay(
            original_image, 
            prediction_mask, 
            alpha=settings["overlay_alpha"],
            show_contours=settings["show_contours"],
            contour_color=hex_to_rgb(settings["contour_color"])
        )
        
        st.image(overlay_image, use_column_width=True)
    
    # Display statistics
    display_statistics(prediction_mask)
    
    # Add download button
    provide_download(original_image, prediction_mask, overlay_image)

def create_overlay(image, mask, alpha=0.5, show_contours=True, contour_color=(255, 0, 0)):
    """
    Create an overlay of the mask on the image.
    
    Args:
        image (numpy.ndarray): Original image
        mask (numpy.ndarray): Binary mask
        alpha (float): Opacity of the overlay
        show_contours (bool): Whether to show contours
        contour_color (tuple): RGB color for contours
        
    Returns:
        numpy.ndarray: Overlay image
    """
    # Make sure mask is binary
    binary_mask = mask > 0
    
    # Create a colored mask (red for polyps)
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask] = [255, 0, 0]  # Red color for polyps
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    # Add contours if requested
    if show_contours:
        # Convert to uint8 for contour finding
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, contour_color, 2)
    
    return overlay

def display_statistics(mask):
    """
    Display statistics about the predicted mask.
    
    Args:
        mask (numpy.ndarray): Binary prediction mask
    """
    # Calculate polyp statistics
    polyp_area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    area_percentage = (polyp_area / total_area) * 100
    
    # Find polyp regions
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_polyps = len(contours)
    
    st.subheader("Detection Results")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Polyps Detected", num_polyps)
    
    with col2:
        st.metric("Polyp Area", f"{polyp_area} pixels")
    
    with col3:
        st.metric("Area Percentage", f"{area_percentage:.2f}%")
    
    # Additional information if polyps are detected
    if num_polyps > 0:
        st.markdown("### Polyp Regions")
        
        # Get information about each polyp
        polyp_info = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity: 4*pi*area/perimeter^2
            # A perfect circle has circularity of 1
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            polyp_info.append({
                "id": i + 1,
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity
            })
        
        # Create a dataframe and display as a table
        if polyp_info:
            st.table(polyp_info)

def provide_download(original, mask, overlay):
    """
    Provide download buttons for the results.
    
    Args:
        original (numpy.ndarray): Original image
        mask (numpy.ndarray): Binary prediction mask
        overlay (numpy.ndarray): Overlay image
    """
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    # Prepare mask image
    mask_image = (mask * 255).astype(np.uint8)
    mask_colored = np.zeros((*mask_image.shape, 3), dtype=np.uint8)
    mask_colored[mask_image > 0] = [255, 0, 0]  # Red color for polyps
    
    # Convert to bytes for download
    def get_image_bytes(img, format="PNG"):
        buffered = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buffered, format=format)
        return buffered.getvalue()
    
    # Download buttons
    with col1:
        st.download_button(
            "Download Mask",
            data=get_image_bytes(mask_colored),
            file_name="polyp_mask.png",
            mime="image/png"
        )
    
    with col2:
        st.download_button(
            "Download Overlay",
            data=get_image_bytes
