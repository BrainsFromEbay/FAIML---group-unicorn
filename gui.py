import streamlit as st
import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Models
try:
    import Mahi.src.inference.cnn_mnist_inference as mahi_model
except ImportError as e:
    st.error(f"Failed to import Mahi's User module: {e}")

try:
    import Oyshe.prediction as oyshe_model
except ImportError as e:
    st.error(f"Failed to import Oyshe's User module: {e}")


try:
    import Jere.predict as jere_model
except ImportError as e:
    st.error(f"Failed to import Jere's User module: {e}")

try:
    import Mahi.src.inference.mlp_mnist_inference as mahi_mlp
except ImportError as e:
    st.error(f"Failed to import Mahi's MLP module: {e}")

try:
    import Mahi.src.inference.rf_mnist_inference as mahi_rf
except ImportError as e:
    st.error(f"Failed to import Mahi's RF module: {e}")


# Page Config
st.set_page_config(
    page_title="FAIML Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Title and Sidebar
st.title("🔢 FAIML Group Unicorn - Digit Recognition")
st.markdown("### Hand-written Digit Recognition System")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.radio(
        "Select Model",
        ("Mahi - CNN", "Mahi - MLP", "Mahi - Random Forest", "Oyshe - HOG + Logistic Regression", "Jere - CNN")
    )
    
    st.info(f"You selected: **{model_choice}**")
    
    st.markdown("---")
    st.write("Current Working Directory:", os.getcwd())

# Helper to load models (Lazy Loading)
@st.cache_resource
def load_mahi_model():
    return mahi_model.load_model()

@st.cache_resource
def load_oyshe_model():
    return oyshe_model.load_model()

@st.cache_resource
def load_jere_model():
    # Use Jere folder as base path
    return jere_model.load_model(base_path="Jere")

@st.cache_resource
def load_mahi_mlp_model():
    return mahi_mlp.load_model()

@st.cache_resource
def load_mahi_rf_model():
    return mahi_rf.load_model()


# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    
    upload_option = st.radio("Input Method", ["Upload Image", "Select from Folder"])
    
    image_path = None
    uploaded_file = None
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    else:
        # Allow user to input folder path directly
        folder_path = st.text_input("Enter folder path", value="", placeholder="e.g., custom_test")
        
        if folder_path:
            if os.path.isdir(folder_path):
                files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if files:
                    selected_file = st.selectbox("Select an image", files)
                    if selected_file:
                        image_path = os.path.join(folder_path, selected_file)
                else:
                    st.warning("No images found in this folder.")
            else:
                st.error(f"Directory '{folder_path}' not found.")
        else:
            st.info("Please enter a folder path to browse images.")



    # Display Image
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        # Save temp for compatibility with scripts that expect paths
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = "temp_image.png"
    elif image_path:
        image = Image.open(image_path)
        st.image(image, caption=f"Selected: {os.path.basename(image_path)}", width=250)

with col2:
    st.subheader("Prediction")
    
    if image_path and st.button("Predict Digit", type="primary"):
        with st.spinner(f"Running inference with {model_choice}..."):
            try:
                prediction = None
                confidence = 0.0
                
                if "Mahi - CNN" in model_choice:
                    model = load_mahi_model()
                    if model:
                        prediction, confidence = mahi_model.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's CNN model.")
                
                elif "Mahi - MLP" in model_choice:
                    model = load_mahi_mlp_model()
                    if model:
                         prediction, confidence = mahi_mlp.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's MLP model.")
                
                elif "Mahi - Random Forest" in model_choice:
                    model = load_mahi_rf_model()
                    if model:
                         prediction, confidence = mahi_rf.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's RF model.")
                        
                elif "Oyshe" in model_choice:

                    weights, bias = load_oyshe_model()
                    prediction, confidence = oyshe_model.predict_digit(image_path, weights, bias)
                    confidence = confidence * 100 # Adjust to percentage
                    
                elif "Jere" in model_choice:
                    model = load_jere_model()
                    # Jere's predict uses GPU/device logic internally, need to reference `jere_model.device`
                    prediction, confidence = jere_model.predict_image(image_path, model, jere_model.device, show_image=False)
                
                if prediction is not None:
                    st.success(f"**Predicted Digit:** {prediction}")
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    if confidence < 50:
                        st.warning("Low confidence alert!")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                import traceback
                st.text(traceback.format_exc())
                
                # Clean up temp file
                if uploaded_file and os.path.exists("temp_image.png"):
                    os.remove("temp_image.png")

# Performance Metrics Section
st.markdown("---")
st.header("Model Performance & Statistics")


if "Mahi - CNN" in model_choice:
    st.markdown("### Mahi's Model Results (CNN)")
    cm_path = "Mahi/results/cnn_mnist/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
        st.info(f"Confusion Matrix not found in path ({cm_path}).")

elif "Mahi - MLP" in model_choice:
    st.markdown("### Mahi's Model Results (MLP)")
    cm_path = "Mahi/results/mlp_mnist/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info(f"Confusion Matrix not found in path ({cm_path}).")

elif "Mahi - Random Forest" in model_choice:
    st.markdown("### Mahi's Model Results (Random Forest)")
    cm_path = "Mahi/results/rf_mnist/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info(f"Confusion Matrix not found in path ({cm_path}).")

elif "Oyshe" in model_choice:

    st.markdown("### Oyshe's Model Results (HOG + Logistic Regression)")
    # Oyshe likely saves to Oyshe/results or root results based on script
    cm_path_1 = "Oyshe/results/confusion_matrix.png"
    cm_path_2 = "results/confusion_matrix.png" 
    
    if os.path.exists(cm_path_1):
        st.image(cm_path_1, caption="Confusion Matrix", width=600)
    elif os.path.exists(cm_path_2):
        st.image(cm_path_2, caption="Confusion Matrix", width=600)
    else:
        st.info("Confusion Matrix not found.")

elif "Jere" in model_choice:
    st.markdown("### Jere's Model Results (CNN)")
    cm_path = "Jere/results/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
        st.info(f"Confusion Matrix not found in 'Jere/results/'. ({cm_path})")
